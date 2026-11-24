"""CFA detection on exported, aligned three-frequency BBDM test maps.

ACT detections measure agreement with the detector, not cluster truth. An
optional sky catalogue supplies a separate reference. Only non-overlapping
windows are used; thresholds must be chosen on validation before testing D56.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS


def _inside_mask(points, mask):
    points = np.asarray(points)
    x, y = np.rint(points).astype(int).T
    inside = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    keep = np.zeros(len(points), dtype=bool)
    keep[inside] = mask[y[inside], x[inside]]
    return keep


def _sky(points, wcs):
    ra, dec = wcs.all_pix2world(points[:, 0], points[:, 1], 0)
    return np.column_stack((ra, dec))


def _counts(matched, n_detected, n_reference):
    return dict(n_detected=n_detected, n_reference=n_reference, matched=matched,
                missed=n_reference - matched, unmatched=n_detected - matched,
                recall=matched / n_reference if n_reference else None,
                matched_fraction=matched / n_detected if n_detected else None)


def match_detections(points, scores, reference, radius_arcmin):
    """Score-ordered one-to-one sky matching; empty denominators stay undefined."""
    matched = 0
    available = np.ones(len(reference), dtype=bool)
    if len(points) and len(reference):
        detected = SkyCoord(points[:, 0] * u.deg, points[:, 1] * u.deg)
        truth = SkyCoord(reference[:, 0] * u.deg, reference[:, 1] * u.deg)
        for i in np.argsort(-np.asarray(scores), kind="stable"):
            distances = detected[i].separation(truth).arcmin
            distances[~available] = np.inf
            j = np.argmin(distances)
            if distances[j] <= radius_arcmin:
                available[j] = False
                matched += 1
    return _counts(matched, len(points), len(reference))


def evaluate_detection(test_set, detector, freq_order, *, threshold=0.1,
                       match_radius_arcmin=2.4, edge_margin=8, catalog=None,
                       ra_column="RADeg", dec_column="decDeg"):
    """Evaluate ACT, Planck, every BBDM sample and the mean of physical maps.

    ``test_set`` is the output of scripts/make_test_set.py. ``freq_order``
    explicitly states the detector's channel order. ``catalog`` is an astropy
    Table with sky coordinates in degrees. Images are processed one at a time.
    No diffusion sampling or training happens here.
    """
    root = Path(test_set)
    if len(freq_order) != 3 or set(freq_order) != {90, 150, 220}:
        raise ValueError("freq_order must contain 90, 150 and 220 exactly once")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if not np.isfinite(match_radius_arcmin) or match_radius_arcmin <= 0:
        raise ValueError("match_radius_arcmin must be positive and finite")
    with (root / "manifest.json").open() as fh:
        manifest = json.load(fh)
    n_samples, p = manifest["n_samples"], manifest["patch_size"]
    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("manifest n_samples must be a positive integer")
    if not isinstance(edge_margin, int) or edge_margin < 0 or 2 * edge_margin >= p:
        raise ValueError("edge_margin must be a nonnegative integer smaller than half a patch")
    with (root / "index.csv").open(newline="") as fh:
        windows = [row for row in csv.DictReader(fh) if row["independent"] == "1"]
    if not windows:
        raise ValueError("No independent windows in index.csv")
    for i, row in enumerate(windows):
        y, x = int(row["y"]), int(row["x"])
        for other in windows[:i]:
            if abs(y - int(other["y"])) < p and abs(x - int(other["x"])) < p:
                raise ValueError("Independent windows overlap; detection counts would be duplicated")

    cat_sky = None
    if catalog is not None:
        cat_sky = np.column_stack((np.asarray(catalog[ra_column], dtype=float),
                                   np.asarray(catalog[dec_column], dtype=float)))
        if not np.isfinite(cat_sky).all() or np.any(np.abs(cat_sky[:, 1]) > 90):
            raise ValueError("Catalogue coordinates must be finite RA/Dec in degrees")

    rows, detections = [], []
    for window in windows:
        name = window["name"]
        wcs = None

        def load(kind, suffix=""):
            nonlocal wcs
            channels = []
            for freq in freq_order:
                path = root / f"f{freq}" / kind / f"{name}{suffix}.fits"
                data, header = fits.getdata(path, header=True)
                if data.shape != (p, p) or not np.isfinite(data).all():
                    raise ValueError(f"{path}: expected a finite {p}x{p} map")
                if header.get("BUNIT") != "uK":
                    raise ValueError(f"{path}: expected BUNIT='uK', not normalized diffusion inputs")
                current = WCS(header)
                if not current.has_celestial:
                    raise ValueError(f"{path}: missing celestial WCS")
                if wcs is None:
                    wcs = current
                xy = np.array([0, 0, p - 1, p - 1, (p - 1) / 2])
                yy = np.array([0, p - 1, 0, p - 1, (p - 1) / 2])
                ra, dec = current.all_pix2world(xy, yy, 0)
                rx, ry = wcs.all_world2pix(ra, dec, 0)
                if not np.isfinite([rx, ry]).all() or max(np.max(abs(rx - xy)), np.max(abs(ry - yy))) > 0.01:
                    raise ValueError(f"{path}: WCS does not align with the other maps")
                channels.append(np.asarray(data, dtype=np.float32))
            return np.stack(channels)

        act, planck = load("true"), load("planck")
        valid = np.all((act != 0) & (planck != 0), axis=0)
        if edge_margin:
            invalid = torch.from_numpy((~valid).astype(np.float32))[None, None]
            valid = F.max_pool2d(invalid, 2 * edge_margin + 1, stride=1,
                                 padding=edge_margin)[0, 0].numpy() == 0
            valid[:edge_margin] = valid[-edge_margin:] = False
            valid[:, :edge_margin] = valid[:, -edge_margin:] = False
        if not valid.any():
            raise ValueError(f"{name}: no common valid interior pixels")
        references = {}
        if cat_sky is not None:
            cx, cy = wcs.all_world2pix(cat_sky[:, 0], cat_sky[:, 1], 0)
            finite = np.isfinite(cx) & np.isfinite(cy)
            candidate = cat_sky[finite]
            references["catalog"] = candidate[_inside_mask(np.column_stack((cx[finite], cy[finite])), valid)]

        samples = [load("pred", f"_s{k}") for k in range(n_samples)]
        variants = [("act", act), ("planck", planck)]
        variants += [(f"bbdm_s{k}", image) for k, image in enumerate(samples)]
        variants.append(("bbdm_mean", np.mean(samples, axis=0, dtype=np.float64).astype(np.float32)))
        for variant, images in variants:
            heatmaps, points, scores = detector.predict(images, batch_size=1, threshold=threshold)
            heatmap, points, scores = heatmaps[0], points[0], scores[0]
            if heatmap.shape != valid.shape or not np.isfinite(heatmap).all():
                raise ValueError(f"{name}/{variant}: detector returned an invalid heatmap")
            keep = _inside_mask(points, valid)
            points, scores = points[keep], scores[keep]
            sky = _sky(points, wcs)
            if variant == "act":
                references["act"] = sky
                act_heatmap = heatmap
            row = dict(name=name, variant=variant, valid_pixels=int(valid.sum()),
                       heatmap_mae_to_act=float(np.mean(abs(heatmap[valid] - act_heatmap[valid]))))
            for label, reference in references.items():
                counts = match_detections(sky, scores, reference, match_radius_arcmin)
                row.update({f"{label}_{key}": value for key, value in counts.items()})
            rows.append(row)
            for (x, y), (ra, dec), score in zip(points, sky, scores):
                detections.append(dict(name=name, variant=variant, x=float(x), y=float(y),
                                       ra=float(ra), dec=float(dec), score=float(score)))

    summary = {}
    for variant, _ in variants:
        selected = [row for row in rows if row["variant"] == variant]
        total_pixels = sum(row["valid_pixels"] for row in selected)
        result = dict(n_windows=len(selected), valid_pixels=total_pixels,
                      heatmap_mae_to_act=sum(row["heatmap_mae_to_act"] * row["valid_pixels"]
                                             for row in selected) / total_pixels)
        for label in references:
            totals = [sum(row[f"{label}_{key}"] for row in selected)
                      for key in ("matched", "n_detected", "n_reference")]
            result.update({f"{label}_{key}": value for key, value in _counts(*totals).items()})
        summary[variant] = result
    return dict(summary=summary, windows=rows, detections=detections,
                settings=dict(freq_order=list(freq_order), threshold=threshold,
                              match_radius_arcmin=match_radius_arcmin, edge_margin=edge_margin,
                              n_samples=n_samples, catalog_used=catalog is not None),
                notes=["ACT metrics measure agreement with CFA detections, not cluster truth.",
                       "Catalogue-unmatched detections are not necessarily false positives.",
                       "Planck 100/143/217 GHz differs from ACT 90/150/220 GHz; this baseline has domain shift.",
                       "Channel order and units must match CFA training; sky matching uses each window WCS.",
                       "CFA scores are not calibrated probabilities; its peak neighbourhood is 9x9 pixels."])
