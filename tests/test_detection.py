"""Detection metric on tiny synthetic FITS maps; no trained networks or sampling."""

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bbdm.metrics.detection import evaluate_detection, match_detections
from cfa_unet_inference import CFAUNetInference


class PeakDetector:
    def __init__(self):
        self.inputs = []

    def predict(self, images, batch_size=None, threshold=0.1):
        assert images.shape == (3, 32, 32) and batch_size == 1
        self.inputs.append(images.copy())
        pred = torch.from_numpy(np.maximum((images[:1] - 10) / 20, 0))[None]
        points, scores = CFAUNetInference._extract_points(pred, threshold)
        return pred[:, 0].numpy(), points, scores


def export_set(root, pixel_arcmin=0.5, pred_x=10):
    root.mkdir()
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [359.99, -1]
    wcs.wcs.crpix = [16, 16]
    wcs.wcs.cdelt = [-pixel_arcmin / 60, pixel_arcmin / 60]
    header = wcs.to_header()
    header["BUNIT"] = "uK"
    base = np.full((32, 32), 10, dtype=np.float32)
    act, planck, s0, s1 = [base.copy() for _ in range(4)]
    act[10, 10] = 30
    s0[10, pred_x] = s1[10, pred_x] = 30
    s0[20, 24] = 30
    act[24, 15] = planck[24, 15] = 0
    s0[24, 15] = s1[24, 15] = 30  # a predicted source inside the input/target mask
    for freq in (90, 150, 220):
        for kind, suffix, image in [("true", "", act), ("planck", "", planck),
                                     ("pred", "_s0", s0), ("pred", "_s1", s1)]:
            directory = root / f"f{freq}" / kind
            directory.mkdir(parents=True, exist_ok=True)
            fits.writeto(directory / f"win_y0_x0{suffix}.fits", image, header)
    (root / "manifest.json").write_text(json.dumps({"n_samples": 2, "patch_size": 32}))
    with (root / "index.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "y", "x", "independent"])
        writer.writeheader()
        writer.writerows([dict(name="win_y0_x0", y=0, x=0, independent=1),
                          dict(name="overlapping_unused", y=0, x=16, independent=0)])
    ra, dec = wcs.all_pix2world([10, 18, 60], [10, 10, 60], 0)
    return Table({"RADeg": ra, "decDeg": dec})


def test_detection_catalogue_act_planck_samples_and_physical_mean(tmp_path):
    root = tmp_path / "set"
    catalogue = export_set(root)
    detector = PeakDetector()
    result = evaluate_detection(root, detector, (90, 150, 220), threshold=0.75,
                                match_radius_arcmin=0.6, edge_margin=1, catalog=catalogue)
    summary = result["summary"]
    assert summary["act"]["catalog_n_reference"] == 2  # outside-crop object was excluded
    assert summary["act"]["catalog_recall"] == 0.5      # calibrates the detector's own miss
    assert summary["planck"]["catalog_recall"] == 0
    assert summary["planck"]["catalog_matched_fraction"] is None
    assert summary["bbdm_s0"]["act_n_detected"] == 2
    assert summary["bbdm_s0"]["act_matched_fraction"] == 0.5
    assert summary["bbdm_mean"]["act_n_detected"] == 1
    assert summary["bbdm_mean"]["act_recall"] == 1
    assert summary["bbdm_mean"]["catalog_recall"] == 0.5
    assert len(detector.inputs) == 5  # ACT, Planck, two samples, mean; independent windows only
    assert detector.inputs[-1][0, 20, 24] == 20  # averaged uK maps before detector preprocessing
    assert all(not (p["x"] == 15 and p["y"] == 24) for p in result["detections"])
    json.dumps(result, allow_nan=False)


def test_sky_matching_is_one_to_one_and_handles_ra_wrap_and_empty_references():
    points = np.array([[359.999, 0], [0.001, 0]])
    matched = match_detections(points, np.array([0.8, 0.9]), np.array([[0, 0]]), 0.2)
    assert matched["matched"] == 1 and matched["unmatched"] == 1
    empty = match_detections(np.empty((0, 2)), np.empty(0), np.empty((0, 2)), 1)
    assert empty["recall"] is None and empty["matched_fraction"] is None


def test_matching_uses_actual_wcs_scale(tmp_path):
    for pixel_arcmin, expected in [(0.5, 1), (2.0, 0)]:
        root = tmp_path / str(pixel_arcmin)
        export_set(root, pixel_arcmin=pixel_arcmin, pred_x=11)
        report = evaluate_detection(root, PeakDetector(), (90, 150, 220),
                                    threshold=0.75, match_radius_arcmin=1, edge_margin=1)
        assert report["summary"]["bbdm_mean"]["act_matched"] == expected


@pytest.mark.parametrize("change", ["wcs", "unit", "missing", "overlap"])
def test_bad_export_is_rejected(tmp_path, change):
    root = tmp_path / "set"
    export_set(root)
    path = root / "f150" / "pred" / "win_y0_x0_s1.fits"
    if change in ("wcs", "unit"):
        with fits.open(path, mode="update") as hdus:
            hdus[0].header["CRVAL1" if change == "wcs" else "BUNIT"] = 10 if change == "wcs" else "K"
    elif change == "missing":
        path.unlink()
    else:
        path = root / "index.csv"
        path.write_text(path.read_text().replace("overlapping_unused,0,16,0", "overlapping_unused,0,16,1"))
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluate_detection(root, PeakDetector(), (90, 150, 220), edge_margin=1)


def test_configurable_cfa_threshold_and_input_validation():
    pred = torch.zeros((1, 1, 8, 8))
    pred[0, 0, 3, 4] = 0.05
    points, _ = CFAUNetInference._extract_points(pred, threshold=0.04)
    assert points[0].tolist() == [[4, 3]]
    assert len(CFAUNetInference._extract_points(pred)[0][0]) == 0
    detector = CFAUNetInference.__new__(CFAUNetInference)
    for threshold in [-1, 2, np.nan]:
        with pytest.raises(ValueError, match="threshold"):
            detector.predict(np.zeros((3, 8, 8)), threshold=threshold)

    class ToyNetwork(torch.nn.Module):
        def forward(self, x):
            out = x.new_zeros((len(x), 1, 8, 8))
            out[:, 0, 3, 4] = 0.05
            return out

    detector.device, detector.model = torch.device("cpu"), ToyNetwork()
    image = np.random.default_rng(0).normal(size=(3, 8, 8)).astype(np.float32)
    assert detector.predict(image)[1][0].shape == (0, 2)  # unchanged old API
    heatmaps, points, _ = detector.predict(image, batch_size=1, threshold=0.04)
    assert heatmaps.shape == (1, 8, 8) and points[0].tolist() == [[4, 3]]


def test_empty_act_does_not_become_perfect_recall(tmp_path):
    root = tmp_path / "set"
    catalogue = export_set(root)
    for path in root.glob("f*/true/*.fits"):
        with fits.open(path, mode="update") as hdus:
            hdus[0].data[:] = 10
    report = evaluate_detection(root, PeakDetector(), (90, 150, 220), threshold=0.75,
                                edge_margin=1, catalog=catalogue)
    assert report["summary"]["act"]["act_recall"] is None
    assert report["summary"]["act"]["catalog_recall"] == 0
    assert report["summary"]["bbdm_mean"]["act_recall"] is None


def test_cli_writes_reports_without_loading_real_weights(tmp_path, monkeypatch):
    import eval_detection

    root = tmp_path / "set"
    catalogue = export_set(root)
    cat_path = tmp_path / "clusters.csv"
    catalogue.write(cat_path)
    monkeypatch.setattr(eval_detection, "CFAUNetInference", lambda *a, **kw: PeakDetector())
    args = ["--test-set", str(root), "--weights", "unused.pt", "--freq-order", "90", "150", "220",
            "--catalog", str(cat_path), "--edge-margin", "1", "--threshold", "0.75"]
    eval_detection.main(args)
    report = json.loads((root / "detection" / "summary.json").read_text())
    assert report["summary"]["planck"]["catalog_recall"] == 0
    assert len(list(csv.DictReader((root / "detection" / "per_window.csv").open()))) == 5
    assert len(list(csv.DictReader((root / "detection" / "detections.csv").open()))) == 5
    with pytest.raises(FileExistsError):
        eval_detection.main(args)
