"""Тестовая выборка для метрики детекции: окна внутри тестового поля, общие для всех частот.

    python scripts/make_test_set.py --data-root /path/to/Diffusion \\
        --runs runs/d56_f90 runs/d56_f150 runs/d56_f220 --out-dir test_set_d56

Окна P x P идут с шагом `--stride` (по умолчанию P/2) по прямоугольнику тестового поля
(`box_px` из splits.json прогонов, для D56 это 6 x 23 = 138 окон). Train-кропы в поле не
заходят (`forbidden_rects`), поэтому ни одно окно модель не видела. Окна перекрываются; в
index.csv `independent = 1` у непересекающегося подмножества (шаг P).

Окно берётся, только если оно лежит в тайле, который есть во всех частотах, и проходит
фильтры нулей / несовпадения масок во ВСЕХ частотах, поэтому набор окон одинаков у всех моделей.

Выход: out-dir/f{freq}/{planck,true,pred}/win_y{Y}_x{X}[_s{k}].fits, по одному HDU, в мкК,
с WCS окна (срез WCS тайла). Плюс index.csv и manifest.json.
"""

import argparse
import csv
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS

from bbdm import config as cfg
from bbdm.data import RandomCropDataset, check_tile_wcs, list_tiles, load_patch_splits
from bbdm.data.splits import parse_tile_pos
from bbdm.sample import sample_batch


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--runs", nargs="+", required=True, help="папки прогонов train_freq.py, по одной на частоту")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--stride", type=int, default=None, help="шаг окон, px (по умолчанию P/2)")
    p.add_argument("--S", type=int, default=200, help="шагов сэмплера")
    p.add_argument("--n-samples", type=int, default=1, help="сэмплов BBDM на окно")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", choices=["metric", "best"], default="metric",
                   help="best_metric.pt (если есть) или best.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_runs(runs, ckpt_kind, device):
    """{freq: (bbdm, mu, sigma, ckpt_path)}, плюс общий (box, patch_size, test positions)."""
    from bbdm.train import load_checkpoint

    models, geom = {}, None
    for run in runs:
        sp_path = os.path.join(run, "splits.json")
        with open(sp_path) as f:
            meta = json.load(f)
        if meta.get("mode") != "region":
            raise RuntimeError(f"{sp_path}: split mode {meta.get('mode')!r}, а не 'region' -- окна "
                               "в тестовом поле могли быть в train этой модели")
        sp, ps = load_patch_splits(sp_path)
        g = (tuple(meta["box_px"]), ps, meta.get("guard_px"))
        if geom is not None and g != geom:
            raise RuntimeError(f"{run}: поле/патч {g} не совпадает с другими прогонами {geom}")
        geom = g
        path = os.path.join(run, "best_metric.pt")
        if ckpt_kind == "best" or not os.path.exists(path):
            path = os.path.join(run, "best.pt")
        bbdm, ck = load_checkpoint(path, device=device)
        extra = ck.get("extra") or {}
        freq = extra["freq"]
        if freq in models:
            raise RuntimeError(f"две модели на {freq} ГГц: {models[freq][3]} и {path}")
        models[freq] = (bbdm, float(extra["mu"]), float(extra["sigma"]), path)
        print(f"{freq} GHz: {path} (epoch {ck['epoch'] + 1}, mu {extra['mu']:.3f}, "
              f"sigma {extra['sigma']:.3f}), field {meta.get('region')}", flush=True)
    return models, geom


def window_grid(box, p, stride):
    y0, y1, x0, x1 = box
    return [(y, x) for y in range(y0, y1 - p + 1, stride) for x in range(x0, x1 - p + 1, stride)]


def build_windows(data_root, freqs, box, p, stride):
    """(окна, {freq: RandomCropDataset}, отчёт): окна, прошедшие фильтры во всех частотах."""
    dirs = {f: cfg.data_dirs(f, data_root) for f in freqs}
    common = set.intersection(*(set(list_tiles(pd)) & set(list_tiles(ad)) for pd, ad in dirs.values()))
    if not common:
        raise RuntimeError("нет тайлов, общих для всех частот")
    readers = {f: RandomCropDataset(pd, ad, sorted(common), mu=0.0, sigma=1.0, forbidden=[],
                                    n_samples=0, patch_size=p, augment=False)
               for f, (pd, ad) in dirs.items()}
    r0 = readers[freqs[0]]
    th, tw = r0.tile_size
    tile_idx = {(ty, tx): i for i, (ty, tx) in enumerate(r0.tile_pos)}
    tys = sorted({ty for ty, _ in tile_idx})
    txs = sorted({tx for _, tx in tile_idx})

    windows, report = [], {"grid": 0, "no_common_tile": 0, **{f"filtered_{f}": 0 for f in freqs}}
    for y, x in window_grid(box, p, stride):
        report["grid"] += 1
        i = next((tile_idx[(ty, tx)] for ty in tys if ty <= y <= ty + th - p
                  for tx in txs if tx <= x <= tx + tw - p and (ty, tx) in tile_idx), None)
        if i is None:
            report["no_common_tile"] += 1
            continue
        ty, tx = r0.tile_pos[i]
        ok = True
        for f, r in readers.items():
            pp, pa = r.read_crop(i, y - ty, x - tx)
            vp, va = pp != 0, pa != 0
            if ((1 - vp.mean()) > cfg.MAX_ZERO_FRAC or (1 - va.mean()) > cfg.MAX_ZERO_FRAC
                    or np.logical_xor(vp, va).mean() > cfg.MAX_MASK_MISMATCH_FRAC):
                report[f"filtered_{f}"] += 1
                ok = False
        if ok:
            windows.append({"y": y, "x": x, "tile_index": i, "oy": y - ty, "ox": x - tx,
                            "tile": r0.tiles[i][0]})
    return windows, readers, report


def window_header(tile_header, oy, ox, p, y, x):
    hdr = WCS(tile_header)[oy:oy + p, ox:ox + p].to_header()
    chk = hdr.copy()
    chk["NAXIS1"], chk["NAXIS2"] = p, p
    dev = check_tile_wcs(chk, f"tile_y{y}_x{x}.fits")
    if dev > 1.0:
        raise RuntimeError(f"WCS окна y{y} x{x} расходится с сеткой карты на {dev:.2f} px")
    return hdr


def main(argv=None):
    args = parse_args(argv)
    models, (box, p, guard) = load_runs(args.runs, args.ckpt, args.device)
    freqs = sorted(models)
    stride = args.stride or p // 2
    windows, readers, report = build_windows(args.data_root, freqs, box, p, stride)
    print(f"Windows: {len(windows)} kept of {report['grid']} on the grid ({report})", flush=True)
    if len(windows) < 100:
        print(f"!! only {len(windows)} windows (< 100): lower --stride", flush=True)
    if not windows:
        raise RuntimeError("ни одного окна")

    for w in windows:
        w["independent"] = int((w["y"] - box[0]) % p == 0 and (w["x"] - box[2]) % p == 0)
        r0 = readers[freqs[0]]
        w["hdr"] = window_header(fits.getheader(os.path.join(r0.act_dir, w["tile"])),
                                 w["oy"], w["ox"], p, w["y"], w["x"])
        w["ra"], w["dec"] = (float(v) for v in WCS(w["hdr"]).wcs_pix2world((p - 1) / 2, (p - 1) / 2, 0))

    os.makedirs(args.out_dir, exist_ok=True)
    for f in freqs:
        bbdm, mu, sigma, ckpt = models[f]
        r = readers[f]
        for sub in ("planck", "true", "pred"):
            os.makedirs(os.path.join(args.out_dir, f"f{f}", sub), exist_ok=True)

        def write(sub, name, data, w, **cards):
            h = w["hdr"].copy()
            h["BUNIT"] = "uK"
            h["FREQ"] = (f, "GHz, ACT channel of this model")
            h["SRCTILE"] = w["tile"]
            for k, v in cards.items():
                h[k] = v
            fits.PrimaryHDU(np.asarray(data, dtype=np.float32), header=h).writeto(
                os.path.join(args.out_dir, f"f{f}", sub, name), overwrite=True)

        for b in range(0, len(windows), args.batch_size):
            batch = windows[b:b + args.batch_size]
            crops = [r.read_crop(w["tile_index"], w["oy"], w["ox"]) for w in batch]
            x0 = torch.from_numpy(np.stack([(pp - mu) / sigma for pp, _ in crops])[:, None]).float()
            for k in range(args.n_samples):
                seed = args.seed + 100003 * k + b
                pred = sample_batch(bbdm, x0, mu, sigma, S=args.S, device=args.device, seed=seed)
                for w, pr in zip(batch, pred):
                    write("pred", f"win_y{w['y']}_x{w['x']}_s{k}.fits", pr, w,
                          CKPT=os.path.basename(ckpt), STEPS=args.S, SEED=seed, SAMPLE=k)
            for w, (pp, pa) in zip(batch, crops):
                write("planck", f"win_y{w['y']}_x{w['x']}.fits", pp, w)
                write("true", f"win_y{w['y']}_x{w['x']}.fits", pa, w)
            print(f"  {f} GHz: {min(b + args.batch_size, len(windows))}/{len(windows)}", flush=True)

    with open(os.path.join(args.out_dir, "index.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["name", "y", "x", "ra_center", "dec_center", "independent", "tile", "oy", "ox"])
        for w in windows:
            wr.writerow([f"win_y{w['y']}_x{w['x']}", w["y"], w["x"], f"{w['ra']:.5f}",
                         f"{w['dec']:.5f}", w["independent"], w["tile"], w["oy"], w["ox"]])
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as fh:
        json.dump({"box_px": list(box), "guard_px": guard, "patch_size": p, "stride": stride,
                   "S": args.S, "n_samples": args.n_samples, "seed": args.seed,
                   "n_windows": len(windows), "n_independent": sum(w["independent"] for w in windows),
                   "window_report": report,
                   "models": {str(f): {"checkpoint": models[f][3], "sha256_16": _sha(models[f][3]),
                                       "mu": models[f][1], "sigma": models[f][2]} for f in freqs}},
                  fh, indent=2)
    print(f"Wrote {len(windows)} windows x {len(freqs)} frequencies to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
