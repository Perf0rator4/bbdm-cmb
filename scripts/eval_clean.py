"""Оценка уже обученной модели на НЕВИДЕННЫХ патчах (и, для сравнения, на утёкших).

Run 5 учился со сплитом по тайлам, поэтому большинство его val/test-патчей --
копии train-патчей. Этот скрипт берёт патчи val+test старого сплита и делит их на

  clean  -- не пересекаются по пикселям ни с одним train-патчем: метрики честные;
  leaked -- совпадают/пересекаются с train-патчем: метрики завышены.

и считает TF / r_ell / r_in / PSNR отдельно на каждой группе. Если r_ell в полосе
0.10-0.30 на clean заметно ниже, чем на leaked, то часть сигнала там --
запоминание, а не реконструкция.

    python scripts/eval_clean.py --ckpt runs/run5/best.pt --freq 90 \\
        --data-root /path/to/Diffusion [--mu -15.17 --sigma 113.47]

Нормировка (mu, sigma) берётся из чекпоинта, если она там есть ("extra"); иначе
из --mu/--sigma или пересчитывается по train-тайлам старого сплита (как делал ноутбук).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bbdm import config as cfg  # noqa: E402
from bbdm.data import (  # noqa: E402
    CMBPatchDataset,
    clean_eval_positions,
    compute_normalization,
    get_tile_splits,
    load_splits,
)
from bbdm.evaluate import evaluate_spectra, print_band_table, spread_indices  # noqa: E402
from bbdm.train import load_checkpoint  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--freq", type=int, default=cfg.FREQ_MASTER, choices=sorted(cfg.FREQ_PAIRS))
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits-json", default=None,
                    help="старый сплит по тайлам (JSON с train/val/test); иначе "
                         "get_tile_splits(папка Planck, 0.8, 0.1, 42), как у run 5")
    ap.add_argument("--mu", type=float, default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--n-patches", type=int, default=50, help="патчей на группу")
    ap.add_argument("--eval-S", type=int, default=cfg.S)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--patch-size", type=int, default=cfg.PATCH_SIZE)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="куда записать JSON с результатами")
    a = ap.parse_args(argv)

    planck_dir, act_dir = cfg.data_dirs(a.freq, a.data_root)
    if a.splits_json:
        tr, va, te = load_splits(a.splits_json)
    else:
        tr, va, te = get_tile_splits(planck_dir, cfg.TRAIN_RATIO, cfg.VAL_RATIO, cfg.SEED)

    bbdm, ck = load_checkpoint(a.ckpt, device=a.device)
    extra = ck.get("extra") or {}
    mu, sigma = a.mu, a.sigma
    if mu is None or sigma is None:
        if "mu" in extra:
            mu, sigma = extra["mu"], extra["sigma"]
        else:
            print("mu/sigma not in checkpoint: recomputing from the legacy train tiles "
                  "(like the notebook did)")
            mu, sigma = compute_normalization(planck_dir, act_dir, tr, patch_size=a.patch_size)
    print(f"checkpoint epoch {ck['epoch'] + 1}; mu={mu:.4f} sigma={sigma:.4f}")

    eval_tiles = sorted(set(va) | set(te))
    clean, leaked = clean_eval_positions(tr, eval_tiles, a.patch_size)
    print(f"val+test patches of the legacy split: {len(clean) + len(leaked)} unique -> "
          f"{len(clean)} clean (never seen), {len(leaked)} leaked (share pixels with a train patch)")

    common = dict(planck_dir=planck_dir, act_dir=act_dir, mu=mu, sigma=sigma,
                  patch_size=a.patch_size, max_zero_frac=cfg.MAX_ZERO_FRAC,
                  max_mask_mismatch_frac=cfg.MAX_MASK_MISMATCH_FRAC)
    results = {"ckpt": a.ckpt, "epoch": ck["epoch"] + 1, "S": a.eval_S}
    for name, positions in (("clean", clean), ("leaked", leaked)):
        if not positions:
            print(f"\n{name}: no patches")
            continue
        ds = CMBPatchDataset(tile_list=eval_tiles, augment=False, keep_positions=positions, **common)
        n = min(a.n_patches, len(ds))
        res = evaluate_spectra(bbdm, ds, mu, sigma, indices=spread_indices(len(ds), n), S=a.eval_S,
                               device=a.device, batch_size=min(a.batch_size, n), progress=False,
                               with_psnr=True)
        print_band_table(res, f"{name} ({res['n_patches']} patches)")
        print(f"  PSNR {res['psnr_mean']:.2f} ± {res['psnr_std']:.2f} dB (n={res['psnr_n']})")
        results[name] = {
            "n_patches": res["n_patches"], "bands": res["bands"],
            "tf_bands": res["tf_bands"], "r_bands": res["r_bands"],
            "r_input_bands": res["r_input_bands"], "psnr_mean": res["psnr_mean"],
            "psnr_std": res["psnr_std"],
            "r_bands_per_patch_std": res["r_bands_per_patch_std"],
        }

    if "clean" in results and "leaked" in results:
        print("\nr_ell, clean vs leaked (a drop on clean means the signal was partly memorised):")
        for (lo, hi), rc, rl in zip(results["clean"]["bands"], results["clean"]["r_bands"],
                                    results["leaked"]["r_bands"]):
            print(f"  {lo:.3f}-{hi:.2f}   clean {rc:7.3f}   leaked {rl:7.3f}   diff {rc - rl:+.3f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
