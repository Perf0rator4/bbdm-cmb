"""Export a joint checkpoint to the same FITS layout consumed by eval_detection.py."""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS

from bbdm.sample import sample_batch
from bbdm.train import load_checkpoint
from make_test_set import _sha, build_windows, window_header


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-samples", type=int, default=1)
    p.add_argument("--S", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    out, run = Path(args.out_dir), Path(args.run)
    if out.exists():
        raise FileExistsError("Choose a new out-dir; exported FITS are never overwritten")
    if args.n_samples < 1 or args.S < 1:
        raise ValueError("n-samples and S must be positive")
    meta = json.loads((run / "splits.json").read_text())
    if meta.get("mode") != "region":
        raise ValueError("Joint export requires a held-out region split")
    ckpt = run / "best_metric.pt"
    if not ckpt.exists():
        ckpt = run / "best.pt"
    model, saved = load_checkpoint(ckpt, device=args.device)
    extra = saved["extra"]
    if not extra.get("joint") or extra.get("frequencies") != [90, 150, 220]:
        raise ValueError("Expected a joint 90/150/220 checkpoint")
    size = meta["patch_size"]
    freqs = (90, 150, 220)
    windows, readers, report = build_windows(args.data_root, freqs, meta["box_px"], size, size // 2)
    if not windows:
        raise ValueError("No common valid test windows")
    for freq in freqs:
        for kind in ("planck", "true", "pred"):
            (out / f"f{freq}" / kind).mkdir(parents=True)
    mu, sigma = np.asarray(extra["mu"], dtype=np.float32), np.asarray(extra["sigma"], dtype=np.float32)
    index = []
    for i, window in enumerate(windows):
        name = f"win_y{window['y']}_x{window['x']}"
        reader = readers[90]
        header = window_header(fits.getheader(Path(reader.act_dir) / window["tile"]),
                               window["oy"], window["ox"], size, window["y"], window["x"])
        ra, dec = WCS(header).all_pix2world((size - 1) / 2, (size - 1) / 2, 0)
        independent = int((window["y"] - meta["box_px"][0]) % size == 0
                          and (window["x"] - meta["box_px"][2]) % size == 0)
        index.append(dict(name=name, y=window["y"], x=window["x"], independent=independent,
                          ra_center=float(ra), dec_center=float(dec)))
        crops = [readers[f].read_crop(window["tile_index"], window["oy"], window["ox"]) for f in freqs]
        planck, act = np.stack([pair[0] for pair in crops]), np.stack([pair[1] for pair in crops])

        def write(kind, maps, suffix="", **cards):
            for channel, freq in enumerate(freqs):
                hdr = header.copy()
                hdr["BUNIT"], hdr["FREQ"] = "uK", freq
                for key, value in cards.items():
                    hdr[key] = value
                fits.writeto(out / f"f{freq}" / kind / f"{name}{suffix}.fits",
                             np.asarray(maps[channel], dtype=np.float32), hdr)

        write("planck", planck)
        write("true", act)
        x = torch.from_numpy(((planck - mu[:, None, None]) / sigma[:, None, None])[None]).float()
        for k in range(args.n_samples):
            seed = args.seed + 100003 * k + i
            prediction = sample_batch(model, x, mu, sigma, S=args.S, device=args.device, seed=seed)[0]
            write("pred", prediction, f"_s{k}", SAMPLE=k, SEED=seed, STEPS=args.S)
        print(f"Joint export {i + 1}/{len(windows)}", flush=True)
    with (out / "index.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    manifest = dict(patch_size=size, n_samples=args.n_samples, S=args.S, seed=args.seed,
                    box_px=meta["box_px"], guard_px=meta.get("guard_px", 0), stride=size // 2,
                    n_windows=len(index), n_independent=sum(row["independent"] for row in index),
                    window_report=report, joint=True,
                    models={str(f): dict(checkpoint=str(ckpt), sha256_16=_sha(ckpt),
                                        mu=float(mu[c]), sigma=float(sigma[c])) for c, f in enumerate(freqs)})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
