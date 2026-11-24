"""Joint 90/150/220 GHz BBDM with an optional frozen-CFA short-chain loss.

Uses train_freq.py's region split, normalization, EMA/checkpoints and stopping.
Run with --detection-weight 0 for the same-architecture MSE control first.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from bbdm import config as cfg, dist as D
from bbdm.data import compute_normalization, crop_audit, forbidden_rects, list_tiles
from bbdm.data.joint import JointPatchDataset
from bbdm.early_stop import EarlyStopper
from bbdm.evaluate import evaluate_spectra
from bbdm.model import BBDM, UNet
from bbdm.model.detection_loss import CFAHeatmapLoss
from bbdm.train import load_checkpoint, train
from train_freq import build_splits, parse_args, region_box


def main(argv=None):
    args = parse_args(argv, joint=True)
    if args.split_mode != "region" or args.freq != 90:
        raise ValueError("Joint training uses the shared region split and all frequencies; omit --freq")
    if (not np.isfinite(args.detection_weight) or args.detection_weight < 0
            or not 2 <= args.detection_steps <= cfg.T):
        raise ValueError("Use finite detection-weight >= 0 and detection-steps in [2, T]")
    if args.detection_weight and not args.detector_weights:
        raise ValueError("--detector-weights is required for the detection loss")
    rank, world, _, device = D.init_distributed()
    try:
        if device == "cpu" and args.patch_size >= cfg.PATCH_SIZE:
            raise ValueError("Full-size joint training requires a CUDA GPU; CPU is for tiny tests only")
        if args.global_batch < 1 or args.global_batch % world:
            raise ValueError("global-batch must be positive and divisible by world size")
        out = Path(args.out_dir)
        if not args.resume and not args.overwrite and any((out / n).exists() for n in ("last.pt", "best.pt")):
            raise FileExistsError("Choose a new out-dir; this run already has checkpoints")
        torch.manual_seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        out.mkdir(parents=True, exist_ok=True)
        splits = build_splits(args, *cfg.data_dirs(90, args.data_root))
        if args.max_tiles:
            splits = {key: set(sorted(value)[:args.max_tiles * 4]) for key, value in splits.items()}
        box, _ = region_box(args)
        dirs = [cfg.data_dirs(freq, args.data_root) for freq in (90, 150, 220)]
        common_tiles = sorted(set.intersection(*[set(list_tiles(d)) for pair in dirs for d in pair]))
        norm_path = out / "norm.json"
        if D.is_main() and not norm_path.exists():
            stats = [compute_normalization(pd, ad, common_tiles, patch_size=args.patch_size,
                                           keep_positions=splits["train"]) for pd, ad in dirs]
            norm_path.write_text(json.dumps(dict(frequencies=[90, 150, 220],
                                                 mu=[s[0] for s in stats], sigma=[s[1] for s in stats]), indent=2))
        D.barrier()
        norm = json.loads(norm_path.read_text())
        if norm.get("frequencies") != [90, 150, 220]:
            raise ValueError("Expected a joint, per-frequency normalization file")
        common = dict(data_root=args.data_root, mu=norm["mu"], sigma=norm["sigma"],
                      patch_size=args.patch_size, max_zero_frac=cfg.MAX_ZERO_FRAC,
                      max_mask_mismatch_frac=cfg.MAX_MASK_MISMATCH_FRAC)
        if args.no_crops:
            train_ds = JointPatchDataset(positions=splits["train"], augment=True, **common)
        else:
            train_ds = JointPatchDataset(
                forbidden=forbidden_rects(splits, box, args.patch_size, args.guard_px),
                n_samples=args.crops_per_epoch or cfg.CROPS_PER_POSITION * len(splits["train"]),
                augment=True, **common)
            if D.is_main():
                audit = crop_audit(train_ds, splits)
                leak = json.loads((out / "leakage.json").read_text())
                leak["train_crops"] = audit
                (out / "leakage.json").write_text(json.dumps(leak, indent=2))
                if audit["crops_touching_val"] or audit["crops_touching_test"]:
                    raise ValueError(f"Joint crop leakage: {audit}")
            D.barrier()
        val_ds = JointPatchDataset(positions=splits["val"], **common)
        if len(train_ds) < args.global_batch:
            raise ValueError("Training data is smaller than one global batch")
        if args.ema_start >= args.epochs * (len(train_ds) // args.global_batch):
            raise ValueError("EMA would not start; reduce ema-start for this experiment")
        model = BBDM(UNet(in_ch=3, cond_ch=3, base_ch=args.base_ch, time_dim=args.time_dim,
                          groups=args.groups, upsample=cfg.UPSAMPLE), T=cfg.T, s=cfg.S_VAR,
                     eta=cfg.ETA, spectral_weight=0)
        objective = (CFAHeatmapLoss(args.detector_weights, norm["mu"], norm["sigma"])
                     if args.detection_weight else None)
        stopper = None if args.no_early_stop or not args.monitor_every else EarlyStopper(**{
            **cfg.EARLY_STOP, "patience": args.patience, "psnr_min_delta": args.psnr_min_delta,
            "r_min_delta": args.r_min_delta, "tf_max": args.tf_max})
        train(model, train_ds, val_ds, n_epochs=args.epochs, batch_size=args.global_batch // world,
              lr=args.lr, ema_start=args.ema_start, spectral_weight=0, device=device,
              checkpoint_dir=str(out), num_workers=args.num_workers, resume=args.resume,
              overwrite=args.overwrite, monitor_dataset=val_ds, monitor_every=args.monitor_every,
              monitor_n=args.monitor_n, monitor_S=args.monitor_S, early_stop=stopper,
              detection_loss=objective, detection_weight=args.detection_weight,
              detection_steps=args.detection_steps,
              extra_state={**norm, "split_mode": "region", "joint": True,
                           "global_batch": args.global_batch, "train_crops": not args.no_crops})
        D.barrier()
        if D.is_main() and not args.skip_eval:
            checkpoint = out / "best_metric.pt"
            if args.eval_ckpt == "loss" or not checkpoint.exists():
                checkpoint = out / "best.pt"
            best, _ = load_checkpoint(checkpoint, device=device)
            results = {}
            for label, ds in [("val", val_ds), ("test", JointPatchDataset(positions=splits["test"], **common))]:
                results[label] = {}
                for channel, freq in enumerate((90, 150, 220)):
                    metrics = evaluate_spectra(best, ds, norm["mu"], norm["sigma"],
                                               n_patches=args.eval_patches, S=args.eval_S,
                                               device=device, batch_size=1, channel=channel,
                                               with_psnr=True)
                    results[label][str(freq)] = {k: metrics[k] for k in
                        ("bands", "tf_bands", "r_bands", "r_input_bands", "psnr_mean")}
            (out / "results.json").write_text(json.dumps(results, indent=2))
        D.barrier()
    finally:
        D.cleanup()


if __name__ == "__main__":
    main()
