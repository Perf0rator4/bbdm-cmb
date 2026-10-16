"""Обучение и оценка BBDM для одной частотной пары (90 / 150 / 220 ГГц).

Один процесс на одной GPU:

    CUDA_VISIBLE_DEVICES=0 python scripts/train_freq.py \\
        --freq 150 --data-root /path/to/Diffusion --out-dir runs/f150

Одна частота на двух GPU (DDP), глобальный батч делится поровну:

    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train_freq.py \\
        --freq 150 --data-root /path/to/Diffusion --out-dir runs/f150

Продолжить оборвавшийся прогон: тот же запуск с `--resume` (число процессов
менять нельзя -- изменится число шагов на эпоху и перетасовка).

Скрипт делает то же, что `notebooks/training+inference.ipynb`: сплит ->
нормализация -> датасеты -> модель -> train -> оценка лучшего чекпоинта на
val и test. Всё складывается в `--out-dir`: best.pt / last.pt, splits.json,
norm.json, results.json, spectra_{val,test}.png.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from bbdm import config as cfg
from bbdm import dist as D
from bbdm.data import (
    CMBPatchDataset,
    compare_tile_grids,
    compute_normalization,
    get_tile_splits,
    inherit_splits,
    list_tiles,
    load_splits,
    save_splits,
)
from bbdm.model import BBDM, UNet


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--freq", type=int, required=True, choices=sorted(cfg.FREQ_PAIRS),
                   help="частота ACT, ГГц (Planck-пара берётся из config.FREQ_PAIRS)")
    p.add_argument("--data-root", required=True,
                   help="папка, внутри которой лежат Planck/f1xx и Planck+ACT/f0xx")
    p.add_argument("--out-dir", required=True, help="папка прогона (чекпоинты, результаты)")
    p.add_argument("--global-batch", type=int, default=cfg.BATCH_SIZE,
                   help="глобальный батч; на один процесс приходится global/число процессов")
    p.add_argument("--epochs", type=int, default=cfg.N_EPOCHS)
    p.add_argument("--lr", type=float, default=cfg.LR)
    p.add_argument("--num-workers", type=int, default=cfg.NUM_WORKERS)
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true",
                   help="разрешить новый прогон поверх существующих best.pt/last.pt")
    p.add_argument("--master-splits", default=None,
                   help="JSON со сплитом главной частоты (например, сохранённым с run 5); "
                        "по умолчанию восстанавливается из папки главной частоты")
    p.add_argument("--no-inherit", action="store_true",
                   help="не наследовать сплит от главной частоты (только для отладки)")
    p.add_argument("--monitor-every", type=int, default=cfg.MONITOR_EVERY)
    p.add_argument("--monitor-n", type=int, default=cfg.MONITOR_N)
    p.add_argument("--monitor-S", type=int, default=cfg.MONITOR_S)
    p.add_argument("--eval-patches", type=int, default=50)
    p.add_argument("--eval-S", type=int, default=cfg.S)
    p.add_argument("--skip-eval", action="store_true")
    # Отладка / малые прогоны.
    p.add_argument("--patch-size", type=int, default=cfg.PATCH_SIZE)
    p.add_argument("--base-ch", type=int, default=cfg.BASE_CH)
    p.add_argument("--time-dim", type=int, default=cfg.TIME_DIM)
    p.add_argument("--groups", type=int, default=cfg.GROUPS)
    p.add_argument("--max-tiles", type=int, default=0,
                   help="взять только первые N тайлов каждого сплита (отладка)")
    return p.parse_args(argv)


def log(*a):
    if D.is_main():
        print(*a, flush=True)


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def build_splits(args, planck_dir, act_dir):
    """Сплит train/val/test для этой частоты; пишется в out-dir/splits.json.

    Считает rank 0, остальные после барьера читают файл -- так у всех
    процессов гарантированно один и тот же сплит.
    """
    path = os.path.join(args.out_dir, "splits.json")
    if D.is_main():
        if os.path.exists(path):
            # И при --resume, и при повторном запуске нужен ТОТ ЖЕ сплит, а не
            # пересчитанный по текущему содержимому папки.
            print(f"Using existing {path}", flush=True)
        else:
            ratios = dict(train_ratio=cfg.TRAIN_RATIO, val_ratio=cfg.VAL_RATIO,
                          seed=cfg.SEED)
            if args.freq == cfg.FREQ_MASTER or args.no_inherit:
                splits = get_tile_splits(planck_dir, **ratios)
            else:
                m_planck, m_act = cfg.data_dirs(cfg.FREQ_MASTER, args.data_root)
                if args.master_splits:
                    master = load_splits(args.master_splits)
                else:
                    master = get_tile_splits(m_planck, **ratios)
                # Имена tile_y{y}_x{x} означают одно небо только на общей сетке.
                bad = compare_tile_grids(m_act, act_dir)
                if bad:
                    raise RuntimeError(
                        f"{len(bad)} одноимённых тайлов ACT {cfg.FREQ_MASTER} и {args.freq} "
                        f"ГГц лежат на РАЗНЫХ пиксельных сетках (например {bad[:3]}). "
                        "Наследование сплита по имени здесь неверно -- нужен сплит по "
                        "координатам тайла (WCS)."
                    )
                splits = inherit_splits(master, list_tiles(planck_dir), **ratios)
            save_splits(path, *splits, freq=args.freq)
    D.barrier()
    return load_splits(path)


def build_norm(args, planck_dir, act_dir, train_tiles):
    """(mu, sigma) по train-тайлам этой частоты; считает rank 0, остальные читают."""
    path = os.path.join(args.out_dir, "norm.json")
    if D.is_main() and not os.path.exists(path):
        mu, sigma = compute_normalization(planck_dir, act_dir, train_tiles,
                                          patch_size=args.patch_size)
        _write_json(path, {"mu": mu, "sigma": sigma, "freq": args.freq})
    D.barrier()
    with open(path) as f:
        d = json.load(f)
    return d["mu"], d["sigma"]


def evaluate_run(args, bbdm, device, val_ds, test_ds, mu, sigma):
    """Оценка лучшего чекпоинта (rank 0): TF / r_ell / r_in, PSNR/SSIM, графики."""
    from bbdm.evaluate import (
        evaluate_image_metrics,
        evaluate_spectra,
        plot_spectral_comparison,
        print_band_table,
    )

    ell_nyq = 180.0 * 60.0 / cfg.PIXEL_ARCMIN
    fwhm = cfg.PLANCK_BEAM_FWHM_ARCMIN[args.freq]
    sigma_b = np.deg2rad(fwhm / 60.0) / (2 * np.sqrt(2 * np.log(2)))
    results = {
        "freq": args.freq,
        "ell_nyquist": ell_nyq,
        "planck_beam_fwhm_arcmin": fwhm,
        "planck_beam_ell_1e": float(np.sqrt(2) / sigma_b),
    }

    for name, ds in (("val", val_ds), ("test", test_ds)):
        n = min(args.eval_patches, len(ds))
        res = evaluate_spectra(bbdm, ds, mu, sigma, indices=list(range(n)),
                               S=args.eval_S, device=device,
                               batch_size=min(4, max(n, 1)), progress=False)
        print_band_table(res, f"{args.freq} GHz, {name}")
        fig = plot_spectral_comparison([res], [f"{args.freq} GHz {name}"],
                                       title=f"{args.freq} GHz, {name}")
        fig.savefig(os.path.join(args.out_dir, f"spectra_{name}.png"), dpi=130)
        results[name] = {
            "n_patches": res["n_patches"],
            "bands": res["bands"],
            "bands_ell": [[lo * ell_nyq, hi * ell_nyq] for lo, hi in res["bands"]],
            "tf_bands": res["tf_bands"],
            "r_bands": res["r_bands"],
            "r_input_bands": res["r_input_bands"],
        }
        if name == "val":
            m = evaluate_image_metrics(bbdm, ds, mu, sigma, indices=list(range(n)),
                                       S=args.eval_S, device=device,
                                       batch_size=min(4, max(n, 1)), progress=False)
            results["val"].update({k: m[k] for k in ("psnr_mean", "psnr_std",
                                                     "ssim_mean", "ssim_std")})
            print(f"PSNR {m['psnr_mean']:.2f} ± {m['psnr_std']:.2f} dB, "
                  f"SSIM {m['ssim_mean']:.4f} ± {m['ssim_std']:.4f}", flush=True)

    _write_json(os.path.join(args.out_dir, "results.json"), results)
    log(f"Wrote {os.path.join(args.out_dir, 'results.json')}")


def main(argv=None):
    args = parse_args(argv)
    # Под nohup / sbatch tqdm пишет в файл, и ежесекундные обновления
    # превращают лог в мусор; раз в полминуты достаточно.
    if not sys.stdout.isatty():
        os.environ.setdefault("TQDM_MININTERVAL", "30")

    rank, world, local_rank, device = D.init_distributed()
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    try:
        if args.global_batch % world:
            raise ValueError(f"--global-batch {args.global_batch} не делится на {world} процессов")
        per_gpu = args.global_batch // world

        os.makedirs(args.out_dir, exist_ok=True)
        planck_dir, act_dir = cfg.data_dirs(args.freq, args.data_root)
        log(f"== {args.freq} GHz: Planck {planck_dir} -> ACT {act_dir}")
        log(f"   {world} process(es), device {device}, batch {per_gpu}/process "
            f"(global {args.global_batch})")

        D.barrier()
        train_tiles, val_tiles, test_tiles = build_splits(args, planck_dir, act_dir)
        if args.max_tiles:
            n = args.max_tiles
            train_tiles, val_tiles, test_tiles = train_tiles[:n], val_tiles[:n], test_tiles[:n]
        log(f"   tiles: train {len(train_tiles)}, val {len(val_tiles)}, test {len(test_tiles)}")

        mu, sigma = build_norm(args, planck_dir, act_dir, train_tiles)
        log(f"   normalization: mu={mu:.4f}, sigma={sigma:.4f}")

        common = dict(
            planck_dir=planck_dir, act_dir=act_dir, mu=mu, sigma=sigma,
            patch_size=args.patch_size, max_zero_frac=cfg.MAX_ZERO_FRAC,
            max_mask_mismatch_frac=cfg.MAX_MASK_MISMATCH_FRAC,
        )
        train_ds = CMBPatchDataset(tile_list=train_tiles, augment=True, **common)
        val_ds = CMBPatchDataset(tile_list=val_tiles, augment=False, **common)
        test_ds = CMBPatchDataset(tile_list=test_tiles, augment=False, **common)
        gb = len(train_ds.pairs) * 2 * args.patch_size ** 2 * 4 / 1024 ** 3
        log(f"   train patches held in RAM: {len(train_ds.pairs)} pairs ≈ {gb:.1f} GB "
            f"per process ({world} process(es) on this node)")

        unet = UNet(in_ch=cfg.IN_CH, base_ch=args.base_ch, time_dim=args.time_dim,
                    groups=args.groups, cond_ch=1 if cfg.COND_ON_INPUT else 0,
                    upsample=cfg.UPSAMPLE)
        bbdm = BBDM(unet, T=cfg.T, s=cfg.S_VAR, eta=cfg.ETA,
                    spectral_weight=cfg.SPECTRAL_WEIGHT)

        from bbdm.train import load_checkpoint, train

        t0 = time.time()
        train(
            bbdm, train_ds, val_ds,
            n_epochs=args.epochs, batch_size=per_gpu, lr=args.lr,
            ema_start=cfg.EMA_START, spectral_weight=cfg.SPECTRAL_WEIGHT,
            device=device, checkpoint_dir=args.out_dir, num_workers=args.num_workers,
            resume=args.resume, overwrite=args.overwrite,
            monitor_dataset=val_ds, monitor_every=args.monitor_every,
            monitor_n=args.monitor_n, monitor_S=args.monitor_S,
            extra_state={
                "freq": args.freq, "mu": mu, "sigma": sigma,
                "splits": {"train": train_tiles, "val": val_tiles, "test": test_tiles},
                "global_batch": args.global_batch,
            },
        )
        log(f"Training finished in {(time.time() - t0) / 3600:.2f} h")

        D.barrier()
        if D.is_main() and not args.skip_eval:
            best, _ = load_checkpoint(os.path.join(args.out_dir, "best.pt"), device=device)
            evaluate_run(args, best, device, val_ds, test_ds, mu, sigma)
        D.barrier()
    finally:
        D.cleanup()


if __name__ == "__main__":
    main()
