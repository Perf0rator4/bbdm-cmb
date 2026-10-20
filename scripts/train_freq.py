"""Обучение и оценка BBDM для одной частотной пары (90 / 150 / 220 ГГц).

Один процесс на одной GPU:

    CUDA_VISIBLE_DEVICES=0 python scripts/train_freq.py \\
        --freq 150 --data-root /path/to/Diffusion --out-dir runs/f150 --global-batch 4

Одна частота на двух GPU (DDP), глобальный батч делится поровну:

    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train_freq.py \\
        --freq 150 --data-root /path/to/Diffusion --out-dir runs/f150 --global-batch 4

Продолжить оборвавшийся прогон: тот же запуск с `--resume` (число процессов
менять нельзя -- изменится число шагов на эпоху и перетасовка).

Скрипт делает то же, что `notebooks/training+inference.ipynb`: сплит ->
нормализация -> датасеты -> модель -> train -> оценка лучшего чекпоинта на
val и test. Всё складывается в `--out-dir`: best.pt / best_metric.pt / last.pt,
splits.json, norm.json, leakage.json, results.json, spectra_{val,test}.png.

Сплит по умолчанию -- по ОБЛАСТИ НЕБА (`--split-mode region`): test -- патчи
целиком внутри поля D56 (config.TEST_REGIONS), патчи на его границе выбрасываются,
val -- пространственные блоки вне поля, train -- всё остальное небо, причём
читается СЛУЧАЙНЫМИ КРОПАМИ с произвольным смещением внутри тайлов, не задевающими
ни поле, ни val (bbdm/data/region.py). Сплит общий для всех частот.
`--split-mode spatial` -- прежний сплит квадрантов по случайным блокам;
`--split-mode legacy` -- сплит по тайлам, как у run 5: тайлы нарезаны с
перекрытием 50%, и он отправляет копии train-патчей в val и test.
`--global-batch` обязателен: от него зависит число шагов на эпоху и эпоха
старта EMA, а значит и сопоставимость частот между собой.
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
    RandomCropDataset,
    check_tile_wcs,
    crop_audit,
    forbidden_rects,
    region_box_px,
    region_splits,
    compare_tile_grids,
    compute_normalization,
    get_patch_splits,
    get_tile_splits,
    inherit_splits,
    legacy_leakage_report,
    list_tiles,
    load_patch_splits,
    load_splits,
    patch_leakage_report,
    save_patch_splits,
    save_splits,
)
from bbdm.early_stop import EarlyStopper
from bbdm.evaluate import spread_indices
from bbdm.model import BBDM, UNet


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--freq", type=int, required=True, choices=sorted(cfg.FREQ_PAIRS),
                   help="частота ACT, ГГц (Planck-пара берётся из config.FREQ_PAIRS)")
    p.add_argument("--data-root", required=True,
                   help="папка, внутри которой лежат Planck/f1xx и Planck+ACT/f0xx")
    p.add_argument("--out-dir", required=True, help="папка прогона (чекпоинты, результаты)")
    p.add_argument("--global-batch", type=int, required=True,
                   help="глобальный батч (обязателен: от него зависят шаги на эпоху и эпоха "
                        "старта EMA; для сопоставимости с run 5 -- его батч); на один "
                        "процесс приходится global/число процессов")
    p.add_argument("--split-mode", choices=("region", "spatial", "legacy"), default="region",
                   help="region: test = поле --test-region, train = случайные кропы вне поля и val; "
                        "spatial: уникальные патчи по случайным пространственным блокам; "
                        "legacy: сплит по тайлам (как у run 5, с утечкой)")
    p.add_argument("--splits-file", default=None,
                   help="общий для всех частот сплит по позициям патчей (JSON); создаётся при "
                        "отсутствии. По умолчанию <data-root>/splits_region_<поле>.json "
                        "(region) или <data-root>/splits_spatial.json (spatial)")
    p.add_argument("--test-region", default=cfg.TEST_REGION,
                   help="[region] имя поля из config.TEST_REGIONS или 'ra_min,ra_max,dec_min,dec_max' "
                        "в градусах")
    p.add_argument("--test-box-px", default=None,
                   help="[region] тестовое поле прямо в пикселях карты 'y0,y1,x0,x1' "
                        "(вместо --test-region; для синтетических данных)")
    p.add_argument("--guard-px", type=int, default=0,
                   help="[region] полоса вокруг поля (px), куда не заходят train и val")
    p.add_argument("--crops-per-epoch", type=int, default=0,
                   help="[region] train-кропов на эпоху; 0 -> config.CROPS_PER_POSITION x "
                        "число train-позиций")
    p.add_argument("--no-crops", action="store_true",
                   help="[region] учить на фиксированных train-квадрантах, а не на случайных кропах")
    p.add_argument("--skip-wcs-check", action="store_true",
                   help="[region] не сверять формулы RA/Dec -> пиксель с WCS заголовков тайлов")
    p.add_argument("--block", type=int, default=2,
                   help="сторона пространственного блока сплита, в патчах (2 -> 8 градусов)")
    p.add_argument("--epochs", type=int, default=cfg.N_EPOCHS)
    p.add_argument("--lr", type=float, default=cfg.LR)
    p.add_argument("--ema-start", type=int, default=cfg.EMA_START,
                   help="шаг, с которого стартует EMA (в ШАГАХ оптимизатора, а не эпохах -- "
                        "поэтому зависит от батча)")
    p.add_argument("--num-workers", type=int, default=cfg.NUM_WORKERS)
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true",
                   help="разрешить новый прогон поверх существующих best.pt/last.pt")
    p.add_argument("--master-splits", default=None,
                   help="[legacy] JSON со сплитом главной частоты (например, сохранённым с run 5); "
                        "по умолчанию восстанавливается из папки главной частоты")
    p.add_argument("--no-inherit", action="store_true",
                   help="[legacy] не наследовать сплит от главной частоты (только для отладки)")
    p.add_argument("--no-early-stop", action="store_true",
                   help="не останавливать по метрикам (только лимит эпох)")
    p.add_argument("--patience", type=int, default=cfg.EARLY_STOP["patience"],
                   help="проверок монитора подряд без улучшения PSNR и r_ell -> стоп")
    p.add_argument("--psnr-min-delta", type=float, default=cfg.EARLY_STOP["psnr_min_delta"],
                   help="минимальный прирост PSNR (дБ), считающийся улучшением")
    p.add_argument("--r-min-delta", type=float, default=cfg.EARLY_STOP["r_min_delta"],
                   help="минимальный прирост r_ell, считающийся улучшением")
    p.add_argument("--tf-max", type=float, default=cfg.EARLY_STOP["tf_max"],
                   help="TF выше этого значения подряд -> стоп (избыток мощности, как run 4)")
    p.add_argument("--eval-ckpt", choices=("metric", "loss"), default="metric",
                   help="чекпоинт для финальной оценки: best_metric.pt (по PSNR при TF в норме; "
                        "если его нет -- best.pt) или best.pt (по val-лоссу)")
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


def _existing_freq_dirs(args):
    """{freq: (planck_dir, act_dir)} для частот, чьи папки реально есть под data-root."""
    out = {}
    for f in sorted(cfg.FREQ_PAIRS):
        pd, ad = cfg.data_dirs(f, args.data_root)
        if os.path.isdir(pd) and os.path.isdir(ad):
            out[f] = (pd, ad)
    return out


def build_splits(args, planck_dir, act_dir):
    """Сплит этой частоты; пишется в out-dir/splits.json (+ leakage.json).

    spatial -> dict {"train"/"val"/"test": множество позиций патчей}, одинаковый для всех
    частот. legacy -> (train, val, test) списки имён тайлов, как у run 5.
    Считает rank 0, остальные после барьера читают файл.
    """
    path = os.path.join(args.out_dir, "splits.json")
    leak_path = os.path.join(args.out_dir, "leakage.json")

    if D.is_main() and not os.path.exists(path):
        if args.split_mode == "legacy":
            splits = _legacy_splits(args, planck_dir, act_dir)
            save_splits(path, *splits, freq=args.freq, mode="legacy")
            rep = legacy_leakage_report(*splits, patch_size=args.patch_size)
            _write_json(leak_path, {"mode": "legacy", **rep})
            for name in ("val", "test"):
                r = rep[name]
                print(f"!! LEGACY SPLIT LEAKS: {r['identical_to_train']}/{r['patch_instances']} "
                      f"{name} patches are exact copies of train patches "
                      f"({r['overlapping_train']} overlap train by pixels). "
                      f"Metrics on them are not out-of-sample.", flush=True)
        elif args.split_mode == "region":
            _region_splits_to(args, path, leak_path)
        else:
            splits_file = args.splits_file or os.path.join(args.data_root, "splits_spatial.json")
            if os.path.exists(splits_file):
                print(f"Using shared patch splits {splits_file}", flush=True)
                sp, ps = load_patch_splits(splits_file)
                if ps != args.patch_size:
                    raise RuntimeError(f"{splits_file} построен для патчей {ps} px, а --patch-size "
                                       f"{args.patch_size}")
            else:
                sp = _make_spatial_splits(args)
                try:
                    save_patch_splits(splits_file, sp, patch_size=args.patch_size,
                                      block=args.block, seed=cfg.SEED)
                    print(f"Saved shared patch splits to {splits_file}", flush=True)
                except OSError as e:
                    print(f"(could not write {splits_file}: {e}; the copy in out-dir is used)",
                          flush=True)
            save_patch_splits(path, sp, patch_size=args.patch_size, block=args.block,
                              seed=cfg.SEED, freq=args.freq, mode="spatial")
            rep = patch_leakage_report(sp, patch_size=args.patch_size)
            _write_json(leak_path, {"mode": "spatial", **rep})
            print(f"Leakage audit: {rep}", flush=True)
            if any(v["overlapping"] for v in rep.values()):
                raise RuntimeError(f"patch splits are not leak-free: {rep}")
    D.barrier()

    if args.split_mode == "legacy":
        return load_splits(path)
    sp, _ = load_patch_splits(path)
    return {k: set(v) for k, v in sp.items()}


def region_box(args):
    """(box, описание поля): прямоугольник тестового поля в пикселях карты."""
    if args.test_box_px:
        box = tuple(int(v) for v in args.test_box_px.split(","))
        if len(box) != 4:
            raise ValueError("--test-box-px: нужно 'y0,y1,x0,x1'")
        return box, {"name": "custom_px", "box_px": list(box)}
    if args.test_region in cfg.TEST_REGIONS:
        name, radec = args.test_region, cfg.TEST_REGIONS[args.test_region]
    else:
        radec = tuple(float(v) for v in args.test_region.split(","))
        if len(radec) != 4:
            raise ValueError("--test-region: имя из config.TEST_REGIONS или 'ra_min,ra_max,dec_min,dec_max'")
        name = "custom"
    return region_box_px(*radec), {"name": name, "ra_dec": list(radec)}


def _check_grid(args, dirs):
    """Формулы RA/Dec -> пиксель должны совпадать с WCS самих тайлов (rank 0)."""
    from astropy.io import fits

    for f, (_, ad) in dirs.items():
        names = list_tiles(ad)
        for n in (names[0], names[len(names) // 2], names[-1]):
            dev = check_tile_wcs(fits.getheader(os.path.join(ad, n)), n)
            if dev > 2.0:
                raise RuntimeError(
                    f"WCS тайла {ad}/{n} расходится с сеткой ACT DR6 на {dev:.1f} px: поле "
                    f"{args.test_region} оказалось бы не там. Проверьте нарезку (или --test-box-px)."
                )
    print(f"WCS check: tile headers agree with the ACT DR6 CAR grid (checked {len(dirs)} freq(s))",
          flush=True)


def _region_splits_to(args, path, leak_path):
    """Сплит по области (общий для частот, в data-root) + его копия и аудит в out-dir."""
    box, desc = region_box(args)
    tag = desc["name"] if desc["name"] != "custom" else "custom_" + "_".join(
        f"{v:g}" for v in desc["ra_dec"])
    splits_file = args.splits_file or os.path.join(args.data_root, f"splits_region_{tag}.json")
    meta = dict(mode="region", region=desc, box_px=list(box), guard_px=args.guard_px,
                block=args.block, seed=cfg.SEED)
    if os.path.exists(splits_file):
        print(f"Using shared region splits {splits_file}", flush=True)
        with open(splits_file) as f:
            saved = json.load(f)
        if saved.get("box_px") != list(box) or saved.get("guard_px") != args.guard_px:
            raise RuntimeError(f"{splits_file} построен для поля {saved.get('box_px')} / guard "
                               f"{saved.get('guard_px')}, а сейчас {list(box)} / {args.guard_px}")
        sp, ps = load_patch_splits(splits_file)
        if ps != args.patch_size:
            raise RuntimeError(f"{splits_file} построен для патчей {ps} px, а --patch-size {args.patch_size}")
        info = saved.get("info", {})
    else:
        dirs = _existing_freq_dirs(args)
        if args.freq not in dirs:
            raise FileNotFoundError(f"нет папок для {args.freq} ГГц под {args.data_root}")
        _check_same_grids(dirs)
        if not args.test_box_px and not args.skip_wcs_check:
            _check_grid(args, dirs)
        tiles = sorted({t for pd, _ in dirs.values() for t in list_tiles(pd)})
        print(f"Splitting the sky once for frequencies {sorted(dirs)} ({len(tiles)} tiles), "
              f"test field {desc} -> pixels y {box[0]}:{box[1]}, x {box[2]}:{box[3]}", flush=True)
        sp, info = region_splits(tiles, box, patch_size=args.patch_size, val_ratio=cfg.VAL_RATIO,
                                 block=args.block, seed=cfg.SEED, guard=args.guard_px)
        try:
            save_patch_splits(splits_file, sp, patch_size=args.patch_size, info=info, **meta)
            print(f"Saved shared region splits to {splits_file}", flush=True)
        except OSError as e:
            print(f"(could not write {splits_file}: {e}; the copy in out-dir is used)", flush=True)
    if not sp["test"]:
        raise RuntimeError(f"в поле {desc} нет ни одного целого патча -- тестировать не на чем")
    save_patch_splits(path, sp, patch_size=args.patch_size, info=info, freq=args.freq, **meta)
    rep = patch_leakage_report(sp, patch_size=args.patch_size)
    _write_json(leak_path, {"mode": "region", "region": desc, "box_px": list(box), **rep})
    print(f"Leakage audit (patch positions): {rep}", flush=True)
    if any(v["overlapping"] for v in rep.values()):
        raise RuntimeError(f"region splits are not leak-free: {rep}")


def _check_same_grids(dirs):
    # Позиции -- в общей пиксельной сетке; одноимённые тайлы разных частот обязаны её делить.
    ref_f = sorted(dirs)[0]
    for f, (pd, ad) in dirs.items():
        if f == ref_f:
            continue
        bad = compare_tile_grids(dirs[ref_f][1], ad)
        if bad:
            raise RuntimeError(
                f"{len(bad)} одноимённых тайлов ACT {ref_f} и {f} ГГц на РАЗНЫХ пиксельных "
                f"сетках (например {bad[:3]}): общий сплит по позициям неприменим."
            )


def _make_spatial_splits(args):
    """Общий сплит по позициям: по объединению тайлов всех частот, что есть на диске."""
    dirs = _existing_freq_dirs(args)
    if args.freq not in dirs:
        raise FileNotFoundError(f"нет папок для {args.freq} ГГц под {args.data_root}")
    _check_same_grids(dirs)
    tiles = sorted({t for pd, _ in dirs.values() for t in list_tiles(pd)})
    print(f"Splitting the sky once for frequencies {sorted(dirs)} ({len(tiles)} tiles)", flush=True)
    return get_patch_splits(tiles, patch_size=args.patch_size, block=args.block,
                            train_ratio=cfg.TRAIN_RATIO, val_ratio=cfg.VAL_RATIO, seed=cfg.SEED)


def _legacy_splits(args, planck_dir, act_dir):
    ratios = dict(train_ratio=cfg.TRAIN_RATIO, val_ratio=cfg.VAL_RATIO, seed=cfg.SEED)
    if args.freq == cfg.FREQ_MASTER or args.no_inherit:
        return get_tile_splits(planck_dir, **ratios)
    m_planck, m_act = cfg.data_dirs(cfg.FREQ_MASTER, args.data_root)
    master = load_splits(args.master_splits) if args.master_splits else get_tile_splits(m_planck, **ratios)
    bad = compare_tile_grids(m_act, act_dir)
    if bad:
        raise RuntimeError(
            f"{len(bad)} одноимённых тайлов ACT {cfg.FREQ_MASTER} и {args.freq} ГГц на РАЗНЫХ "
            f"пиксельных сетках (например {bad[:3]}); наследование сплита по имени неверно."
        )
    return inherit_splits(master, list_tiles(planck_dir), **ratios)


def build_norm(args, planck_dir, act_dir, train_tiles, train_positions=None):
    """(mu, sigma) по train-патчам этой частоты; считает rank 0, остальные читают.

    В spatial-режиме по уникальным позициям (патч из четырёх тайлов -- один раз).
    """
    path = os.path.join(args.out_dir, "norm.json")
    if D.is_main() and not os.path.exists(path):
        mu, sigma = compute_normalization(planck_dir, act_dir, train_tiles,
                                          patch_size=args.patch_size,
                                          keep_positions=train_positions)
        _write_json(path, {"mu": mu, "sigma": sigma, "freq": args.freq})
    D.barrier()
    with open(path) as f:
        d = json.load(f)
    return d["mu"], d["sigma"]


def _jsonable(x):
    """-inf/nan из состояния останова -> None, чтобы results.json был валидным JSON."""
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return None
    return x


def evaluate_run(args, bbdm, device, val_ds, test_ds, mu, sigma, extra=None):
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
    results.update(extra or {})

    for name, ds in (("val", val_ds), ("test", test_ds)):
        n = min(args.eval_patches, len(ds))
        idx = spread_indices(len(ds), n)
        res = evaluate_spectra(bbdm, ds, mu, sigma, indices=idx,
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
            m = evaluate_image_metrics(bbdm, ds, mu, sigma, indices=idx,
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
        splits = build_splits(args, planck_dir, act_dir)
        all_tiles = list_tiles(planck_dir)
        pos = {}
        if args.split_mode == "legacy":
            train_tiles, val_tiles, test_tiles = splits
            if args.max_tiles:
                n = args.max_tiles
                train_tiles, val_tiles, test_tiles = train_tiles[:n], val_tiles[:n], test_tiles[:n]
            tile_lists = {"train": train_tiles, "val": val_tiles, "test": test_tiles}
            pos = {"train": None, "val": None, "test": None}
            log(f"   [legacy split] tiles: train {len(train_tiles)}, val {len(val_tiles)}, "
                f"test {len(test_tiles)}")
        else:
            if args.max_tiles:
                splits = {k: set(sorted(v)[: args.max_tiles * 4]) for k, v in splits.items()}
            tile_lists = {k: all_tiles for k in ("train", "val", "test")}
            pos = splits
            log(f"   [{args.split_mode} split] patch positions: train {len(splits['train'])}, "
                f"val {len(splits['val'])}, test {len(splits['test'])} "
                f"(shared by all frequencies, leak-free; audit in leakage.json)")

        mu, sigma = build_norm(args, planck_dir, act_dir, tile_lists["train"], pos["train"])
        log(f"   normalization: mu={mu:.4f}, sigma={sigma:.4f}")

        common = dict(
            planck_dir=planck_dir, act_dir=act_dir, mu=mu, sigma=sigma,
            patch_size=args.patch_size, max_zero_frac=cfg.MAX_ZERO_FRAC,
            max_mask_mismatch_frac=cfg.MAX_MASK_MISMATCH_FRAC,
        )
        use_crops = args.split_mode == "region" and not args.no_crops
        if use_crops:
            box, _ = region_box(args)
            n_crops = args.crops_per_epoch or cfg.CROPS_PER_POSITION * len(pos["train"])
            train_ds = RandomCropDataset(
                tile_list=tile_lists["train"], n_samples=n_crops, augment=True,
                forbidden=forbidden_rects(splits, box, args.patch_size, args.guard_px), **common)
        else:
            train_ds = CMBPatchDataset(tile_list=tile_lists["train"], augment=True,
                                       keep_positions=pos["train"], **common)
        val_ds = CMBPatchDataset(tile_list=tile_lists["val"], augment=False,
                                 keep_positions=pos["val"], **common)
        test_ds = CMBPatchDataset(tile_list=tile_lists["test"], augment=False,
                                  keep_positions=pos["test"], **common)
        if use_crops:
            if D.is_main():
                audit = crop_audit(train_ds, splits)
                leak_path = os.path.join(args.out_dir, "leakage.json")
                with open(leak_path) as f:
                    leak = json.load(f)
                leak["train_crops"] = audit
                _write_json(leak_path, leak)
                print(f"Leakage audit (random train crops): {audit}", flush=True)
                if audit["crops_touching_val"] or audit["crops_touching_test"]:
                    raise RuntimeError(f"train crops touch val/test: {audit}")
            log(f"   train = random crops read from disk ({len(train_ds.tiles)} tiles), nothing "
                f"held in RAM")
        else:
            gb = len(train_ds.pairs) * 2 * args.patch_size ** 2 * 4 / 1024 ** 3
            log(f"   train patches held in RAM: {len(train_ds.pairs)} pairs ≈ {gb:.1f} GB "
                f"per process ({world} process(es) on this node)")

        steps_per_epoch = len(train_ds) // args.global_batch
        ema_epoch = args.ema_start / max(steps_per_epoch, 1)
        log(f"   {len(train_ds)} train samples/epoch "
            f"({'random crops' if use_crops else 'x4 augmentation'}) / global batch "
            f"{args.global_batch} = {steps_per_epoch} steps/epoch; EMA starts at step {args.ema_start} "
            f"-> EMA starts at epoch {ema_epoch:.1f} of {args.epochs}")
        if ema_epoch >= args.epochs:
            raise ValueError(f"EMA would never start (epoch {ema_epoch:.1f} >= {args.epochs}); "
                             "lower --global-batch or pass a smaller --ema-start")
        if args.global_batch != cfg.BATCH_SIZE:
            log(f"   note: --global-batch {args.global_batch} differs from config.BATCH_SIZE "
                f"({cfg.BATCH_SIZE}); keep it equal to the batch of the run you compare with")

        unet = UNet(in_ch=cfg.IN_CH, base_ch=args.base_ch, time_dim=args.time_dim,
                    groups=args.groups, cond_ch=1 if cfg.COND_ON_INPUT else 0,
                    upsample=cfg.UPSAMPLE)
        bbdm = BBDM(unet, T=cfg.T, s=cfg.S_VAR, eta=cfg.ETA,
                    spectral_weight=cfg.SPECTRAL_WEIGHT)

        from bbdm.train import load_checkpoint, train

        stopper = None
        if not args.no_early_stop and args.monitor_every:
            stopper = EarlyStopper(**{
                **cfg.EARLY_STOP,
                "patience": args.patience,
                "psnr_min_delta": args.psnr_min_delta,
                "r_min_delta": args.r_min_delta,
                "tf_max": args.tf_max,
            })
            log(f"   early stop: patience {args.patience} checks x every {args.monitor_every} "
                f"epoch(s), PSNR delta {args.psnr_min_delta} dB, r_ell delta {args.r_min_delta}, "
                f"TF guard > {args.tf_max}")
        else:
            log("   early stop: OFF (epoch limit only)")

        t0 = time.time()
        train(
            bbdm, train_ds, val_ds,
            n_epochs=args.epochs, batch_size=per_gpu, lr=args.lr,
            ema_start=args.ema_start, spectral_weight=cfg.SPECTRAL_WEIGHT,
            device=device, checkpoint_dir=args.out_dir, num_workers=args.num_workers,
            resume=args.resume, overwrite=args.overwrite,
            monitor_dataset=val_ds, monitor_every=args.monitor_every,
            monitor_n=args.monitor_n, monitor_S=args.monitor_S,
            extra_state={
                "freq": args.freq, "mu": mu, "sigma": sigma,
                "split_mode": args.split_mode,
                "train_crops": use_crops,
                "split_sizes": {"train": len(train_ds) if use_crops else len(train_ds.pairs),
                                "train_positions": len(pos["train"]) if pos["train"] else None,
                                "val": len(val_ds.pairs), "test": len(test_ds.pairs)},
                "global_batch": args.global_batch,
            },
            early_stop=stopper,
        )
        log(f"Training finished in {(time.time() - t0) / 3600:.2f} h")

        D.barrier()
        if D.is_main() and not args.skip_eval:
            metric_ckpt = os.path.join(args.out_dir, "best_metric.pt")
            if args.eval_ckpt == "metric" and os.path.exists(metric_ckpt):
                ckpt_path = metric_ckpt
            else:
                ckpt_path = os.path.join(args.out_dir, "best.pt")
                if args.eval_ckpt == "metric":
                    log("   no best_metric.pt (TF never in range together with a PSNR): "
                        "evaluating best.pt (best val loss)")
            best, ck = load_checkpoint(ckpt_path, device=device)
            last = torch.load(os.path.join(args.out_dir, "last.pt"), map_location="cpu",
                              weights_only=False)
            log(f"   evaluating {os.path.basename(ckpt_path)} (epoch {ck['epoch'] + 1}); "
                f"run ended at epoch {last['epoch'] + 1}"
                + (f", early stop: {last['stop_reason']}" if last.get("stop_reason") else ""))
            evaluate_run(args, best, device, val_ds, test_ds, mu, sigma, extra=_jsonable({
                "eval_checkpoint": os.path.basename(ckpt_path),
                "eval_checkpoint_epoch": ck["epoch"] + 1,
                "last_epoch": last["epoch"] + 1,
                "stop_reason": last.get("stop_reason"),
                "early_stop": last.get("early_stop"),
                "monitor": last.get("monitor"),
            }))
        D.barrier()
    finally:
        D.cleanup()


if __name__ == "__main__":
    main()
