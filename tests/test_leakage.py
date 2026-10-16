"""Утечка между train и test из-за перекрывающихся тайлов и сплит по позициям патчей.

Тайлы 2P x 2P нарезаны со страйдом P, патчи -- их квадранты P x P, поэтому один и тот
же патч лежит в нескольких тайлах. Здесь это проверяется на синтетической «карте
неба», из которой вырезаны тайлы, а не на утверждении в комментарии.

Запуск: `pytest tests/` или `python tests/test_leakage.py`. Данные не нужны.
"""

import os
import random
import shutil
import sys
import tempfile

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bbdm.data import (  # noqa: E402
    CMBPatchDataset,
    clean_eval_positions,
    compute_normalization,
    get_patch_splits,
    get_tile_splits,
    legacy_leakage_report,
    load_patch_splits,
    patch_leakage_report,
    save_patch_splits,
    split_tile_names,
    tile_patch_positions,
    unique_patch_positions,
)
from bbdm.data.dataset import tile_to_patches  # noqa: E402
from bbdm.evaluate import spread_indices  # noqa: E402

P = 32                      # размер патча в тестах
TILE = 2 * P
H, W = 7 * P, 9 * P         # «карта неба»: 7 x 9 патчей


def _sky(seed=0, scale_gradient=True):
    """Карта, у которой статистика заметно меняется по полю (чтобы дубли были видны)."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((H, W)).astype(np.float32)
    if scale_gradient:
        base *= np.linspace(1.0, 6.0, W, dtype=np.float32)[None, :]
    return base + 10.0     # нет нулей -> фильтры маски не срабатывают


def _tile_names(offsets_y=None, offsets_x=None):
    ys = offsets_y if offsets_y is not None else list(range(0, H - TILE + 1, P))
    xs = offsets_x if offsets_x is not None else list(range(0, W - TILE + 1, P))
    return [f"tile_y{y}_x{x}.fits" for y in ys for x in xs]


def _write_tree(root, sky, names, freq_dirs=(("Planck/f100", "Planck+ACT/f090"),), scale=1.0):
    from astropy.io import fits as F

    for pd, ad in freq_dirs:
        os.makedirs(os.path.join(root, pd), exist_ok=True)
        os.makedirs(os.path.join(root, ad), exist_ok=True)
        for n in names:
            y, x = tile_patch_positions(n, P)[0]
            h = F.Header()
            h["CRVAL1"], h["CRVAL2"] = 10.0 + x * 0.01, -30.0 + y * 0.01
            h["CRPIX1"], h["CRPIX2"] = 1.0, 1.0
            h["CDELT1"], h["CDELT2"] = -1 / 120, 1 / 120
            tile = sky[y:y + TILE, x:x + TILE]
            F.PrimaryHDU((tile * scale).astype(np.float32), header=h).writeto(
                os.path.join(root, pd, n))
            F.PrimaryHDU(tile.astype(np.float32), header=h).writeto(os.path.join(root, ad, n))


# ------------------------------------------------------------------ геометрия


def test_overlapping_tiles_contain_bit_identical_patches():
    """Правый нижний квадрант тайла (y, x) == левый верхний квадрант тайла (y+P, x+P)."""
    sky = _sky()
    a = sky[0:TILE, 0:TILE]
    b = sky[P:P + TILE, P:P + TILE]
    pa, pb = tile_to_patches(a, P), tile_to_patches(b, P)
    assert np.array_equal(pa[3], pb[0])          # квадранты 3 = (1,1) и 0 = (0,0)


def test_tile_patch_positions_match_tile_to_patches_order():
    """Позиция k-го квадранта -- ровно то место глобальной карты, откуда он вырезан."""
    sky = _sky(1)
    for name in ("tile_y0_x0.fits", f"tile_y{P}_x{2 * P}.fits", f"tile_y{3 * P}_x{4 * P}.fits"):
        y, x = tile_patch_positions(name, P)[0]
        quads = tile_to_patches(sky[y:y + TILE, x:x + TILE], P)
        for k, (py, px) in enumerate(tile_patch_positions(name, P)):
            assert np.array_equal(quads[k], sky[py:py + P, px:px + P]), (name, k)


def test_bad_tile_name_is_rejected():
    try:
        tile_patch_positions("tile_000.fits", P)
    except ValueError:
        return
    raise AssertionError("имя без y/x должно давать ValueError, а не молча пропускаться")


# ------------------------------------------------------- утечка старого сплита


def test_legacy_tile_split_leaks_and_report_matches_pixel_comparison():
    sky = _sky(2)
    names = _tile_names()
    tr, va, te = get_tile_splits_from_names(names, seed=0)

    rep = legacy_leakage_report(tr, va, te, P)

    # независимый подсчёт: по самим ПИКСЕЛЯМ (хэш патча), а не по позициям
    def patch_bytes(name):
        y, x = tile_patch_positions(name, P)[0]
        return [q.tobytes() for q in tile_to_patches(sky[y:y + TILE, x:x + TILE], P)]

    train_hashes = {b for n in tr for b in patch_bytes(n)}
    for name, tiles in (("val", va), ("test", te)):
        q = [b for n in tiles for b in patch_bytes(n)]
        copies = sum(b in train_hashes for b in q)
        assert rep[name]["identical_to_train"] == copies
        assert rep[name]["patch_instances"] == len(q)

    # и это не мелочь: на плотной сетке копий большинство
    total = rep["val"]["patch_instances"] + rep["test"]["patch_instances"]
    copies = rep["val"]["identical_to_train"] + rep["test"]["identical_to_train"]
    assert copies / total > 0.5, f"{copies}/{total}"
    assert rep["unique_train_patches"] < rep["train_patch_instances"]


def get_tile_splits_from_names(names, train_ratio=0.8, val_ratio=0.1, seed=42):
    """get_tile_splits без файловой системы: тот же алгоритм на списке имён."""
    tmp = tempfile.mkdtemp(prefix="names_")
    try:
        for n in names:
            open(os.path.join(tmp, n), "w").close()
        return get_tile_splits(tmp, train_ratio, val_ratio, seed)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_split_tile_names_equals_get_tile_splits_and_check_script_accepts_a_names_file():
    names = _tile_names()
    assert split_tile_names(names, seed=7) == get_tile_splits_from_names(names, seed=7)
    # имена с путями и пустыми строками (вывод `ls`, полные пути) не ломают сплит
    messy = ["", "/some/dir/" + names[0]] + names[1:] + ["notes.txt", "  "]
    assert split_tile_names(messy, seed=7) == split_tile_names(names, seed=7)

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import check_leakage

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "names.txt")
        with open(path, "w") as f:
            f.write("\n".join(names))
        check_leakage.main(["--names-file", path, "--patch-size", str(P)])


# --------------------------------------------------------- сплит по позициям


def test_spatial_split_has_no_identical_or_overlapping_patches():
    names = _tile_names()
    sp = get_patch_splits(names, patch_size=P, block=2, seed=0)
    rep = patch_leakage_report(sp, P)
    for pair, r in rep.items():
        assert r["identical"] == 0 and r["overlapping"] == 0, (pair, r)
    assert all(len(sp[k]) > 0 for k in ("train", "val", "test"))


def test_spatial_split_handles_misaligned_rows_and_columns():
    """Последний ряд/столбец тайлов в batching_planck смещён (h - tile_size): патчи
    пересекаются со сдвигом, а не совпадают. Пересечения между сплитами должны быть сняты."""
    ys = list(range(0, H - TILE + 1, P)) + [H - TILE - 7]
    xs = list(range(0, W - TILE + 1, P)) + [W - TILE - 11]
    names = _tile_names(ys, xs)
    for seed in range(5):
        sp = get_patch_splits(names, patch_size=P, block=1, seed=seed)
        rep = patch_leakage_report(sp, P)
        assert all(r["overlapping"] == 0 for r in rep.values()), (seed, rep)


def test_spatial_split_is_deterministic_and_seed_dependent():
    names = _tile_names()
    a = get_patch_splits(names, patch_size=P, seed=1)
    b = get_patch_splits(names, patch_size=P, seed=1)
    c = get_patch_splits(names, patch_size=P, seed=2)
    assert a == b
    assert a != c


def test_spatial_split_ratios_and_coverage():
    names = _tile_names()
    n_unique = len(unique_patch_positions(names, P))
    sp = get_patch_splits(names, patch_size=P, block=1, train_ratio=0.8, val_ratio=0.1, seed=3)
    used = sum(len(v) for v in sp.values())
    assert used <= n_unique
    assert used >= 0.8 * n_unique                    # снято только пересечение, а не пол-датасета
    assert abs(len(sp["train"]) / used - 0.8) < 0.15


def test_blocks_stay_together():
    """Блок 2x2 патча целиком в одном сплите (кроме снятых пересечений)."""
    names = _tile_names()
    sp = get_patch_splits(names, patch_size=P, block=2, seed=0)
    owner = {p: k for k, v in sp.items() for p in v}
    by_block = {}
    for p, k in owner.items():
        by_block.setdefault((p[0] // (2 * P), p[1] // (2 * P)), set()).add(k)
    assert all(len(v) == 1 for v in by_block.values())


def test_same_split_serves_every_frequency():
    """Позиции -- в общей сетке: объединение тайлов частот даёт один сплит для всех."""
    names_a = _tile_names()
    names_b = names_a[: len(names_a) - 9]            # на другой частоте валидны не все тайлы
    sp_union = get_patch_splits(sorted(set(names_a) | set(names_b)), patch_size=P, seed=0)
    sp_alone = get_patch_splits(names_b, patch_size=P, seed=0)
    # на другой частоте нужные позиции просто частично отсутствуют, а не пересчитываются
    avail = unique_patch_positions(names_b, P)
    for k in ("train", "val", "test"):
        assert set(sp_union[k]) & avail <= avail
    assert sp_union != sp_alone, "сплит по подмножеству тайлов -- другой (поэтому делим по объединению)"


def test_save_and_load_patch_splits_roundtrip():
    sp = get_patch_splits(_tile_names(), patch_size=P, seed=0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.json")
        save_patch_splits(path, sp, patch_size=P, block=2, seed=0)
        back, ps = load_patch_splits(path)
        assert ps == P and back == sp


# -------------------------------------------------- честная оценка старого сплита


def test_clean_eval_positions_are_disjoint_from_train():
    names = _tile_names()
    tr, va, te = get_tile_splits_from_names(names, seed=5)
    clean, leaked = clean_eval_positions(tr, va + te, P)
    train_pos = unique_patch_positions(tr, P)
    assert clean and leaked
    assert not (clean & train_pos)
    assert clean | leaked == unique_patch_positions(va + te, P)
    # clean не пересекается по пикселям с train-патчами
    for (cy, cx) in clean:
        assert not any(abs(cy - ty) < P and abs(cx - tx) < P for ty, tx in train_pos)


# --------------------------------------------------------------- датасет


def test_dataset_with_keep_positions_dedups_and_records_positions():
    sky = _sky(3)
    names = _tile_names()
    root = tempfile.mkdtemp(prefix="leak_ds_")
    try:
        _write_tree(root, sky, names)
        pd, ad = os.path.join(root, "Planck/f100"), os.path.join(root, "Planck+ACT/f090")
        keep = unique_patch_positions(names, P)

        legacy = CMBPatchDataset(pd, ad, names, 0.0, 1.0, patch_size=P)
        spatial = CMBPatchDataset(pd, ad, names, 0.0, 1.0, patch_size=P, keep_positions=keep)

        assert len(legacy.pairs) == 4 * len(names)
        assert len(spatial.pairs) == len(keep) < len(legacy.pairs)
        assert sorted(spatial.positions) == sorted(keep)
        assert len(set(spatial.positions)) == len(spatial.positions)

        # содержимое: патч в позиции (py, px) -- ровно этот кусок карты
        k = len(spatial.pairs) // 2
        py, px = spatial.positions[k]
        assert np.array_equal(spatial.pairs[k][1], sky[py:py + P, px:px + P])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_dataset_keep_positions_selects_only_requested_and_skips_absent():
    sky = _sky(4)
    names = _tile_names()
    root = tempfile.mkdtemp(prefix="leak_ds2_")
    try:
        _write_tree(root, sky, names)
        pd, ad = os.path.join(root, "Planck/f100"), os.path.join(root, "Planck+ACT/f090")
        want = set(sorted(unique_patch_positions(names, P))[:10]) | {(10 ** 6, 10 ** 6)}
        ds = CMBPatchDataset(pd, ad, names, 0.0, 1.0, patch_size=P, keep_positions=want)
        assert set(ds.positions) == want - {(10 ** 6, 10 ** 6)}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_normalization_counts_each_unique_patch_once():
    sky = _sky(5)                       # статистика растёт по x -> повторы смещают среднее
    names = _tile_names()
    root = tempfile.mkdtemp(prefix="leak_norm_")
    try:
        _write_tree(root, sky, names)
        pd, ad = os.path.join(root, "Planck/f100"), os.path.join(root, "Planck+ACT/f090")
        keep = unique_patch_positions(names, P)

        mu_u, sig_u = compute_normalization(pd, ad, names, patch_size=P, keep_positions=keep)
        mu_l, sig_l = compute_normalization(pd, ad, names, patch_size=P)

        # эталон: каждая уникальная позиция ровно один раз, оба канала (здесь они совпадают)
        vals = np.concatenate([sky[py:py + P, px:px + P].ravel() for py, px in sorted(keep)] * 2)
        assert abs(mu_u - vals.mean()) < 1e-3 and abs(sig_u - vals.std()) < 1e-3
        assert abs(mu_u - mu_l) > 1e-3, "с дублями среднее другое -- иначе тест ничего не проверяет"
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ------------------------------------------------------------- индексы монитора


def test_spread_indices_cover_the_dataset_evenly():
    idx = spread_indices(1000, 32)
    assert len(idx) == 32 and idx == sorted(set(idx))
    assert idx[0] == 0 and idx[-1] == 999
    assert max(b - a for a, b in zip(idx, idx[1:])) <= 33      # без дыр: не «первые 32»
    assert spread_indices(5, 10) == [0, 1, 2, 3, 4]
    assert spread_indices(10, 1) == [0]
    assert spread_indices(100, 8) != list(range(8))


# ------------------------------------------------------- скрипт, сквозной прогон


def test_train_script_runs_with_spatial_split_for_two_frequencies():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import train_freq

    sky = _sky(6)
    names = _tile_names()
    root = tempfile.mkdtemp(prefix="leak_e2e_")
    out = tempfile.mkdtemp(prefix="leak_out_")
    try:
        _write_tree(root, sky, names, freq_dirs=(("Planck/f100", "Planck+ACT/f090"),
                                                  ("Planck/f143", "Planck+ACT/f150")))
        common = ["--data-root", root, "--patch-size", str(P), "--base-ch", "8", "--time-dim", "64",
                  "--groups", "4", "--epochs", "2", "--global-batch", "4", "--num-workers", "0",
                  "--eval-patches", "3", "--eval-S", "3", "--monitor-n", "3", "--monitor-S", "3",
                  "--monitor-every", "1", "--block", "1", "--ema-start", "20"]
        train_freq.main(["--freq", "90", "--out-dir", os.path.join(out, "f90")] + common)
        train_freq.main(["--freq", "150", "--out-dir", os.path.join(out, "f150")] + common)

        import json
        shared = os.path.join(root, "splits_spatial.json")
        assert os.path.exists(shared), "общий сплит должен создаваться один раз в data-root"
        for f in ("f90", "f150"):
            leak = json.load(open(os.path.join(out, f, "leakage.json")))
            assert all(v["identical"] == 0 and v["overlapping"] == 0
                       for k, v in leak.items() if isinstance(v, dict)), leak
            assert os.path.exists(os.path.join(out, f, "results.json"))
        a, _ = load_patch_splits(os.path.join(out, "f90", "splits.json"))
        b, _ = load_patch_splits(os.path.join(out, "f150", "splits.json"))
        assert a == b, "у обеих частот один и тот же сплит неба"
    finally:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)


def test_train_script_requires_explicit_global_batch():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import train_freq

    try:
        train_freq.parse_args(["--freq", "90", "--data-root", "x", "--out-dir", "y"])
    except SystemExit:
        return
    raise AssertionError("--global-batch должен быть обязательным")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
