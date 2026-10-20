"""Сплит по области неба (тест = D56) и случайные train-кропы.

Запуск: `pytest tests/` или `python tests/test_region.py`. Данные не нужны.
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bbdm import config as cfg  # noqa: E402
from bbdm.data import (  # noqa: E402
    RandomCropDataset,
    check_tile_wcs,
    crop_audit,
    forbidden_rects,
    load_patch_splits,
    patch_leakage_report,
    radec_to_pixel,
    region_box_px,
    region_splits,
    tile_patch_positions,
)
from bbdm.data.region import _intersects  # noqa: E402

P = 32
TILE = 2 * P
H, W = 8 * P, 12 * P


def _full_map_wcs():
    """WCS полной карты ACT DR6, как у pixell fullsky CAR 0.5' (CRPIX 21601 / 7561)."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---CAR", "DEC--CAR"]
    w.wcs.crval = [0.0, 0.0]
    w.wcs.crpix = [21601.0, 7561.0]
    w.wcs.cdelt = [-1 / 120, 1 / 120]
    return w


def test_pixel_formulas_match_the_tiling_notebook():
    """batching_planck.ipynb, ячейка 35: dec_to_row / ra_to_col и их вывод для D56 (-6..2, 15..40)."""
    y, _ = radec_to_pixel(0.0, -6.0)
    assert round(float(y)) == 6840            # ноутбук печатает 6839 = int(6839.9999...)
    _, x = radec_to_pixel(40.0, 0.0)
    assert round(float(x)) == 16800           # 16799 там же
    _, x = radec_to_pixel(352.0, 0.0)         # RA 352 = -8: справа от RA = 0
    assert round(float(x)) == 21600 + 960
    y, _ = radec_to_pixel(0.0, -63.0)
    assert round(float(y)) == 0               # низ карты


def test_d56_box_in_pixels():
    box = region_box_px(*cfg.TEST_REGIONS["D56"])
    assert box == (6504, 8185, 16728, 22561), box
    # 680 кв. град. в описании поля: (14 град x 48.6 град) с cos(dec) ~ 0.99
    area = (box[1] - box[0]) * (box[3] - box[2]) / 120 ** 2
    assert 670 < area < 690, area


def test_region_crossing_the_map_seam_is_rejected():
    try:
        region_box_px(170.0, -170.0, -5, 5)
    except ValueError:
        return
    raise AssertionError("переход через RA = 180 должен давать ошибку")


def test_check_tile_wcs_agrees_with_a_tile_cut_like_the_notebook():
    w = _full_map_wcs()
    for y, x in ((6720, 16800), (0, 0), (9360, 42240)):
        hdr = w.slice((slice(y, y + 960), slice(x, x + 960))).to_header()
        hdr["NAXIS"], hdr["NAXIS1"], hdr["NAXIS2"] = 2, 960, 960
        assert check_tile_wcs(hdr, f"tile_y{y}_x{x}.fits") < 0.01
    hdr = w.slice((slice(6720, 7680), slice(16800, 17760))).to_header()
    hdr["NAXIS"], hdr["NAXIS1"], hdr["NAXIS2"] = 2, 960, 960
    assert check_tile_wcs(hdr, "tile_y7200_x16800.fits") > 400    # имя не от этого тайла


def _names(ys=None, xs=None):
    ys = ys if ys is not None else range(0, H - TILE + 1, P)
    xs = xs if xs is not None else range(0, W - TILE + 1, P)
    return [f"tile_y{y}_x{x}.fits" for y in ys for x in xs]


BOX = (2 * P + 5, 5 * P - 3, 3 * P - 7, 8 * P + 2)    # поле, не выровненное по сетке патчей


def test_region_split_test_inside_rest_outside_and_boundary_dropped():
    sp, info = region_splits(_names(), BOX, patch_size=P, val_ratio=0.2, block=1, seed=0)
    assert sp["test"], "в поле должны быть целые патчи"
    for q in sp["test"]:
        assert BOX[0] <= q[0] and q[0] + P <= BOX[1] and BOX[2] <= q[1] and q[1] + P <= BOX[3]
    for k in ("train", "val"):
        assert not any(_intersects(q, BOX, P) for q in sp[k]), k
    assert info["n_dropped_on_region_boundary"] > 0
    rep = patch_leakage_report(sp, patch_size=P)
    assert all(v["identical"] == 0 and v["overlapping"] == 0 for v in rep.values()), rep
    n_all = len({q for t in _names() for q in tile_patch_positions(t, P)})
    assert len(sp["train"]) + len(sp["val"]) + len(sp["test"]) + info["n_dropped_on_region_boundary"] \
        + info["n_train_dropped_for_touching_val"] == n_all


def test_region_split_guard_keeps_train_away():
    sp, _ = region_splits(_names(), BOX, patch_size=P, val_ratio=0.1, block=1, seed=0, guard=P)
    g = (BOX[0] - P, BOX[1] + P, BOX[2] - P, BOX[3] + P)
    assert not any(_intersects(q, g, P) for q in sp["train"] + sp["val"])


def _sky(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((H, W)) * np.linspace(1, 5, W)[None, :] + 10).astype(np.float32)


def _write_tree(root, sky, names, freq_dirs=(("Planck/f100", "Planck+ACT/f090"),), hole=None):
    for pd, ad in freq_dirs:
        for d in (pd, ad):
            os.makedirs(os.path.join(root, d), exist_ok=True)
        for n in names:
            y = int(n.split("_y")[1].split("_")[0])
            x = int(n.split("_x")[1].split(".")[0])
            tile = sky[y:y + TILE, x:x + TILE].copy()
            tile_a = tile.copy()
            if hole is not None:      # дыра только в ACT -> несовпадение масок
                hy0, hy1, hx0, hx1 = hole
                sub = tile_a[max(hy0 - y, 0):max(hy1 - y, 0), max(hx0 - x, 0):max(hx1 - x, 0)]
                sub[...] = 0
            h = fits.Header()
            h["CDELT1"], h["CDELT2"] = -1 / 120, 1 / 120
            fits.PrimaryHDU(tile * 0.5, header=h).writeto(os.path.join(root, pd, n))
            fits.PrimaryHDU(tile_a, header=h).writeto(os.path.join(root, ad, n))


def _crop_ds(root, sp, names, **kw):
    return RandomCropDataset(
        os.path.join(root, "Planck/f100"), os.path.join(root, "Planck+ACT/f090"), names,
        mu=0.0, sigma=1.0, forbidden=forbidden_rects(sp, BOX, P), n_samples=50,
        patch_size=P, **kw)


def test_random_crops_read_the_right_pixels_and_avoid_val_and_test():
    sky = _sky(1)
    names = _names()
    root = tempfile.mkdtemp(prefix="region_")
    try:
        _write_tree(root, sky, names)
        sp, _ = region_splits(names, BOX, patch_size=P, val_ratio=0.2, block=1, seed=0)
        ds = _crop_ds(root, sp, names, augment=False)
        rng = np.random.default_rng(0)
        offsets = set()
        for _ in range(200):
            i, oy, ox = ds.sample_window(rng)
            ty, tx = ds.tile_pos[i]
            assert not _intersects((ty + oy, tx + ox), BOX, P)
            offsets.add((oy % P, ox % P))
            pp, pa = ds.read_crop(i, oy, ox)
            np.testing.assert_array_equal(pa, sky[ty + oy:ty + oy + P, tx + ox:tx + ox + P])
            np.testing.assert_allclose(pp, 0.5 * pa)
        assert len(offsets) > 50, "смещения должны быть произвольными, а не кратными патчу"
        audit = crop_audit(ds, sp, n=3000)
        assert audit["crops_touching_val"] == 0 and audit["crops_touching_test"] == 0, audit
        x0, y = ds[0]
        assert x0.shape == (1, P, P) and y.shape == (1, P, P)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_crop_audit_catches_a_dataset_without_the_forbidden_area():
    """Аудит не тривиален: датасет без запрета на поле действительно залезает в test."""
    sky = _sky(2)
    names = _names()
    root = tempfile.mkdtemp(prefix="region_")
    try:
        _write_tree(root, sky, names)
        sp, _ = region_splits(names, BOX, patch_size=P, val_ratio=0.2, block=1, seed=0)
        ds = RandomCropDataset(os.path.join(root, "Planck/f100"), os.path.join(root, "Planck+ACT/f090"),
                               names, 0.0, 1.0, forbidden=[], n_samples=10, patch_size=P)
        audit = crop_audit(ds, sp, n=2000)
        assert audit["crops_touching_test"] > 0 and audit["crops_touching_val"] > 0, audit
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_random_crops_apply_the_mask_mismatch_filter():
    sky = _sky(3)
    names = _names()
    root = tempfile.mkdtemp(prefix="region_")
    try:
        hole = (0, H, 0, 2 * P)      # левые два столбца патчей: ACT пустой, Planck нет
        _write_tree(root, sky, names, hole=hole)
        sp, _ = region_splits(names, BOX, patch_size=P, val_ratio=0.0, block=1, seed=0)
        ds = _crop_ds(root, sp, names, augment=False)
        torch.manual_seed(0)
        for _ in range(100):
            x0, y = ds[0]
            assert (y != 0).float().mean() > 0.95
        assert ds.n_rejected["zero"] + ds.n_rejected["mismatch"] > 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_workers_draw_different_crops():
    sky = _sky(4)
    names = _names()
    root = tempfile.mkdtemp(prefix="region_")
    try:
        _write_tree(root, sky, names)
        sp, _ = region_splits(names, BOX, patch_size=P, val_ratio=0.2, block=1, seed=0)
        ds = _crop_ds(root, sp, names)
        torch.manual_seed(0)
        dl = torch.utils.data.DataLoader(ds, batch_size=5, num_workers=2)
        batches = [y for _, y in dl]
        flat = torch.cat(batches).flatten(1)
        assert len({tuple(r[:8].tolist()) for r in flat}) == len(flat), "кропы повторяются"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_train_script_region_mode_end_to_end():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import json

    import train_freq

    sky = _sky(5)
    names = _names()
    root = tempfile.mkdtemp(prefix="region_e2e_")
    out = tempfile.mkdtemp(prefix="region_out_")
    try:
        _write_tree(root, sky, names, freq_dirs=(("Planck/f100", "Planck+ACT/f090"),
                                                  ("Planck/f143", "Planck+ACT/f150")))
        common = ["--data-root", root, "--patch-size", str(P), "--base-ch", "8", "--time-dim", "64",
                  "--groups", "4", "--epochs", "2", "--global-batch", "4", "--num-workers", "0",
                  "--eval-patches", "3", "--eval-S", "3", "--monitor-n", "3", "--monitor-S", "3",
                  "--monitor-every", "1", "--block", "1", "--ema-start", "5",
                  "--test-box-px", ",".join(str(v) for v in BOX), "--crops-per-epoch", "16"]
        train_freq.main(["--freq", "90", "--out-dir", os.path.join(out, "f90")] + common)
        train_freq.main(["--freq", "150", "--out-dir", os.path.join(out, "f150")] + common)
        assert os.path.exists(os.path.join(root, "splits_region_custom_px.json"))
        for f in ("f90", "f150"):
            leak = json.load(open(os.path.join(out, f, "leakage.json")))
            assert leak["mode"] == "region"
            assert all(v["overlapping"] == 0 for k, v in leak.items()
                       if isinstance(v, dict) and "overlapping" in v), leak
            assert leak["train_crops"]["crops_touching_test"] == 0
            assert leak["train_crops"]["crops_touching_val"] == 0
            res = json.load(open(os.path.join(out, f, "results.json")))
            assert res["test"]["n_patches"] > 0
        a, _ = load_patch_splits(os.path.join(out, "f90", "splits.json"))
        b, _ = load_patch_splits(os.path.join(out, "f150", "splits.json"))
        assert a == b
        for q in a["test"]:
            assert BOX[0] <= q[0] and q[0] + P <= BOX[1] and BOX[2] <= q[1] and q[1] + P <= BOX[3]
        ck = torch.load(os.path.join(out, "f90", "last.pt"), map_location="cpu", weights_only=False)
        assert ck["extra"]["train_crops"] is True and ck["extra"]["split_sizes"]["train"] == 16
    finally:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)


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
