"""scripts/make_test_set.py на синтетическом небе: окна, общие для частот, пиксели и WCS.

Запуск: `pytest tests/` или `python tests/test_make_test_set.py`. Данные не нужны.
"""

import csv
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from bbdm.data import radec_to_pixel, save_patch_splits  # noqa: E402
from bbdm.model import BBDM, UNet  # noqa: E402
from test_region import BOX, P, TILE, _full_map_wcs, _names, _sky  # noqa: E402

FREQS = {90: ("Planck/f100", "Planck+ACT/f090"), 150: ("Planck/f143", "Planck+ACT/f150"),
         220: ("Planck/f217", "Planck+ACT/f220")}
HOLE = (BOX[0], BOX[0] + 6, BOX[2], BOX[2] + 6)    # дыра в ACT 150 в углу поля


def _write_tiles(root, sky):
    full = _full_map_wcs()
    for f, (pd, ad) in FREQS.items():
        for d in (pd, ad):
            os.makedirs(os.path.join(root, d), exist_ok=True)
        for n in _names():
            y = int(n.split("_y")[1].split("_")[0])
            x = int(n.split("_x")[1].split(".")[0])
            h = full[y:y + TILE, x:x + TILE].to_header()
            tile = sky[y:y + TILE, x:x + TILE].copy()
            act = tile.copy()
            if f == 150:
                act[max(HOLE[0] - y, 0):max(HOLE[1] - y, 0), max(HOLE[2] - x, 0):max(HOLE[3] - x, 0)] = 0
            fits.PrimaryHDU(tile * 0.5, header=h).writeto(os.path.join(root, pd, n))
            fits.PrimaryHDU(act, header=h).writeto(os.path.join(root, ad, n))


def _write_run(run, freq):
    os.makedirs(run)
    save_patch_splits(os.path.join(run, "splits.json"), {"train": [], "val": [], "test": []},
                      patch_size=P, mode="region", box_px=list(BOX), guard_px=0)
    torch.manual_seed(freq)
    unet = UNet(in_ch=1, base_ch=8, time_dim=64, groups=4, cond_ch=1, upsample="resize_conv")
    bbdm = BBDM(unet, T=20)
    torch.save({"model": bbdm.state_dict(), "epoch": 0, "hparams": {"groups": 4},
                "extra": {"freq": freq, "mu": 10.0, "sigma": 3.0}}, os.path.join(run, "best.pt"))


def test_make_test_set_end_to_end():
    import make_test_set

    sky = _sky(7)
    root = tempfile.mkdtemp(prefix="testset_data_")
    out = tempfile.mkdtemp(prefix="testset_out_")
    try:
        _write_tiles(root, sky)
        runs = [os.path.join(out, f"run{f}") for f in FREQS]
        for r, f in zip(runs, FREQS):
            _write_run(r, f)
        dest = os.path.join(out, "set")
        make_test_set.main(["--data-root", root, "--runs", *runs, "--out-dir", dest,
                            "--S", "3", "--device", "cpu"])

        rows = list(csv.DictReader(open(os.path.join(dest, "index.csv"))))
        grid = make_test_set.window_grid(BOX, P, P // 2)
        assert 0 < len(rows) < len(grid)                       # дыра в 150 ГГц выкинула окна
        for row in rows:
            y, x = int(row["y"]), int(row["x"])
            assert BOX[0] <= y and y + P <= BOX[1] and BOX[2] <= x and x + P <= BOX[3]
            assert not (y < HOLE[1] and HOLE[0] < y + P and x < HOLE[3] and HOLE[2] < x + P)
            gy, gx = radec_to_pixel(float(row["ra_center"]), float(row["dec_center"]))
            assert abs(gy - (y + (P - 1) / 2)) < 0.01 and abs(gx - (x + (P - 1) / 2)) < 0.01
        man = json.load(open(os.path.join(dest, "manifest.json")))
        assert man["window_report"]["filtered_150"] > 0 and man["window_report"]["filtered_90"] == 0

        names = sorted(r["name"] for r in rows)
        for f in FREQS:
            for sub, suffix in (("true", ""), ("planck", ""), ("pred", "_s0")):
                files = sorted(os.listdir(os.path.join(dest, f"f{f}", sub)))
                assert files == sorted(n + suffix + ".fits" for n in names), (f, sub)
        r = rows[len(rows) // 2]
        y, x = int(r["y"]), int(r["x"])
        with fits.open(os.path.join(dest, "f90", "true", r["name"] + ".fits")) as h:
            np.testing.assert_array_equal(h[0].data, sky[y:y + P, x:x + P])
            assert h[0].header["BUNIT"] == "uK"
            ra, dec = WCS(h[0].header).wcs_pix2world(0, 0, 0)
            gy, gx = radec_to_pixel(ra, dec)
            assert abs(gy - y) < 0.01 and abs(gx - x) < 0.01
        with fits.open(os.path.join(dest, "f220", "pred", r["name"] + "_s0.fits")) as h:
            assert h[0].data.shape == (P, P) and np.isfinite(h[0].data).all()
    finally:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)


def test_rerun_needs_overwrite_and_leaves_no_stale_files():
    import make_test_set

    root = tempfile.mkdtemp(prefix="testset_data_")
    out = tempfile.mkdtemp(prefix="testset_out_")
    try:
        _write_tiles(root, _sky(7))
        runs = [os.path.join(out, f"run{f}") for f in FREQS]
        for r, f in zip(runs, FREQS):
            _write_run(r, f)
        dest = os.path.join(out, "set")
        base = ["--data-root", root, "--runs", *runs, "--out-dir", dest, "--S", "3", "--device", "cpu"]
        make_test_set.main(base + ["--n-samples", "2"])
        try:
            make_test_set.main(base)
        except FileExistsError:
            pass
        else:
            raise AssertionError("a rerun into a filled out-dir was accepted")

        # упавший запуск: FITS есть, index.csv / manifest.json нет -- тоже отказ
        for name in ("index.csv", "manifest.json"):
            os.remove(os.path.join(dest, name))
        try:
            make_test_set.main(base)
        except FileExistsError:
            pass
        else:
            raise AssertionError("a rerun over a crashed run's FITS was accepted")

        # --overwrite с ошибкой в --runs падает до удаления: старые FITS на месте
        before = sorted(os.listdir(os.path.join(dest, "f90", "pred")))
        bad = ["--data-root", root, "--runs", os.path.join(out, "nope"), "--out-dir", dest,
               "--S", "3", "--device", "cpu", "--overwrite"]
        try:
            make_test_set.main(bad)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("a nonexistent run was accepted")
        assert sorted(os.listdir(os.path.join(dest, "f90", "pred"))) == before and before

        make_test_set.main(base + ["--overwrite", "--stride", str(P)])   # меньше окон, 1 сэмпл
        names = sorted(r["name"] for r in csv.DictReader(open(os.path.join(dest, "index.csv"))))
        assert 0 < len(names) <= len(make_test_set.window_grid(BOX, P, P))
        for f in FREQS:
            for sub, suffix in (("true", ""), ("planck", ""), ("pred", "_s0")):
                files = sorted(os.listdir(os.path.join(dest, f"f{f}", sub)))
                assert files == sorted(n + suffix + ".fits" for n in names), (f, sub)
    finally:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(out, ignore_errors=True)


def test_runs_without_region_split_are_refused():
    import make_test_set

    out = tempfile.mkdtemp(prefix="testset_bad_")
    try:
        save_patch_splits(os.path.join(out, "splits.json"), {"train": [], "val": [], "test": []},
                          patch_size=P, mode="spatial")
        try:
            make_test_set.load_runs([out], "best", "cpu")
        except RuntimeError as e:
            assert "region" in str(e)
        else:
            raise AssertionError("a spatial-split run was accepted")
    finally:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    test_make_test_set_end_to_end()
    test_rerun_needs_overwrite_and_leaves_no_stale_files()
    test_runs_without_region_split_are_refused()
    print("ok")
