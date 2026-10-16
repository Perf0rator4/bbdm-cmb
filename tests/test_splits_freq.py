"""Согласованные сплиты между частотами: inherit_splits и compare_tile_grids.

Запуск: `pytest tests/` или `python tests/test_splits_freq.py`. Данные не нужны.
"""

import os
import shutil
import sys
import tempfile

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bbdm.data.splits import (  # noqa: E402
    compare_tile_grids,
    inherit_splits,
    load_splits,
    save_splits,
)


def _names(prefix, n):
    return [f"tile_y{prefix}{i:03d}_x0.fits" for i in range(n)]


def _master(n=100, seed=0):
    names = _names("", n)
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    return names[:80], names[80:90], names[90:]


def test_known_tiles_keep_their_split():
    """Тайл, уже лежащий в сплите главной частоты, остаётся в том же сплите."""
    m_train, m_val, m_test = _master()
    # на другой частоте валидны не все тайлы
    tiles = m_train[:50] + m_val[:6] + m_test[:7]
    train, val, test = inherit_splits((m_train, m_val, m_test), tiles)

    assert set(train) == set(m_train[:50])
    assert set(val) == set(m_val[:6])
    assert set(test) == set(m_test[:7])


def test_test_tiles_of_master_never_land_in_train():
    """Главное свойство: test главной частоты не попадает в train/val другой."""
    m_train, m_val, m_test = _master()
    new = _names("new", 40)
    train, val, test = inherit_splits((m_train, m_val, m_test),
                                      m_train + m_val + m_test + new)
    assert not (set(m_test) & (set(train) | set(val)))
    assert not (set(m_train) & (set(val) | set(test)))
    assert not (set(m_val) & (set(train) | set(test)))


def test_new_tiles_are_split_deterministically_and_by_ratio():
    m = _master()
    new = _names("new", 100)
    a = inherit_splits(m, list(m[0]) + new, seed=7)
    b = inherit_splits(m, list(m[0]) + new, seed=7)
    c = inherit_splits(m, list(m[0]) + new, seed=8)
    assert a == b
    assert a != c, "другой сид -- другая раздача новых тайлов"

    n_new = [set(s) & set(new) for s in a]
    assert [len(x) for x in n_new] == [80, 10, 10]


def test_splits_are_disjoint_and_cover_everything():
    m = _master()
    new = _names("new", 33)
    tiles = list(m[0][:30]) + list(m[1]) + list(m[2][:5]) + new
    train, val, test = inherit_splits(m, tiles)
    assert sorted(train + val + test) == sorted(set(tiles))
    assert len(set(train) & set(val)) == len(set(train) & set(test)) == len(set(val) & set(test)) == 0


def test_master_frequency_reproduces_itself():
    m = _master()
    train, val, test = inherit_splits(m, m[0] + m[1] + m[2])
    assert (train, val, test) == tuple(sorted(s) for s in m)


def test_overlapping_master_is_rejected():
    try:
        inherit_splits((["a"], ["a"], ["b"]), ["a", "b"])
    except ValueError:
        return
    raise AssertionError("пересекающиеся сплиты главной частоты должны давать ошибку")


def test_save_and_load_roundtrip():
    m = _master()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "splits.json")
        save_splits(path, *m, freq=150)
        assert load_splits(path) == m


def _write_tile(path, crval=(10.0, -30.0), crpix=(480.0, 480.0), shape=(8, 8)):
    h = fits.Header()
    h["CRVAL1"], h["CRVAL2"] = crval
    h["CRPIX1"], h["CRPIX2"] = crpix
    h["CDELT1"], h["CDELT2"] = -1 / 120, 1 / 120
    fits.PrimaryHDU(np.zeros(shape, np.float32), header=h).writeto(path)


def test_compare_tile_grids_detects_a_shifted_grid():
    root = tempfile.mkdtemp(prefix="grids_")
    try:
        a, b = os.path.join(root, "a"), os.path.join(root, "b")
        os.makedirs(a)
        os.makedirs(b)
        for n in ("tile_y0_x0.fits", "tile_y0_x480.fits", "tile_y480_x0.fits"):
            _write_tile(os.path.join(a, n))
            _write_tile(os.path.join(b, n))
        assert compare_tile_grids(a, b) == []

        os.remove(os.path.join(b, "tile_y0_x480.fits"))
        _write_tile(os.path.join(b, "tile_y0_x480.fits"), crval=(10.5, -30.0))
        assert compare_tile_grids(a, b) == ["tile_y0_x480.fits"]

        # tile из одной папки, которого нет в другой, не считается расхождением
        _write_tile(os.path.join(a, "tile_y960_x0.fits"))
        assert compare_tile_grids(a, b) == ["tile_y0_x480.fits"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


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
