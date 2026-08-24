"""Проверки пути данных: фильтры патчей, нормализация, аугментации.

Работает на синтетических FITS во временной папке -- реальные тайлы не нужны.
Запуск: `pytest tests/` или `python tests/test_dataset.py`.
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import torch
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bbdm.data.dataset import (  # noqa: E402
    CMBPatchDataset,
    compute_normalization,
    tile_to_patches,
)

TILE = 128
PATCH = 64


class _FakeTiles:
    """Контекст с парой папок синтетических тайлов."""

    def __init__(self, planck_tiles, act_tiles):
        self.planck_tiles = planck_tiles
        self.act_tiles = act_tiles

    def __enter__(self):
        self.root = tempfile.mkdtemp(prefix="bbdm_test_")
        self.planck_dir = os.path.join(self.root, "planck")
        self.act_dir = os.path.join(self.root, "act")
        os.makedirs(self.planck_dir)
        os.makedirs(self.act_dir)
        self.names = []
        for i, (p, a) in enumerate(zip(self.planck_tiles, self.act_tiles)):
            name = f"tile_{i:03d}.fits"
            fits.PrimaryHDU(p.astype(np.float32)).writeto(
                os.path.join(self.planck_dir, name)
            )
            fits.PrimaryHDU(a.astype(np.float32)).writeto(
                os.path.join(self.act_dir, name)
            )
            self.names.append(name)
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.root, ignore_errors=True)


def _clean_tile(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((TILE, TILE)).astype(np.float32) * 100 - 15


def _dataset(planck, act, **kwargs):
    with _FakeTiles([planck], [act]) as t:
        return CMBPatchDataset(
            planck_dir=t.planck_dir,
            act_dir=t.act_dir,
            tile_list=t.names,
            mu=0.0,
            sigma=1.0,
            patch_size=PATCH,
            **kwargs,
        )


def test_tile_to_patches_covers_tile_without_overlap():
    tile = np.arange(TILE * TILE, dtype=np.float32).reshape(TILE, TILE)
    patches = tile_to_patches(tile, PATCH)
    assert len(patches) == 4
    assert all(p.shape == (PATCH, PATCH) for p in patches)
    assert sorted(np.concatenate([p.ravel() for p in patches])).__len__() == TILE * TILE
    assert len(set(np.concatenate([p.ravel() for p in patches]))) == TILE * TILE


def test_mask_mismatch_filter_drops_planck_only_hole():
    """Дыра только в Planck при валидном ACT -- пара обязана отсеяться.

    Это и есть артефакт разных футпринтов: сеть на таких парах учится
    восстанавливать край маски, а не небо, и на спектре это даёт ложную
    высокочастотную мощность.
    """
    planck, act = _clean_tile(1), _clean_tile(2)
    # 10% нижне-правого патча -- ниже MAX_ZERO_FRAC, так что отсеять его
    # может ТОЛЬКО фильтр рассогласования масок.
    planck[-21:, -20:] = 0.0

    kept = _dataset(planck, act, max_zero_frac=0.2, max_mask_mismatch_frac=0.01)
    assert len(kept.pairs) == 3, "патч с дырой только в Planck должен выпасть"

    off = _dataset(planck, act, max_zero_frac=0.2, max_mask_mismatch_frac=1.0)
    assert len(off.pairs) == 4, "с выключенным порогом остаются все четыре"


def test_matching_masks_survive_the_filter():
    """Одинаковая маска в обоих каналах -- рассогласования нет, пара остаётся."""
    planck, act = _clean_tile(3), _clean_tile(4)
    planck[:20, :20] = 0.0
    act[:20, :20] = 0.0

    ds = _dataset(planck, act, max_zero_frac=0.2, max_mask_mismatch_frac=0.01)
    assert len(ds.pairs) == 4


def test_zero_fraction_filter():
    planck, act = _clean_tile(5), _clean_tile(6)
    planck[:PATCH, :PATCH] = 0.0
    act[:PATCH, :PATCH] = 0.0  # маски согласованы, срабатывает только zero-frac

    ds = _dataset(planck, act, max_zero_frac=0.2, max_mask_mismatch_frac=0.01)
    assert len(ds.pairs) == 3


def test_augmentation_is_lazy_and_paired():
    """Аугментация применяется на лету и ОДИНАКОВО к входу и цели."""
    planck, act = _clean_tile(7), _clean_tile(8)
    ds = _dataset(planck, act, augment=True, max_mask_mismatch_frac=1.0)

    assert len(ds) == 4 * len(ds.pairs)
    # Список пар не раздут в 4 раза -- иначе на train-сплите это ~19 ГБ RAM.
    assert len(ds.pairs) == 4

    base_x, base_y = ds[0]
    flip_x, flip_y = ds[1]
    assert torch.equal(base_x.flip(-1), flip_x)
    assert torch.equal(base_y.flip(-1), flip_y), "цель должна флипаться так же"

    plain = _dataset(planck, act, augment=False, max_mask_mismatch_frac=1.0)
    assert len(plain) == len(plain.pairs)


def test_normalization_matches_direct_computation():
    """Потоковая (mu, sigma) должна совпасть с прямым подсчётом по всем пикселям."""
    planck, act = _clean_tile(9), _clean_tile(10)
    planck[:10, :10] = 0.0  # нули исключаются из статистики

    with _FakeTiles([planck], [act]) as t:
        mu, sigma = compute_normalization(
            t.planck_dir, t.act_dir, t.names, patch_size=PATCH
        )

    both = np.concatenate([planck.ravel(), act.ravel()])
    both = both[both != 0].astype(np.float64)
    assert abs(mu - both.mean()) < 1e-4
    assert abs(sigma - both.std()) < 1e-4


def test_sigma_clipped_normalization_is_narrower():
    """С n_sigma выбросы отбрасываются, sigma обязана уменьшиться."""
    planck, act = _clean_tile(11), _clean_tile(12)
    planck[0, :50] = 5000.0  # выбросы

    with _FakeTiles([planck], [act]) as t:
        plain = compute_normalization(
            t.planck_dir, t.act_dir, t.names, patch_size=PATCH
        )
        clipped = compute_normalization(
            t.planck_dir, t.act_dir, t.names, patch_size=PATCH, n_sigma=3.0
        )
    assert clipped[1] < plain[1]


def test_denormalize_roundtrip():
    planck, act = _clean_tile(13), _clean_tile(14)
    ds = _dataset(planck, act, max_mask_mismatch_frac=1.0)
    ds.mu, ds.sigma = -15.0, 113.0

    x0, _ = ds[0]
    restored = ds.denormalize(x0[0].numpy())
    assert np.allclose(restored, ds.pairs[0][0], atol=1e-2)


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
