"""Датасет патчей и нормализация.

Тайлы 960x960 режутся на четыре патча 480x480. Патчи отбрасываются, если
слишком много замаскированных (нулевых) пикселей ИЛИ если маски Planck и
ACT заметно не совпадают -- у двух инструментов разные футпримы, и
несовпадение проявляется как срезанный угол у одного из патчей пары.
"""

import os

import numpy as np
import torch
from astropy.io import fits
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def load_tile(path):
    with fits.open(path) as hdul:
        data = hdul[0].data
        if data is None:
            raise ValueError(f"No image data in primary HDU: {path}")
        return np.asarray(data, dtype=np.float32)


def tile_to_patches(tile, patch_size=480):
    h, w = tile.shape
    assert (
        h >= patch_size * 2 and w >= patch_size * 2
    ), f"Patch too small: {tile.shape}"

    patches = []
    for i in range(2):
        for j in range(2):
            patch = tile[
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
            ]
            patches.append(patch)
    return patches


def compute_normalization(
    planck_dir, act_dir, train_tiles, patch_size=480, n_sigma=None
):
    """Общие (mu, sigma) по ненулевым пикселям train-тайлов обоих каналов.

    Считается потоково (сумма / сумма квадратов / счётчик), а не через
    конкатенацию всех значений: на ~650 train-тайлах конкатенация -- это
    порядка 5 ГБ во временном массиве.

    Args:
        n_sigma: если задан, делается второй проход с отбрасыванием
            выбросов |x - mu| > n_sigma * sigma (в тексте курсовой описана
            именно clipped-статистика; на Transfer Function это не влияет,
            т.к. предсказание и цель денормализуются одной парой (mu, sigma)
            и нормировка сокращается в отношении).
    """

    def _accumulate(mu_ref=None, sigma_ref=None):
        count = 0
        total = 0.0
        total_sq = 0.0
        for fname in tqdm(train_tiles, desc="Calculating normalization"):
            for directory in (planck_dir, act_dir):
                tile = load_tile(os.path.join(directory, fname))
                for patch in tile_to_patches(tile, patch_size):
                    valid = patch[patch != 0].astype(np.float64)
                    if mu_ref is not None:
                        valid = valid[np.abs(valid - mu_ref) <= n_sigma * sigma_ref]
                    if valid.size == 0:
                        continue
                    count += valid.size
                    total += valid.sum()
                    total_sq += np.square(valid).sum()
        if count == 0:
            raise ValueError("No valid (nonzero) pixels found in train tiles")
        mu = total / count
        var = max(total_sq / count - mu * mu, 0.0)
        return float(mu), float(np.sqrt(var))

    mu, sigma = _accumulate()
    if n_sigma is not None:
        mu, sigma = _accumulate(mu_ref=mu, sigma_ref=sigma)

    if sigma <= 0:
        raise ValueError(f"Degenerate normalization: sigma={sigma}")

    print(f"Normalization: mu={mu:.4f}, sigma={sigma:.4f}")
    return mu, sigma


# Патч и цель обязаны получать ОДНО И ТО ЖЕ преобразование, иначе
# геометрическая связь между входом и целью разрушается.
AUGMENTATIONS = [
    lambda x: x,
    lambda x: np.fliplr(x).copy(),
    lambda x: np.flipud(x).copy(),
    lambda x: np.rot90(x, 2).copy(),
]


class CMBPatchDataset(Dataset):
    """Пары (Planck, ACT+Planck) патчей, нормализованные общими (mu, sigma).

    Аугментации применяются на лету в __getitem__, а не материализуются в
    списке: материализация держала бы в RAM в 4 раза больше патчей
    (~19 ГБ на train-сплите) при нулевом выигрыше в скорости.
    """

    def __init__(
        self,
        planck_dir,
        act_dir,
        tile_list,
        mu,
        sigma,
        augment=False,
        patch_size=480,
        max_zero_frac=0.2,
        max_mask_mismatch_frac=0.01,
    ):
        self.planck_dir = planck_dir
        self.act_dir = act_dir
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.patch_size = patch_size
        self.max_zero_frac = max_zero_frac
        self.max_mask_mismatch_frac = max_mask_mismatch_frac
        self.augmentations = AUGMENTATIONS if augment else AUGMENTATIONS[:1]

        print(f"Loading {len(tile_list)} tiles to RAM...")
        self.pairs = []
        n_dropped_zero = 0
        n_dropped_mismatch = 0

        for fname in tqdm(tile_list, desc="Downloading tiles"):
            tile_p = load_tile(os.path.join(planck_dir, fname))
            tile_a = load_tile(os.path.join(act_dir, fname))

            patches_p = tile_to_patches(tile_p, patch_size)
            patches_a = tile_to_patches(tile_a, patch_size)

            for pp, pa in zip(patches_p, patches_a):
                valid_p = pp != 0
                valid_a = pa != 0

                if (1.0 - valid_p.mean()) > max_zero_frac or (
                    1.0 - valid_a.mean()
                ) > max_zero_frac:
                    n_dropped_zero += 1
                    continue

                # Разные футпринты Planck и ACT DR6: там, где валиден только
                # один из двух, сеть учится восстанавливать край маски, а не
                # небо, и на спектре это даёт ложную высокочастотную мощность.
                mismatch = np.logical_xor(valid_p, valid_a).mean()
                if mismatch > max_mask_mismatch_frac:
                    n_dropped_mismatch += 1
                    continue

                self.pairs.append((np.ascontiguousarray(pp), np.ascontiguousarray(pa)))

        print(
            f"  Loaded {len(self.pairs)} patch pairs "
            f"(dropped {n_dropped_zero} by zero-fraction, "
            f"{n_dropped_mismatch} by Planck/ACT mask mismatch)"
        )
        if augment:
            print(f"  With on-the-fly augmentation: {len(self)} samples")

    def normalize(self, patch):
        return (patch - self.mu) / self.sigma

    def denormalize(self, patch_norm):
        return patch_norm * self.sigma + self.mu

    def __len__(self):
        return len(self.pairs) * len(self.augmentations)

    def __getitem__(self, idx):
        n_aug = len(self.augmentations)
        pp, pa = self.pairs[idx // n_aug]
        aug = self.augmentations[idx % n_aug]

        x0 = torch.from_numpy(self.normalize(aug(pp))).unsqueeze(0).float()
        y = torch.from_numpy(self.normalize(aug(pa))).unsqueeze(0).float()
        return x0, y
