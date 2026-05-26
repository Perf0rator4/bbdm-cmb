import os
import numpy as np
import torch
from torch.utils.data import Dataset
from astropy.io import fits
from tqdm import tqdm


def load_tile(path):
    with fits.open(path) as hdul:
        return hdul[0].data.astype(np.float32)


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


def compute_normalization(planck_dir, act_dir, train_tiles, patch_size=480):
    all_values = []

    for fname in tqdm(train_tiles, desc="Calculating normalization"):
        tile_p = load_tile(os.path.join(planck_dir, fname))
        tile_a = load_tile(os.path.join(act_dir, fname))

        for patch in tile_to_patches(tile_p, patch_size):
            valid = patch[patch != 0]
            if valid.size > 0:
                all_values.append(valid)

        for patch in tile_to_patches(tile_a, patch_size):
            valid = patch[patch != 0]
            if valid.size > 0:
                all_values.append(valid)

    all_values = np.concatenate(all_values)
    mu = float(np.mean(all_values))
    sigma = float(np.std(all_values))

    print(f"Normalization: mu={mu:.4f}, sigma={sigma:.4f}")
    return mu, sigma


AUGMENTATIONS = [
    lambda x: x,
    lambda x: np.fliplr(x).copy(),
    lambda x: np.flipud(x).copy(),
    lambda x: np.rot90(x, 2).copy(),
]


class CMBPatchDataset(Dataset):
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
    ):
        self.planck_dir = planck_dir
        self.act_dir = act_dir
        self.mu = mu
        self.sigma = sigma
        self.augment = augment
        self.patch_size = patch_size
        self.max_zero_frac = max_zero_frac

        print(f"Loading {len(tile_list)} tiles to RAM...")
        self.pairs = []

        for fname in tqdm(tile_list, desc="Downloading tiles"):
            tile_p = load_tile(os.path.join(planck_dir, fname))
            tile_a = load_tile(os.path.join(act_dir, fname))

            patches_p = tile_to_patches(tile_p, patch_size)
            patches_a = tile_to_patches(tile_a, patch_size)

            for pp, pa in zip(patches_p, patches_a):
                if (pp == 0).mean() > max_zero_frac or (pa == 0).mean() > max_zero_frac:
                    continue
                self.pairs.append((pp, pa))

        print(f"  Loaded {len(self.pairs)} patches after filtering")

        if augment:
            augmented = []
            for pp, pa in tqdm(self.pairs, desc="Augmenting"):
                for aug in AUGMENTATIONS:
                    augmented.append((aug(pp), aug(pa)))
            self.pairs = augmented
            print(f"After augmentation: {len(self.pairs)} patches")

    def normalize(self, patch):
        return (patch - self.mu) / self.sigma

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pp, pa = self.pairs[idx]
        pp_norm = self.normalize(pp)
        pa_norm = self.normalize(pa)
        x0 = torch.tensor(pp_norm, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(pa_norm, dtype=torch.float32).unsqueeze(0)
        return x0, y
