"""Aligned three-frequency crops; one sky window and augmentation for all channels."""

import numpy as np
import torch

from bbdm.config import data_dirs
from bbdm.data.region import RandomCropDataset
from bbdm.data.splits import compare_tile_grids, list_tiles


class JointPatchDataset(RandomCropDataset):
    frequencies = (90, 150, 220)

    def __init__(self, data_root, mu, sigma, *, positions=None, forbidden=(),
                 n_samples=0, augment=False, **kwargs):
        self.directories = [data_dirs(freq, data_root) for freq in self.frequencies]
        tiles = sorted(set.intersection(*[
            set(list_tiles(directory)) for pair in self.directories for directory in pair]))
        if not tiles:
            raise ValueError("No common Planck/ACT tiles at all three frequencies")
        reference = self.directories[0][1]
        for pair in self.directories:
            for directory in pair:
                if compare_tile_grids(reference, directory):
                    raise ValueError(f"{directory}: frequency maps do not share a WCS grid")
        super().__init__(*self.directories[0], tiles, 0.0, 1.0, forbidden,
                         n_samples, augment=augment, **kwargs)
        self.mu, self.sigma = np.asarray(mu, dtype=np.float32), np.asarray(sigma, dtype=np.float32)
        if self.mu.shape != (3,) or self.sigma.shape != (3,) or not np.isfinite(self.mu).all() or not np.isfinite(self.sigma).all() or np.any(self.sigma <= 0):
            raise ValueError("Joint normalization requires three finite means and positive standard deviations")
        self.records = None
        self.positions = []
        if positions is not None:
            self.records = []
            p = self.patch_size
            th, tw = self.tile_size
            for y, x in sorted(positions):
                record = next(((i, y - ty, x - tx) for i, (_, ty, tx) in enumerate(self.tiles)
                               if ty <= y <= ty + th - p and tx <= x <= tx + tw - p), None)
                if record is None:
                    continue
                pp, pa = self.read_crop(*record)
                vp, va = pp != 0, pa != 0
                if (np.any(1 - vp.mean((-2, -1)) > self.max_zero_frac)
                        or np.any(1 - va.mean((-2, -1)) > self.max_zero_frac)
                        or np.any(np.logical_xor(vp, va).mean((-2, -1)) > self.max_mask_mismatch_frac)):
                    continue
                self.records.append(record)
                self.positions.append((y, x))
            print(f"Joint patches: {len(self.records)} of {len(positions)} positions pass all frequencies", flush=True)
            if not self.records:
                raise ValueError("No common valid fixed patches at all frequencies")

    def __len__(self):
        return (len(self.records) * len(self.augmentations)
                if self.records is not None else super().__len__())

    def read_crop(self, i, oy, ox):
        name = self.tiles[i][0]
        pairs = [(self._read_rows(pd, name, oy, ox), self._read_rows(ad, name, oy, ox))
                 for pd, ad in self.directories]
        if not all(np.isfinite(image).all() for pair in pairs for image in pair):
            raise ValueError(f"{name}: non-finite pixels in joint crop")
        return np.stack([pair[0] for pair in pairs]), np.stack([pair[1] for pair in pairs])

    def normalize(self, patch):
        return (patch - self.mu[:, None, None]) / self.sigma[:, None, None]

    def denormalize(self, patch):
        return patch * self.sigma[:, None, None] + self.mu[:, None, None]

    def __getitem__(self, idx):
        if self.records is None:
            return super().__getitem__(idx)
        aug = self.augmentations[idx % len(self.augmentations)]
        pp, pa = self.read_crop(*self.records[idx // len(self.augmentations)])
        return (torch.from_numpy(self.normalize(aug(pp))).float(),
                torch.from_numpy(self.normalize(aug(pa))).float())
