from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from astropy.stats import mad_std, sigma_clip

from cfa_unet import CrossFrequencyAttentionUNet


TRAIN_CHANNEL_MIN = np.array([
    -519.5077514648438,
    -537.7560424804688,
    -1109.919921875,
], dtype=np.float32)

TRAIN_CHANNEL_MAX = np.array([
    457.490966796875,
    507.04071044921875,
    1114.9049072265625,
], dtype=np.float32)


class CFAUNetInference:
    def __init__(self, weights_path, device=None):
        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(weights_path)

        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = CrossFrequencyAttentionUNet()
        self.model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
        self.model = self.model.to(self.device).eval()

    @staticmethod
    def _preprocess_tile(tile):
        result = np.empty_like(tile, dtype=np.float32)

        for ch in range(3):
            channel = tile[ch]
            valid = np.isfinite(channel)
            if not valid.any():
                raise ValueError(f"Channel {ch} contains no finite pixels")

            values = channel[valid]
            clipped = sigma_clip(
                values,
                sigma=3.0,
                maxiters=5,
                cenfunc="median",
                stdfunc="mad_std",
            )
            kept = clipped.data[~np.ma.getmaskarray(clipped)]

            channel_clipped = channel.copy()
            if kept.size > 0:
                center = np.median(kept)
                sigma = mad_std(kept)
                channel_clipped[valid] = np.clip(
                    values,
                    center - 3.0 * sigma,
                    center + 3.0 * sigma,
                )

            result[ch] = np.clip(
                (channel_clipped - TRAIN_CHANNEL_MIN[ch])
                / (TRAIN_CHANNEL_MAX[ch] - TRAIN_CHANNEL_MIN[ch]),
                0.0,
                1.0,
            )

        if not np.isfinite(result).all():
            raise ValueError("Non-finite values after preprocessing")
        return result

    @staticmethod
    def _as_batch(images):
        if isinstance(images, (list, tuple)):
            if not images:
                raise ValueError("At least one image is required")
            images = np.stack([
                image.detach().cpu().numpy()
                if isinstance(image, torch.Tensor) else np.asarray(image)
                for image in images
            ])
        elif isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()
        else:
            images = np.asarray(images)

        if images.ndim == 3:
            images = images[None]
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"Expected (3, H, W) or (N, 3, H, W), got {images.shape}"
            )
        if len(images) == 0:
            raise ValueError("At least one image is required")
        if min(images.shape[-2:]) < 8 or images.shape[-2] % 8 or images.shape[-1] % 8:
            raise ValueError("Image height and width must be at least 8 and divisible by 8")

        return images

    @staticmethod
    def _extract_points(pred, threshold=0.1):
        heatmaps = pred[:, 0]
        pooled = F.max_pool2d(
            heatmaps.unsqueeze(1),
            kernel_size=9,
            stride=1,
            padding=4,
        )[:, 0]
        peaks = (
            (heatmaps == pooled)
            & torch.isfinite(heatmaps)
            & (heatmaps >= threshold)
        )

        points = []
        scores = []
        for heatmap, peak_mask in zip(heatmaps, peaks):
            points.append(
                torch.nonzero(peak_mask, as_tuple=False)[:, [1, 0]]
                .float().cpu().numpy()
            )
            scores.append(
                heatmap[peak_mask].float().cpu().numpy()
            )

        return points, scores

    @torch.inference_mode()
    def predict(self, images, batch_size=None, threshold=0.1):
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be finite and between 0 and 1")
        images = self._as_batch(images)
        if batch_size is None:
            batch_size = len(images)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer or None")

        heatmaps = []
        points = []
        scores = []

        for start in range(0, len(images), batch_size):
            batch = np.stack([
                self._preprocess_tile(np.asarray(tile, dtype=np.float32))
                for tile in images[start:start + batch_size]
            ])
            batch = torch.from_numpy(batch).to(self.device)
            pred = self.model(batch)

            heatmaps.append(pred[:, 0].float().cpu().numpy())
            batch_points, batch_scores = self._extract_points(pred, threshold=threshold)
            points.extend(batch_points)
            scores.extend(batch_scores)

        return np.concatenate(heatmaps, axis=0), points, scores
