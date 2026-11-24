"""Frozen CFA perceptual loss on jointly sampled ACT maps (90, 150, 220 GHz)."""

import hashlib
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from cfa_unet import CrossFrequencyAttentionUNet
from cfa_unet_inference import TRAIN_CHANNEL_MIN, TRAIN_CHANNEL_MAX


def preprocess_cfa(images):
    """CFA's sigma clipping with detached robust bounds; pixel gradients survive.

    Matches the inference preprocessing, including median/MAD and five clipping
    iterations. Statistics are constants in backward, not trainable quantities.
    """
    if images.ndim != 4 or images.shape[1] != 3 or not torch.isfinite(images).all():
        raise ValueError("CFA requires finite (B, 3, H, W) maps in uK")
    with torch.no_grad():
        values = images.detach().float().flatten(2)
        kept = values.clone()
        for _ in range(5):
            center = torch.nanquantile(kept, 0.5, dim=-1, keepdim=True)
            sigma = 1.482602218505602 * torch.nanquantile(abs(kept - center), 0.5, dim=-1, keepdim=True)
            kept = kept.masked_fill((kept < center - 3 * sigma) | (kept > center + 3 * sigma), float("nan"))
        center = torch.nanquantile(kept, 0.5, dim=-1, keepdim=True)
        sigma = 1.482602218505602 * torch.nanquantile(abs(kept - center), 0.5, dim=-1, keepdim=True)
        lower = (center - 3 * sigma).unsqueeze(-1)
        upper = (center + 3 * sigma).unsqueeze(-1)
    clipped = torch.maximum(torch.minimum(images.float(), upper), lower)
    lo = torch.as_tensor(TRAIN_CHANNEL_MIN, device=images.device).view(1, 3, 1, 1)
    hi = torch.as_tensor(TRAIN_CHANNEL_MAX, device=images.device).view(1, 3, 1, 1)
    return ((clipped - lo) / (hi - lo)).clamp(0, 1)


class CFAHeatmapLoss(nn.Module):
    def __init__(self, weights_path, mu, sigma, edge_margin=8):
        super().__init__()
        path = Path(weights_path)
        self.detector = CrossFrequencyAttentionUNet()
        self.detector.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.detector.requires_grad_(False).eval()
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32).reshape(1, 3, 1, 1))
        self.register_buffer("sigma", torch.as_tensor(sigma, dtype=torch.float32).reshape(1, 3, 1, 1))
        if not torch.isfinite(self.mu).all() or not torch.isfinite(self.sigma).all() or (self.sigma <= 0).any():
            raise ValueError("CFA loss requires three finite training means and positive standard deviations")
        if not isinstance(edge_margin, int) or edge_margin < 0:
            raise ValueError("edge_margin must be a nonnegative integer")
        self.edge_margin = edge_margin
        self.config = dict(weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           mu=self.mu.flatten().tolist(), sigma=self.sigma.flatten().tolist(),
                           edge_margin=edge_margin, objective="balanced_soft_heatmap_mse_v1",
                           frequencies=[90, 150, 220])

    def train(self, mode=True):
        super().train(mode)
        self.detector.eval()
        return self

    def forward(self, pred, target, *, planck):
        if pred.shape != target.shape or pred.shape != planck.shape or pred.ndim != 4 or pred.shape[1] != 3:
            raise ValueError("Detection loss needs aligned (B, 3, H, W) prediction, ACT and Planck")
        if min(pred.shape[-2:]) < 8 or any(n % 8 for n in pred.shape[-2:]):
            raise ValueError("CFA spatial dimensions must be divisible by 8")
        m = self.edge_margin
        if min(pred.shape[-2:]) <= 2 * m:
            raise ValueError("Patches are too small for the detector edge margin")
        physical_target = target * self.sigma + self.mu
        physical_planck = planck * self.sigma + self.mu
        valid = ((physical_target.abs() > 1e-4) & (physical_planck.abs() > 1e-4)).all(1, keepdim=True)
        if m:
            valid = F.max_pool2d((~valid).float(), 2 * m + 1, 1, m) == 0
            valid[:, :, :m] = valid[:, :, -m:] = False
            valid[:, :, :, :m] = valid[:, :, :, -m:] = False
        if not valid.flatten(1).any(1).all():
            raise ValueError("No common valid interior for CFA loss in a batch image")
        with torch.no_grad():
            reference = self.detector(preprocess_cfa(physical_target))
        predicted = self.detector(preprocess_cfa(pred * self.sigma + self.mu))
        error = (predicted - reference).square()
        # Balance soft foreground/background so rare peaks are not lost in blank sky.
        fg, bg = reference * valid, (1 - reference) * valid
        axes = (1, 2, 3)
        loss = ((error * fg).sum(axes) / fg.sum(axes).clamp_min(1e-8)
                + (error * bg).sum(axes) / bg.sum(axes).clamp_min(1e-8)).mean() / 2
        if not torch.isfinite(loss):
            raise ValueError("Non-finite CFA heatmap loss")
        return loss
