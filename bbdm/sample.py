"""Инференс и визуализация.

В сэмплер попадает ТОЛЬКО Planck-вход. Целевая карта (ACT+Planck) здесь не
принимается ни в каком виде -- это была одна из исходных ошибок (утечка
цели на инференсе), и отсутствие такого аргумента -- защита от её возврата.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def denormalize(patch_norm, mu, sigma):
    return patch_norm * sigma + mu


def _as_batch(x, device):
    """Приводит (H,W) / (1,H,W) / (B,1,H,W) к тензору (B, 1, H, W) на device."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 2:
        x = x[None, None]
    elif x.ndim == 3:
        x = x[:, None] if x.shape[0] != 1 else x[None]
    elif x.ndim != 4:
        raise ValueError(f"Expected 2D/3D/4D input, got shape {x.shape}")

    return torch.from_numpy(np.ascontiguousarray(x)).to(device)


@torch.no_grad()
def run_inference(
    bbdm, x0_norm, mu, sigma, S=200, device="cuda", n_samples=1, seed=None,
    progress=False,
):
    """n независимых стохастических сэмплов для ОДНОГО Planck-патча.

    Args:
        x0_norm: нормализованный Planck-патч, (H, W) или (1, H, W).
        mu, sigma: статистики нормализации, чтобы вернуть результат в мкК.
        n_samples: сколько независимых реализаций сгенерировать.
        seed: сид шума обратного процесса (для воспроизводимости).

    Returns:
        Список из n_samples numpy-массивов (H, W) в исходных единицах.
    """
    bbdm = bbdm.to(device).eval()

    planck = _as_batch(x0_norm, device)
    if planck.shape[0] != 1:
        raise ValueError(
            "run_inference ожидает один патч; для набора патчей используйте "
            "sample_batch"
        )
    # repeat, а не expand: expand даёт вью с нулевым шагом по батчу, и любая
    # in-place операция над ним затронула бы все "копии" разом.
    planck = planck.repeat(n_samples, 1, 1, 1)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=planck.device)
        generator.manual_seed(seed)

    pred_norm = bbdm.sample(planck, S=S, generator=generator, progress=progress)
    pred_norm = pred_norm.cpu().numpy()

    return [denormalize(pred_norm[i, 0], mu, sigma) for i in range(n_samples)]


@torch.no_grad()
def sample_batch(bbdm, x0_norm, mu, sigma, S=200, device="cuda", seed=None,
                 progress=False):
    """Один сэмпл для каждого патча из батча -- для массовой оценки метрик.

    Args:
        x0_norm: (B, 1, H, W) или (B, H, W) нормализованных Planck-патчей.

    Returns:
        numpy-массив (B, H, W) в исходных единицах.
    """
    bbdm = bbdm.to(device).eval()
    planck = _as_batch(x0_norm, device)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=planck.device)
        generator.manual_seed(seed)

    pred_norm = bbdm.sample(planck, S=S, generator=generator, progress=progress)
    return denormalize(pred_norm[:, 0].cpu().numpy(), mu, sigma)


def visualize_inference(x0_patch, y_patch, pred_patches, shared_scale=True):
    """Вход, цель и сэмплы в одном ряду.

    Args:
        shared_scale: рисовать все панели в единой шкале, взятой по цели.
            Это важно: при индивидуальной автошкале на каждой панели
            амплитудная ошибка модели (та самая, что видна в Transfer
            Function) визуально полностью исчезает.
    """
    n = len(pred_patches)
    fig, axes = plt.subplots(1, n + 2, figsize=((n + 2) * 5, 5.4))

    if shared_scale:
        valid = y_patch[y_patch != 0]
        limits = np.percentile(valid, [2, 98]) if valid.size else (0.0, 1.0)
    else:
        limits = None

    def show(ax, data, title):
        if limits is None:
            valid = data[data != 0]
            vmin, vmax = np.percentile(valid, [2, 98]) if valid.size else (0.0, 1.0)
        else:
            vmin, vmax = limits
        im = ax.imshow(data, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(title)
        ax.axis("off")
        return im

    show(axes[0], x0_patch, "Planck (input)")
    im = show(axes[1], y_patch, "ACT+Planck (target)")
    for i, pred in enumerate(pred_patches):
        show(axes[2 + i], pred, f"BBDM sample {i+1}")

    if shared_scale:
        fig.colorbar(im, ax=axes, shrink=0.8, label="uK")
    else:
        plt.tight_layout()
    plt.show()
