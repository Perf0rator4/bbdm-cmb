import torch
import numpy as np
import matplotlib.pyplot as plt


def denormalize(patch_norm, mu, sigma):
    return patch_norm * sigma + mu


@torch.no_grad()
def run_inference(bbdm, x0_norm, y_norm, mu, sigma, S=200, device="cuda", n_samples=1):
    bbdm = bbdm.to(device).eval()

    if y_norm.ndim == 2:
        y_norm = y_norm[None, None]
    elif y_norm.ndim == 3:
        y_norm = y_norm[None]

    y = torch.tensor(y_norm, dtype=torch.float32).to(device)
    y = y.expand(n_samples, -1, -1, -1)

    pred_norm = bbdm.sample(y, S=S)  # <- теперь передаём y

    samples = []
    for i in range(n_samples):
        patch = pred_norm[i, 0].cpu().numpy()
        samples.append(denormalize(patch, mu, sigma))

    return samples


def visualize_inference(x0_patch, y_patch, pred_patches):
    n = len(pred_patches)
    fig, axes = plt.subplots(1, n + 2, figsize=((n + 2) * 5, 5))

    def show(ax, data, title):
        valid = data[data != 0]
        vmin, vmax = np.percentile(valid, [2, 98]) if valid.size > 0 else (0, 1)
        ax.imshow(data, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(title)
        ax.axis("off")

    show(axes[0], x0_patch, "Planck (input)")
    show(axes[1], y_patch,  "ACT+Planck (target)")
    for i, pred in enumerate(pred_patches):
        show(axes[2 + i], pred, f"BBDM sample {i+1}")

    plt.tight_layout()
    plt.show()