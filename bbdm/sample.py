import torch
import numpy as np
import matplotlib.pyplot as plt


def denormalize(patch_norm, mu, sigma):
    return patch_norm * sigma + mu


@torch.no_grad()
def run_inference(bbdm, x0_norm, mu, sigma, S=200, device="cuda", n_samples=1):

    bbdm = bbdm.to(device).eval()

    if isinstance(x0_norm, torch.Tensor):
        x0_norm = x0_norm.cpu().numpy()

    if x0_norm.ndim == 2:
        x0_norm = x0_norm[None, None]
    elif x0_norm.ndim == 3:
        x0_norm = x0_norm[None]

    planck = torch.tensor(x0_norm, dtype=torch.float32).to(device)
    planck = planck.expand(n_samples, -1, -1, -1)

    pred_norm = bbdm.sample(planck, S=S)

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
    show(axes[1], y_patch, "ACT+Planck (target)")
    for i, pred in enumerate(pred_patches):
        show(axes[2 + i], pred, f"BBDM sample {i+1}")

    plt.tight_layout()
    plt.show()