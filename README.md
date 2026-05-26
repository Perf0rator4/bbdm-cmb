# BBDM for CMB Super-Resolution

Brownian Bridge Diffusion Model (BBDM) for enhancing Planck microwave sky maps using paired ACT+Planck observations as ground truth. Part of a broader project on galaxy cluster detection via the Sunyaev–Zeldovich effect.

> **Status:** Training pipeline complete. Two implementation bugs identified and fixed (bridge direction mismatch + subsampling coefficient mismatch). Retraining in progress.

---

## Overview

The model learns a stochastic mapping from low-resolution **Planck** patches (100 GHz) to high-resolution **ACT+Planck** patches (90 GHz) using a Brownian Bridge diffusion process. Unlike standard conditional DDPMs, BBDM pins the forward process at both endpoints — the source image at `t=0` and the target image at `t=T` — which is a more natural formulation for paired image translation.

---

## Requirements

```bash
pip install torch numpy astropy pixell scikit-image tqdm matplotlib
```

Trained on Google Colab Pro+ with an NVIDIA A40 (40 GB VRAM). Batch size 32 requires ~18 GB VRAM — reduce `BATCH_SIZE` in `config.py` if needed.

---

## Data

| Source | Frequency | Resolution |
|--------|-----------|------------|
| Planck HFI (input) | 100 GHz | ~5 arcmin beam |
| ACT+Planck (target) | 90 GHz | ~1.4 arcmin beam |

- Planck maps: [NASA/IPAC Infrared Science Archive](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/)
- ACT+Planck maps: [Naess et al. 2020](https://doi.org/10.1088/1475-7516/2020/12/046)
- Galactic mask: [Chandran et al. 2023](https://doi.org/10.5281/zenodo.7947597)

Download the data and set the paths in `config.py` before running.

---

## Project Structure

```
bbdm-cmb/
├── README.md
├── requirements.txt
├── .gitignore
│
├── bbdm/                          # Core model and training code
│   ├── __init__.py
│   ├── config.py                  # All hyperparameters and paths
│   ├── bbdm.py                    # Brownian Bridge Diffusion Model
│   ├── unet.py                    # U-Net backbone (33M parameters)
│   ├── dataset.py                 # CMBPatchDataset, tiling, normalisation
│   ├── splits.py                  # Train/val/test tile splits
│   ├── train.py                   # Training loop with EMA
│   ├── sample.py                  # Inference and visualisation
│   └── power_spectrum.py          # RAPSD and Transfer Function metrics
│
└── notebooks/                     # Jupyter notebooks for workflows
    ├── batching_planck.ipynb 
    ├── matching_planck_and_act.ipynb 
    └── training+inference.ipynb 
```
## Quickstart

**1. Set paths in `config.py`:**
```python
PLANCK_DIR     = "/path/to/planck/tiles"
ACT_DIR        = "/path/to/act_planck/tiles"
CHECKPOINT_DIR = "/path/to/checkpoints"
```

**2. Run the Colab notebook end-to-end:**

Open `training+inference.ipynb` in Google Colab. The notebook covers:
- Data loading and sanity checks
- Model initialisation
- Training (100 epochs, 32 hours on A40)
- Inference with 3 stochastic samples
- RAPSD and Transfer Function evaluation

---

## Model

**U-Net backbone** — 33M parameters:

| Stage | Channels | Resolution |
|-------|----------|------------|
| init_conv | 64 | 480×480 |
| down1 | 128 | 480×480 |
| down2 | 256 | 240×240 |
| down3 | 512 | 120×120 |
| down4 + Self-Attn | 512 | 60×60 |
| Bottleneck | 512 | 60×60 |

**Training hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Optimiser | Adam |
| Learning rate | 1e-4 |
| LR schedule | ReduceLROnPlateau ×0.5, patience 3000 |
| Batch size | 32 |
| Epochs | 100 |
| EMA decay | 0.995 |
| Diffusion steps T | 1000 |
| Sampling steps S | 200 |
| Bridge scale s | 0.5 |

---

## Results

Current results (pre-bugfix checkpoint):

| Metric | Value |
|--------|-------|
| PSNR | 2.78 ± 0.64 dB |
| SSIM | 0.0148 ± 0.0092 |
| Transfer Function | 8–120× (ideal: 1.0) |

Low scores are caused by two implementation bugs (now fixed) rather than fundamental model failure. Visual inspection shows the model correctly reproduces large-scale CMB structure.


---

## Known Issues (Fixed)

**Bug 1 — Bridge direction mismatch:** The reverse process was initialised from `x_T ≈ x0` (Planck) instead of `x_T ≈ y` (ACT+Planck), causing an out-of-distribution mismatch at every inference step.

**Bug 2 — Subsampling coefficient mismatch:** Posterior coefficients were precomputed assuming step size Δt=1, but inference used S=200 steps with effective Δt≈5. Fixed by computing coefficients dynamically per step in `_posterior_coeffs()`.

---

