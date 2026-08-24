# BBDM for CMB Super-Resolution

Brownian Bridge Diffusion Model (BBDM, [Li et al., CVPR 2023](https://arxiv.org/abs/2205.07680))
for enhancing Planck microwave sky maps using paired ACT+Planck observations as
ground truth. Part of a broader project on galaxy cluster detection via the
Sunyaev–Zel'dovich effect.

> **Status:** the sampler and metric bugs that dominated earlier results are
> fixed and covered by tests. The current run targets **spectral fidelity**
> (Transfer Function ≈ 1 *together with* high cross-correlation `r_ℓ`), not just
> pixel-space scores.

---

## Overview

The model learns a stochastic mapping from low-resolution **Planck** patches
(100 GHz) to high-resolution **ACT+Planck** patches (90 GHz) through a Brownian
Bridge diffusion process. Unlike a standard conditional DDPM, BBDM pins the
forward process at both endpoints, which is a more natural formulation for
paired image translation.

**Direction convention** — the single most important thing to keep straight:

```
t = 0  ->  y   ACT+Planck   (generation target)
t = T  ->  x0  Planck       (conditional input)
```

`x0` in the code means "the Planck patch", *not* "the model's own t=0 state".
The target never enters `BBDM.sample()`; the sampler's signature has no
parameter for it, and a test enforces that.

Forward process and schedule (`s` = `S_VAR`):

```
m_t     = t / T
delta_t = 2 * s * (m_t - m_t**2)
x_t     = (1 - m_t) * y + m_t * x0 + sqrt(delta_t) * eps
```

The network predicts the clean target directly (x0-parametrization), so the
reverse step contracts the current state before adding the prediction:

```
x_{t_prev} = (c_xt - c_et) * x_t + c_et * pred + c_yt * y_cond
             + sqrt(delta_tilde) * z
```

`c_xt` comes from the paper's eps-parametrization and already absorbs a `+c_et`
term; dropping the subtraction double-counts the signal on every one of the ~200
sampling steps and inflates amplitude geometrically.

---

## Loss

```
L = MSE(pred, y) + spectral_weight * L1( log RAPSD(pred), log RAPSD(y) )
```

Plain MSE pulls predictions toward the conditional mean, which is smoother than
any single true realization, so high-ℓ power is under-produced. The spectral
term penalizes that deviation directly, on the *same* Hanning-windowed, radially
binned spectrum that `metrics/power_spectrum.py` later measures — so `TF ≈ 1`
becomes a consequence of the training objective rather than a coincidence.
Spectra are averaged over the batch before comparison: the goal is matching
*ensemble* power, not the χ²-noisy spectrum of each individual realization.

`ETA` (noise injected into the reverse process's start state) is **0**, matching
the original paper's `x_T = y`. It is deliberately not used as a knob for
fixing TF: an earlier `ETA=0.01` produced a TF near 1 in a narrow band only
because its white start-noise inflated high-ℓ power by roughly as much as the
MSE objective suppressed it. Tuning ETA re-balances those two opposite defects
instead of removing either, and shows up as a collapsing `r_ℓ` wherever the
extra power is uncorrelated with the target.

---

## Requirements

```bash
pip install -r requirements.txt
```

Trained on Google Colab Pro+ with an NVIDIA A40 (40 GB). Batch size 32 needs
~18 GB VRAM — lower `BATCH_SIZE` in `config.py` if needed.

---

## Data

| Source | Frequency | Resolution |
|--------|-----------|------------|
| Planck HFI (input) | 100 GHz | ~5 arcmin beam |
| ACT+Planck (target) | 90 GHz | ~1.4 arcmin beam |

- Planck maps: [NASA/IPAC Infrared Science Archive](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/)
- ACT+Planck maps: [Naess et al. 2020](https://doi.org/10.1088/1475-7516/2020/12/046)
- Galactic mask: [Chandran et al. 2023](https://doi.org/10.5281/zenodo.7947597)

960×960 tiles are cut into four 480×480 patches. A patch pair is dropped if
either side has more than `MAX_ZERO_FRAC` masked pixels, **or** if the Planck
and ACT valid masks disagree on more than `MAX_MASK_MISMATCH_FRAC` of pixels —
the two instruments have different footprints, and a patch valid in only one of
them teaches the network to reconstruct a mask edge instead of sky.

Splits are made at the **tile** level before patching: the four patches of a
tile are adjacent on the sky, so a patch-level split would leak nearly identical
sky between train and test.

---

## Project structure

```
bbdm-cmb/
├── README.md
├── requirements.txt
│
├── bbdm/
│   ├── config.py                  # paths and hyperparameters
│   ├── train.py                   # training loop, EMA, deterministic validation
│   ├── sample.py                  # inference and visualisation
│   ├── evaluate.py                # TF / r_ell / PSNR / SSIM averaged over patches
│   ├── model/
│   │   ├── bbdm.py                # bridge, loss, sampler
│   │   └── unet.py                # U-Net backbone (33M parameters)
│   ├── data/
│   │   ├── dataset.py             # CMBPatchDataset, tiling, normalisation
│   │   └── splits.py              # tile-level train/val/test splits
│   └── metrics/
│       └── power_spectrum.py      # RAPSD, Transfer Function, cross-correlation
│
├── tests/
│   ├── test_bbdm_math.py          # bridge/sampler/metric math (no data needed)
│   └── test_dataset.py            # patch filters, normalisation, augmentation
│
└── notebooks/
    ├── batching_planck.ipynb      # tiling the full-sky maps
    ├── matching_planck_and_act.ipynb  # reprojection / footprint matching
    └── training+inference.ipynb   # main workflow
```

---

## Quickstart

**1. Set paths** in `bbdm/config.py` (or override them in the notebook):

```python
PLANCK_DIR     = "/path/to/planck/tiles"
ACT_DIR        = "/path/to/act_planck/tiles"
CHECKPOINT_DIR = "/path/to/checkpoints"
```

**2. Run the tests** — no data required, a few seconds on CPU:

```bash
python tests/test_bbdm_math.py && python tests/test_dataset.py
```

**3. Run `notebooks/training+inference.ipynb` end to end.** It covers data
loading and sanity checks, training (100 epochs, ~32 h on an A40, resumable via
`last.pt`), inference with three stochastic samples, the TF / `r_ℓ` evaluation
averaged over 50 patches, and the ETA ablation.

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
| LR schedule | ReduceLROnPlateau ×0.5, patience 5 **epochs** |
| Batch size | 32 |
| Epochs | 100 |
| EMA decay / start | 0.995 / step 10000 |
| Diffusion steps `T` | 1000 |
| Sampling steps `S` | 200 |
| Bridge scale `s` | 0.5 |
| Start noise `ETA` | 0.0 |
| `spectral_weight` | 0.1 |

`SCHEDULER_PATIENCE` is in epochs because the scheduler is stepped once per
epoch. The paper's value of 3000 is in optimizer *steps*; copied verbatim it
meant the LR never decayed across a 100-epoch run.

---

## Evaluating results

Read `TF` and `r_ℓ` together, always averaged over ≥30–60 patches:

```python
from bbdm.evaluate import evaluate_spectra, print_band_table, plot_spectral_comparison

res = evaluate_spectra(bbdm, val_ds, mu, sigma, n_patches=50, S=200, device="cuda")
print_band_table(res)
plot_spectral_comparison([res], ["ETA=0 + spectral loss"])
```

Two rules that changed conclusions in this project:

- **Never judge a spectrum from one patch.** At high ℓ the power is small and a
  single-patch estimate swings enough that a real systematic and pure estimator
  noise look identical.
- **Average the spectra, then divide.** Averaging per-patch ratios blows up when
  any patch has near-zero target power in a bin.

`TF ≈ 1` on its own proves nothing: power that is spectrally right but spatially
uncorrelated with the target (white noise, hallucinated point sources) gives
`TF ≈ 1` with a collapsing `r_ℓ`. `evaluate_spectra` returns both, and
`plot_spectral_comparison` puts them side by side for exactly this reason.

Changing `BBDM.sample` / `_posterior_coeffs` is sampling-only and can be tested
by re-sampling an existing checkpoint (that is how the ETA and S ablations work —
`evaluate_spectra(eta=...)` swaps it temporarily). Changing `BBDM.loss` or
`BBDM.q_sample` changes what the network is trained to predict and requires a
full retrain.

---

## Fixed bugs

Each of these is covered by a regression test in `tests/`.

| Bug | Symptom | Fix |
|-----|---------|-----|
| Bridge direction inverted | Target leaked into inference; model solved the inverse task | Direction convention above; `sample()` takes only Planck |
| Mixed parametrization in the sampler | TF inflated 25–120×; oracle model recovered ×6.58 amplitude | Contract with `(c_xt - c_et)` before adding `c_et * pred` |
| Sampler conditioned on the noisy start state | Every step's `c_yt` term used `x_T` instead of clean input | `sample()` keeps a separate clean `y_cond` |
| First-step degeneracy at `t=T` | `m_t = 1` made 0/0 drop the model's prediction entirely | `m_t.clamp(max=1-1e-4)`, verified against the analytic limit |
| LR scheduler never fired | `patience=3000` epochs on a 100-epoch run | `SCHEDULER_PATIENCE = 5` epochs |
| EMA staleness | Early "best" checkpoints stored near-initialization weights in the `ema` branch | Snapshot live weights into the shadow at `ema_start` |
| Biased `r_ℓ` estimator | Per-pixel ratio averaged per bin; no window on the cross term | Bin numerator and both spectra first, divide after; same Hanning window throughout |
| Normalization loaded every pixel into RAM | ~5 GB temporary array over the train split | Streaming sum / sum-of-squares |
| Augmentation materialized 4× copies | ~19 GB of patches held in RAM | Applied on the fly in `__getitem__` |
| Stochastic validation loss | Best-checkpoint choice and `ReduceLROnPlateau` driven by estimator noise | Fixed seed for `t` and bridge noise in validation |
| Per-panel autoscaling in plots | Amplitude error invisible in every figure | Shared colour scale taken from the target |
