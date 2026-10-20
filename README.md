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

**Test set = the ACT Deep56 field.** Tiles overlap by 50% (960 px, stride 480), so the
same patch sits in up to four tiles and no split *by tiles* is leak-free. The default split
(`--split-mode region`) holds out one contiguous field of sky instead: every patch lying
entirely inside D56 (352° < RA < 40.6°, −8.8° < Dec < 5.2°, 680 deg²) is test, patches
crossing its border are dropped, val is a set of 8°×8° blocks elsewhere, and the rest of
the sky is train. Train is read as **random 480-px crops at arbitrary offsets** inside the
tiles (never touching D56 or a val patch) instead of the ~1600 fixed tile quadrants.
Details in "Training the frequencies" below.

---

## Project structure

```
bbdm-cmb/
├── README.md
├── requirements.txt
│
├── bbdm/
│   ├── config.py                  # paths and hyperparameters
│   ├── train.py                   # training loop, EMA, deterministic validation, DDP
│   ├── dist.py                    # torch.distributed helpers (no-ops without torchrun)
│   ├── early_stop.py              # stop / checkpoint selection by sampler metrics (PSNR, TF, r_ell)
│   ├── sample.py                  # inference and visualisation
│   ├── evaluate.py                # TF / r_ell / PSNR / SSIM averaged over patches
│   ├── model/
│   │   ├── bbdm.py                # bridge, loss, sampler
│   │   └── unet.py                # U-Net backbone (33M parameters)
│   ├── data/
│   │   ├── dataset.py             # CMBPatchDataset, tiling, normalisation
│   │   ├── region.py              # D56 test field: RA/Dec -> map pixels, region split, random train crops
│   │   └── splits.py              # patch-position split (leak-free), legacy tile split, leakage audit
│   └── metrics/
│       └── power_spectrum.py      # RAPSD, Transfer Function, cross-correlation
│
├── scripts/
│   ├── train_freq.py              # train + evaluate one frequency pair (90 / 150 / 220 GHz)
│   ├── check_leakage.py           # how many val/test patches of the tile split are copies of train patches
│   └── eval_clean.py              # evaluate a trained model on unseen vs leaked patches
│
├── tests/
│   ├── test_bbdm_math.py          # bridge/sampler/metric math (no data needed)
│   ├── test_dataset.py            # patch filters, normalisation, augmentation
│   ├── test_phase1.py             # cond channel, upsampling, monitor, checkpoint loading
│   ├── test_splits_freq.py        # split inheritance and tile-grid checks
│   ├── test_early_stop.py         # early-stopping logic and its place in train()
│   ├── test_leakage.py            # overlapping tiles, patch-position split, dedup, script end to end
│   ├── test_region.py             # D56 pixel box, region split, random crops, script end to end
│   └── test_ddp.py                # 2-process DDP on CPU (gloo)
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

## Training the frequencies (90, 150 and 220 GHz)

Three channel pairs share one pipeline (`config.FREQ_PAIRS`): Planck 100 -> ACT 90,
Planck 143 -> ACT 150, Planck 217 -> ACT 220. `scripts/train_freq.py` does what the
notebook does (split -> normalisation -> datasets -> train -> evaluate best checkpoint
on val and test) for one pair and writes everything into `--out-dir`
(`best.pt`, `last.pt`, `splits.json`, `norm.json`, `results.json`, `spectra_*.png`).

```bash
# the three frequencies in parallel, one GPU each (D56 test field, random train crops)
DATA=/path/to/Diffusion      # the folder that contains Planck/ and Planck+ACT/
mkdir -p runs logs
CUDA_VISIBLE_DEVICES=0 nohup python scripts/train_freq.py --freq 90  --data-root $DATA \
    --out-dir runs/d56_f90  --global-batch 4 --num-workers 4 > logs/d56_f90.log  2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python scripts/train_freq.py --freq 150 --data-root $DATA \
    --out-dir runs/d56_f150 --global-batch 4 --num-workers 4 > logs/d56_f150.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python scripts/train_freq.py --freq 220 --data-root $DATA \
    --out-dir runs/d56_f220 --global-batch 4 --num-workers 4 > logs/d56_f220.log 2>&1 &

# two GPUs for one frequency (DDP); the global batch is split between the processes
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train_freq.py \
    --freq 150 --data-root $DATA --out-dir runs/d56_f150 --global-batch 4

# resume an interrupted run: same command plus --resume (same number of processes)
```

Things worth knowing:

- **The D56 split (`--split-mode region`, default).** The tiles were cut by
  `batching_planck.ipynb` from the full ACT DR6 / Planck CAR maps (shape 10320 x 43200,
  0.5'/px), and `tile_y{y}_x{x}` is the tile's pixel offset in that grid, so a field on the sky
  is a pixel rectangle: `x = 21600 - 120*RA` (RA in [-180, 180)), `y = 7560 + 120*Dec`.
  D56 (`config.TEST_REGIONS`) becomes y 6504-8185, x 16728-22561; on the real tile list that is
  36 test patches (3 rows x 12), 33 patches on its border are dropped. Before splitting, the
  script checks these formulas against the WCS stored in the tile headers and refuses to start if
  they disagree by more than 2 px. Val is ~10% of the remaining patch positions in 2x2-patch
  blocks (it drives the monitor and early stopping; the test field is never looked at during
  training). The split is made once over the union of all frequencies and saved to
  `<data-root>/splits_region_D56.json`; `leakage.json` in every run holds the audit (patch
  positions and 20k sampled train crops vs val/test, all must be 0). Other fields:
  `--test-region ra_min,ra_max,dec_min,dec_max`; a guard band: `--guard-px`.
- **Random train crops.** `RandomCropDataset` draws a tile uniformly and an offset uniformly in
  [0, 480]^2, rejects windows touching D56 or a val patch and windows that fail the zero-fraction
  / mask-mismatch filters, then applies a random flip / 180° rotation. An epoch is
  `--crops-per-epoch` crops (default `CROPS_PER_POSITION`=4 x the number of train positions, the
  same count the x4 augmentation of fixed quadrants gave). Crops are read straight from the FITS
  files (only the needed rows), so nothing is held in RAM and the OS page cache is shared between
  runs. `--no-crops` trains on the fixed quadrants instead (ablation). D56 is one of ACT's deep
  fields (lower noise than the average sky), which is worth remembering when comparing test and
  val numbers.
- **The older leak-free split (`--split-mode spatial`).** Tiles are 960 px cut with stride 480 px and a
  patch is a 480x480 quadrant of a tile, so the same patch (bit for bit) sits in up to four
  tiles. A random split *by tiles* (what run 5 was trained with, `--split-mode legacy`) therefore
  puts copies of train patches into val and test: 97% of them on the real data (882 tiles: 1050 unique train
  patches stored 2.69x; measured with `scripts/check_leakage.py`). `--split-mode spatial` identifies a patch by its
  absolute position in the ACT pixel grid, splits the unique positions in 2x2-patch blocks, and
  drops patches that share pixels with a higher-priority split (test > val > train). The split is
  made once over the union of the tiles of all frequencies and saved to
  `<data-root>/splits_spatial.json`, so every frequency model is evaluated on the same unseen sky.
  `leakage.json` in each run directory holds the audit (identical / overlapping pairs, must be 0).
  This requires tiles with the same name to lie on the same pixel grid; the script checks the WCS
  headers and refuses to start otherwise.
- **What this means for run 5.** Its val/test numbers (and the monitor, early stopping and
  `best.pt` selection that use val) were measured largely on patches the network trained on, so
  they are not out-of-sample. `scripts/eval_clean.py --ckpt <run 5 best.pt> --data-root ...`
  evaluates it on the val/test patches that share no pixels with any train patch and, for
  comparison, on the leaked ones; a lower r_ell on the clean group means part of the signal was
  memorised.
- **Batch size.** `--global-batch` is required: it fixes steps per epoch and the epoch at which the
  EMA starts (`EMA_START` is counted in optimizer steps), so a different batch silently changes the
  recipe. The script prints both numbers. Keep it equal to the batch of the run you compare with.
- **Normalisation** (`mu`, `sigma`) is computed per frequency from its own unique train patches and
  stored in `norm.json` and in every checkpoint (`extra`).
- **Memory.** With random crops (default) train is read from disk; val and test (~200 patches)
  are held in RAM. With `--no-crops` / `spatial` / `legacy` every process holds its training set
  in RAM; the script prints the estimate.
- **Reading the bands.** Planck's beam narrows from 100 to 217 GHz, so the multipole where
  `r_ell` falls moves outwards; `results.json` lists the band edges in ell
  (`bands_ell`, `planck_beam_ell_1e`).

## Early stopping by metrics

Training ends when the epoch limit is reached **or** the metrics of the real sampler say stop
(`bbdm/early_stop.py`, settings in `config.EARLY_STOP`, flags `--patience`, `--psnr-min-delta`,
`--r-min-delta`, `--tf-max`, `--no-early-stop`). The val loss is not used for this: it is the MSE
of one diffusion step, and in run 4 it looked fine for 100 epochs while TF at the sampler output
was 9.5. Every `MONITOR_EVERY` epochs (default 2) the monitor samples 32 validation patches with the
EMA weights and computes PSNR, TF, r_ell and r_in; the stopper looks at those. The patches are spread
evenly over the whole validation set (not the first N, which would be a few neighbouring tiles). The monitor
uses S=50 while the TF thresholds were reasoned from S=200 numbers; compare the last monitor entry with the
final S=200 evaluation in `results.json` and retune if they differ.

- **Plateau.** A check counts as an improvement if PSNR grew by more than 0.05 dB **or** r_ell grew
  by more than 0.01 in the 0.03-0.10 or 0.10-0.30 band. 8 checks in a row without one -> stop
  (16 epochs at the default cadence, comfortably more than the 5-epoch LR scheduler patience).
  r_ell is included because PSNR is dominated by large scales that are learned early, while the
  cluster-relevant ell grows later; a PSNR-only rule would stop while it is still improving.
- **TF guard.** TF above 2 in 2 checks in a row -> stop (excess power at the sampler output).
- **Warm-up.** Nothing stops during the first 3 checks.
- **Best checkpoint by metric.** `best_metric.pt` is the check with the highest PSNR among those where
  TF is acceptable: within 1 +/- 0.15 in the bands up to f = 0.10 (where the input carries information)
  and not above 2 everywhere. TF is not required to be 1 above the Planck beam, where an honest sample
  has TF ~ 1 with r_ell ~ 0 and a deficit is fine. `best.pt` (lowest val loss) is still written.
  The final evaluation uses `best_metric.pt` (`--eval-ckpt loss` for `best.pt`); if TF was never in
  range together with a PSNR, `best_metric.pt` does not exist and the script falls back to `best.pt`
  and says so.
- The reason for stopping, the stopper state and all monitor checks are stored in the checkpoint and
  in `results.json`. `--resume` of an already stopped run returns immediately. Under DDP rank 0 decides
  and broadcasts the decision to the other ranks.

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
