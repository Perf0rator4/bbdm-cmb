# Project Context: BBDM for CMB Super-Resolution ("Galaxy Hackers")

This file is a working context dump for Claude Code, covering the project's
goal, the current (fixed) state of the code, and prioritized next steps.
It reflects a long chat history of debugging and diagnosis — treat the
"Fixed bugs" section as ground truth for how the pipeline is *supposed* to
work now; don't reintroduce the old behavior described there.

## 1. Project

Student: Fedor Gromakov (Teo), 2nd-year DSBA, HSE Moscow.
Scientific supervisor: Svetlana Voskresenskaia (Faculty of Physics).
Repo: github.com/Perf0rator4/bbdm-cmb

Goal: apply a Brownian Bridge Diffusion Model (BBDM; Li et al., CVPR 2023,
arXiv:2205.07680) to CMB map super-resolution — generate high-resolution
ACT+Planck maps from low-resolution Planck-only input. Downstream use case:
aiding galaxy cluster detection.

The coursework defense (based on this project) has already concluded
successfully. The project is now transitioning from "coursework deliverable"
to "research paper" — the standard is now physical accuracy and spectral
fidelity (Transfer Function ≈ 1, high cross-correlation r_ℓ), not just
coursework-grade metrics. There is now freedom to experiment more
aggressively than a coursework would allow.

## 2. Data

- Paired **Planck 100 GHz** (low-res input) and **ACT DR6 90 GHz** (PA6,
  s17s22; high-res target, referred to as "ACT+Planck") HEALPix tiles,
  cut into 480×480 patches (`PATCH_SIZE = 480`).
- ~10,384 training pairs, 89 held-out test tiles.
- Split: `TRAIN_RATIO=0.8`, `VAL_RATIO=0.1`, seeded (`SEED=42`), done at the
  *tile* level in `data/splits.py` before patching, to avoid leakage between
  train/val/test from overlapping patches of the same tile.
- `MAX_ZERO_FRAC = 0.2` filters out patches with too many zero/masked pixels.
- **Planck/ACT footprint mismatch — now filtered** (was an open TODO). Planck
  and ACT DR6 have different sky coverage masks, so some patch pairs had a
  cropped/masked region on one side that the other side did not have. A patch
  valid in only one channel teaches the network to reconstruct a mask edge
  rather than sky, and that edge adds spurious high-ℓ power to the spectrum.
  `CMBPatchDataset` now drops a pair when
  `(planck_valid XOR act_valid).mean() > MAX_MASK_MISMATCH_FRAC` (default
  `0.01`), and prints how many pairs each filter dropped. **Watch that
  printout on the first real run** — if it removes a large fraction of the
  data, raise the threshold deliberately rather than leaving it silent.
  Covered by `tests/test_dataset.py::test_mask_mismatch_filter_drops_planck_only_hole`.
- Normalization: plain mean/std over nonzero pixels (mu ≈ -15.17,
  sigma ≈ 113.47 in one run), computed **streaming** (sum / sum-of-squares /
  count) — the old version concatenated every pixel of the train split into
  one array, roughly 5 GB. The coursework text describes σ-clipped statistics;
  `compute_normalization(..., n_sigma=3.0)` now provides that as an opt-in
  second pass. Which one is used does not affect Transfer Function anyway:
  prediction and target are denormalized with the same (mu, sigma) and the
  normalization cancels in the ratio.

## 3. Repo layout

Everything lives under the `bbdm/` package; imports are absolute
(`from bbdm.model import BBDM, UNet`). There are no implicit relative imports
left — `train.py` used to do `from config import ...`, which broke as soon as
the module was imported as part of a package.

```
bbdm/config.py              - paths and hyperparameters
bbdm/model/unet.py          - U-Net backbone, ~33M params
bbdm/model/bbdm.py          - bridge, loss (MSE + spectral), sampler
bbdm/data/splits.py         - tile-level train/val/test split
bbdm/data/dataset.py        - patch dataset, filters, streaming normalization
bbdm/metrics/power_spectrum.py - RAPSD / TF / cross-correlation primitives
bbdm/train.py               - training loop, EMA, deterministic validation, resume
bbdm/sample.py              - run_inference / sample_batch / visualize_inference
bbdm/evaluate.py            - TF, r_ell, PSNR/SSIM averaged over many patches
tests/test_bbdm_math.py     - bridge/sampler/metric math, no data needed
tests/test_dataset.py       - patch filters, normalization, augmentation
```

Notebook: `notebooks/training+inference.ipynb` (rewritten; the old
`bbdm_cmb_2.ipynb` cell map no longer applies). Structure:
1. Colab setup (commented out), 2. imports + paths + device, 3. tile split,
4. normalization, 5. datasets, 6. pair sanity check, 7. model, 8. one-batch
sanity check, 9. training, 10. best-checkpoint load, 11. inference with 3
samples, 12. **TF + r_ℓ over 50 patches**, 13. **ETA ablation**, 14. PSNR/SSIM,
15. test-split visuals, 16. test-split spectra.

Checkpoint convention (unchanged): `best.pt` holds both `model` and `ema`
state dicts; the notebook loads `model` and then **overwrites** the network
weights with the `ema` shadow, so the live weights at inference are the EMA
ones. `last.pt` is written every epoch and carries optimizer + scheduler +
`global_step`, so `train(..., resume=True)` resumes a dropped Colab session.
Both now also carry a `hparams` dict (`T`, `s`, `eta`, `spectral_weight`) so
inference cannot silently run with a different process than training did.

The two preprocessing notebooks (`batching_planck.ipynb`,
`matching_planck_and_act.ipynb`) are standalone Colab workflows that do not
import the package.

## 4. BBDM math (as implemented, corrected direction)

Direction convention:
```
t = 0  ->  y   (ACT+Planck, the GENERATION TARGET)
t = T  ->  x0  (Planck, the CONDITIONAL INPUT)
```
This is the opposite of what a naive reading of variable names might suggest
— `x0` in the codebase means "the Planck patch", not "the model's own t=0
state". This convention was the subject of "Bug A" below; don't invert it.

Schedule (from the paper, `s` = `S_VAR` in config, default 0.5):
```
m_t     = t / T
delta_t = 2 * s * (m_t - m_t**2)
```

Forward process:
```
x_t = (1 - m_t) * y + m_t * x0 + sqrt(delta_t) * eps,   eps ~ N(0, I)
```
At t=T: x_T = x0 (Planck) exactly. At t=0: x_0 = y (ACT) exactly.

Reverse posterior (arbitrary jump t -> t_prev, for subsampled S<T inference),
computed in `BBDM._posterior_coeffs`:
```
m_t, m_t1           = t/T, t_prev/T
delta_t, delta_t1   = schedule(m_t), schedule(m_t1)
delta_t_given_t1    = delta_t - delta_t1 * ((1-m_t)/(1-m_t1))**2
delta_tilde         = delta_t_given_t1 * delta_t1 / delta_t

c_xt = (delta_t1/delta_t) * (1-m_t)/(1-m_t1) + (delta_t_given_t1/delta_t) * (1-m_t1)
c_yt = m_t1 - m_t * (1-m_t)/(1-m_t1) * (delta_t1/delta_t)
c_et = (1-m_t1) * (delta_t_given_t1/delta_t)
```
These `(c_xt, c_yt, c_et)` are the paper's *eps-parametrization* coefficients
(Algorithm 2), where `c_et` is meant to multiply a noise-like residual
`m_t(y-x0) + sqrt(delta_t)*eps`, subtracted with a minus sign.

The network in this project is **not** trained to predict that residual —
it's trained to predict the clean image `y` directly (x0-parametrization,
verified consistent with the training loss below). The correct sampling
update in this parametrization is:
```
x_{t_prev} = (c_xt - c_et) * x_t + c_et * pred + c_yt * y_cond + sqrt(delta_tilde) * z
```
where `pred = model(x_t, t)` is the predicted clean ACT+Planck image and
`y_cond` is the **clean** Planck input (constant across all steps, not the
evolving bridge state). This is what `BBDM.sample()` currently implements.

Training loss (`BBDM.loss`) — **now MSE + spectral term** (was plain MSE):
```python
t = randint(1, T+1)
x_t, _ = q_sample(x0=planck, y=act, t)      # x0=Planck, y=ACT (see direction convention)
pred = model(x_t, t)

mse  = F.mse_loss(pred, act)                 # predicts the clean t=0 endpoint
spec = F.l1_loss(log(rapsd(pred)), log(rapsd(act)))
loss = mse + spectral_weight * spec
```

`rapsd` here is `BBDM.rapsd`: a Hanning-windowed, radially binned power
spectrum computed on GPU, averaged over batch and channel, using bins
**identical** to `metrics/power_spectrum.py` (a test asserts the two agree to
the FFT normalization factor). Design decisions and why:

- **Batch-averaged spectrum, not per-sample.** The target is matching
  *ensemble* power — which is exactly what TF≈1 means. A single realization's
  spectrum is χ²-noisy, so per-sample matching would be over-constraining.
- **Log-space.** True power falls by orders of magnitude from low to high ℓ;
  an unweighted spectral loss would be dominated by the lowest bins alone.
- **L1, not L2.** Log-power residuals are heavy-tailed.
- **Bin 0 dropped.** That's the DC mode (mean level), not spectral fidelity,
  and MSE already covers it.
- **`real**2 + imag**2`, not `.abs()**2`.** `abs()` of a complex tensor has a
  NaN gradient at zero.

`spectral_weight` is `0.1` by default and is the main thing to sweep — see
§7.1. `spectral_weight=0` reproduces the old pure-MSE objective exactly.

## 5. Fixed bugs (history — do not reintroduce)

These were diagnosed and fixed across a long debugging chat, verified with
Monte-Carlo checks against the empirical forward-process conditional law,
closed-form boundary checks, and a real mini training run on synthetic data
(TF went from ~25-120x inflation down to ~0.1-1.3x on a toy blur task).

**Every item below now has a regression test** in `tests/` — run
`python tests/test_bbdm_math.py && python tests/test_dataset.py` (no data
needed, a few seconds on CPU) before trusting any change to the math.

**Bug A — bridge direction inverted.** An earlier version had `q_sample`
built as ACT at t=0 -> Planck at t=T being backwards, and/or inference fed
the ground-truth ACT target into `sample()` instead of the Planck input.
Symptom: data leakage at inference / the model solved the inverse task.
Fixed: see direction convention above; `run_inference` in `sample.py` now
only ever receives `x0_norm` (Planck), never the target.

**Bug B — mixed parametrization in the sampler, causing severe amplitude
inflation.** An earlier version used the *eps*-parametrization coefficients
`c_xt` (which already algebraically absorb a `+ c_et` term) together with
`+ c_et * pred` added on top, and fed an image-valued prediction where the
formula expected a noise-residual. This is a genuine double-count: each of
the ~200 sampling steps deposited an extra copy of the signal that the
correct contraction `(c_xt - c_et)` would have removed, compounding
geometrically. Verified: with an oracle "model" that returns the exact
clean target, the old formula recovered target amplitude ×6.58 (i.e. ~×43
in power / Transfer-Function terms — this alone reproduced the observed TF
inflation of 25-120x). Fixed formula recovers amplitude exactly ×1.0 with
an oracle model. This bug also let the (buggy) `c_yt` term leak the full
starting map through un-contracted, adding correlated power on top.

**Bug C — sampler conditioned on the noisy start state, not the clean
input.** Every reverse step's `c_yt` term multiplied `x_T = planck +
sqrt(eta)*noise` rather than the clean `planck`. Minor with eta=0.01, but
fixed for correctness: `sample()` now keeps a separate clean `y_cond`
reference used in every step's conditioning term.

**First-step degeneracy.** At t=T, m_t=1 exactly makes several coefficients
divide 0/0 (regularized to ~0), which used to make the very first reverse
step drop the model's prediction entirely. Fixed with a small clamp
(`m_t.clamp(max=1-1e-4)`) so the first step computes the proper limit
`x_{T-1} ≈ m_{T-1}*planck + (1-m_{T-1})*pred + noise` instead.

**LR scheduler never fired.** `ReduceLROnPlateau` is stepped once per
*epoch* in `train.py`, but `SCHEDULER_PATIENCE` was copied from the paper's
value of 3000, which is in *optimizer steps*, not epochs. Over 100 epochs
the LR never decayed. Fixed: `SCHEDULER_PATIENCE = 5` (epochs) in
`config.py`.

**EMA staleness.** `ema.shadow` used to be a copy of init weights that
wasn't updated until `global_step >= ema_start`, but every "best"
checkpoint saved `ema.shadow.state_dict()` regardless of whether EMA had
actually started — an early-training "best" checkpoint's EMA branch could
contain near-initialization weights. Fixed: `train.py` now snapshots the
live model weights into the EMA shadow at the exact step `ema_start` is
reached (`ema.copy_from(model)`), and only starts exponential averaging
after that. `config.EMA_START = 10000` is also now actually threaded
through from `config.py` into `train()` (previously the function's own
default of 10000 was used regardless of config, since the notebook did pass
`ema_start=EMA_START`, so this was latent rather than active — but worth
knowing).

**Biased cross-correlation estimator.** `metrics/power_spectrum.py`'s
`cross_correlation` used to compute a per-pixel ratio
`Re(F_pred F_true*)/sqrt(|F_pred|^2 |F_true|^2)` and then bin-average the
ratio — a biased estimator, and inconsistent windowing vs
`compute_power_spectrum` (no Hanning window on the cross term). Fixed: now
bins the numerator and both power spectra *first*, divides afterward
(`r_ell = <Re(F_pred F_true*)>_ell / sqrt(<|F_pred|^2>_ell <|F_true|^2>_ell)`),
and applies the same Hanning window as `compute_power_spectrum` throughout.
Sanity-checked: `cross_correlation(p, p)` ≡ 1 for any `p`.

**Additional fixes from the cleanup pass (2026-08-24).** None of these change
the diagnosed math above; they were correctness/robustness problems found
while making the whole repo run end to end.

- *Broken imports.* The notebook imported `bbdm.splits` / `bbdm.dataset` /
  `bbdm.unet` / `bbdm.bbdm` / `bbdm.power_spectrum` (none of which exist after
  the restructure), and `train.py` used `from config import ...`, an implicit
  relative import that fails under package import. All imports are now absolute
  and there is a real `bbdm/__init__.py`.
- *`run_inference` signature mismatch.* The notebook passed `y_norm=...`, which
  the function never accepted — every inference cell would have raised
  `TypeError`. Removed rather than added: the target must not reach the
  sampler. `tests/test_bbdm_math.py::test_sampler_ignores_target_beyond_the_input`
  enforces that `BBDM.sample` has no target-shaped parameter.
- *PSNR/SSIM cell used a stale `y_test`* left over from a previous cell as the
  (nonexistent) `y_norm` argument. That whole cell is now
  `evaluate.evaluate_image_metrics`.
- *Normalization held ~5 GB in RAM*; now streaming. *Augmentation materialized
  4× copies of every patch* (~19 GB on the train split); now applied on the fly
  in `__getitem__`.
- *Validation loss was stochastic* — `t` and the bridge noise were redrawn every
  epoch, so the val curve wobbled by roughly as much as real improvement, and
  both the best-checkpoint choice and `ReduceLROnPlateau` were partly driven by
  estimator noise. Now seeded with `VAL_SEED`.
- *Sampler step list could contain duplicates* after rounding when `S` is close
  to `T` (identity steps that only cost a forward pass); now deduplicated.
- *`UNet.forward` hard-coded `dim=256`* for the sinusoidal embedding instead of
  using `time_dim`, which would silently mismatch `TimeEmbedding` for any other
  `TIME_DIM`.
- *`ALPHA` was dead* — stored on `BBDM`, never read. Removed from the config,
  the constructor, and the notebook.
- *`visualize_inference` autoscaled every panel independently*, which made the
  model's amplitude error — the exact thing Transfer Function measures —
  invisible in every figure ever produced. Now one shared scale taken from the
  target.
- *No resume.* A dropped Colab session lost the whole run; `train.py` now writes
  `last.pt` every epoch and `train(..., resume=True)` picks it up.

## 6. Diagnostic state at run 3 (the run that motivated the current loss)

These numbers are from the **pure-MSE, `ETA=0.01`** configuration and are kept
as the baseline the next run must beat. They are not the current state of the
code.

With bugs A/B/C fixed, `S_VAR (s) = 0.5`, `ETA = 0.01`, retrained from
scratch:

```
PSNR: 23.44 ± 1.32 dB
SSIM: 0.2563 ± 0.0616   (low SSIM expected/documented as unsuitable for
                          one-to-many stochastic generation)
```

Transfer Function, **averaged over 60 validation patches** (mean-of-spectra
ratio, not mean-of-ratios — the latter has enormous variance on the tail
where `ps_target` is small):

| Frequency band | TF (mean ± std across patches, of per-band mean) |
|---|---|
| 0.005–0.03 | 0.995 ± 0.005 |
| 0.03–0.10  | 1.033 ± 0.026 |
| 0.10–0.30  | 1.199 ± 0.133 |
| 0.30–1.00  | 2.047 ± 0.203 |

This confirms the high-ℓ excess is a **real systematic**, not just
single-patch estimator noise (an earlier single-patch TF plot showed wild
oscillation between 1.5–2.25 in that range, which looked like it might be
noise — the 60-patch average shows it's a genuine ~2x excess, the
oscillation was estimator variance riding on top of a real bias).

**ETA ablation (resampling only, no retrain), 50 patches:**

| Frequency band | TF, ETA=0.01 | TF, ETA=0.0 |
|---|---|---|
| 0.005–0.03 | 0.995 | 0.989 |
| 0.03–0.10  | 1.033 | 0.985 |
| 0.10–0.30  | 1.199 | 0.755 |
| 0.30–1.00  | 2.047 | 0.734 |

**Diagnosis:** the reverse process's injected start-noise
`sqrt(ETA)*randn` is spectrally *white* (flat power). At `ETA=0.01` it adds
roughly uniform extra power across all ℓ; at low ℓ the real signal
dominates and this is negligible (TF≈1), but at high ℓ where the real
signal is intrinsically faint, this flat addition roughly doubles power
(TF≈2). At `ETA=0`, removing this noise **reveals the underlying MSE-loss
defect**: the model itself under-produces high-ℓ power (TF drops to
~0.73–0.76), because MSE pulls predictions toward the conditional mean,
which is smoother than any single true realization.

**Conclusion: `ETA=0.01` was accidentally cancelling two independent
defects (MSE under-power + white-noise over-power) into a TF that looked
closer to 1 in a narrow band — not genuine physical accuracy.** Tuning ETA
to some intermediate value (discussed and rejected) would just re-tune this
cancellation to a different lucky band; it doesn't fix either underlying
defect and would very likely show weak `r_ℓ` (cross-correlation) at high ℓ
even where TF≈1, because the extra power is spatially uncorrelated with the
target (white noise) rather than genuine recovered structure. **This has
not yet been numerically confirmed** — see recommended next step #1 below.

**Hallucinated point sources:** predicted samples show more small
dark/bright compact spots than the target, and they don't spatially match
the target's real compact sources (radio sources / clusters visible in the
90 GHz ACT map). Mechanism: (a) MSE-driven amplitude compensation — the
model can't know the true position of compact sources from Planck alone
(genuinely one-to-many), so under MSE it either blurs them out or
"invents" plausible-looking ones in the wrong place; (b) partially amplified
by the ETA start-noise being a source of spurious high-frequency structure
that the network learned to lean on. Expect this to reduce, but likely not
disappear, when retraining with `ETA=0`.

**Known preprocessing artifact** (see Data section above): the Planck/ACT
footprint mismatch causing a cropped corner on at least one test patch. **Now
addressed** by `MAX_MASK_MISMATCH_FRAC` in `CMBPatchDataset` — independent of
the loss/bug work above, but it also removes one source of spurious high-ℓ
power (mask edges), so it slightly changes the baseline the next run is
compared against.

## 7. Next experiments, in priority order

**Status as of 2026-08-24: the code for 7.0 and 7.1 is written, tested, and
in the repo. Neither has been *run on real data* yet — both need GPU time.**

### 7.0 (No retrain, do this FIRST): confirm the r_ℓ hypothesis

Before spending compute on retraining, confirm numerically that `ETA=0` is
the physically correct regime — i.e. that cross-correlation `r_ℓ` is higher
at `ETA=0` than at `ETA=0.01` at high ℓ, despite TF being farther from 1
there. If `r_ℓ` is *also* worse at `ETA=0`, the diagnosis above needs
revisiting before proceeding.

**Tooling: done.** `bbdm/evaluate.py::evaluate_spectra` computes TF and `r_ℓ`
together over N patches (mean-of-spectra, then ratio), and takes `eta=` to
swap `bbdm.eta` temporarily and restore it — resample only, no retrain.
Notebook cell 12 runs the ablation and `plot_spectral_comparison` draws RAPSD,
TF and `r_ℓ` side by side. Pass the **same `indices` and `seed`** to both
arms, otherwise you are comparing different patches of sky.

```python
abl = {e: evaluate_spectra(bbdm, val_ds, mu, sigma, indices=idx,
                           S=200, device="cuda", eta=e, seed=0)
       for e in (0.0, 0.01)}
plot_spectral_comparison([abl[0.0], abl[0.01]], ["ETA = 0", "ETA = 0.01"])
```

**This needs to be run against the run-3 checkpoint** (the pure-MSE,
`ETA=0.01` one) — that is the checkpoint the hypothesis is about. Running it
against a newly retrained spectral-loss model answers a different question.

Expected result if the diagnosis is right: `r_ℓ` roughly flat/high across
all ℓ for `ETA=0`, and visibly degraded at high ℓ for `ETA=0.01` (because
that extra power there is white noise, uncorrelated with target structure).

### 7.1 Retrain with `ETA=0` + a spectral loss term (primary recommendation)

Do **not** tune ETA to some intermediate value to cosmetically fix TF —
that just re-balances the same two cancelling defects (see §6 diagnosis).
Instead: `ETA=0` (matches the original BBDM paper's `x_T = y` exactly anyway —
ETA was never part of the published method, it's a local addition), plus a
term in `BBDM.loss` that directly penalizes power-spectrum deviation, so that
TF≈1 becomes a consequence of what the network is optimized for rather than a
side effect of noise cancellation.

**Implemented.** `config.ETA = 0.0`, `config.SPECTRAL_WEIGHT = 0.1`, and
`BBDM.loss` / `BBDM.spectral_loss` / `BBDM.rapsd` as described in §4. The
implementation differs from the original sketch in two ways, both deliberate:

- It bins **radially** with the exact bins of `metrics/power_spectrum.py`
  (the sketch used raw 2D `rfft2` bins). This costs one `index_add` per batch
  and makes the loss penalize precisely the quantity that is later measured.
- It applies the **same Hanning window** as the metric. Without it, the
  patch-boundary discontinuity — and the edges of masked regions — dump
  spurious high-ℓ power into the loss, which is exactly the band in question.

Still open, and the main thing to sweep: **`spectral_weight`**. Pick it so
`spectral_weight * spec` is comparable to `mse` once `mse` has reached its
late-training plateau — the training bar prints `mse` and `spec` separately
for this. Sweep `{0.05, 0.1, 0.3}` with short runs (~20-30 epochs) before
committing to a full 100-epoch run, and choose on TF **and** `r_ℓ`, not TF
alone: a weight high enough to force TF to 1 by manufacturing uncorrelated
high-ℓ texture would show up as `r_ℓ` regressing while TF looks perfect.
Report the final numbers on the **test** split — validation is used for
checkpoint selection and for this sweep.

A caveat worth checking during the sweep: at large `t` the MSE-optimal
prediction is `E[y | x_t]`, which is genuinely smooth, and the spectral term
pushes against that. Because the loss compares batch-averaged spectra over a
batch with mixed `t`, the pressure is soft rather than a hard per-`t`
constraint — but if training destabilizes at higher `spectral_weight`, a
`t`-dependent weight (weaker at large `t`) is the first thing to try.

This requires a full retrain (loss function changes what the network is
trained to predict), not just a resample.

### 7.2 Frequency-dependent bridge variance schedule (bigger methodological contribution, if pursuing a paper)

Current `delta_t` schedule is scalar / frequency-independent, meaning
signal-to-noise during the bridge is uniform across ℓ. Since the true CMB
power spectrum falls steeply with ℓ, the same absolute noise level implies
very different *relative* SNR at low vs high ℓ — likely part of why the
model struggles more at high ℓ. A frequency-dependent
`delta_t(ell)` (smaller injected noise at high ℓ, where the signal is
already faint) would need the forward/reverse process reformulated in
Fourier space. This is a substantial rewrite (not a drop-in fix) but is a
more original contribution for a paper than the spectral loss addition
above. Treat as a stretch goal after 7.1 is validated.

### 7.3 Condition the network on a local power-spectrum estimate of the input

Cheaper structural idea if the paper wants a novel-but-small architecture
change: append a per-patch estimate of the input Planck patch's local power
spectrum (e.g. a handful of radially-binned RAPSD values) as an extra
conditioning input to the U-Net (e.g. concatenated to the time embedding).
Lets the network calibrate its high-ℓ output amplitude based on what it
knows about the input's own spectral content, rather than a single global
behavior learned across all patches.

### 7.4 Adversarial / learned perceptual loss (larger undertaking, do last)

If 7.1 doesn't fully fix high-ℓ behavior: train a small discriminator CNN
to distinguish real vs. predicted ACT patches, and use intermediate-layer
features as a perceptual loss (analogous to LPIPS, but trained on this
domain since no pretrained CMB feature extractor exists). This pushes the
project from "diffusion model" toward "diffusion + adversarial", which is a
bigger change in character and needs its own stabilization work (loss
balancing, discriminator capacity) — only worth it if the simpler spectral
loss doesn't close the gap.

## 8. Style/process notes for whoever (human or Claude Code) touches this repo next

- Prefer diffs / isolated functions over full-file rewrites when editing;
  explain *why* a change fixes something, not just what changed.
- Any change to `BBDM.loss` or `BBDM.q_sample` changes what the network is
  trained to predict — requires a full retrain, not just re-sampling an
  existing checkpoint.
- Any change to `BBDM.sample` / `_posterior_coeffs` (sampling-only) can be
  tested by re-sampling an *existing* checkpoint — no retrain needed. Use
  this for ETA ablations, S (sampling steps) ablations, etc.
- Always evaluate TF and r_ℓ **averaged over ≥30-60 patches**, never a
  single patch — single-patch spectral estimates are extremely noisy at
  high ℓ where power is low, and can look like real systematics or look
  like noise depending on which single patch you happen to plot.
- When averaging TF across patches: average `ps_pred` and `ps_target`
  separately first, then take the ratio of means — not the mean of
  per-patch ratios (the latter blows up if any single patch has near-zero
  target power in some bin).
- Validate any new sampler math with an oracle-model test before trusting
  it on the real network: feed `BBDM.sample()` a fake "model" that returns
  the exact known target, and check the output amplitude/shape matches
  exactly. This caught Bug B immediately and cheaply. That test now lives in
  `tests/test_bbdm_math.py` (`test_sampler_oracle_*`) alongside a deliberate
  reconstruction of the buggy update that asserts it *does* inflate — so the
  check stays meaningful rather than silently passing on a no-op.
- Run the tests before and after any math change:
  `python tests/test_bbdm_math.py && python tests/test_dataset.py`. Neither
  needs data or a GPU; both finish in seconds.
- Note the trap in the oracle test: the final reverse step has `c_xt == c_et`
  and `c_yt == 0`, so it returns `pred` exactly no matter what the trajectory
  did. Checking only the final output would pass even with a badly broken
  sampler. That's why the oracle also records the intermediate `std(x_t)` and
  asserts the trajectory doesn't diverge.
- TF≈1 alone is not a result. Report `r_ℓ` next to it, always. Power that is
  spectrally correct but spatially uncorrelated with the target (white noise,
  hallucinated point sources) produces exactly TF≈1 with collapsing `r_ℓ` —
  which is how the `ETA=0.01` cancellation went unnoticed for a whole run.
