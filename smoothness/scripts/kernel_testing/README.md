# kernel_testing/ — r_e-in-kpc-vs-arcsec smoothing kernel investigation

This directory answers a question raised while revisiting the Hα clumpiness
statistic `S` (`sami_sf_pipeline/src/03_metrics/smoothness.py`), whose
smoothing kernel is `sigma = 0.35 x r_e`: should `r_e` be in arcsec (the
current, production convention) or kpc?

Short answer, established by `01_unit_invariance.py`: **it doesn't matter -
the two are algebraically and numerically identical**, because `sigma` ends
up in pixels either way, and the `kpc_per_arcsec(z)` conversion factor
cancels exactly. That is not the real methodological question. The real
question, which the rest of these scripts address, is whether `S` is
contaminated by a **resolution artifact** (PSF-fixed-in-arcsec vs
galaxy-angular-size-shrinking-with-z), and whether that contamination could
corrupt the ML "key drivers of SF morphology" analysis `S` is meant to feed.

All scripts import the actual production functions from
`sami_sf_pipeline/src/03_metrics/smoothness.py` (loaded by file path in
`common.py`, since that module's name starts with a digit and can't be
`import`-ed normally) - so every test here exercises the real pipeline
code, not a hand-copied re-implementation that could silently drift from
it.

## Order and how to run

Each script is self-contained; run them from this directory in the order
below (this is also the order they were actually run in - later scripts
build interpretive context from earlier ones, though none depend on
another's output files):

```bash
cd smoothness/scripts/kernel_testing
python3 01_unit_invariance.py
python3 02_noise_floor.py
python3 03_redshift_degradation.py
python3 04_resolution_matched_validation.py
python3 05_kernel_scale_sensitivity.py
python3 06_fixed_kpc_comparison.py
```

Each writes its own `outputs/<NN_name>/` subfolder (CSVs + PNGs) and prints
a summary to the console.

Default sample: `in_full_sam` (Wang 2022's 6-cut purity sample, n=705 - see
[[wang_2022_full_sam_selection_criteria]]), loaded via
`common.load_full_sam()`. `01_unit_invariance.py` deliberately uses the
full merged catalogue (n=2259) instead, since the unit-cancellation claim
it tests is a general property of the formula, not specific to the science
sample.

## What each script tests, and what it found

### 01 — Unit invariance (T1)
Recomputes `S` for every galaxy via both an arcsec route and an explicit
kpc round-trip (through the real cosmology calculation, not a no-op),
using the production `process_one()` on the real FITS maps.

**Finding:** identical to floating-point precision for 2258/2259 galaxies.
One galaxy (CATID 47254) differs by `delta_S = -3.6e-5` (5e-5 relative to
its S=0.72) - traced to a ~1e-13 float epsilon from the cosmology
round-trip crossing `scipy.ndimage.gaussian_filter`'s internal kernel-
truncation-radius boundary (`int(4*sigma + 0.5)`), giving a 1-pixel-wider
kernel on one route. This is a benign numerical artifact, not a units
effect, and five orders of magnitude below typical S values.
**Conclusion: r_e-in-arcsec vs r_e-in-kpc is not a real choice for this
kernel definition - it cancels.**

### 02 — Noise floor (T6)
For every `in_full_sam` galaxy, draws pure white noise from its own error
map (`noise ~ N(0, err)`), runs it through the identical production
formula (`I_clean = noise - err`, smoothed at the galaxy's own production
`sigma_px`), and computes `S_noise` - a quantity built from a map with
*zero real flux information* by construction.

**Finding:** `S_noise` tracks the same resolution proxies (`n_res`,
`re_used`, `z`, `sigma_over_psf`) with the same sign and similar magnitude
as the real `S_actual` does. More importantly: **a linear fit of
`S_actual` on `S_noise` alone has R² = 0.94** (Spearman rho = 0.937,
rho² = 0.88). A quantity containing no real signal explains ~94% of the
variance in the measured clumpiness statistic across the science sample.
(Note: `S_noise` is systematically *larger* than `S_actual` at fixed
kernel geometry - this is expected, not a bug: white noise has no spatial
coherence, so it is close to the *maximum* achievable roughness under a
given kernel, whereas real Hα flux has genuine spatial coherence from real
HII regions/diffuse gas even in a clumpy galaxy. The `S_noise/S_actual`
ratio should NOT be read as "X% of S is literally noise" - see the NOTE in
`02_noise_floor.py` for why that decomposition doesn't hold. The
R²/correlation evidence is the load-bearing result, not the ratio.)
**Conclusion: S's between-galaxy dynamic range is dominated by kernel-
geometry effects, not by differences in real spatial structure.**

### 03 — Redshift degradation (T2) — the core causal test
Takes 25 real, well-resolved, low-z anchor galaxies (`n_res` 5.8-8.8,
`z_native < 0.06`) and simulates observing the *same physical galaxy* at a
grid of higher redshifts out to z=0.15: shrinks the flux/error maps by
`kpc_per_arcsec(z_native)/kpc_per_arcsec(z_target)`, **re-convolves with
the native PSF** (fixed in pixels - see the script docstring for why this
step, initially omitted, is physically essential and why skipping
PSF deconvolution beforehand is a stated, deliberately conservative
simplification), shrinks the kernel with the galaxy exactly as production
does, and recomputes S.

**Finding:** S declines smoothly and (for 23/25 anchors, 92%) monotonically
as simulated resolution degrades - down to ~27% of its native value by
`n_res~1`. The simulated degradation tracks, overlaid on the real sample's
own S-vs-n_res scatter, trace directly along it (`outputs/03_.../
degradation_plot.png`, right panel). **This is a direct causal
demonstration that the resolution effect seen in the real sample is real
and mechanistic, not a spurious correlation with some other confound.**

### 04 — Resolution-matched validation (T4)
No ML model exists in this repo yet, so this is the currently-implementable
stand-in for "does the driver ranking survive resolution control": compares
raw Spearman rho(S, driver) on the full sample, on a resolution-matched
subsample (`n_res` in [2.5, 4.5], n=271), and a partial correlation
controlling `n_res` on the full sample, for candidate drivers already in
`master_final.csv` (`log_Mstar`, `delta_ms`, `log_SFR`, `log_sSFR`, `5NN`,
`Dfil`, `Dnode`, `Dvoid`, `ellip`) plus `z` as a deliberate negative
control (not a physical driver of SF morphology).

**Finding:** `log_SFR`, `log_Mstar`, and `delta_ms` (SFMS offset) all
*survive* resolution control - if anything their |rho| slightly
*increases* once the resolution confound is removed (e.g. `log_SFR`:
-0.565 raw -> -0.669 partial). These are not resolution artifacts.
`log_sSFR` and `ellip` weaken substantially and lose significance in the
matched subsample - flagged. `z` itself retains almost its full raw
correlation under `n_res`-control (-0.327 -> -0.322) - meaning `n_res`
alone does not fully capture the resolution effect T2/T6 demonstrate (T2's
direct simulation is the more trustworthy diagnostic for the mechanism;
this partial-correlation approach is a complementary, not a replacement,
check). **Conclusion: the primary candidate drivers for the planned ML
step (SFR, stellar mass, SFMS offset) are not artifacts of the resolution
confound; sSFR and ellipticity need caution.**

### 05 — Kernel-scale sensitivity (T5)
Recomputes S at seven kernel scales (0.10-0.50 x r_e, bracketing the
original `smoothness_test_02.py` exploration) and checks rank stability
against the production 0.35 value, resolution-safety-fraction trade-off,
and driver-correlation stability.

**Finding:** rank stability is very high across the whole range tested
(Spearman rho vs S(0.35) >= 0.939, and >=0.98 for 0.20-0.50). No candidate
driver changes sign at any tested scale; magnitudes shift smoothly and
mildly (e.g. `log_SFR`: -0.517 at 0.10 to -0.582 at 0.50). Resolution-
safety fraction rises smoothly from 0.1% (scale 0.10) to 85% (scale 0.50).
**Conclusion: 0.35 is not a special, load-bearing choice - the science
conclusions are robust to it, which is the strongest part of its defence.**

### 06 — Fixed-kpc kernel comparison (T3)
Directly tests the alternative that *would* actually differ from
production: a kernel with a fixed physical width (0.5, 1.0, 2.0 kpc for
every galaxy, not tied to each galaxy's own `r_e`) instead of the
self-similar `0.35 x r_e`.

**Finding, and a genuine correction to the a priori prediction going in:**
the fixed-kpc kernel's correlation with `n_res` is actually *weaker* than
production's (rho ~ 0.01-0.08 vs production's 0.52) - so by that specific
metric it looked *less* resolution-confounded. But its resolution-safety
fraction is worse at small physical scales (0.5 kpc: 7% safe vs
production's 53%), and its correlation with `z` and with `log_Mstar` /
`log_SFR` is dramatically *larger* than production's (`z`: -0.77 vs -0.33;
`log_Mstar`: -0.88 vs -0.45). Mechanism: because
`sigma_arcsec = X_kpc / kpc_per_arcsec(z)` is not tied to any individual
galaxy's own size, it swings from very wide (over-smoothing, inflating S)
at low z to very narrow (under-smoothing, collapsing S toward 0) at high z,
purely as a function of distance - a mechanical anti-correlation with z
that then aliases into any variable correlated with z in a flux-limited
survey (which stellar mass and SFR are, via Malmquist-type selection).
**Conclusion: a fixed-kpc kernel is not more physical - it trades a
resolution confound for a more severe, more direct distance confound.
This is the concrete, data-backed answer to "wouldn't kpc be more
physical?": no.**

## Bottom line for the thesis

1. r_e-in-arcsec vs r_e-in-kpc was never a real choice (01) - don't relitigate it.
2. S's dynamic range is substantially driven by resolution/kernel geometry, not real structure alone (02) - a real limitation to state.
3. That resolution effect is causally confirmed, not just correlated (03).
4. The primary planned ML drivers (SFR, stellar mass, SFMS offset) survive resolution control; sSFR and ellipticity do not, and should be treated with caution or residualized against `n_res` before use (04).
5. The 0.35 scale choice is not fragile - conclusions hold across 0.10-0.50 (05).
6. Switching to a fixed-kpc kernel would make things worse, not better (06).

**Recommended follow-up, not yet implemented here:** when the ML "key
drivers" step is built, feed it `n_res` (or `sigma_over_psf`) explicitly as
a covariate, or residualize `S` against it first, rather than relying on
post-hoc checks like 04 alone - see the parent conversation for the
reasoning.
