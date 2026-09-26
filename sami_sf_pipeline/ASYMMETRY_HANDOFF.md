# Handoff: add the missing background-asymmetry term (`A_bg`) to `asymmetry.py`

**Purpose of this document**: context for a separate Claude Code session
working in this repo (`sami_sf_pipeline`, inside `sami copy/`) to fix a
confirmed, audited gap in the Asymmetry (`A`) statistic — the same class of
problem Smoothness (`S`) already had and already had fixed via `B_S`. This
was surfaced by an ML-side diagnostic session working in a sibling `sf_ml`
repo, which is not visible to this session; everything needed is
reproduced below. Nothing in this repo has been changed yet — this is
scope + a validated recipe to follow, mirroring the `S`/`B_S` precedent.

**Naming convention wanted for the output** (agreed with the user, keep this
exact mapping — note it's the reverse letter order from `S`'s convention,
so don't copy that pattern by habit):

| New name | Meaning |
|---|---|
| `A` | The **corrected** asymmetry value (background-subtracted). Becomes the new default/production `A`. |
| `A_no_sub` | The **original, uncorrected** value — i.e. today's `A_min` renamed. Kept for comparison/backward compatibility. |
| `A_BS` | The estimated background-asymmetry term itself (the quantity subtracted). |

(Contrast with `S`'s existing convention in `smoothness.py`/`master_final.csv`:
`S` = corrected, `S_no_B_S_sub` = original, `B_S` = the term. Same idea,
different label choices for `A` — `A_no_sub` and `A_BS` specifically, not
`A_no_A_BS_sub`/`B_A`.)

---

## 1. The confirmed problem

`src/03_metrics/asymmetry.py` computes the CAS Asymmetry index following
Conselice (2003) eq. 1:

```
A = sum(|I0 - I180|) / sum(|I0|)
```

with a grid-search over rotation centre to find `A_min` (the production
value, currently renamed to `A` in `master_final.csv` by
`src/04_build_master_csv.py`).

**This repo's own `AUDIT_REPORT.md`, §3.4, already documents the gap**:

> "The full Conselice definition subtracts a background-asymmetry term,
> `A = min(Σ|I₀−I₁₈₀|/Σ|I₀|) − A_bg`, estimated by rotating a blank sky
> region the same way. **This repo's implementation has no `A_bg` term at
> all.** Given the data is already continuum-masked/zeroed outside the
> galaxy footprint, this may be a deliberate and reasonable simplification,
> but it is a real deviation from the formal definition and isn't flagged
> as such anywhere in the docstrings (which cite 'Conselice 2003, eq. 1' as
> if it's the complete definition)."

`asymmetry.py`'s own docstring repeats this same admission almost verbatim.

This is structurally the **same missing term** that `S` had before it was
fixed — see `src/03_metrics/smoothness.py`'s docstring and `compute_B_S()`
for the already-shipped precedent (Conselice's background-smoothness term
`B_S`, validated in
`../smoothness/scripts/kernel_testing/08_background_subtraction_test.py`).
That fix used each galaxy's own per-pixel error map (no blank-sky region
needed — SAMI hexabundle pointings don't reliably have one) rather than the
literal Conselice blank-sky-patch method. The same approach should port
directly to `A`.

**One difference to be aware of**: `asymmetry.py`'s own noise handling
already does something *related* but distinct — `apply_error_subtraction`
computes `I_clean = clip(flux - error, 0)`, a hard per-pixel S/N cut. This
is not the same mechanism as `A_bg`/`B_S`: it's a pre-smoothing/pre-rotation
per-pixel bias correction, not a post-computation roughness/asymmetry-floor
subtraction. `smoothness.py`'s own comments (and
`07_negative_clipping_test.py` in the kernel_testing folder) found this kind
of clipping does *not* fix the analogous problem for `S` — don't assume it
already covers this for `A` either. Worth a quick equivalent check
(does `A` correlate with a pure-noise-through-the-same-formula quantity the
way `S_no_B_S_sub` did at rho≈0.94? `02_noise_floor.py` in kernel_testing is
the template) before/alongside implementing the fix, so there's a clear
"before" number to validate the "after" against — this repo doesn't seem to
have that baseline test for `A` yet.

---

## 2. Why this matters now (the ML-side evidence that surfaced it)

A diagnostic session in the sibling `sf_ml` repo ran `re[arcsec]` vs.
`re_kpc` vs. `n_res` (`= re_used/seeing`, resolution elements across the
galaxy) feature swaps across RF/XGBoost/CatBoost/NN models predicting `A`
(the current `A_min`-derived, uncorrected value). Result, consistently
across **all four model types**:

- `n_res` **outperformed both** `re[arcsec]` and `re_kpc` on both CV R² and
  permutation importance (e.g. permutation importance ~0.10–0.16 for
  `n_res` vs. ~0.01–0.02 for `re_kpc`, which was barely above a `RANDOM`
  noise-baseline feature).
- This is the *opposite* pattern from the same test run against Gini (`G`),
  where `re_kpc` won cleanly and `n_res` was the weakest of the three — and
  different again from Concentration (`C`), where raw `re[arcsec]` won and
  `n_res` was weakest. `A` is the one metric where the resolution proxy
  cleanly dominates.
- No formula-level use of `re` was found in `asymmetry.py` (it uses the
  image's own geometric centroid + grid search, not `re`), so this isn't a
  literal kernel-width mechanism the way it was for `S` pre-fix. But a
  missing noise/background correction produces exactly this signature:
  noisier, more poorly-resolved galaxies get a spuriously inflated `A`
  (residual doesn't cancel between the image and its noisy rotated copy),
  and that bias scales with how few resolution elements the galaxy has —
  which is exactly what `n_res` measures. This is consistent with, though
  not fully identical to, the mechanism `02_noise_floor.py` demonstrated
  for `S`.

**Practical implication**: fixing this at the ML feature-engineering level
(e.g. controlling for `n_res`) would only be a stopgap. The `sf_ml` session
is treating current `A`/`A_min` values the same way `S_no_B_S_sub` was
treated before `B_S` existed — i.e. a known-partial statistic — pending
this fix.

---

## 3. The validated recipe to adapt (`B_S` → `A_BS`)

From `src/03_metrics/smoothness.py::compute_B_S`:

```python
def compute_B_S(err: np.ndarray, sigma_px: float, catid: int, n_draws: int = N_DRAWS_BS) -> float:
    """Median, over n_draws white-noise realisations drawn from this galaxy's
    own error map, of sum(|I_noise - I_noise_smoothed|) at sigma_px."""
    rng = np.random.default_rng((BS_SEED, catid))
    scale = np.where(err > 0, err, 0.0)
    numerators = []
    for _ in range(n_draws):
        noise = rng.normal(loc=0.0, scale=scale)
        I_noise = noise - err
        smoothed = gaussian_filter(I_noise, sigma=sigma_px)
        numerators.append(float(np.nansum(np.abs(I_noise - smoothed))))
    return float(np.median(numerators))
```

then: `S = max((sum(residual) - B_S) / sum(|I_clean|), 0)`.

The `A` analog should follow the same shape — draw noise from the error
map, push it through **the same formula used for the real signal**
(rotate-and-differance, not smoothing), and subtract the median simulated
numerator from the real numerator before dividing:

```python
def compute_A_BS(err, cy, cx, catid, n_draws=N_DRAWS_A_BS):
    """Median, over n_draws white-noise realisations drawn from this
    galaxy's own error map, of the A-numerator (sum|noise_180 - noise|
    over the common footprint) at this galaxy's best-fit centre."""
    rng = np.random.default_rng((A_BS_SEED, catid))
    scale = np.where(err > 0, err, 0.0)
    numerators = []
    for _ in range(n_draws):
        noise = rng.normal(loc=0.0, scale=scale)
        I_noise = apply_error_subtraction(noise, err)  # same clip(., 0) as real I_clean -- open Q, see below
        _, num = compute_A_numerator(I_noise, cy, cx)   # same rotate/shift/common-mask logic as compute_A
        numerators.append(num)
    return float(np.median(numerators))
```

then something like: `A = max((real_numerator - A_BS) / real_denom, 0)`,
with `A_no_sub` = today's `A_min` (renamed, unchanged).

**Open design questions this session should resolve (not prescribed here
— use judgement, document the choice)**:

1. **Which centre to use for the noise draws?** `compute_A` currently runs
   inside a 10×10 grid search (100 candidate centres) to find `A_min`. Doing
   50 noise draws *per candidate centre* would be 5,000 extra rotate/shift
   operations per galaxy — likely too slow (smoothness's B_S at 50 draws
   already added ~6–7 min for the whole sample doing 1 operation per draw,
   not 100). The natural mirror of `B_S` (computed once, at the real
   signal's fixed `sigma_px`) is probably: run the grid search on the real
   data first to get `best_centre`/`A_min` as today, **then** compute
   `A_BS` once, at that fixed `best_centre`, and subtract. Flag this choice
   explicitly in the new script's docstring the way `smoothness.py` does.
2. **Should the noise draws go through `apply_error_subtraction`'s clip, or
   the unclipped `noise - err`?** `smoothness.py`'s `B_S` noise draws use
   `noise - err` unclipped, matching its own (unclipped) `I_clean`
   construction. `asymmetry.py`'s real `I_clean` **is** clipped
   (`clip(flux - err, 0)`), so for consistency the noise draws should
   probably go through the identical `apply_error_subtraction(noise, err)`
   call, not an unclipped variant — but confirm this doesn't degenerate
   (e.g. an all-zero noise draw after clipping) before committing to it.
3. **Footprint handling for noise draws.** `compute_A`'s common-mask logic
   depends on `img != 0` and a rotated threshold relative to `np.max(img)`.
   A pure-noise draw has a different footprint/max than the real image —
   decide whether the noise draw should reuse the *real* image's footprint
   mask (more analogous to `B_S`'s reuse of the real `sigma_px`) or compute
   its own from the noise realisation. Reusing the real mask is probably
   more defensible (keeps the comparison apples-to-apples: "how much
   spurious asymmetry would this same footprint show if it were pure
   noise") — but worth a quick sensitivity check either way.
4. **Draw count.** `smoothness.py` validated `N_DRAWS_BS=50` as stable
   (~4% median relative draw-to-draw scatter). Don't assume the same count
   is sufficient for `A`'s different formula — check the analogous
   convergence criterion before fixing a number.

---

## 4. Validation checklist (mirror `08_background_subtraction_test.py`)

That script (`../smoothness/scripts/kernel_testing/08_background_subtraction_test.py`)
is the template — same four criteria should be checked for the `A` fix,
ideally in an analogous new test script before touching production code:

1. **Held-out noise draws** (a *second*, independent set of draws never
   used to fit `A_BS`) pushed through the corrected formula should cluster
   near `A≈0`, and lose most of their correlation with `n_res`/`z` — this
   is the real test, not a self-consistency check against the same draws
   used to fit the correction.
2. **Rank relationship**: `A` (corrected) vs. `A_no_sub` should still
   correlate sensibly (changes things, especially for poorly-resolved
   galaxies, without scrambling the sample).
3. **Do the real drivers survive?** Whatever `A`'s established physical
   correlates are in this sample (check the `sf_ml`/ML_HANDOFF-side work,
   or the equivalent of `log_SFR`/`log_Mstar`/`delta_ms` used for the `S`
   validation) should survive or strengthen under the correction, the way
   they did for `S` (`log_SFR` -0.565→-0.552, `log_Mstar` -0.450→-0.453,
   `delta_ms` -0.426→-0.401 — barely moved, which was read as "the
   correction doesn't damage genuine signal").
4. **Negative-value fraction and estimate stability**: how many galaxies
   go negative pre-clipping (0/705 for `S`), and the draw-to-draw scatter
   of the `A_BS` estimate itself.

`S`'s validation criterion for a go/no-go was roughly: R² of
corrected-statistic-on-held-out-noise should drop well below the
pre-correction baseline (0.94→0.12 for `S`) and |rho| < 0.4 with resolution
proxies. Establish `A`'s own pre-correction baseline (§1's suggested
`02_noise_floor.py`-equivalent check) first so there's a real number to
compare the "after" against, rather than assuming 0.94 carries over.

---

## 5. Downstream propagation once validated

`src/04_build_master_csv.py` currently does:

```python
master = master.merge(asym[["CATID", "A_min"]], on="CATID", how="left")
...
master = master.rename(columns={"gini_halpha": "G", "A_min": "A"})
```

This needs to change to mirror how `S`/`B_S`/`S_no_B_S_sub` are already
merged in (same file, a few lines below):

```python
smooth_cols = ["CATID", "S", "B_S", "S_no_B_S_sub", "re_used", "re_source"]
smooth_present = [c for c in smooth_cols if c in smooth.columns]
master = master.merge(smooth[smooth_present], on="CATID", how="left")
```

i.e. once `asymmetry.py` outputs `A`, `A_BS`, `A_no_sub` columns directly
(analogous to `smoothness.py` now outputting `S`, `B_S`, `S_no_B_S_sub`),
update the merge to pull all three, conditionally (`c in asym.columns`) so
nothing breaks if this is run before the fix lands — same pattern used for
the `S` merge's own comment about being "a no-op until then."

After that:
- Re-run `src/04_build_master_csv.py` to regenerate `master_final.csv` with
  the new `A`/`A_no_sub`/`A_BS` columns.
- The `sf_ml` repo's `data/clean_master_sfms.py` will need re-running too
  (mirrors what already happened for the `S` fix — see its
  `ML_HANDOFF.md`), and its four Asymmetry ML models (`RF_A.ipynb`,
  `XGBoost_A.ipynb`, `CatBoost_A.ipynb`, `Neural_Net_A.ipynb`) should be
  retrained/re-evaluated against the corrected `A`, same as was planned for
  `S`.
- Worth pinging back to the `sf_ml` session/user once this lands, the same
  way the `S`/`B_S` fix was communicated across repos via `ML_HANDOFF.md`'s
  "Update" notes at the top of that file.

## 6. Files referenced here

- `src/03_metrics/asymmetry.py` — the file to modify.
- `src/03_metrics/smoothness.py` — the validated precedent (`compute_B_S`,
  the module docstring's rationale, output column pattern).
- `src/04_build_master_csv.py` (lines ~216–231) — the merge to update.
- `AUDIT_REPORT.md` §3.4 — the original audit finding this fix addresses.
- `../smoothness/scripts/kernel_testing/02_noise_floor.py` — template for
  establishing `A`'s own pre-correction noise-floor baseline.
- `../smoothness/scripts/kernel_testing/07_negative_clipping_test.py` —
  template for checking whether the existing `clip(flux-err, 0)` step
  already mitigates this (it didn't, for `S`).
- `../smoothness/scripts/kernel_testing/08_background_subtraction_test.py`
  — template for the `A_BS` validation script.
