# Handoff: r_e / ellip feature-importance investigation → ML models

**Purpose of this document**: context for a Claude Code session working in the
ML models folder, carried over from a separate conversation that investigated
the smoothing-kernel methodology behind the Hα Smoothness (clumpiness)
statistic `S`, and which surfaced a broader concern about several features
used across all four CAS+Gini ML models (Concentration, Gini, Asymmetry,
Smoothness). That conversation did not have access to the ML model code, so
everything below is context and a diagnostic plan, not a description of the
actual feature sets/model code - establish those first, then apply this.

**Scope agreed with the user**: diagnose across all four metrics' models
first and report findings; do not change any model code until the user has
reviewed the diagnostics. Smoothness's leakage mechanism (below) is already
confirmed at the pipeline-formula level, but the actual ML-side diagnostic
(how much it actually distorts that model's importances/predictions) should
still be run and reported before anything is changed, per the user's choice.

**Data status (updated)**: `sami_sf_pipeline/outputs/csv/master_final.csv` is
now fully up to date with everything referenced below - `re_used`,
`re_source`, `re_kpc`, `n_res`, `B_S`, `S_no_B_S_sub`, the corrected `S`, and
`Dfil`/`Dnode`/`Dvoid` are all present and verified (spot-checked against the
values this investigation derived independently). Nothing here is pending
regeneration any more - see Section 6 for the exact column set.

---

## 1. Background

This repo computes four morphology statistics per SAMI galaxy from Hα flux
maps and an external catalogue: Concentration (C), Gini (G), Asymmetry (A),
Smoothness/clumpiness (S). The end goal (stated by the user) is to feed these
into ML models to identify the key physical drivers of spatially-resolved
star-formation distribution. The user noticed `re` (effective radius, in
arcsec) consistently ranks with very high feature importance across all four
metrics' models, and asked whether that's a genuine result or a methodological
artifact. Investigation (in another folder, `smoothness/scripts/kernel_testing/`,
which can provide the full T1-T8 test sequence on request) found real,
mechanism-specific problems - summarised below - and the user then also asked
about `ellip` (ellipticity) as a possible alternative/size proxy.

Sample: `in_full_sam` = Wang (2022)'s 6-cut purity sample. In the current
`master_final.csv`, `in_full_sam == True` for 837 galaxies, of which 705 also
have a valid Smoothness measurement (`S`/`S_no_B_S_sub` notna) - **705 is the
denominator used throughout every rho/R² number quoted below**, not 837.

### IMPORTANT: naming convention in `master_final.csv` (read this before anything else)

The user renamed these columns from what the diagnostic scripts originally
called them, specifically so their existing ML code needs minimal changes.
**`S_corrected` from the investigation is now just called `S`** in
`master_final.csv`, and **the original, uncorrected clumpiness value (what
the diagnostic scripts called `S`/`S_production`) is now called
`S_no_B_S_sub`**. This is the opposite of what you'd guess from the names at
a glance, so:

| Column in `master_final.csv` | What it is |
|---|---|
| `S` | The **corrected** clumpiness value: `max((residual_sum − B_S) / flux_sum, 0)`. Use this as your default Smoothness target/feature going forward. |
| `S_no_B_S_sub` | The **original, uncorrected** clumpiness value (`residual_sum / flux_sum`, no background-smoothness term subtracted). Kept for comparison/backward compatibility. |
| `B_S` | The estimated background-smoothness term itself (see Section 2). |

Every historical number quoted in this document from before the B_S fix was
derived (Sections 2, 3, 5) describes the behaviour of what is now called
`S_no_B_S_sub`, **not** the current `S` column - relabelled accordingly
below. Where a number describes the corrected value, it's labelled `S`
(current) explicitly.

---

## 2. Per-metric diagnosis of `re`

| Metric | Formula-level leakage? | Mechanism | Confidence |
|---|---|---|---|
| **Smoothness (S / S_no_B_S_sub)** | **Yes, confirmed** | `sigma_px = 0.35 * re / plate_scale` is a literal input parameter to the Gaussian kernel that defines both `S` and `S_no_B_S_sub`. `re` doesn't correlate with either, it partly *determines* how they're computed. | Confirmed by reading `sami_sf_pipeline/src/03_metrics/smoothness.py` directly. |
| **Concentration (C)** | **Suspected, not confirmed** | `concentration.py`'s own docstring notes Wang (2022)'s external CIndex (`log(r50,Ha/r50,cont)`) was measured with "their own elliptical-aperture" methodology - `re` and/or `ellip`/PA may have sized or oriented those apertures. Needs checking against Wang's actual methodology, which is out of this repo's scope. | Flagged, unverified. |
| **Gini (G)** | No | Checked directly (`grep` for `re`/`ellip` in `src/03_metrics/gini.py`) - not used. Gini operates on finite/positive masked pixels only. | Confirmed absent from this repo's formula. |
| **Asymmetry (A)** | No | Checked directly in `src/03_metrics/asymmetry.py` - uses the image's own geometric centre + grid search, not `re`. | Confirmed absent from this repo's formula. |

**Even where there's no formula leakage (Gini, Asymmetry), `re` is a strong
proxy for the shared continuum mask's footprint pixel count** (bigger galaxy
on sky → more mask pixels), which is a literature-documented source of bias
in CAS-type statistics generally (Conselice 2003, Lotz 2004 both discuss
minimum-pixel-count requirements). So even the two "clean" metrics have a
plausible indirect channel for `re` to pick up importance without being a
real physical driver.

### The resolution/noise mechanism behind this (established rigorously for Smoothness)

All of the following describe `S_no_B_S_sub` (the original, uncorrected
value) unless stated otherwise - these tests predate the B_S fix:

- `S_no_B_S_sub` vs `re` (arcsec): Spearman rho = +0.538 (in_full_sam, n=705)
- `S_no_B_S_sub` vs `n_res` (`= re/seeing`, resolution elements across the
  galaxy): rho = +0.520
- A quantity built from **pure white noise** (zero real flux information),
  pushed through the identical formula at each galaxy's own kernel width,
  explained **R²=0.94** of the variance in `S_no_B_S_sub` across the sample
  (`kernel_testing/02_noise_floor.py`).
- This was causally confirmed (not just correlational) by artificially
  redshifting 25 well-resolved real galaxies and re-measuring the statistic
  as simulated resolution degraded: it dropped to ~27% of its native value by
  `n_res~1`, monotonic in 92% of anchors, tracking the real cross-sample
  relation almost exactly (`kernel_testing/03_redshift_degradation.py`).
- **This noise-floor problem has since been fixed**: a proper
  background-smoothness term `B_S` (Conselice 2003's missing term, estimated
  from each galaxy's own per-pixel error map via Monte Carlo noise draws, no
  blank-sky region needed) was validated on held-out noise draws never used
  to fit it: R² of the corrected value regressed on that held-out noise
  dropped from 0.94 to 0.12, and held-out pure noise pushed through the
  corrected formula clustered at ≈0, as it should. This is implemented in
  `sami_sf_pipeline/src/03_metrics/smoothness.py` and is now live in
  `master_final.csv` as `B_S` and `S` (see naming table above) -
  **confirmed in production**: within `in_full_sam` (n=705), 0 galaxies
  clip to `S=0`, exactly matching the validation run.
- **Caveat that still applies after the B_S fix**: `B_S` removes the
  noise/kernel-geometry artifact specifically. It does NOT remove the
  separate, genuine resolution/information limit the redshift-degradation
  test demonstrated (real structure that coarser sampling genuinely cannot
  recover) - in a direct rho comparison, the corrected `S` still correlated
  with `n_res` at essentially the same level `S_no_B_S_sub` did (+0.526 vs
  +0.520 in the validation run). A resolution proxy (`n_res` or `seeing`)
  should still be treated as a covariate/control in the ML step, not assumed
  solved by the B_S correction alone.
- The real drivers barely moved under the correction, in a direct check:
  `log_SFR` rho with `S_no_B_S_sub` = -0.565 vs with `S` = -0.552;
  `log_Mstar` -0.450 vs -0.453; `delta_ms` -0.426 vs -0.401. The correction
  doesn't appear to damage genuine signal.

### Other mechanisms that inflate `re`'s apparent importance, independent of resolution

- **Distance/redshift entanglement**: `re` in arcsec blends true physical size
  with angular-diameter-distance (`kpc_per_arcsec(z)` spans a ~20x range
  across this sample's z=0.004-0.11). In a flux/volume-limited survey,
  distance correlates with stellar mass (Malmquist-type selection), SFR, and
  survey depth - so high `re` importance can quietly mean "distance/selection
  effects matter," not "size drives morphology."
- **Statistical artifact in tree-based impurity importance (MDI)**:
  continuous, fine-grained numeric features get inflated MDI/"Gini importance"
  (unrelated to the astrophysical Gini coefficient - watch for this name
  collision) purely from offering more split points, independent of real
  predictive power (Strobl et al. 2007). `re` is exactly this kind of
  feature. Permutation importance or SHAP on held-out data are more robust to
  this.
- **Unit choice does NOT matter for Smoothness's own kernel definition**:
  `01_unit_invariance.py` proved `0.35 x re[arcsec]` and an equivalent
  kpc-based definition give numerically identical `sigma_px` (they cancel
  algebraically via `kpc_per_arcsec(z)`) - so this is not "arcsec vs kpc" as
  a methodological choice for the kernel (applies to both `S` and
  `S_no_B_S_sub`, since both use the same kernel). It IS a live question for
  what value you feed an ML model as a raw feature, though: `re[arcsec]`
  (blends size+distance) vs `re[kpc]` (isolates physical size, decoupled
  from distance to the extent it's used consistently) are genuinely
  different ML features, even though they produce the same kernel/S.

---

## 3. `ellip` - not a size proxy, different (real) confound

`ellip` measures projected axis ratio, not size - confirmed by this
pipeline's own convention (`wang_quality_cuts.py` labels the `ellip <= 0.7`
cut "Inclination"). For a disk galaxy this is driven overwhelmingly by
viewing angle: face-on looks round, edge-on looks flattened, independent of
whether the galaxy is physically large or small. **Do not use it as a
substitute for `re`** - it answers a different question.

- Confirmed no formula-level use of `ellip` in Gini/Asymmetry/Smoothness
  (same grep check as above). Concentration is the one to check (see table
  above - same "their own elliptical-aperture" methodology note applies to
  `ellip`/PA, possibly even more directly than to `re`).
- Real, legitimate astrophysical reason `ellip` could still matter: Hα dust
  attenuation scales with inclination (more disk path length, edge-on). High
  `ellip` importance could reflect genuine viewing-angle/dust geometry
  effects on the *observed* light distribution, which is real physics but
  different from what "key drivers of intrinsic SF morphology" is presumably
  trying to isolate - worth distinguishing in interpretation, not necessarily
  "fixing."
- Training-sample range is truncated at `ellip <= 0.7` by the `in_full_sam`
  cut - any relationship is only valid within that truncated range; watch for
  edge effects near the boundary.
- Bounded/narrower range than `re` → less susceptible to the MDI
  high-cardinality artifact above, for whatever that's worth.
- Flagged as resolution-sensitive for `S_no_B_S_sub` specifically: `ellip`'s
  correlation weakened substantially and lost significance in a
  resolution-matched subsample test (Section 5) - that test predates the B_S
  fix and has not yet been re-run against the corrected `S`.

If what's actually wanted is a clean, resolution-safe "how big/massive is
this galaxy" control variable, **`log_Mstar` is probably the better
candidate** - it isn't entangled in any of these metrics' own measurement
apparatus the way `re` (kernel width for S) or `ellip` (possibly apertures
for C) are.

---

## 4. Diagnostic checklist to run per model (this is the deliverable for this pass)

For **each of the four metrics' models** (adapt to however your actual code/
feature sets are structured - this doc doesn't know that):

1. **Leakage audit**: for every feature currently in that model, check
   whether it (or something deterministic from it) is a literal input to
   that metric's own formula. Section 2's table gives you `re`/`ellip` for
   free; check any *other* features you're using that this investigation
   didn't cover (e.g. `seeing`, `PA`, `z`, `Mstar` - are any of these used
   inside the C/G/A/S calculation itself, directly or via a shared
   intermediate like the continuum mask?).
2. **MDI vs. permutation/SHAP importance**, on held-out data, for `re`
   specifically in each model. A large gap between them is the
   fingerprint of the statistical artifact in Section 2, separate from
   anything physical.
3. **Refit with `re[arcsec]` swapped for `re[kpc]` and `n_res`** - both are
   already columns in `master_final.csv` (`re_kpc`, `n_res` - see Section 6),
   no need to derive them. Also worth trying raw mask footprint pixel count
   if you want that comparison too (would need deriving from the masked FITS
   in `data/masked/halpha/`, not currently a column). See which one "absorbs"
   the importance currently sitting on `re[arcsec]` - that tells you which
   story is true (real physical size vs. resolution artifact vs.
   distance/selection confound).
4. **Correlation/VIF matrix** among `re`, `re_kpc`, `n_res`, `z`, `seeing`,
   `log_Mstar`, `ellip` (and whatever else is in each feature set) before
   trusting any single feature's importance ranking.
5. **Partial dependence / SHAP dependence plot shape** for `re` and `ellip`
   in each model - a monotonic, saturating shape that tracks resolution
   proxies would support the artifact story over a genuine physical
   relationship.
6. **For Smoothness specifically**: check whether your existing model(s)
   were trained against `S_no_B_S_sub` (likely, if trained before this fix)
   or the corrected `S`. If the former, consider retraining/re-evaluating
   against `S` given the noise-floor finding above, and note that Section 5's
   resolution-matched-subsample results (T4) have only been validated
   against `S_no_B_S_sub`, not yet re-run against `S` - worth doing if time
   allows, since the correlations could shift slightly even though the T8
   spot-check (previous section) found them largely stable.
7. Report back per-metric: is `re`/`ellip`'s importance mostly artifact,
   mostly real, or mixed - and what changed when you controlled for it.

## 5. Useful reference numbers/results already established (don't re-derive)

All numbers in this section describe `S_no_B_S_sub` (pre-B_S-correction)
unless stated otherwise - see the naming note in Section 1/2 before using
any of these.

- Plate scale: confirmed exactly 0.5"/px across **all 2530** masked Hα FITS
  files, zero deviation - `PLATE_SCALE` hardcoded in `smoothness.py` is safe.
- `S_no_B_S_sub` vs `re`[arcsec]: rho=+0.538; vs `n_res`: rho=+0.520; vs `z`:
  rho=-0.327 (all in_full_sam, n=705).
- Real drivers that **survive** resolution control (partial correlation
  controlling `n_res`, and a resolution-matched `n_res in [2.5,4.5]`
  subsample, both against `S_no_B_S_sub`): `log_SFR` (-0.565 raw → -0.669
  partial), `log_Mstar` (-0.450 → -0.608), `delta_ms`/SFMS offset (-0.426 →
  -0.424, essentially unchanged). These look like genuine signal, not
  resolution artifacts. A direct (not resolution-matched) check against the
  corrected `S` found these three barely moved under the B_S correction
  (Section 2) - consistent with, though not a full re-run of, this result.
- Flagged as resolution-sensitive against `S_no_B_S_sub`: `log_sSFR` and
  `ellip` both weaken substantially and lose significance in the
  resolution-matched subsample - treat their importance with caution until
  you've run the checklist above on your actual models, ideally against `S`.
- Kernel scale (0.35 x re) is not fragile: rank-stability rho >= 0.939 across
  the whole 0.10-0.50 range tested (against `S_no_B_S_sub`), no driver
  changes sign - not a live concern for the ML work, mentioned for
  completeness.

## 6. Relevant files in this repo, if useful to reference or reuse

- `sami_sf_pipeline/outputs/csv/master_final.csv` - **the** master table,
  fully up to date (3031 rows, 60 columns). Relevant columns for this work:
  `C`, `G`, `A`, `S` (corrected), `S_no_B_S_sub` (original), `B_S`,
  `re`, `re_com` (raw Wang values), `re_used`, `re_source` (fallback-resolved,
  unphysical-excluded - the value actually used to size the Smoothness
  kernel), `re_kpc`, `n_res` (both derived from `re_used`), `ellip`, `z`,
  `seeing`, `log_Mstar`, `log_SFR`, `log_sSFR`, `delta_ms`, `environment`,
  `Dfil`, `Dnode`, `Dvoid` (cosmic web distances, matched for 1161/3031
  galaxies, NaN elsewhere), `in_full_sam`, and the individual `cut_*` columns.
- `sami_sf_pipeline/src/03_metrics/{smoothness,gini,asymmetry,concentration}.py`
  - the actual formula implementations (leakage ground-truth).
- `sami_sf_pipeline/outputs/csv/smoothness.csv` - lower-level Smoothness-only
  table (same `re_used`/`re_source`/`B_S`/`S`/`S_no_B_S_sub` info, plus
  `sigma_arcsec`, `sigma_px`, `kernel_scale`). Not usually needed now that
  `master_final.csv` carries `re_used`/`re_kpc`/`n_res` directly, but useful
  if you want the raw kernel-width values too.
- `smoothness/scripts/kernel_testing/` - the full investigation (8 scripts,
  README.md summarising all findings - note the README predates the B_S
  column rename, so it still uses the original `S`/`S_corrected` names from
  when the tests were written). `common.py` has reusable loaders
  (`load_full_sample()`, `load_full_sam()`, `spearman_table()`,
  `CANDIDATE_DRIVERS` dict, `kpc_per_arcsec()`) that may be directly useful
  for building the diagnostics above rather than rewriting them - but note
  `common.py` also predates the rename, so check what it actually loads
  (`S` there means the original script's `S`, i.e. `master_final.csv`'s
  `S_no_B_S_sub`) before reusing it against the current CSV.

## 7. What this document can't tell you - establish these first

Actual current feature sets per model, model type(s) (tree-based? linear?
other?), importance method currently in use, whether `re`/`ellip` appear in
all four models or a subset, which Smoothness target your existing model(s)
were trained against (`S_no_B_S_sub` vs the corrected `S` - see Section 4
point 6), train/test methodology, and which sample (`in_full_sam` or
broader) each model actually trains on. Start by establishing these from the
real code before applying anything above.
