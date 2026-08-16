# Changelog

All entries describe the move from the old repo (`../` relative to this
folder) to `sami_sf_pipeline/`. See `AUDIT_REPORT.md` for the original
diagnostic findings this is based on.

## [2026-08-10] Fixed-scale flux map notebook added

`notebooks/flux_map_fixed_scale.ipynb`: notebook version of `01_flux_map.py`
(same raw-Hα-cube data source), with a fixed colour scale
(`FLUX_VMIN`/`FLUX_VMAX`, default 0-40) instead of auto-scaling each galaxy
to its own min/max - for thesis figures where the point is comparing
absolute Hα brightness between galaxies, not just each one's own internal
structure. Includes a side-by-side comparison-panel section (shared scale +
one colorbar) that defaults to the same three example CATIDs used in
`sfms_explorer.ipynb` and pulls each one's `delta_ms`/SFMS-position label
from that notebook's output if available.

`src/utils/plotting.py::plot_map` gained `vmin`/`vmax`/`cmap`/`ax`
parameters to support this (all default to the previous behaviour -
`vmin=vmax=None` auto-scales exactly as before, `ax=None` creates its own
figure exactly as before) - `01_flux_map.py` and its CLI behaviour are
unchanged.
## [2026-08-10] Sentinel-value bug found and fixed in Wang SFR/Mstar loading

Found while building `notebooks/sfms_explorer.ipynb`: the M*-sSFR plot
showed an obviously-too-tight diagonal streak of `composite_SF` galaxies at
low mass. Traced it to `src/utils/wang.py::load_wang_txt` (the loader
centralised for `03_metrics/concentration.py` and `04_build_master_csv.py`
in the original rebuild): it was missing sentinel-value cleaning that the
old `master/scripts/build_csv.py::load_wang_txt` had -

```python
# old build_csv.py, never carried into utils/wang.py:
df.loc[df[c] == 0, c] = np.nan       # SFR columns: 0 = "no valid detection" in Wang
df.loc[df[c] <= -999, c] = np.nan    # explicit missing-data sentinel
```

Without it, a sentinel `SFR_n2ha_s = 0` (stored as a log10 value in Wang's
raw catalogue) was being read as a literal `log10(SFR) = 0` -> `SFR = 1
Msun/yr` "measurement", identical across every affected galaxy regardless
of mass - hence the deterministic-looking streak (sSFR = 1/M* traces a
straight line in log-log space).

**Confirmed scope:** 747 of 1758 `composite_SF` galaxies (42%) had a
sentinel-zero `SFR_n2ha_s`; 445 of 1201 `pure_SF` galaxies (37%) had a
sentinel-zero plain `SFR`, though none of those happened to coincide with a
galaxy actually classified `pure_SF` for the SFR-resolution step, so the
practical impact there was zero. `Mstar` had no `<=-999` sentinels in the
current data (checked, none found) - only the SFR-family columns were
affected.

**Fix:** `src/utils/wang.py::load_wang_txt` now replicates the old
`build_csv.py` cleaning exactly. Re-ran the full downstream chain -
`03_metrics/concentration.py` -> `03_metrics/smoothness.py` (unaffected
numerically, re-run only for consistency - it doesn't touch SFR/Mstar) ->
`04_build_master_csv.py` -> `05_sfms.py`. Net effect: 747 `composite_SF`
galaxies that previously had a fabricated `SFR=1`/`sfr_flag='n2ha_s'` now
correctly get `SFR=NaN`/`sfr_flag='nan_no_n2ha'` (`master_results`'s
"SFR present" count: 2959 -> 2212). The SFMS refit shifted slightly now that
the contaminating points are gone (1657 vs. 2404 galaxies anchoring the fit;
`S0=2.87, a1=0.22, Mt=9.89` vs. the previous entry's `S0=2.90, a1=0.20,
Mt=9.92`) - a small, expected shift from removing bad data, not a
regression.

Because this loader is shared (the whole point of centralising it - see the
"SFR/Mstar log-detection heuristic" entry below), fixing it once corrects
`concentration.csv`, `master_<date>.csv`, and `master_sfms_<date>.csv`
together. Anyone with an older dated CSV from before 2026-08-10 should
regenerate rather than keep using it.

## [2026-08-10] SFMS Explorer notebook added

`notebooks/sfms_explorer.ipynb`: plots log(M*) vs log(sSFR) for every
galaxy in Wang's catalogue (`in_C == True`), colored by `spectral_class`,
with the fitted SFMS line and a shaded on-MS band (default ±0.3 dex)
overlaid. A config cell (`HIGHLIGHT_CATIDS`) lets you highlight any number
of galaxies by CATID; a summary table below the plot reports each
highlighted galaxy's `delta_ms` and on/above/below classification. Reads
whichever `master_sfms_<date>.csv` (+ its paired fit-params/ridge-points
files) is most recent - no editing needed as new pipeline runs are added.

## [2026-08-03] Pipeline rebuild

### Gini coefficient - formula corrected

**What was wrong:** every Gini implementation in the old repo
(`src/gini.py::gini_from_mask`, `src/morphology.py::gini_from_map`) used the
generic population-statistics Gini formula:

```
G = 2*sum(i * x_i) / (n * sum(x_i))  -  (n+1)/n        [x sorted ascending]
```

This is the textbook income-inequality Gini estimator. Galaxy-morphology CAS
work (Abraham et al. 2003; Lotz et al. 2004) uses a different,
astronomy-specific convention:

```
G = 1/(mean(x) * n*(n-1)) * sum( (2i - n - 1) * x_i )   [x sorted ascending]
```

Algebraically, `G_astro = G_old * n/(n-1)` - the old values were
systematically too low, and by an amount that depends on how many valid
pixels (`n`) a galaxy had after masking. At the `min_pixels=10` floor the
old value was ~11% too low; at n=500 the effect shrinks to ~0.2%. Because
`n` correlates with galaxy size/brightness/S-N, this wasn't just an additive
offset - it distorted any Gini-vs-mass/size/environment trend.

**What changed:** `src/03_metrics/gini.py` now implements the Lotz (2004)
formula (`gini_lotz2004()`). Verified against the old formula directly on
identical pixel data (CATID 7139, n=175): old value 0.492614, new value
0.495446, and `0.492614 * 175/174 = 0.495446` exactly.

**Which files/columns are affected:** every Gini value anywhere in the old
repo, since both implementations shared the bug and nothing else computed
Gini independently:

| Old file | Old column(s) |
|---|---|
| `gini/gini_outputs/tables/Halpha_gini.csv` | `gini_all`, `gini_sn3` |
| `gini/gini_outputs/tables/continuum_gini.csv` | `gini_all`, `gini_sn3` |
| `gini/gini_outputs/tables/compare_gini.csv` | `gini_all_halpha`, `gini_all_cont`, `gini_ratio`, `gini_delta`, `log_gini_ratio` |
| `master/outputs/tables/master_results.csv` | `G` |
| `master/outputs/tables/master_sfms.csv` | `gini_all_halpha` |
| `master/outputs/tables/star_formation_ml.csv`, `star_formation_ml_02.csv` | `G` |

All of the above are superseded by `outputs/csv/gini.csv` and
`outputs/csv/master_<date>.csv` in this pipeline, which are the only
correct Gini values going forward. Any downstream figure or notebook cell
that used `G`/`gini_all`/`gini_all_halpha`/etc. needs to be re-run against
the new outputs.

### Gini - redundant S/N recut removed

The old `Halpha_gini.py`/`continuum_gini.py` computed two Gini values per
map: `gini_all` (finite & positive pixels only) and `gini_sn3` (the same,
plus a second S/N>=3 cut recomputed from the flux/error layers). That second
cut was redundant - the input FITS were already continuum-masked at S/N>=3
before either script ever saw them. `src/03_metrics/gini.py` now computes a
single `gini_halpha`/`gini_continuum` value per galaxy directly from the
pre-masked data.

### Concentration - now sourced consistently, one bug fixed

Concentration (`C`) continues to be sourced entirely from Wang's external
`SAMI_CIndex_data` catalogue, as confirmed in the audit - this pipeline does
not compute it from FITS. `src/03_metrics/concentration.py` carries over
`src/concentration_merging.py`'s merge logic unchanged, with one fix found
during porting (not part of the original audit): the old script's initial
column filter dropped `BCGCATAID` and `MassA_1` before they ever reached the
SF/Composite merge step, even though both columns exist in the raw
`gamaout_com_v0.12.txt`. The old `concentration_results.csv` is missing
these two columns as a result; the new `outputs/csv/concentration.csv`
recovers them from source (1003 galaxies now have real `BCGCATAID`/`MassA_1`
values instead of being silently absent).

### Standardized data sources - AR2022 dropped entirely

`04_build_master_csv.py` no longer reads
`AR2022_SAMI_DR3_Mstar_SFR_GSWL_rescaled.fits` at all (previously used as a
Mstar fallback and as one of the sources defining the master CATID
universe). Every galaxy's Mstar/SFR/spectral_class now comes from Wang's
catalogue only; the CATID universe is the union of Concentration, Gini,
Asymmetry, and Smoothness CATIDs. A galaxy with no Wang data gets `NaN`
Mstar rather than a silently-mixed-in AR2022 value. GAMA pairs/morphology
FITS are still used (confirmed as in-scope - they're environment metadata,
not an SFR/Mstar source).

Effect on row count: 3070 galaxies (old `master_results.csv`) -> 3031
(new `master_<date>.csv`). The 39-galaxy difference is entirely AR2022-only
galaxies with no Wang or masked-Hα data, which no longer appear.

### SFR resolution policy - unchanged, now the only path

Confirmed against your review: pure_SF galaxies use Wang's own SFR;
composite_SF galaxies use `SFR_n2ha_s` (the BPT-isolated star-forming
component); composite_SF galaxies with no valid `SFR_n2ha_s` get `NaN`
(never silently backfilled with the AGN/LINER-contaminated SF-file SFR).
This logic already existed in the old `build_csv.py` and is carried over
unchanged in `04_build_master_csv.py`.

### SFR/Mstar log-detection heuristic - centralized

The "if median > 100, treat as linear and take log10" heuristic used to be
independently duplicated in six places (`gini/scripts/galaxy_stats.py`,
`gini/scripts/representative_galaxies.py`,
`smoothness/scripts/representative_galaxies.py`,
`smoothness/scripts/smoothness_plotting.py`,
`asymmetry/scripts/asymmetry_plotting.py`, `master/scripts/build_csv.py`).
It now lives in `src/utils/units.py::to_log10`/`to_log_sfr`, used by
`04_build_master_csv.py`.

### Asymmetry, Smoothness - carried over unchanged (confirmed correct)

Per your review of AUDIT_REPORT.md section 3.4, no formula changes were
made:

- **Asymmetry** (`src/03_metrics/asymmetry.py`): same Conselice (2003)
  eq. 1 ratio, same grid-search centring, same error-subtraction-and-clip
  masking on top of the shared continuum mask. Verified bit-for-bit against
  the old `asymmetry_04.py` output for CATID 7139
  (A_geom=0.329444, A_min=0.338992, delta_A=-0.009548, exact match).
- **Smoothness** (`src/03_metrics/smoothness.py`): same "partial S" formula
  (no background-smoothness term, no central-pixel exclusion - both
  documented as open items in every old iteration and left as-is here), same
  hardcoded `PLATE_SCALE = 0.5` arcsec/px, same `KERNEL_SCALE = 0.35 * r_e`.
  Verified bit-for-bit against the old `smoothness_04.py` output for CATID
  23117 (S=0.666332, sigma_px=5.117, exact match).

### Masking - single shared implementation, Gini's redundant recut removed

`src/02_continuum_mask.py` builds the continuum S/N>=3 mask once per galaxy
and writes it into `data/masked/`; every metric in `src/03_metrics/` reads
from there rather than re-deriving its own footprint. Verified the masked
output is bit-identical to the old repo's `masked_halpha_maps/` for CATID
7139 (`np.allclose` on the full flux cube, max abs diff = 0.0). Per your
review: Asymmetry's error-subtract-and-clip and Smoothness's
error-subtract-without-clip are both left exactly as they were (audit
finding on masking-treatment differences noted, not changed).

### Dead code removed from the active pipeline

`src/other.py` and `visualiser/loader.py`, confirmed dead
(unreferenced/broken) in the audit, are not part of this pipeline. Both are
kept in `archive/` rather than deleted (see `archive/README.md`).

### Known gap - `5NN` / `is_isolated` / `5NN_err` not reproduced

The old `master_results.csv` contained `5NN`, `5NN_err`, and `is_isolated`
columns (5th-nearest-neighbour surface density / isolation flag), used by
`src/sf_ml_build.py`, `src/ml_data.py`, and the master notebooks. No script
anywhere in the old repo computes them - they were only ever read. One
notebook cell (`master/notebooks/environment.ipynb`) patches a corrupted
string value in this column (`'Falsea'` for CATID 7969), which suggests it
was hand-edited outside any script at some point.

Per instruction 4 ("do not touch any values by hand; all corrected outputs
must come from actually running the new scripts"), `04_build_master_csv.py`
does not fabricate or carry these columns forward - they are simply absent
from `outputs/csv/master_<date>.csv`. If there is a source for these values
(another script, a notebook cell not currently in the repo, an external
group-catalogue cross-match), it should become a genuine pipeline stage
rather than be reintroduced by hand.

### Not ported (out of scope for this rebuild)

The generic plotting scaffold (`master/scripts/03_plotting.py`) and the
ML-dataset builders (`src/sf_ml_build.py`, `src/ml_data.py`) were not part
of the requested 01-04 pipeline and have not been touched or moved. They
still point at the old `master/outputs/tables/master_results.csv` path and
column names (`gini_all_halpha`, `CIndex`, `A_min`, `in_gini`) and will need
updating to the new `outputs/csv/master_<date>.csv` schema (`G`, `C`, `A`,
`in_G`, ...) before they can be re-run.

## [2026-08-08] SFMS fitting rebuilt as 05_sfms.py

The old `master/scripts/sfms_prep.py` (which used `src/sfms.py`'s Lee et al.
2015 ridge-line fit to compute each galaxy's offset from the star-forming
main sequence, `delta_ms`) was flagged "not ported" in the previous entry.
It's now rebuilt as `src/05_sfms.py` (driver) + `src/utils/sfms.py` (fitting
logic, moved from the old `src/sfms.py`), reading the current
`outputs/csv/master_<date>.csv` instead of the old `master_results.csv`.

**Sample changed, so refit parameters aren't expected to match exactly:**
because AR2022 was dropped (see the AR2022 entry above), the galaxy sample
here (3031 galaxies, 2404 used to anchor the fit) differs from what the old
`sfms_prep.py` fit against (3070 galaxies, 1656 used to anchor the fit) -
this is a resample effect, not a code change.

**Bug found and fixed while porting - `curve_fit` parameter degeneracy:**
`lee2015_sfms(log_mstar, S0, a1, Mt, t=13.5)` fixes `t` at a single value
for every fit point (this repo only ever fits one cosmic epoch, never
varies `t` across galaxies). That makes `S0` and `a1` individually
non-identifiable: the fitted curve only depends on the combination
`S0 - a1*t`, so `curve_fit`, run unbounded, can trade `S0` and `a1` off
against each other freely without changing the fit at all. Confirmed this
is a **pre-existing issue**, not introduced by this port: re-running the
original unmodified `src/sfms.py` against the original `master_results.csv`
reproduces the same pathology (`S0=3261`, `a1=242` - nonsense values for a
form whose own initial guess is `p0=[2.9, 0.2, 9.9]`).

The fitted curve/residuals themselves were never wrong - `delta_ms` was
always computed from the actual fitted curve, not from `S0`/`a1`
individually, and the curve's residuals are identical with or without the
fix below. But the *reported* `S0`/`a1` values were meaningless (in the
thousands), which matters if they're ever quoted, plotted, or written into
a figure legend/thesis text - exactly the kind of thing this element exists
for.

**Fix:** added physically-motivated bounds to the `curve_fit` call in
`fit_ridge_line()` (`S0` in [0, 20], `a1` in [0, 2], `Mt` in [8, 12] -
matching the scale of the pre-existing initial guess). Verified this
changes nothing about the fitted curve or its residuals (checked
point-by-point against the unbounded fit) - it only selects the physically
sane member of the family of equivalent `(S0, a1)` pairs. After the fix,
fitting against the current `master_2026-08-03.csv`:

```
S0 = 2.8964
a1 = 0.2017
Mt = 9.9178
```

**Output:** `outputs/csv/master_sfms_<date>.csv` (master + `delta_ms`),
`outputs/csv/sfms_fit_params_<date>.csv` (fit provenance),
`outputs/csv/sfms_ridge_points_<date>.csv` (the ridge points used).
