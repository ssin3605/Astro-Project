# Archive

Everything in this folder is superseded, duplicate, or scratch code from
before the pipeline was rebuilt (see `../AUDIT_REPORT.md` and
`../CHANGELOG.md`). Nothing here is deleted outright - it's kept so the
history of how each metric evolved is still readable, and so nothing that
turns out to be needed later has to be dug out of git history. Files are
copies of what existed at the time this pipeline was rebuilt; the originals
also remain in the working repo one level up.

None of this is imported by anything in `../src/`. If you need to run
something from here, treat it as a historical reference implementation, not
a maintained tool - paths, column names, and formulas may not match the
current pipeline.

## gini/

Superseded by `../src/03_metrics/gini.py`, which fixes the Gini formula
(AUDIT_REPORT.md section 3.1) and drops the redundant S/N recut
(section 6 of the summary).

| File | Why archived |
|---|---|
| `apply_continuum_mask.py` | Superseded by `../src/02_continuum_mask.py` (same masking logic, ported). |
| `Halpha_gini.py`, `continuum_gini.py`, `compare_gini.py` | Superseded by `../src/03_metrics/gini.py`, which computes both maps' Gini and the comparison columns (ratio/delta) in one script with the corrected formula. |
| `Halpha_flux_gini_map.py`, `continuum_flux_gini_map.py` | One-off single-galaxy diagnostic plot scripts; `continuum_flux_gini_map.py` used the now-deleted duplicate `gini_from_map` (see `src/morphology.py` below). Not ported - `../src/01_flux_map.py` covers the "plot one galaxy's flux map" use case. |
| `galaxy_stats.py`, `representative_galaxies.py`, `plot_classifications.py`, `plot_Halpha_gini.py`, `plot_continuum_gini.py`, `SFR.py` | Lookup/plotting utilities downstream of the old Gini CSVs. Not part of the requested 01-04 pipeline scope; would need updating to read `outputs/csv/gini.csv`'s new (single-column) schema if revived. |

## asymmetry/

Superseded by `../src/03_metrics/asymmetry.py` (formula/methodology
unchanged - carried over as confirmed-correct per AUDIT_REPORT.md
section 3.4 - only refactored for clarity).

| File | Why archived |
|---|---|
| `individual_asymmetry_01.py`, `individual_asymmetry_02.py` | Early iterations (no centring, then grid-search centring added) superseded by `individual_asymmetry_03.py`'s approach, which is what's carried over. |
| `individual_asymmetry_03.py` | Its `compute_A`/`grid_search_centre` logic is what `../src/03_metrics/asymmetry.py` now implements - kept here as the original reference version. |
| `asymmetry_04.py` | The old batch/production script; superseded by `../src/03_metrics/asymmetry.py`. |
| `masking_diagnostic.py` | One-off diagnostic, explicitly marked in its own docstring as "run once then set aside." |
| `asymmetry_plotting.py` | Plotting utility downstream of the old asymmetry CSV. Not part of the requested pipeline scope. |

## smoothness/

Superseded by `../src/03_metrics/smoothness.py` (methodology unchanged,
including the two documented simplifications relative to full Conselice S -
no background term, no central exclusion - which were confirmed to stay
as-is; hardcoded plate scale also confirmed to stay as-is).

| File | Why archived |
|---|---|
| `smoothness_test_01.py`, `smoothness_test_02.py`, `smoothness_test_03.py` | Iterative development versions (R_Petrosian kernel -> r_e-based kernel -> error subtraction). `smoothness_test_03.py`'s approach is what's carried over. |
| `smoothness_04.py` | The old batch/production script; superseded by `../src/03_metrics/smoothness.py`. |
| `kernel_resolution.py` | Diagnostic comparing kernel sigma to PSF sigma across the sample. Not part of the requested pipeline scope. |
| `smoothness_plotting.py`, `representative_galaxies.py`, `plot_representative_galaxies.py`, `galaxy_stats.py` | Lookup/plotting utilities downstream of the old smoothness CSV. Not part of the requested pipeline scope. |

## master/

| File | Why archived |
|---|---|
| `build_csv.py` | Superseded by `../src/04_build_master_csv.py`, which drops the AR2022 fallback entirely (per your decision) and reads from the new metric CSVs. |
| `sfms_prep.py` | AUDIT_REPORT.md flagged this as stale against the current `master_results.csv` schema even before this rebuild (its output, `master_sfms.csv`, still has old column names like `in_gini`/`CIndex`/`A_min`). SFMS fitting was not part of the requested 01-04 scope, so it hasn't been ported or fixed - it would need rewriting against `outputs/csv/master_<date>.csv`'s schema to be usable again. |
| `03_plotting.py` | Also flagged stale in the audit (default `COLOUR_COL="gini_all_halpha"` doesn't exist in the current schema - would `KeyError`). Not part of the requested scope; not fixed. |

## src/

| File | Why archived |
|---|---|
| `gini.py` | The buggy Gini implementation (population-statistics formula instead of Lotz 2004). Superseded by `../src/03_metrics/gini.py`. |
| `morphology.py` | Duplicate of `gini.py`'s (buggy) Gini logic, used only by the now-archived `continuum_flux_gini_map.py`. Confirmed for deletion in your review of AUDIT_REPORT.md - archived rather than hard-deleted, per instruction 3. |
| `masking.py`, `io.py`, `signal_noise.py`, `plotting.py` | Logic carried over unchanged into `../src/utils/{masking,io,signal_noise,plotting}.py`. |
| `gini_merging.py` | The "merge Wang classifications onto an existing CSV after the fact" pattern is obsolete - the new pipeline builds classifications in from the start (see `../src/utils/wang.py`). |
| `gini_merging_new.py` | **Confirmed broken** (AUDIT_REPORT.md section 2): contains a duplicated, half-finished port of `concentration_merging.py`'s logic that calls `load_txt()` with a mismatched signature and references several undefined names (`SHARED_COLS`, `OUTPUT_CSV`, etc.). Running it raises an exception immediately. Never a usable alternative to `gini_merging.py`. |
| `concentration_merging.py` | Its Wang-file-merge logic was audited as correct and is what `../src/03_metrics/concentration.py` now implements (with one genuine bug fix - see CHANGELOG.md - that recovers `BCGCATAID`/`MassA_1`, which this version silently dropped). |
| `asymmetry_merging.py` | Same obsolete "merge after the fact" pattern as `gini_merging.py`. |
| `other.py` | Confirmed dead code: hardcoded a stale path (`master/master_results.csv`, which never existed - the real path was `master/outputs/tables/master_results.csv`) and referenced old column names. |

## visualiser/

| File | Why archived |
|---|---|
| `loader.py` | Confirmed dead code - not imported anywhere else in the old repo (verified with a repo-wide grep for "visualiser" during the audit). |

## external_wang_pipeline/

Not part of this repo's own pipeline at all - this is the collaborator
(Di/Wang)'s own code for computing `CIndex`, `SFR`, and `ha_r50`/`cont_r50`
from raw LZIFU maps on their own infrastructure (hardcoded paths to
`/suphys/dwan9990/di`, Python-2-style syntax, depends on an unincluded
`useful_functions` module). Kept purely for provenance - so it's clear where
the numbers in `data/catalogues/SAMI_Cindex_data/*.txt` actually came from -
not because it's runnable here.

| File | What it did upstream |
|---|---|
| `calculate_r50_example_cluster.py` | Computed CIndex/SFR/r50 for cluster SF galaxies -> `clusterout_sf_hbsn3_v0.12.txt`. |
| `calculate_r50_example_gama.py` | Same, for GAMA SF galaxies -> `gamaout_sf_hbsn3_v0.12.txt`. |
| `calculate_r50_example_clu_com_all.py` | Same, for cluster composite (SF+AGN/LINER) galaxies -> `cluout_com_v0.12.txt`. |
| `calculate_r50_example_gama_com_all.py` | Same, for GAMA composite galaxies -> `gamaout_com_v0.12.txt`. |
| `plot_C_vs_M_errbarAGN.py` | The collaborator's own C-vs-Mstar plotting script over these outputs. |
