# Repository Audit Report

**Scope:** diagnostic only — no files were edited or moved. This covers every `.py` and `.ipynb` file in the repo, the Gini/Concentration/Asymmetry/Smoothness formulas, unit handling, mask consistency, and which output CSVs are affected by the Gini formula issue.

---

## 1. Repository map by pipeline stage

### Data extraction / I/O
| File | Role |
|---|---|
| `src/io.py` | Reads raw SAMI Hα cubes (`read_sami_Halpha_layers`) and continuum maps (`read_sami_continuum_layers`) from FITS. |
| `src/signal_noise.py` | `compute_sn`, `sn_mask` — shared S/N utilities used by masking and by the Gini scripts. |
| `visualiser/loader.py` | Standalone FITS-file discovery/lookup utility (env-var driven paths, pattern matching for map types). **Not imported anywhere else in the repo** (`grep` for `visualiser` returns nothing) → unused / scratch. |

### Masking
| File | Role |
|---|---|
| `src/masking.py` | Core masking library: loads continuum flux+error, builds the continuum S/N≥3 mask (`make_continuum_mask`), applies it to 2D/3D arrays, saves masked FITS. |
| `gini/scripts/apply_continuum_mask.py` | Driver script. Builds the continuum mask per galaxy and applies the **same 2D mask** to both the continuum map and the Hα cube, writing `gini/gini_outputs/tables/masked_continuum_maps/` and `masked_halpha_maps/`. These masked FITS files are the shared input for Gini, Asymmetry, and Smoothness. |
| `asymmetry/scripts/masking_diagnostic.py` | One-off diagnostic (docstring: "run on a representative galaxy, then set aside") comparing three candidate masks for the asymmetry rotation. Not part of the production pipeline. |

### Gini calculation
| File | Role |
|---|---|
| `src/gini.py` (`gini_from_mask`) | **Production** Gini implementation. Used by `Halpha_gini.py` and `continuum_gini.py`, i.e. it is what actually produces the saved Gini CSVs. |
| `src/morphology.py` (`gini_from_map`) | Near-duplicate Gini implementation with a slightly different call signature (mask built internally from flux+error+S/N cut rather than passed in). Used only by `plot_continuum_gini.py`'s counterpart `continuum_flux_gini_map.py` for one single-galaxy diagnostic print/plot — **not** part of the CSV-building pipeline. |
| `gini/scripts/Halpha_gini.py`, `continuum_gini.py` | Loop over the masked FITS files, compute Gini (all-pixel and S/N≥3 variants), merge in SFR/Mstar, write `Halpha_gini.csv` / `continuum_gini.csv`. |
| `gini/scripts/compare_gini.py` | Merges the two CSVs above, computes `gini_ratio`, `gini_delta`, `log_gini_ratio`, writes `compare_gini.csv`. |
| `gini/scripts/Halpha_flux_gini_map.py`, `continuum_flux_gini_map.py` | Single-galaxy flux-map plotting/diagnostic scripts (Gini call is commented out in the Hα version). |
| `gini/scripts/galaxy_stats.py`, `representative_galaxies.py`, `plot_classifications.py`, `plot_Halpha_gini.py`, `plot_continuum_gini.py`, `SFR.py` | Lookup / plotting / catalogue-inspection utilities downstream of the CSVs above. |

### Concentration
| File | Role |
|---|---|
| `data/catalogs/SAMI_Cindex_data/calculate_r50_example_*.py` (4 files) | **External, third-party pipeline** (collaborator "Di"/"Wang" — hardcoded `/suphys/dwan9990/di` paths, Python-2-style code, `useful_functions` module not in this repo). Computes `CIndex = log10(r50,Hα / r50,continuum)` from curve-of-growth radii on **raw, non-continuum-masked** LZIFU maps, with its own elliptical aperture, BPT dust correction, and S/N handling. Produces the four `.txt` files in `data/catalogs/SAMI_Cindex_data/`. |
| `src/concentration_merging.py` | Loads the four Wang `.txt` files, resolves SF vs Composite conflicts, writes `concentration/outputs/tables/concentration_results.csv`. |
| `data/catalogs/SAMI_Cindex_data/plot_C_vs_M_errbarAGN.py` | Collaborator's own plotting script over the raw `.txt` outputs (not part of this repo's pipeline). |

**This confirms your note: Concentration is not computed anywhere in this repo's own FITS/masking pipeline — it is taken verbatim from the external Wang/CIndex catalogue**, which uses a different dataset version, aperture, and masking convention than Gini/Asymmetry/Smoothness.

### Asymmetry
| File | Role |
|---|---|
| `asymmetry/scripts/individual_asymmetry_01.py` | First iteration: A at geometric centre only, no centre search, no error subtraction. |
| `individual_asymmetry_02.py` | Adds 10×10 grid search over rotation centre (±2 px); still no error subtraction. |
| `individual_asymmetry_03.py` | Adds error subtraction (`I_clean = clip(flux − error, 0)`) before rotating; refines the "common footprint" mask logic. |
| `asymmetry_04.py` | **Production/batch script.** Imports functions directly from `individual_asymmetry_03.py` and iterates over the full masked-Hα sample, writing `asymmetry/outputs/tables/asymmetry_results.csv`/`asymmetry_test.csv`. |
| `asymmetry_plotting.py` | Plots A_min vs Mstar/sSFR/SFR/Gini, merging in classifications and Gini. |
| `src/asymmetry_merging.py` | Merges Wang classifications onto the asymmetry CSV (mirrors `gini_merging.py`). |

**Final version: `asymmetry_04.py`** (built directly on `individual_asymmetry_03.py`'s functions). 01→02→03 are documented, self-superseding iterations on a single test galaxy (CATID 7139); each docstring states what changed from the previous version.

### Smoothness
| File | Role |
|---|---|
| `smoothness_test_01.py` | First iteration ("clumpiness_start.py" per its own docstring): Gaussian kernel σ = 0.2×1.5×R_Petrosian (Conselice's own scale), no background subtraction, no central exclusion — explicitly labelled "partial S", "NOT the final Conselice S". |
| `smoothness_test_02.py` | Switches kernel scale to fractions of `r_e` (0.10/0.25/0.35/0.50 × r_e) instead of R_Petrosian; reads plate scale from `CDELT1`. Still no background subtraction. |
| `smoothness_test_03.py` | Adds error subtraction (`I_clean = I − err`, **no clipping** of negative values). Still "no central exclusion applied yet" per its own docstring. |
| `smoothness_04.py` | **Production/batch script**, explicitly "following the methodology established in `smoothness_test_03.py`". Fixes kernel scale at 0.35×r_e and hardcodes `PLATE_SCALE = 0.5` arcsec/px (rather than reading `CDELT1` per file, as `test_02`/`test_03` did). Writes `smoothness/outputs/tables/smoothness_results.csv`. |
| `kernel_resolution.py` | Diagnostic comparing kernel σ to the PSF σ across the sample — not part of the production chain. |
| `smoothness_plotting.py`, `representative_galaxies.py`, `plot_representative_galaxies.py`, `galaxy_stats.py` | Downstream plotting/lookup utilities. |

**Final version: `smoothness_04.py`.** Important: it still computes the same **"partial S"** as the test scripts — `S = Σ|I_clean − I_clean,smoothed| / Σ|I_clean|` — with no background-clumpiness term and no central-region exclusion. Every test script explicitly flagged these two omissions as open TODOs; none of them were resolved before `smoothness_04.py` was promoted to the batch/production script. See §3 below for why this matters against the Conselice definition.

### Catalogue merge / master build
| File | Role |
|---|---|
| `src/gini_merging.py` | Merges Wang classifications onto `Halpha_gini.csv`, `continuum_gini.csv`, `compare_gini.csv` (in place). **Working / final.** |
| `src/gini_merging_new.py` | **Broken.** Contains the Step-1 lookup logic from `gini_merging.py` duplicated, followed by a second "STEP 1: Load all four files" block copy-pasted from `concentration_merging.py` that calls `load_txt(path, label, spectral_class, environment)` — a 4-argument call against the 2-argument `load_txt` defined earlier in the *same file* — and references `SHARED_COLS`, `SF_EXTRA_COLS`, `COMPOSITE_CINDEX_COLS`, `CLUSTER_EXTRA_COLS`, `META_COLS`, `OUTPUT_CSV`, none of which are ever defined in this file. **Running it would raise `TypeError`/`NameError` immediately.** This is an abandoned, unfinished port of `concentration_merging.py`'s structure onto the Gini use case — not usable, should be treated as scratch/dead code (or deleted). |
| `src/concentration_merging.py` | Builds `concentration/outputs/tables/concentration_results.csv` from the Wang `.txt` files (clean, complete, final). |
| `src/asymmetry_merging.py` | Merges Wang classifications onto `asymmetry_results.csv` (mirrors `gini_merging.py`, working). |
| `master/scripts/build_csv.py` | **The** master-build script. Merges Wang TXT (SFR/Mstar/spectral_class anchor) + `concentration_results.csv` (C) + `compare_gini.csv` (G) + `asymmetry_results.csv` (A) + `smoothness_results.csv` (S) + AR2022 FITS (Mstar fallback) + GAMA morphology/pairs FITS → `master/outputs/tables/master_results.csv`. Very well documented (docstring explains SFR/Mstar resolution logic in detail). |
| `master/scripts/sfms_prep.py` | Fits the SFMS ridge line (`src/sfms.py`) on `master_results.csv` and writes `delta_ms` into `master/outputs/tables/master_sfms.csv`. **Stale relative to the current `build_csv.py`:** `master_sfms.csv`'s header still has `in_gini`, `gini_all_halpha`, `CIndex`, `A_min`, `in_wang_conc`, `in_asym`, `in_smooth` — an older `build_csv.py` schema — whereas the current `master_results.csv` uses `in_G/in_C/in_A/in_S`, `G/C/A/S`. `sfms_prep.py` needs to be re-run against the current `master_results.csv` before it's trustworthy. |
| `src/sf_ml_build.py` | Builds `star_formation_ml.csv` — pure_SF galaxies only, binary `gama_close_pair_50kpc`. |
| `src/ml_data.py` | Builds `star_formation_ml_02.csv` — **newer/more complete iteration**: includes both pure_SF and composite_SF, replaces the binary pair flags with a continuous `closest_companion_kpc`, adds `ellip`/`z`/`re`. This is the untracked new file in `git status`, i.e. the one currently being iterated on. |
| `master/scripts/03_plotting.py` | Generic plotting scaffold over `master_results.csv`. Its default `COLOUR_COL = "gini_all_halpha"` **does not exist** in the current `master_results.csv` schema (it's `G` now) — running it as-is would `KeyError`. Same staleness pattern as `sfms_prep.py`/`master_sfms.csv`. |
| `master/notebooks/environment.ipynb`, `exploration.ipynb`, `multi_space_exploration.ipynb` | All three read `master_results.csv` (current `G/C/A/S` schema) and only **plot** the metrics — no independent Gini/CAS calculation exists anywhere in the notebooks. |
| `src/z_plot_config.py`, `z_plotting.py`, `z_plotting_multi.py`, `z_samples.py` | Active plotting infrastructure imported by the notebooks (`plot_scatter`, `plot_corner`, etc.) — despite the `z_` prefix these are **not** scratch, they're the current notebook plotting backend. |
| `src/other.py` | One-off diagnostic; hardcoded path `master/master_results.csv` (doesn't exist — actual path is `master/outputs/tables/master_results.csv`) and references old column names. Dead/scratch. |

---

## 2. Duplicate / iteration scripts — which is "final"

| Group | Final version | Superseded versions | What differs |
|---|---|---|---|
| Gini core formula | `src/gini.py::gini_from_mask` (feeds the actual CSVs) | `src/morphology.py::gini_from_map` (same formula, only used for one diagnostic plot) | Mask passed in vs. built internally from S/N cut; **identical (incorrect) formula** in both. |
| Gini CSV merge | `src/gini_merging.py` | `src/gini_merging_new.py` | `_new` is broken/unfinished (see above) — not a valid alternative. |
| Asymmetry | `asymmetry_04.py` (batch, built on `_03`) | `individual_asymmetry_01/02/03.py` | 01: no centring/error handling. 02: + grid-search centring. 03: + error subtraction & refined common-footprint mask. 04 = 03's functions run over the full sample. |
| Smoothness | `smoothness_04.py` (batch, built on `_test_03`) | `smoothness_test_01/02/03.py` | 01: R_Petrosian-based kernel, no error subtraction. 02: r_e-based kernel, multiple scale factors, dynamic plate scale. 03: + error subtraction (unclipped). 04: fixes kernel scale at 0.35×r_e, **hardcodes** plate scale instead of reading it per-file. All four still compute the same "partial S" (no background term, no central exclusion). |
| ML dataset build | `src/ml_data.py` → `star_formation_ml_02.csv` | `src/sf_ml_build.py` → `star_formation_ml.csv` | `_02` adds composite_SF galaxies, continuous companion distance, structural columns; strictly a superset/refinement. |
| Master SFMS/plotting | needs re-run | `sfms_prep.py`→`master_sfms.csv`, `03_plotting.py` | Both reference the pre-refactor `master_results.csv` column names (`gini_all_halpha`, `CIndex`, `A_min`, `in_gini`) and will silently carry stale data or `KeyError` against the current `master_results.csv`. |

---

## 3. Formula verification

### 3.1 Gini coefficient — confirmed discrepancy

Every Gini implementation in the repo (`src/gini.py::gini_from_mask`, `src/morphology.py::gini_from_map`) computes:

```
x_sorted = sort(flux, ascending)
i = 1..n
G_repo = 2·Σ(i·x_i) / (n·Σx_i)  −  (n+1)/n
```

This is the standard **generic/statistics (population) Gini coefficient** — the textbook formula for an income-inequality-style Gini computed from ranked data, with an **`n`** in the denominator.

The formula you cited — the one actually used in galaxy-morphology CAS work (Abraham et al. 2003; Lotz et al. 2004) — is:

```
G_astro = [1 / (X̄ · n(n−1))] · Σ_{i=1}^{n} (2i − n − 1)·X_i
```

Expanding both algebraically (with X̄ = ΣX_i/n, so X̄·n(n−1) = (n−1)·ΣX_i):

```
G_astro = [2·Σ(i·X_i)/ΣX_i − (n+1)] / (n−1)
G_repo  = [2·Σ(i·X_i)/ΣX_i − (n+1)] / n
```

so:

```
G_astro = G_repo · n / (n−1)
```

**Confirmed: the repo uses the wrong denominator** (`n` instead of `n−1`). This is not a cosmetic difference — it is the difference between the general population-Gini estimator and the astronomy convention that Abraham/Lotz derived specifically so that G is computed consistently for a *finite, discretely-sampled pixel/spaxel* population rather than a continuous income distribution.

**Magnitude of the effect:** the correction factor `n/(n−1)` shrinks as `n` grows, but SAMI Hα maps after the S/N≥3 continuum mask often retain only tens to a few hundred valid spaxels per galaxy (the code enforces `min_pixels=10` as the floor). At `n=10` the true value is **11% higher** than what's stored; at `n=50` it's **~2% higher**; at `n=500` it's ~0.2% higher. Since `n` (valid pixel count) varies galaxy-to-galaxy (it's stored as `n_sn3_pixels` in the CSVs), **the size of the error itself varies systematically with galaxy size/S/N** — small, low-S/N, or heavily-masked galaxies are under-estimated by more than large, well-resolved ones. This means the error isn't just an additive bias — it will also distort any Gini vs. mass/size/environment trend, since fainter/smaller galaxies (which tend to have fewer valid pixels) are pulled down disproportionately.

### 3.2 Unit handling

- No explicit flux-unit conversion happens anywhere in the Gini/masking pipeline; SAMI Hα/continuum maps carry `10⁻¹⁶ erg/s/cm²` units implicitly through the FITS headers, but the header units are never read or asserted. This is largely harmless for Gini itself since the formula is scale-invariant (a ratio of a weighted sum to the total sum), so a global flux-unit choice doesn't bias G — only the **wrong denominator issue above** does.
- No per-galaxy normalisation (e.g. dividing by total flux or by aperture area) is applied before computing Gini in any script — again appropriate, since Gini doesn't need it.
- Where units *do* matter — SFR — handling is fragile: at least **six different call sites** (`gini/scripts/galaxy_stats.py`, `gini/scripts/representative_galaxies.py`, `smoothness/scripts/representative_galaxies.py`, `smoothness/scripts/smoothness_plotting.py`, `asymmetry/scripts/asymmetry_plotting.py`, `master/scripts/build_csv.py::to_log_sfr`) independently re-implement the same "if median > 100, assume linear and take log10" heuristic to decide whether a loaded SFR/Mstar column is already log-scaled. It's duplicated rather than centralised, so a future change to one copy won't propagate, and the heuristic itself is fragile (any subsample with a genuinely unusual value distribution could silently mis-classify).
- `smoothness_04.py` hardcodes `PLATE_SCALE = 0.5` arcsec/px for converting `r_e` to pixels, whereas `smoothness_test_02.py`/`test_03.py` read `CDELT1` from each galaxy's own FITS header (and convert deg→arcsec if needed). If any galaxy's actual plate scale differs from 0.5″/px, `smoothness_04.py`'s kernel width (and therefore S) would be silently wrong for that galaxy — this wasn't validated across the full sample, only asserted "confirmed from FITS header" for one test galaxy.

### 3.3 Continuum mask — generation and consistency across the four metrics

**Generation** (`src/masking.py::make_continuum_mask`): built once per galaxy from the **continuum** flux/error map — keep pixel if flux is finite & positive, error is finite & positive, and S/N ≥ 3 (`sn_cut=3.0`). `apply_continuum_mask.py` applies this **single 2D mask** to both the continuum map and (broadcast across both Hα slices) the Hα cube, and writes the masked FITS files that everything downstream reads from.

**Consistency across metrics:**

| Metric | Uses the shared continuum-masked FITS as input? | Additional masking/cleaning on top |
|---|---|---|
| Gini | ✅ (`masked_halpha_maps/`, `masked_continuum_maps/`) | Re-derives its own mask on top: `mask_all` (finite & positive) or `mask_sn3` (S/N≥3 recomputed from the already-masked flux/error) — effectively re-applying an S/N cut that's largely redundant with the upstream continuum mask. |
| Asymmetry | ✅ (`masked_halpha_maps/`) | Applies its **own** error-subtraction (`I_clean = clip(flux − error, 0)`) and its own footprint logic (`original_mask = img != 0`, `rotated_mask = rotated > 0` or a small threshold depending on version) — an intersection mask that is *not* the continuum S/N mask, built after further shrinking the footprint via clipping. |
| Smoothness | ✅ (`masked_halpha_maps/`) | Applies error-subtraction **without clipping** (`flux_clean = flux − err`, negative values retained — explicitly noted in `smoothness_test_03.py`: *"Negative pixels remain as-is — existing mask handles these"*). |
| **Concentration** | ❌ | Not derived from these FITS files at all — computed by the external Wang pipeline on raw LZIFU maps with its own elliptical aperture, dust correction, and S/N handling (see §1). |

**Findings:**
1. Concentration is the clear outlier — it bypasses the shared continuum mask entirely, as you suspected, so C is not on the same footing as G/A/S with respect to masking, aperture, or even the underlying SAMI data-release version.
2. Even among G/A/S, which do share the same upstream continuum-masked FITS, the *treatment of low-S/N pixels differs*: Gini re-applies a hard S/N cut, Asymmetry clips negative post-subtraction pixels to zero, Smoothness leaves negative post-subtraction pixels in place. This means the effective pixel population feeding each metric's sum is not identical even for the same galaxy — worth deciding whether that's intentional.

### 3.4 Concentration / Asymmetry / Smoothness vs. standard (Conselice) definitions

- **Asymmetry** — matches the core Conselice (2003) eq. 1 ratio: `A = Σ|I₀−I₁₈₀| / Σ|I₀|`, with a grid-search minimisation over rotation centre (methodologically consistent with the literature). **Omission:** the full Conselice definition subtracts a background-asymmetry term, `A = min(Σ|I₀−I₁₈₀|/Σ|I₀|) − A_bg`, estimated by rotating a blank sky region the same way. This repo's implementation has no `A_bg` term at all. Given the data is already continuum-masked/zeroed outside the galaxy footprint, this may be a deliberate and reasonable simplification, but it is a real deviation from the formal definition and isn't flagged as such anywhere in the docstrings (which cite "Conselice 2003, eq. 1" as if it's the complete definition).
- **Smoothness** — the docstrings are honest about this: every iteration (`test_01` → `test_03` → `smoothness_04.py`) explicitly labels its output "partial S" and lists the two missing pieces of the standard Conselice smoothness statistic: (1) a background-smoothness term `B_S` subtracted from the residual sum, and (2) exclusion of the central `~0.25×R_petrosian` region (the smoothed core always self-residuals strongly regardless of real clumpiness, so Conselice excludes it). **Neither omission was resolved before `smoothness_04.py` was promoted to the production/batch script** — it computes the same partial-S formula as the very first test script, just batched over the full sample and normalised to a different kernel definition (r_e-based rather than R_Petrosian-based, and without the customary `×10` scaling some CAS papers apply). So the `S` column in every output CSV is explicitly a partial/simplified clumpiness statistic, not the standard Conselice `S`.
- **Concentration** — as you flagged, this is **not calculated by any script in this repo**. `CIndex` (renamed `C` in the master CSV) is pulled directly from the external `SAMI_Cindex_data` catalogue, where it is defined as `log10(r50,Hα / r50,continuum)` — a Petrosian/curve-of-growth half-light-radius ratio between the Hα and continuum light profiles. This is a legitimate concentration index used in some SAMI Hα literature (Wang et al. 2022 convention), but it is **conceptually different** from the classic Conselice `C = 5·log₁₀(r80/r20)` (a single-band light-concentration measure), and it is computed on a different dataset/mask/aperture than G/A/S, as discussed in §3.3.

---

## 4. Output CSVs affected by the Gini formula issue

**Yes — every CSV in the repo that contains a Gini value is affected**, because they all trace back to `src/gini.py::gini_from_mask`, which is the sole formula used to build the base Gini tables:

```
Halpha_gini.csv  ──┐
continuum_gini.csv ─┴─► compare_gini.csv ─► master_results.csv ─► master_sfms.csv
                                                  │
                                                  ├─► star_formation_ml.csv
                                                  └─► star_formation_ml_02.csv
```

| File | Affected column(s) | How they're affected |
|---|---|---|
| `gini/gini_outputs/tables/Halpha_gini.csv` | `gini_all`, `gini_sn3` | Computed directly by `gini_from_mask`. |
| `gini/gini_outputs/tables/continuum_gini.csv` | `gini_all`, `gini_sn3` | Computed directly by `gini_from_mask`. |
| `gini/gini_outputs/tables/compare_gini.csv` | `gini_all_halpha`, `gini_all_cont`, and everything derived from them: `gini_ratio`, `gini_delta`, `log_gini_ratio` | Copied from the two CSVs above; the derived ratio/delta columns compound the error (a ratio of two independently-biased values does *not* cancel the bias, since the bias factor `n/(n−1)` differs between the Hα and continuum masks' respective valid-pixel counts). |
| `master/outputs/tables/master_results.csv` | `G` | Renamed straight from `compare_gini.csv::gini_all_halpha`. |
| `master/outputs/tables/master_sfms.csv` | `gini_all_halpha` | Carried through from an older `master_results.csv` build (also flagged above as schema-stale, but same underlying Gini values). |
| `master/outputs/tables/star_formation_ml.csv` | `G` | Carried through from `master_results.csv`. |
| `master/outputs/tables/star_formation_ml_02.csv` | `G` | Carried through from `master_results.csv`. |

**Rows affected:** all rows in all of the above (there is no code path anywhere that computes Gini with the correct `n(n−1)` astronomy formula — `src/morphology.py::gini_from_map`, the only other implementation in the repo, has the identical `n`-denominator bug). Every one of these tables would need Gini/`G`/`gini_all`/`gini_all_halpha`/`gini_all_cont`/`gini_ratio`/`gini_delta`/`log_gini_ratio` recomputed once the formula is fixed, and every figure built from them (2×2 classification panels, Gini-vs-A/S scatter plots, corner plots, ML targets) would need regenerating.

Any figure or notebook cell that colours by, filters on, or fits against Gini (`plot_classifications.py`, `plot_Halpha_gini.py`, `plot_continuum_gini.py`, `compare_gini.py`'s plots, `asymmetry_plotting.py`'s Gini-vs-A panel, `smoothness_plotting.py`'s Gini-vs-S panel, and the `master/notebooks/*.ipynb` cells using column `G`) is downstream of the same bug and will need re-running once the formula is corrected.

---

## Summary of findings, ranked by impact

1. **Gini formula uses population-Gini denominator `n` instead of the astronomy convention `n−1`** — affects every Gini value in every CSV in the repo (confirmed, both implementations share the bug).
2. **Smoothness's "final" batch script (`smoothness_04.py`) still computes an explicitly partial/incomplete `S`** (no background term, no central exclusion) — every test script flagged this as unresolved and it was never fixed before being promoted to production.
3. **Concentration bypasses the repo's own masking/data pipeline entirely**, sourced from an external, differently-masked/aperture'd catalogue — confirmed per your note.
4. **`src/gini_merging_new.py` is broken** (undefined names, mismatched function signature) — dead code, not a usable alternative to `gini_merging.py`.
5. **`master_sfms.csv` and `03_plotting.py` are stale** against the current `master_results.csv` column schema (`gini_all_halpha`/`CIndex`/`A_min`/`in_gini` vs. current `G`/`C`/`A`/`in_G`) — need re-running/updating.
6. **Masking treatment of low-S/N pixels differs between Gini, Asymmetry, and Smoothness** even though all three share the same upstream continuum-masked FITS input (hard S/N recut vs. clipped error-subtraction vs. unclipped error-subtraction).
7. **`smoothness_04.py` hardcodes plate scale (0.5″/px)** rather than reading it per-file as the test scripts did.
8. SFR/Mstar log-vs-linear detection heuristic (`median > 100`) is duplicated across ~6 files rather than centralised.
9. `visualiser/loader.py` and `src/other.py` appear to be unused/dead scratch code.
