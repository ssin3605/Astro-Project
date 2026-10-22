# SAMI Star-Forming Galaxy Hα Morphology Pipeline

Computes four non-parametric morphology metrics of the Hα emission - Gini (G),
Concentration (C), Asymmetry (A) and Clumpiness (S) - for SAMI DR3
star-forming and composite galaxies. It then merges them with Wang et al.
(2022)'s classification, SFR and stellar mass, the Full_sam quality cuts, a
star-forming main sequence (SFMS) fit, GAMA environment measures and
cosmic-web distances. The result is one table, `outputs/pipeline/master.csv`
(one row per galaxy), which Chapters 2-4 and the ML analysis in `ml/` all use.

## Repo layout

```
<repo>/
  data/                    inputs (not produced by this pipeline, except masked/)
    fits/                  SAMI DR3 Halpha + continuum maps, GAMA and cosmic-web FITS
    catalogs/SAMI_Cindex_data/   Wang et al. (2022) catalogue txt files
    masked/                continuum-masked Halpha + continuum maps (made by step 02)
    optical/               optical cutouts for the Ch. 2 figure
  sf_statistics/           this folder
    src/                   pipeline steps 01-06 + utils/
    notebooks/ch2, ch3/    thesis figures and sample table
    external_wang_pipeline/  Wang's original scripts (reference only)
  outputs/
    pipeline/              master.csv + SFMS fit files
    ch2/, ch3/, ch4/       chapter figures and tables
  ml/                      Chapter 4 ML analysis (reads master.csv)
```

All paths are set in one place, `src/utils/paths.py`.

## Running the pipeline

Run from the repo root:

```bash
python sf_statistics/src/02_continuum_mask.py          # shared continuum mask -> data/masked/
python sf_statistics/src/03_metrics/concentration.py   # must run before clumpiness (provides r_e)
python sf_statistics/src/03_metrics/gini.py
python sf_statistics/src/03_metrics/asymmetry.py
python sf_statistics/src/03_metrics/clumpiness.py
python sf_statistics/src/04_build_master_csv.py        # merge -> master.csv
python sf_statistics/src/05_sfms.py                    # adds delta_ms
python sf_statistics/src/06_add_cosmic_web.py          # adds Dfil, Dnode, Dvoid
```

```
02_continuum_mask ──► gini, asymmetry, clumpiness ──┐
concentration ──────► clumpiness (r_e)              ├──► 04 ──► 05 ──► 06 ──► master.csv
concentration ──────────────────────────────────────┘
```

`01_flux_map.py`, `02_continuum_mask.py`, `gini.py`, `asymmetry.py` and
`clumpiness.py` accept `--catid <CATID>` to run one galaxy and print the
result without writing the full-sample CSV. Re-running any step is safe:
04 rebuilds `master.csv` from scratch, and 05 and 06 replace their own columns.

## Outputs

| Step | Writes |
|---|---|
| `01_flux_map.py` | `outputs/ch2/figures/` (single-galaxy flux map) |
| `02_continuum_mask.py` | `data/masked/{halpha,continuum}/*_masked.fits` |
| `03_metrics/*.py` | `outputs/ch3/csv/{concentration,gini,asymmetry,clumpiness}.csv` |
| `04_build_master_csv.py` | `outputs/pipeline/master.csv` |
| `05_sfms.py` | adds `delta_ms` to `master.csv`; writes `sfms_fit_params.csv`, `sfms_ridge_points.csv` |
| `06_add_cosmic_web.py` | adds `Dfil`, `Dnode`, `Dvoid` to `master.csv` |

## Metrics

G, A and S are measured on the summed-flux Hα slice, after the shared
continuum mask (keep spaxels with finite, positive continuum flux and
continuum S/N ≥ 3). So all three use the same pixels.

- **G - Gini** (Lotz et al. 2004), over the masked Hα pixels (minimum 10 pixels).
- **C - Concentration** = log10(r50,Hα / r50,cont). Taken directly from Wang
  et al. (2022)'s catalogue, not computed here, and does not use the shared mask.
- **A - Asymmetry** (Conselice 2003). 180° rotation, minimised over a 10×10
  grid of centres within ±2 px. A background term (the median of 50 noise
  images drawn from each pixel's error) is subtracted, and A is floored at 0.
- **S - Clumpiness** (Conselice 2014). Positive residuals after Gaussian
  smoothing with σ = 0.35 r_e, minus a background term from 50 noise draws
  (the same method as A). S is not floored at 0.
- **delta_ms** - each galaxy's vertical offset in log SFR from the SFMS. The
  SFMS uses the Lee et al. (2015) form, fitted to the ridge line (the peak of
  the log SFR distribution in each mass bin, 9 ≤ log M* < 11; Tuntipong et al.
  2024).

## Sample selection

`master.csv` has 3031 galaxies, 2959 of them in Wang's catalogue (`in_C`).
`src/utils/wang_quality_cuts.py` applies Wang (2022)'s six Full_sam cuts. Each
cut gets one `cut_*` column, and `in_full_sam` is True when a galaxy passes all six:

1. size: r_e ≤ 15″
2. inclination: ellip ≤ 0.7
3. seeing: seeing/r_e < 0.75 and seeing < 4″
4. star-forming: EW(Hα) ≥ 1 Å
5. not pure AGN: `galclass != 4`
6. sSFR: log sSFR > −11.25

This leaves a **clean sample of 837 galaxies** (paper: 778). The difference
comes from a smaller starting catalogue (2959 vs 2988) and the sSFR step.
Metrics are computed for every galaxy with a usable map, not just the clean
sample, so filter on `in_full_sam` (and `in_G`/`in_C`/`in_A`/`in_S`)
yourself. Valid values in the clean sample: G 698, C 837, A 705, S 705.

## Master table columns

- **Flags:** `in_C`, `in_G`, `in_A`, `in_S`, `cut_*`, `in_full_sam`
- **Metrics:** `G`, `C`, `A`, `S`
- **Properties:** `spectral_class` (pure_SF / composite_SF), `environment`
  (GAMA / Cluster), `galclass`, `z`, `re`, `re_kpc`, `ellip`, `seeing`,
  `n_res`, `log_Mstar`, `log_SFR`, `log_sSFR`, `sfr_flag`, `delta_ms`
- **Environment:** `group_status` (central / satellite / isolated; GAMA only,
  from Wang's `BCGCATAID`), `5NN`, `gama_pair_sep_kpc`, `gama_hubble_type`,
  `gama_final_class`
- **Cosmic web:** `Dfil`, `Dnode`, `Dvoid` (NaN where there is no match)

Composite galaxies use Wang's `SFR_n2ha_s` as their SFR; pure-SF galaxies use
Wang's `SFR`.

## Notebooks (thesis figures)

- `notebooks/ch2/sample_summary.py` - sample table: raw maps → Wang
  classification → the six cuts → clean sample, plus metric and environment coverage.
- `notebooks/ch2/thesis_galaxies.py` - optical + Hα comparison of three
  galaxies above, on and below the SFMS.
- `notebooks/ch2/masking_visual.ipynb` - continuum S/N, raw Hα and masked Hα for one galaxy.
- `notebooks/ch2/flux_map_variable_scale.ipynb` - Hα flux maps, each panel on its own colour scale.
- `notebooks/ch3/chapter_3_plots.ipynb` - Chapter 3 Figures 1-4 (SFMS, GCAS
  vs global properties, GCAS vs environment, sSFR-M* plane coloured by metric).

Figures are saved to `outputs/ch2/figures/` and `outputs/ch3/figures/`.

## external_wang_pipeline/

Wang et al.'s original scripts for computing r50 / CIndex (GAMA and Cluster,
SF and composite) and plotting C vs M*. They are kept to document where C
comes from. They are **not runnable here** because their paths point to
Wang's own machines.

## Requirements

Python 3.11. Core: `astropy`, `numpy`, `pandas`, `scipy`, `matplotlib`, `tqdm`.
The notebooks also need `Pillow` and `statsmodels`.
