"""
Shared folder and file locations, so every script agrees on where inputs
and outputs live.

Repo layout:
    <repo>/
        data/                       all input data (DATA_DIR)
            fits/                   raw SAMI maps + GAMA / cosmic web FITS
            catalogs/SAMI_Cindex_data/   Wang et al. catalogue txt files
            masked/                 continuum-masked maps (made by 02)
            optical/                optical cutouts used in Ch. 2 figures
        sf_statistics/              this pipeline (PIPELINE_ROOT)
            src/
        outputs/                    all results (OUTPUTS_DIR)
            pipeline/               master tables shared across chapters
            ch2/, ch3/, ch4/        chapter-specific figures and tables
"""

from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PIPELINE_ROOT.parent

# Inputs
DATA_DIR = REPO_ROOT / "data"
FITS_DIR = DATA_DIR / "fits"
RAW_HALPHA_DIR = FITS_DIR / "sami_dr3_Halpha_maps"
RAW_CONT_DIR = FITS_DIR / "continuum_maps"

GAMA_MORPH_FITS = FITS_DIR / "gama_morphology.fits"
GAMA_PAIRS_FITS = FITS_DIR / "gama_pairs.fits"
SURFACE_DENSITY_FITS = FITS_DIR / "surface_density.fits"
COSMIC_WEB_FITS = FITS_DIR / "cosmic_web_metrics.fits"

WANG_CATALOGUE_DIR = DATA_DIR / "catalogs" / "SAMI_Cindex_data"

MASKED_DIR = DATA_DIR / "masked"
MASKED_HALPHA_DIR = MASKED_DIR / "halpha"
MASKED_CONT_DIR = MASKED_DIR / "continuum"

OPTICAL_DIR = DATA_DIR / "optical"

# Outputs
OUTPUTS_DIR = REPO_ROOT / "outputs"
PIPELINE_DIR = OUTPUTS_DIR / "pipeline"
CH3_CSV_DIR = OUTPUTS_DIR / "ch3" / "csv"

# Per-metric tables (03_metrics/), merged by 04_build_master_csv.py.
# concentration.csv is also read by clumpiness.py for r_e.
CONCENTRATION_CSV = CH3_CSV_DIR / "concentration.csv"
GINI_CSV = CH3_CSV_DIR / "gini.csv"
ASYMMETRY_CSV = CH3_CSV_DIR / "asymmetry.csv"
CLUMPINESS_CSV = CH3_CSV_DIR / "clumpiness.csv"

# Master table: 04 builds master.csv, 05 adds delta_ms (and writes the two
# SFMS fit files), 06 adds the cosmic web columns. The finished master.csv is
# the table every chapter and the ML analysis use.
MASTER_CSV = PIPELINE_DIR / "master.csv"
SFMS_FIT_PARAMS_CSV = PIPELINE_DIR / "sfms_fit_params.csv"
SFMS_RIDGE_POINTS_CSV = PIPELINE_DIR / "sfms_ridge_points.csv"
