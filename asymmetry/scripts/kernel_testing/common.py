"""
kernel_testing/common.py
=========================
Shared utilities for the A_bg / background-asymmetry investigation
(asymmetry/scripts/kernel_testing/*), mirroring the equivalent module in
smoothness/scripts/kernel_testing/common.py that the S / B_S fix used.
See ../../../sami_sf_pipeline/ASYMMETRY_HANDOFF.md for the full plan.

Design choices made once here, applied everywhere below:

  - Cosmology: FlatLambdaCDM(H0=70, Om0=0.3), matching
    sami_sf_pipeline/src/04_build_master_csv.py::COSMO exactly.
  - HALPHA_LAYER, apply_error_subtraction, compute_A, grid_search_centre:
    pulled directly from the production module
    (sami_sf_pipeline/src/03_metrics/asymmetry.py) via
    import_production_asymmetry(), not re-typed here. That module's
    filename starts with a digit ("03_metrics") so it can't be imported
    with a normal `import` statement - it's loaded by file path with
    importlib instead, same pattern smoothness's kernel_testing/common.py
    uses for smoothness.py. This guarantees these tests exercise the
    actual production formulas rather than a hand-copied re-implementation
    that could silently drift.
  - Analysis sample: in_full_sam (Wang 2022's 6-cut purity sample - see
    [[wang_2022_full_sam_selection_criteria]]) is the default sample.
  - Resolution proxy: n_res = re_used / seeing (resolution elements per
    r_e), read directly from master_final.csv (already computed there).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM

# =============================================================================
# PATHS
# =============================================================================

THIS_DIR = Path(__file__).resolve().parent                       # .../kernel_testing
ASYMMETRY_DIR = THIS_DIR.parent.parent                            # .../asymmetry
REPO_ROOT = ASYMMETRY_DIR.parent                                  # .../sami copy
PIPELINE_ROOT = REPO_ROOT / "sami_sf_pipeline"                    # .../sami_sf_pipeline

MASKED_HALPHA_DIR = PIPELINE_ROOT / "data" / "masked" / "halpha"
MASTER_CSV = PIPELINE_ROOT / "outputs" / "csv" / "master_final.csv"
ASYMMETRY_CSV = PIPELINE_ROOT / "outputs" / "csv" / "asymmetry.csv"
PRODUCTION_ASYMMETRY_PY = PIPELINE_ROOT / "src" / "03_metrics" / "asymmetry.py"

OUTPUTS_DIR = THIS_DIR / "outputs"

# =============================================================================
# COSMOLOGY (matches sami_sf_pipeline/src/04_build_master_csv.py::COSMO)
# =============================================================================

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)


def kpc_per_arcsec(z):
    """Proper kpc per arcsecond at redshift z (scalar or array-like)."""
    z = np.asarray(z, dtype=float)
    return COSMO.kpc_proper_per_arcmin(z).value / 60.0


# =============================================================================
# PRODUCTION MODULE (loaded by path - see module docstring)
# =============================================================================

_production_module = None


def import_production_asymmetry():
    """
    Load sami_sf_pipeline/src/03_metrics/asymmetry.py as a module and
    return it, so these scripts call the *actual* production functions
    (apply_error_subtraction, compute_A, grid_search_centre, HALPHA_LAYER)
    rather than re-implementing them.

    Cached after first call.
    """
    global _production_module
    if _production_module is not None:
        return _production_module

    if not PRODUCTION_ASYMMETRY_PY.exists():
        raise FileNotFoundError(f"Production module not found: {PRODUCTION_ASYMMETRY_PY}")

    spec = importlib.util.spec_from_file_location(
        "production_asymmetry", PRODUCTION_ASYMMETRY_PY
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _production_module = mod
    return mod


# =============================================================================
# SAMPLE LOADING
# =============================================================================

def find_halpha_path(catid) -> Path | None:
    """Locate the masked Halpha FITS for a CATID (mirrors asymmetry.py's own lookup)."""
    matches = sorted(MASKED_HALPHA_DIR.glob(f"{catid}_*_Halpha_masked.fits"))
    return matches[0] if matches else None


def load_full_sample() -> pd.DataFrame:
    """
    master_final.csv left-merged with asymmetry.csv's own A_geom/A_min/
    delta_A/best_cy/best_cx (master_final only carries A_min renamed to
    'A' - the grid-search centre isn't in master_final.csv, but every
    script here needs it, so it's pulled back in from asymmetry.csv).

    Returns the full merged table; filter on in_full_sam yourself where
    the science sample (not the full catalogue) is wanted.
    """
    master = pd.read_csv(MASTER_CSV)
    asym = pd.read_csv(ASYMMETRY_CSV)

    master["CATID"] = master["CATID"].astype(int)
    asym["CATID"] = asym["CATID"].astype(int)

    keep = ["CATID", "A_geom", "A_min", "delta_A", "best_cy", "best_cx"]
    df = master.merge(asym[keep], on="CATID", how="inner")
    return df


def load_full_sam() -> pd.DataFrame:
    """load_full_sample() restricted to Wang 2022's in_full_sam purity cut."""
    df = load_full_sample()
    return df[df["in_full_sam"] == True].copy()  # noqa: E712


# =============================================================================
# SMALL HELPERS
# =============================================================================

def print_header(title: str):
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def ensure_outdir(name: str) -> Path:
    out = OUTPUTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    return out
