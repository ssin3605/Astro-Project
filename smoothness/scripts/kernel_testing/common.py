"""
kernel_testing/common.py
=========================
Shared utilities for the r_e-in-kpc-vs-arcsec kernel investigation
(smoothness/scripts/kernel_testing/*). See README.md for the full plan and
the order the six scripts are meant to be run in.

Design choices made once here, applied everywhere below:

  - Cosmology: FlatLambdaCDM(H0=70, Om0=0.3), matching
    sami_sf_pipeline/src/04_build_master_csv.py::COSMO exactly, so kpc
    conversions here are consistent with the rest of the pipeline.
  - r_e, plate scale, kernel scale: pulled directly from the production
    module (sami_sf_pipeline/src/03_metrics/smoothness.py) via
    import_production_smoothness(), not re-typed here. That module's
    filename starts with a digit ("03_metrics") so it can't be imported
    with a normal `import` statement - it's loaded by file path with
    importlib instead. This guarantees these tests exercise the actual
    production formulas (subtract_error, compute_S, process_one) rather
    than a hand-copied re-implementation that could silently drift.
  - Analysis sample: in_full_sam (Wang 2022's 6-cut purity sample, 705
    galaxies per wang_2022_full_sam_selection_criteria) is used as the
    default sample for every script except 01_unit_invariance.py, which
    deliberately uses the full merged catalogue (~2259 galaxies) because
    the algebraic-identity claim it's testing should hold everywhere, not
    just on the science sample.
  - Resolution proxy: n_res = re_used / seeing (resolution elements per
    r_e). This is the reciprocal of Wang's own `seeing_re` column, which
    is already gatekept at < 0.75 (i.e. n_res > 1.33) in the in_full_sam
    cut - so n_res is not a new quantity invented for this investigation,
    it's the same ratio Wang's own quality cut already uses, just phrased
    the other way up.
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

THIS_DIR = Path(__file__).resolve().parent                      # .../kernel_testing
SMOOTHNESS_DIR = THIS_DIR.parent.parent                          # .../smoothness
REPO_ROOT = SMOOTHNESS_DIR.parent                                 # .../sami copy
PIPELINE_ROOT = REPO_ROOT / "sami_sf_pipeline"                    # .../sami_sf_pipeline

MASKED_HALPHA_DIR = PIPELINE_ROOT / "data" / "masked" / "halpha"
MASTER_CSV = PIPELINE_ROOT / "outputs" / "csv" / "master_final.csv"
SMOOTHNESS_CSV = PIPELINE_ROOT / "outputs" / "csv" / "smoothness.csv"
CONCENTRATION_CSV = PIPELINE_ROOT / "outputs" / "csv" / "concentration.csv"
PRODUCTION_SMOOTHNESS_PY = PIPELINE_ROOT / "src" / "03_metrics" / "smoothness.py"

OUTPUTS_DIR = THIS_DIR / "outputs"

# =============================================================================
# COSMOLOGY (matches sami_sf_pipeline/src/04_build_master_csv.py::COSMO)
# =============================================================================

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

# PSF FWHM -> sigma conversion
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def kpc_per_arcsec(z):
    """Proper kpc per arcsecond at redshift z (scalar or array-like)."""
    z = np.asarray(z, dtype=float)
    return COSMO.kpc_proper_per_arcmin(z).value / 60.0


# =============================================================================
# PRODUCTION MODULE (loaded by path - see module docstring)
# =============================================================================

_production_module = None


def import_production_smoothness():
    """
    Load sami_sf_pipeline/src/03_metrics/smoothness.py as a module and
    return it, so these scripts call the *actual* production functions
    (subtract_error, compute_S, process_one, load_re_lookup, KERNEL_SCALE,
    PLATE_SCALE, RE_MAX_ARCSEC) rather than re-implementing them.

    Cached after first call.
    """
    global _production_module
    if _production_module is not None:
        return _production_module

    if not PRODUCTION_SMOOTHNESS_PY.exists():
        raise FileNotFoundError(f"Production module not found: {PRODUCTION_SMOOTHNESS_PY}")

    spec = importlib.util.spec_from_file_location(
        "production_smoothness", PRODUCTION_SMOOTHNESS_PY
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _production_module = mod
    return mod


# =============================================================================
# SAMPLE LOADING
# =============================================================================

def find_halpha_path(catid) -> Path | None:
    """Locate the masked Halpha FITS for a CATID (mirrors smoothness.py's own lookup)."""
    matches = sorted(MASKED_HALPHA_DIR.glob(f"{catid}_*_Halpha_masked.fits"))
    return matches[0] if matches else None


def load_full_sample() -> pd.DataFrame:
    """
    master_final.csv left-merged with smoothness.csv's own r_e/sigma columns
    (re_used, re_source, sigma_arcsec, sigma_px - the values production
    actually used, not master_final's raw, un-resolved 're'), plus derived
    resolution-diagnostic columns:

        kpc_per_as        : kpc per arcsec at each galaxy's own z
        re_kpc             : re_used converted to kpc
        psf_sigma_arcsec    : seeing (FWHM) converted to a Gaussian sigma
        n_res               : re_used / seeing (resolution elements per r_e;
                               reciprocal of Wang's own seeing_re)
        sigma_over_psf      : sigma_arcsec / psf_sigma_arcsec (>1 means the
                               smoothing kernel is at least nominally above
                               the PSF floor)

    Returns the full merged table (2259 rows); filter on in_full_sam
    yourself where the science sample (not the full catalogue) is wanted.
    """
    master = pd.read_csv(MASTER_CSV)
    smooth = pd.read_csv(SMOOTHNESS_CSV)

    master["CATID"] = master["CATID"].astype(int)
    smooth["CATID"] = smooth["CATID"].astype(int)

    keep = ["CATID", "re_used", "re_source", "sigma_arcsec", "sigma_px", "kernel_scale"]
    df = master.merge(smooth[keep], on="CATID", how="inner")

    df["kpc_per_as"] = kpc_per_arcsec(df["z"].values)
    df["re_kpc"] = df["re_used"] * df["kpc_per_as"]
    df["psf_sigma_arcsec"] = df["seeing"] * FWHM_TO_SIGMA
    df["n_res"] = df["re_used"] / df["seeing"]
    df["sigma_over_psf"] = df["sigma_arcsec"] / df["psf_sigma_arcsec"]

    return df


def load_full_sam() -> pd.DataFrame:
    """load_full_sample() restricted to Wang 2022's in_full_sam purity cut."""
    df = load_full_sample()
    return df[df["in_full_sam"] == True].copy()  # noqa: E712


# =============================================================================
# CANDIDATE "DRIVER" COLUMNS (for 04_resolution_matched_validation.py and
# 05_kernel_scale_sensitivity.py's driver-correlation tables)
# =============================================================================

CANDIDATE_DRIVERS = {
    "log_Mstar": "Stellar mass (log M*)",
    "delta_ms": "SFMS offset (delta_ms)",
    "log_SFR": "log SFR",
    "log_sSFR": "log sSFR",
    "5NN": "5th-nearest-neighbour density",
    "Dfil": "Distance to filament (cosmic web)",
    "Dnode": "Distance to node (cosmic web)",
    "Dvoid": "Distance to void (cosmic web)",
    "ellip": "Ellipticity (inclination proxy)",
    "z": "Redshift (sanity/negative control - not a physical driver of S)",
}


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


def spearman_table(df: pd.DataFrame, target: str, candidates: dict, control: str | None = None):
    """
    Spearman rho (and, if `control` is given, the partial Spearman rho
    controlling for that column) between `target` and each candidate
    column, via rank-transform + partial Pearson on ranks.
    """
    from scipy.stats import spearmanr, rankdata

    rows = []
    for col, label in candidates.items():
        sub_cols = [target, col] + ([control] if control else [])
        sub = df[sub_cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < 10:
            rows.append({"column": col, "label": label, "n": len(sub), "rho": np.nan, "p": np.nan, "partial_rho": np.nan})
            continue

        rho, p = spearmanr(sub[target], sub[col])

        partial_rho = np.nan
        if control is not None:
            X = rankdata(sub[target])
            Y = rankdata(sub[col])
            Z = rankdata(sub[control])
            rxy = np.corrcoef(X, Y)[0, 1]
            rxz = np.corrcoef(X, Z)[0, 1]
            ryz = np.corrcoef(Y, Z)[0, 1]
            denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
            partial_rho = (rxy - rxz * ryz) / denom if denom > 0 else np.nan

        rows.append({
            "column": col, "label": label, "n": len(sub),
            "rho": round(rho, 4), "p": p, "partial_rho": round(partial_rho, 4) if control else np.nan,
        })

    return pd.DataFrame(rows)
