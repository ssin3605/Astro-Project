#!/usr/bin/env python3
"""
03_metrics/smoothness.py
==========================
Clumpiness (S) parameter for the Hα emission-line flux map, following the
methodology carried over unchanged from smoothness_04.py (audited, and per
your review of AUDIT_REPORT.md this is left as-is - including the two
documented simplifications relative to the full Conselice (2003) S
statistic: no background-smoothness term (B_S) is subtracted, and the
central pixel region is not excluded before summing the residual. This
script computes the same "partial S" as the old pipeline:

    S = sum(|I_clean - I_clean_smoothed|) / sum(|I_clean|)

Pipeline per galaxy:
    1. Load flux + error from the continuum-masked Halpha FITS (summed
       Gaussian slice, HDU0/HDU1 index 1)
    2. Error-subtract, NOT clipped: I_clean = flux - error (negative values
       are retained - the shared continuum mask already zero-fills outside
       the galaxy footprint, so this is left exactly as smoothness_04.py had
       it; see AUDIT_REPORT.md section 3.3 for why this differs from
       asymmetry.py's clipped subtraction, and why that's being left alone)
    3. Smooth I_clean with a Gaussian kernel: sigma = KERNEL_SCALE * r_e (px)
    4. residual = |I_clean - I_clean_smoothed|
    5. S = sum(residual) / sum(|I_clean|)

Kernel scale and plate scale are unchanged from smoothness_04.py:
    KERNEL_SCALE = 0.35 (fraction of r_e)
    PLATE_SCALE  = 0.5 arcsec/px (hardcoded - left as-is per your instruction,
                   rather than reading CDELT1 per-file as the smoothness_test_02/03
                   iterations did)

r_e is read from the Concentration table (03_metrics/concentration.py must
be run first), with the same SF-preferred / composite-fallback logic and the
same >50" unphysical-r_e exclusion as smoothness_04.py.

Input : data/masked/halpha/, outputs/csv/concentration.csv
Output: outputs/csv/smoothness.csv (CATID, S, kernel_scale, sigma_arcsec,
        sigma_px, re_used, re_source)

Usage
-----
    python src/03_metrics/smoothness.py                # full sample
    python src/03_metrics/smoothness.py --catid 23117   # single galaxy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_ROOT))

from src.utils.io import read_masked_halpha  # noqa: E402

MASKED_HALPHA_DIR = PIPELINE_ROOT / "data" / "masked" / "halpha"
CONCENTRATION_CSV = PIPELINE_ROOT / "outputs" / "csv" / "concentration.csv"
OUT_CSV = PIPELINE_ROOT / "outputs" / "csv" / "smoothness.csv"

HALPHA_LAYER = 1  # summed-flux slice, matching gini.py and asymmetry.py

KERNEL_SCALE = 0.35   # fraction of r_e used as the Gaussian sigma
PLATE_SCALE = 0.5     # arcsec/px - hardcoded, left as-is (see module docstring)
RE_MAX_ARCSEC = 50.0  # excludes unphysical r_e (failed Sersic fits)


def load_re_lookup(concentration_csv: Path) -> pd.DataFrame:
    """
    Build the CATID -> r_e (arcsec) lookup, SF value preferred, composite
    ('re_com') used as fallback. Excludes unphysical r_e > 50".
    Mirrors smoothness_04.py::load_catalogue exactly.
    """
    df = pd.read_csv(concentration_csv)
    df["CATID"] = df["CATID"].astype(str)

    df["re_used"] = df["re"]
    df["re_source"] = "re_sf"
    fallback_mask = df["re_used"].isna() & df["re_com"].notna()
    df.loc[fallback_mask, "re_used"] = df.loc[fallback_mask, "re_com"]
    df.loc[fallback_mask, "re_source"] = "re_com"

    n_total = len(df)
    bad_re = df["re_used"] > RE_MAX_ARCSEC
    if bad_re.sum() > 0:
        print(f"  Excluding {bad_re.sum()} galaxies with unphysical r_e > {RE_MAX_ARCSEC}\"")
    df = df[~bad_re]

    print(f"  r_e lookup: {len(df)} / {n_total} galaxies with valid r_e")
    return df.set_index("CATID")


def subtract_error(flux: np.ndarray, err: np.ndarray) -> np.ndarray:
    """I_clean = flux - error. Negative values retained (not clipped)."""
    return flux - err


def compute_S(flux_clean: np.ndarray, residual: np.ndarray) -> float:
    """S = sum(|I_clean - I_clean_smoothed|) / sum(|I_clean|)."""
    denom = np.nansum(np.abs(flux_clean))
    if denom == 0:
        return np.nan
    return float(np.nansum(residual) / denom)


def galaxy_id_from_masked_halpha_name(filename: str) -> str:
    return filename.replace("_Halpha_masked.fits", "")


def process_one(halpha_path: Path, re_arcsec: float, kernel_scale: float, plate_scale: float) -> dict:
    flux_cube, err_cube = read_masked_halpha(str(halpha_path))
    flux = flux_cube[HALPHA_LAYER]
    err = err_cube[HALPHA_LAYER]

    r_e_px = re_arcsec / plate_scale
    sigma_px = kernel_scale * r_e_px

    flux_clean = subtract_error(flux, err)
    flux_smoothed = gaussian_filter(flux_clean, sigma=sigma_px)
    residual = np.abs(flux_clean - flux_smoothed)
    S = compute_S(flux_clean, residual)

    return {"S": S, "sigma_arcsec": kernel_scale * re_arcsec, "sigma_px": round(sigma_px, 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catid", type=int, default=None, help="Compute and print S for a single CATID (does not write the CSV).")
    args = parser.parse_args()

    if not CONCENTRATION_CSV.exists():
        raise FileNotFoundError(f"{CONCENTRATION_CSV} not found - run 03_metrics/concentration.py first.")

    re_lookup = load_re_lookup(CONCENTRATION_CSV)

    if args.catid is not None:
        catid_str = str(args.catid)
        if catid_str not in re_lookup.index:
            print(f"CATID {args.catid}: no valid r_e in concentration.csv.")
            return
        matches = sorted(MASKED_HALPHA_DIR.glob(f"{args.catid}_*_Halpha_masked.fits"))
        if not matches:
            raise FileNotFoundError(f"No masked Halpha FITS for CATID {args.catid}. Run 02_continuum_mask.py first.")
        row = re_lookup.loc[catid_str]
        result = process_one(matches[0], row["re_used"], KERNEL_SCALE, PLATE_SCALE)
        print(f"CATID {args.catid}: S={result['S']:.4f}  sigma={result['sigma_px']:.2f}px  (r_e source: {row['re_source']})")
        return

    halpha_files = sorted(MASKED_HALPHA_DIR.glob("*_Halpha_masked.fits"))
    if not halpha_files:
        raise FileNotFoundError(f"No masked Halpha FITS found in {MASKED_HALPHA_DIR}. Run 02_continuum_mask.py first.")

    print(f"Found {len(halpha_files)} masked Halpha files.")

    rows = []
    failures = {"no_catid_in_re_lookup": 0, "nan_re": 0, "nan_S": 0}

    for halpha_path in tqdm(halpha_files, desc="Smoothness"):
        galaxy_id = galaxy_id_from_masked_halpha_name(halpha_path.name)
        catid_str = galaxy_id.split("_")[0]

        if catid_str not in re_lookup.index:
            failures["no_catid_in_re_lookup"] += 1
            continue

        re_row = re_lookup.loc[catid_str]
        if isinstance(re_row, pd.DataFrame):  # duplicate CATID guard
            re_row = re_row.iloc[0]

        if pd.isna(re_row["re_used"]):
            failures["nan_re"] += 1
            continue

        result = process_one(halpha_path, re_row["re_used"], KERNEL_SCALE, PLATE_SCALE)
        if np.isnan(result["S"]):
            failures["nan_S"] += 1
            continue

        rows.append({
            "CATID": int(catid_str),
            "galaxy_id": galaxy_id,
            "S": round(result["S"], 6),
            "kernel_scale": KERNEL_SCALE,
            "sigma_arcsec": round(result["sigma_arcsec"], 4),
            "sigma_px": result["sigma_px"],
            "re_used": re_row["re_used"],
            "re_source": re_row["re_source"],
        })

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("\n--- Summary ---")
    print(f"  Galaxies processed         : {len(df)}")
    print(f"  No CATID in concentration  : {failures['no_catid_in_re_lookup']}")
    print(f"  NaN r_e                    : {failures['nan_re']}")
    print(f"  NaN S (zero flux)          : {failures['nan_S']}")
    print(f"  Wrote                      : {OUT_CSV}")


if __name__ == "__main__":
    main()
