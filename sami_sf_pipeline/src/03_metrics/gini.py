#!/usr/bin/env python3
"""
03_metrics/gini.py
====================
Corrected Gini coefficient of Hα and continuum surface brightness.

AUDIT_REPORT.md (section 3.1) found that the old repo computed Gini using
the generic population-statistics formula (denominator n), not the
astronomy convention used in galaxy-morphology CAS work (Abraham 2003;
Lotz et al. 2004), which uses n(n-1). This module implements the Lotz 2004
form:

    G = 1 / (X_bar * n * (n-1)) * sum_{i=1}^{n} (2i - n - 1) * X_i

with X sorted ascending and X_bar the mean of the included pixel values.

Masking: this script does NOT re-derive a S/N mask. It reads directly from
data/masked/, which already has the shared continuum S/N>=3 mask baked in
(zero-filled outside the footprint - see 02_continuum_mask.py). The only
selection applied here is "finite and > 0", to drop the zero-fill and any
non-physical negative pixels. The old repo additionally recomputed a second,
redundant S/N>=3 cut on top of the already-masked data (the "gini_sn3"
column) - that redundant recut has been removed per audit finding #6; there
is now a single gini value per map.

Output: outputs/csv/gini.csv, one row per galaxy with:
    CATID, gini_halpha, n_pixels_halpha, gini_continuum, n_pixels_continuum,
    gini_ratio (halpha/continuum), gini_delta (halpha - continuum),
    log_gini_ratio

Usage
-----
    python src/03_metrics/gini.py                # full sample
    python src/03_metrics/gini.py --catid 7139    # single galaxy (prints only)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_ROOT))

from src.utils.io import read_masked_continuum, read_masked_halpha  # noqa: E402

MASKED_HALPHA_DIR = PIPELINE_ROOT / "data" / "masked" / "halpha"
MASKED_CONT_DIR = PIPELINE_ROOT / "data" / "masked" / "continuum"
OUT_CSV = PIPELINE_ROOT / "outputs" / "csv" / "gini.csv"

MIN_PIXELS = 10

# Halpha cube slice convention (see src/utils/io.py / 02_continuum_mask.py):
#   0 = Gaussian-fit flux, 1 = summed flux. Summed flux is used for Gini,
# matching the audited old pipeline's default (LAYER = 1 in Halpha_gini.py).
HALPHA_LAYER = 1


def gini_lotz2004(flux_values: np.ndarray, min_pixels: int = MIN_PIXELS) -> tuple[float, int]:
    """
    Gini coefficient of a set of (already-masked) flux values, using the
    Lotz et al. (2004) astronomy convention.

    Only finite, positive values are included. Returns (nan, n) if fewer
    than min_pixels values survive that cut.
    """
    x = np.asarray(flux_values, dtype=float).ravel()
    x = x[np.isfinite(x) & (x > 0)]
    n = x.size

    if n < min_pixels or n < 2:
        return np.nan, n

    x_sorted = np.sort(x)
    mean_x = np.mean(x_sorted)
    if mean_x <= 0:
        return np.nan, n

    i = np.arange(1, n + 1)
    g = np.sum((2 * i - n - 1) * x_sorted) / (mean_x * n * (n - 1))
    return float(g), n


def galaxy_id_from_masked_halpha_name(filename: str) -> str:
    """'7139_A_Halpha_masked.fits' -> '7139_A'"""
    return filename.replace("_Halpha_masked.fits", "")


def process_one(halpha_path: Path) -> dict | None:
    galaxy_id = galaxy_id_from_masked_halpha_name(halpha_path.name)
    catid = int(galaxy_id.split("_")[0])

    cont_path = MASKED_CONT_DIR / f"{galaxy_id}_continuum_masked.fits"
    if not cont_path.exists():
        return None

    flux_cube, _err_cube = read_masked_halpha(str(halpha_path))
    cont_flux, _cont_err = read_masked_continuum(str(cont_path))

    g_ha, n_ha = gini_lotz2004(flux_cube[HALPHA_LAYER])
    g_cont, n_cont = gini_lotz2004(cont_flux)

    return {
        "CATID": catid,
        "galaxy_id": galaxy_id,
        "gini_halpha": g_ha,
        "n_pixels_halpha": n_ha,
        "gini_continuum": g_cont,
        "n_pixels_continuum": n_cont,
    }


def add_comparison_columns(df: pd.DataFrame) -> pd.DataFrame:
    """gini_ratio, gini_delta, log_gini_ratio - mirrors the old compare_gini.py."""
    eps = 1e-6
    df["gini_delta"] = df["gini_halpha"] - df["gini_continuum"]
    df["gini_ratio"] = df["gini_halpha"] / (df["gini_continuum"] + eps)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["log_gini_ratio"] = np.log10(df["gini_ratio"])
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catid", type=int, default=None, help="Compute and print Gini for a single CATID (does not write the CSV).")
    args = parser.parse_args()

    if args.catid is not None:
        matches = sorted(MASKED_HALPHA_DIR.glob(f"{args.catid}_*_Halpha_masked.fits"))
        if not matches:
            raise FileNotFoundError(f"No masked Halpha FITS for CATID {args.catid}. Run 02_continuum_mask.py first.")
        result = process_one(matches[0])
        if result is None:
            print(f"CATID {args.catid}: no matching masked continuum file found.")
            return
        print(f"CATID {args.catid}:")
        print(f"  Gini (Hα)        = {result['gini_halpha']:.4f}  (n={result['n_pixels_halpha']})")
        print(f"  Gini (continuum) = {result['gini_continuum']:.4f}  (n={result['n_pixels_continuum']})")
        return

    halpha_files = sorted(MASKED_HALPHA_DIR.glob("*_Halpha_masked.fits"))
    if not halpha_files:
        raise FileNotFoundError(f"No masked Halpha FITS found in {MASKED_HALPHA_DIR}. Run 02_continuum_mask.py first.")

    print(f"Found {len(halpha_files)} masked Halpha files.")

    rows = []
    n_missing_cont = 0
    for halpha_path in tqdm(halpha_files, desc="Gini"):
        result = process_one(halpha_path)
        if result is None:
            n_missing_cont += 1
            continue
        rows.append(result)

    df = pd.DataFrame(rows)
    df = add_comparison_columns(df)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("\n--- Summary ---")
    print(f"  Galaxies processed      : {len(df)}")
    print(f"  Missing continuum match : {n_missing_cont}")
    print(f"  NaN Gini (Hα)           : {df['gini_halpha'].isna().sum()}")
    print(f"  NaN Gini (continuum)    : {df['gini_continuum'].isna().sum()}")
    print(f"  Wrote                   : {OUT_CSV}")


if __name__ == "__main__":
    main()
