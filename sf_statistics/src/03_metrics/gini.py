#!/usr/bin/env python3
"""
Gini coefficient of masked Hα maps.

    G = 1 / (X_bar * n * (n-1)) * sum_{i=1}^{n} (2i - n - 1) * X_i

with X sorted ascending and X_bar the mean of the included pixel values.

Output: outputs/ch3/csv/gini.csv

Usage
    python sf_statistics/src/03_metrics/gini.py
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

from src.utils.io import read_masked_halpha  
from src.utils.paths import GINI_CSV, MASKED_HALPHA_DIR 

OUT_CSV = GINI_CSV

MIN_PIXELS = 10

# Halpha cube slices: 0 = Gaussian-fit flux, 1 = summed flux.
# Gini uses the summed flux.
HALPHA_LAYER = 1

def gini_lotz2004(flux_values: np.ndarray, min_pixels: int = MIN_PIXELS) -> tuple[float, int]:
    # Gini coefficient of a set of (already-masked) flux values, using the
    # Lotz et al. (2004) astronomy convention.
    # Only finite, positive values are included. Returns (nan, n) if fewer
    # than min_pixels values survive that cut.

    x = np.asarray(flux_values, dtype=float).ravel()
    x = x[np.isfinite(x) & (x > 0)]
    n = x.size

    if n < min_pixels:
        return np.nan, n

    x_sorted = np.sort(x)
    mean_x = np.mean(x_sorted)

    i = np.arange(1, n + 1)
    g = np.sum((2 * i - n - 1) * x_sorted) / (mean_x * n * (n - 1))
    return float(g), n

def galaxy_id_from_masked_halpha_name(filename: str) -> str:
    return filename.replace("_Halpha_masked.fits", "")

def process_one(halpha_path: Path) -> dict:
    galaxy_id = galaxy_id_from_masked_halpha_name(halpha_path.name)
    catid = int(galaxy_id.split("_")[0])

    flux_cube, _err_cube = read_masked_halpha(str(halpha_path))
    g_ha, n_ha = gini_lotz2004(flux_cube[HALPHA_LAYER])

    return {
        "CATID": catid,
        "galaxy_id": galaxy_id,
        "gini_halpha": g_ha,
        "n_pixels_halpha": n_ha,
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catid", type=int, default=None, help="Compute and print Gini for a single CATID (does not write the CSV).")
    args = parser.parse_args()

    if args.catid is not None:
        matches = sorted(MASKED_HALPHA_DIR.glob(f"{args.catid}_*_Halpha_masked.fits"))
        if not matches:
            raise FileNotFoundError(f"No masked Halpha FITS for CATID {args.catid}. Run 02_continuum_mask.py first.")
        result = process_one(matches[0])
        print(f"CATID {args.catid}:")
        print(f"  Gini (Hα) = {result['gini_halpha']:.4f}  (n={result['n_pixels_halpha']})")
        return

    halpha_files = sorted(MASKED_HALPHA_DIR.glob("*_Halpha_masked.fits"))
    if not halpha_files:
        raise FileNotFoundError(f"No masked Halpha FITS found in {MASKED_HALPHA_DIR}. Run 02_continuum_mask.py first.")

    print(f"Found {len(halpha_files)} masked Halpha files.")

    rows = [process_one(halpha_path) for halpha_path in tqdm(halpha_files, desc="Gini")]
    df = pd.DataFrame(rows)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("\n--- Summary ---")
    print(f"  Galaxies processed : {len(df)}")
    print(f"  NaN Gini (Hα)      : {df['gini_halpha'].isna().sum()}")
    print(f"  Wrote              : {OUT_CSV}")


if __name__ == "__main__":
    main()
