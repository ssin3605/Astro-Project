"""
Iterates through the full SAMI masked Hα FITS catalogue, applying the
error-subtracted asymmetry pipeline from asymmetry_03.py.

For per-galaxy figures use asymmetry_03.py directly.
Merge Wang's classifications over the output CSV using asymmetry_merging.py.
"""

## Note: I merge Wang's classifications over the .csv created in this script


import os
import numpy as np
import pandas as pd
from glob import glob

from asymmetry.scripts.individual_asymmetry_03 import (
    extract_flux_and_error,
    apply_error_subtraction,
    compute_A,
    grid_search_centre,
)

# ── Configuration ──────────────────────────────────────────────────────────────

MAPS_DIR = "gini/gini_outputs/tables/masked_halpha_maps"
OUTDIR   = "asymmetry/outputs/tables"
N_TEST   = None   # set to an integer (e.g. 10) for a test run, None for full dataset

os.makedirs(OUTDIR, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_galaxy_id(fits_path):
    """Extract galaxy ID from filename e.g. '7139_A_HALPHA_masked.fits' → '7139'."""
    return os.path.basename(fits_path).split("_")[0]


def process_galaxy(fits_path):
    """
    Run the full error-subtracted asymmetry pipeline on a single galaxy.
    Returns a results dict, or None if the cleaned map is empty.
    """
    # 1. Load flux and error maps
    flux, error = extract_flux_and_error(fits_path)
    flux  = np.nan_to_num(flux,  nan=0.0, posinf=0.0, neginf=0.0)
    error = np.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. Subtract error, clip negatives to zero
    img = apply_error_subtraction(flux, error)

    if np.sum(img != 0) == 0:
        return None

    # 3. Asymmetry at geometric centre
    A_geom, _, _             = compute_A(img, cy=24.5, cx=24.5)

    # 4. Grid search for minimum A
    best_centre, A_min, _, _ = grid_search_centre(img)
    delta_A                  = A_geom - A_min

    return {
        "catid"   : get_galaxy_id(fits_path),
        "A_geom"  : round(A_geom,  6),
        "A_min"   : round(A_min,   6),
        "delta_A" : round(delta_A, 6),
        "best_cy" : round(best_centre[0], 3),
        "best_cx" : round(best_centre[1], 3),
    }

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    all_files  = sorted(glob(os.path.join(MAPS_DIR, "*_Halpha_masked.fits")))

    if len(all_files) == 0:
        raise FileNotFoundError(f"No FITS files found in {MAPS_DIR}")

    test_files = all_files[:N_TEST] if N_TEST is not None else all_files
    print(f"Found {len(all_files)} galaxies — processing {len(test_files)}\n")

    results = []
    failed  = []

    for i, fits_path in enumerate(test_files, 1):
        galaxy_id = get_galaxy_id(fits_path)
        print(f"[{i}/{len(test_files)}] {galaxy_id}", end="  ")
        try:
            result = process_galaxy(fits_path)
            if result is None:
                print("SKIP — empty map after error subtraction")
            else:
                results.append(result)
                print(f"A_geom = {result['A_geom']:.4f}   "
                      f"A_min = {result['A_min']:.4f}   "
                      f"ΔA = {result['delta_A']:.4f}")
        except Exception as e:
            print(f"FAIL — {e}")
            failed.append(galaxy_id)

    # Save results
    df       = pd.DataFrame(results)
    csv_path = os.path.join(OUTDIR, "asymmetry_test.csv")
    df.to_csv(csv_path, index=False)

    print(f"\nSummary table saved → {csv_path}")
    print(f"Processed: {len(results)}   "
          f"Skipped: {len(test_files) - len(results) - len(failed)}   "
          f"Failed: {len(failed)}")
    if failed:
        print(f"Failed galaxies: {failed}")


if __name__ == "__main__":
    main()
