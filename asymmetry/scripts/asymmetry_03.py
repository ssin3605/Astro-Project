"""
Batch Hα Asymmetry Analysis — Test Run of 10 Galaxies
======================================================
Scales up individual_asymmetry_02.py to iterate through a test sample
before running through the full SAMI dataset.

Outputs a single CSV table of A_geom, A_min, and delta_A per galaxy.
For per-galaxy figures, use individual_asymmetry_02.py directly.
"""

## Note: I merge Wang's classifications over the .csv created in this script

import os
import numpy as np
import pandas as pd
from glob import glob

from individual_asymmetry_02 import (
    extract_flux_map,
    compute_A,
    grid_search_centre,
)

# ── Configuration ──────────────────────────────────────────────────────────────

MAPS_DIR = "gini/gini_outputs/tables/masked_halpha_maps"
OUTDIR   = "asymmetry/outputs/tables"
N_TEST   = None                                   # set to None to run full dataset

os.makedirs(OUTDIR, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_galaxy_id(fits_path):
    """Extract galaxy ID from filename e.g. '7139_A_HALPHA_masked.fits' → '7139'."""
    return os.path.basename(fits_path).split("_")[0]


def process_galaxy(fits_path):
    """
    Run the full asymmetry pipeline on a single galaxy.
    Returns a results dict, or None if the map is empty.
    """
    img = extract_flux_map(fits_path)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    if np.sum(img != 0) == 0:
        return None

    A_geom, _, _              = compute_A(img, cy=24.5, cx=24.5)
    best_centre, A_min, _, _  = grid_search_centre(img)
    delta_A                   = A_geom - A_min

    return {
        "catid" : get_galaxy_id(fits_path),
        "A_geom"    : round(A_geom,  6),
        "A_min"     : round(A_min,   6),
        "delta_A"   : round(delta_A, 6),
        "best_cy"   : round(best_centre[0], 3),
        "best_cx"   : round(best_centre[1], 3),
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
                print("SKIP — empty flux map")
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
    csv_path = os.path.join(OUTDIR, "asymmetry_results.csv")
    df.to_csv(csv_path, index=False)

    print(f"\nSummary table saved → {csv_path}")
    print(f"Processed: {len(results)}   Skipped: {len(test_files) - len(results) - len(failed)}   Failed: {len(failed)}")
    if failed:
        print(f"Failed galaxies: {failed}")


if __name__ == "__main__":
    main()