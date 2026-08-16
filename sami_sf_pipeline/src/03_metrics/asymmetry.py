#!/usr/bin/env python3
"""
03_metrics/asymmetry.py
=========================
CAS Asymmetry index (A) for the Hα emission-line flux map, following
Conselice (2003), eq. 1:

    A = sum(|I0 - I180|) / sum(|I0|)

AUDIT_REPORT.md section 3.4 confirmed this core ratio and the grid-search
minimisation over rotation centre are methodologically consistent with the
literature; the one documented deviation is that no background-asymmetry
term (A_bg) is subtracted, since the input is already continuum-masked and
zero-filled outside the galaxy footprint. Per your review of the audit, this
logic is carried over unchanged - only refactored for clarity - from
individual_asymmetry_03.py / asymmetry_04.py, including its own
error-subtraction and footprint-intersection masking on top of the shared
continuum mask (kept deliberately independent from Gini's and Smoothness's
masking, per audit finding #6).

Pipeline per galaxy:
    1. Load flux + error from the continuum-masked Halpha FITS (summed
       Gaussian slice, HDU0/HDU1 index 1)
    2. Error-subtract and clip: I_clean = clip(flux - error, 0)
    3. Rotate 180 deg about a candidate centre (shift -> rot90 -> shift back)
    4. A = sum(|I_clean - I_180|) / sum(|I_clean|), over the intersection of
       the original and rotated non-zero footprints
    5. Grid-search candidate centres within +/-2 px of the geometric centre
       to find A_min

Input : data/masked/halpha/
Output: outputs/csv/asymmetry.csv (CATID, A_geom, A_min, delta_A, best_cy, best_cx)

Usage
-----
    python src/03_metrics/asymmetry.py                # full sample
    python src/03_metrics/asymmetry.py --catid 7139    # single galaxy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import shift
from tqdm import tqdm

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_ROOT))

from src.utils.io import read_masked_halpha  # noqa: E402

MASKED_HALPHA_DIR = PIPELINE_ROOT / "data" / "masked" / "halpha"
OUT_CSV = PIPELINE_ROOT / "outputs" / "csv" / "asymmetry.csv"

HALPHA_LAYER = 1  # summed-flux slice, matching gini.py and smoothness.py
GRID_POINTS = 10  # 10x10 = 100 candidate centres
GRID_RANGE = 2.0  # search +/- this many pixels from the geometric centre


def apply_error_subtraction(flux: np.ndarray, error: np.ndarray) -> np.ndarray:
    """I_clean = clip(flux - error, 0). Down-weights low-S/N periphery pixels."""
    return np.clip(flux - error, 0, None)


def compute_A(img: np.ndarray, cy: float, cx: float) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Asymmetry A from rotating img 180 deg about the point (cy, cx).

    Shifts (cy, cx) onto the array midpoint, rotates exactly with np.rot90
    (no interpolation error), then shifts back. Uses the intersection of the
    original non-zero footprint and a thresholded rotated footprint
    (rotated > 1e-3 * max(img), falling back to rotated > 0 if img is all
    zero) so edge spaxels smeared by the sub-pixel shift don't spuriously
    inflate A - this threshold is what individual_asymmetry_03.py /
    asymmetry_04.py actually used in production, refined from an earlier
    `rotated > 0` cut in individual_asymmetry_02.py.
    """
    array_centre = (img.shape[0] - 1) / 2.0
    dy = array_centre - cy
    dx = array_centre - cx

    shifted = shift(img, shift=(dy, dx), order=3, mode="constant", cval=0.0)
    rotated = np.rot90(shifted, 2)
    rotated = shift(rotated, shift=(-dy, -dx), order=3, mode="constant", cval=0.0)

    original_mask = img != 0
    rotated_mask = rotated > 1e-3 * np.max(img) if np.max(img) > 0 else rotated > 0
    common_mask = original_mask & rotated_mask

    residual = img - rotated
    denom = np.sum(np.abs(img[common_mask]))

    if denom == 0:
        return np.nan, rotated, residual

    A = np.sum(np.abs(residual[common_mask])) / denom
    return A, rotated, residual


def grid_search_centre(img: np.ndarray, grid_points: int = GRID_POINTS, grid_range: float = GRID_RANGE):
    """Search a grid of candidate centres for the one that minimises A."""
    array_centre = (img.shape[0] - 1) / 2.0
    offsets = np.linspace(-grid_range, grid_range, grid_points)

    best_A = np.inf
    best_centre = (array_centre, array_centre)

    for dy in offsets:
        for dx in offsets:
            cy = array_centre + dy
            cx = array_centre + dx
            A, _, _ = compute_A(img, cy, cx)
            if A < best_A:
                best_A = A
                best_centre = (cy, cx)

    return best_centre, best_A


def galaxy_id_from_masked_halpha_name(filename: str) -> str:
    return filename.replace("_Halpha_masked.fits", "")


def process_one(halpha_path: Path) -> dict | None:
    galaxy_id = galaxy_id_from_masked_halpha_name(halpha_path.name)
    catid = int(galaxy_id.split("_")[0])

    flux_cube, err_cube = read_masked_halpha(str(halpha_path))
    flux = np.nan_to_num(flux_cube[HALPHA_LAYER], nan=0.0, posinf=0.0, neginf=0.0)
    error = np.nan_to_num(err_cube[HALPHA_LAYER], nan=0.0, posinf=0.0, neginf=0.0)

    img = apply_error_subtraction(flux, error)
    if np.sum(img != 0) == 0:
        return None

    array_centre = (img.shape[0] - 1) / 2.0
    A_geom, _, _ = compute_A(img, array_centre, array_centre)
    best_centre, A_min = grid_search_centre(img)

    return {
        "CATID": catid,
        "galaxy_id": galaxy_id,
        "A_geom": round(A_geom, 6),
        "A_min": round(A_min, 6),
        "delta_A": round(A_geom - A_min, 6),
        "best_cy": round(best_centre[0], 3),
        "best_cx": round(best_centre[1], 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catid", type=int, default=None, help="Compute and print A for a single CATID (does not write the CSV).")
    args = parser.parse_args()

    if args.catid is not None:
        matches = sorted(MASKED_HALPHA_DIR.glob(f"{args.catid}_*_Halpha_masked.fits"))
        if not matches:
            raise FileNotFoundError(f"No masked Halpha FITS for CATID {args.catid}. Run 02_continuum_mask.py first.")
        result = process_one(matches[0])
        if result is None:
            print(f"CATID {args.catid}: empty map after error subtraction.")
            return
        print(f"CATID {args.catid}: A_geom={result['A_geom']:.4f}  A_min={result['A_min']:.4f}  delta_A={result['delta_A']:.4f}")
        return

    halpha_files = sorted(MASKED_HALPHA_DIR.glob("*_Halpha_masked.fits"))
    if not halpha_files:
        raise FileNotFoundError(f"No masked Halpha FITS found in {MASKED_HALPHA_DIR}. Run 02_continuum_mask.py first.")

    print(f"Found {len(halpha_files)} masked Halpha files.")

    rows = []
    n_empty = 0
    for halpha_path in tqdm(halpha_files, desc="Asymmetry"):
        result = process_one(halpha_path)
        if result is None:
            n_empty += 1
            continue
        rows.append(result)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("\n--- Summary ---")
    print(f"  Galaxies processed          : {len(df)}")
    print(f"  Empty after error subtract  : {n_empty}")
    print(f"  Wrote                       : {OUT_CSV}")


if __name__ == "__main__":
    main()
