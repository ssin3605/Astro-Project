#!/usr/bin/env python3
"""
CAS Asymmetry (A) of the Halpha emission-line flux map, following
Conselice (2003):

    A = max( min[ \Sigma|I_0 - I_\phi| / \Sigma|I_0| ]
             - median[ \Sigma|B_0 - B_\phi| / \Sigma|I_0| ] , 0 )

where I_0 is the masked Halpha flux map, I_\phi is I_0 rotated 180 deg about
the best-fit centre, B_0 is a noise image and B_\phi is B_0 rotated the same
way. The first term (A_raw) is minimised over rotation centre; the second
(A_bg) is the background asymmetry.

The background term differs from Conselice (2003), who measures it on a
blank-sky region: here it is the median over N_NOISE_DRAWS noise images, each
pixel drawn from Normal(0, that pixel's measured error), rotated about the
galaxy's best-fit centre and summed over the galaxy's own pixels.

Steps per galaxy:
    1. Load the masked Halpha flux and error maps (summed-flux slice).
    2. Grid-search 100 candidate rotation centres within +/-2 px of the
       image centre, and keep the centre that gives the lowest
       \Sigma|I_0 - I_\phi| / \Sigma|I_0|. This minimum is A_raw.
    3. Background term A_bg: make N_NOISE_DRAWS (50) noise images B_0,
       rotate each about the same best-fit centre, and compute
       \Sigma|B_0 - B_\phi| / \Sigma|I_0| over the same pixels used for the
       real galaxy. A_bg is the median over all draws.
    4. A = max(A_raw - A_bg, 0).

All sums are over the "common mask": pixels that are non-zero in the
original map and still above a small threshold after rotation.

Input : data/masked/halpha/<CATID>_<IFU>_Halpha_masked.fits
Output: outputs/ch3/csv/asymmetry.csv

    A_raw             first term, at the best-fit centre (no background term)
    A_bg              background asymmetry term
    A                 final asymmetry, max(A_raw - A_bg, 0)
    best_cy, best_cx  best-fit rotation centre (pixels)

Usage
    python sf_statistics/src/03_metrics/asymmetry.py
    python src/03_metrics/asymmetry.py --catid 7139   # single galaxy, prints only
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
from src.utils.paths import ASYMMETRY_CSV, MASKED_HALPHA_DIR  # noqa: E402

OUT_CSV = ASYMMETRY_CSV

HALPHA_LAYER = 1     # summed-flux slice
GRID_POINTS = 10     # 10 x 10 = 100 candidate centres
GRID_RANGE = 2.0     # search +/- this many pixels from the image centre
N_NOISE_DRAWS = 50   # number of noise images used for A_bg
NOISE_SEED = 44      # combined with CATID so each galaxy's noise is reproducible


def rotate_180(img: np.ndarray, cy: float, cx: float) -> np.ndarray:
    # Rotate img by 180 deg about (cy, cx): shift centre to middle, rot90 twice, shift back.
    array_centre = (img.shape[0] - 1) / 2.0
    dy = array_centre - cy
    dx = array_centre - cx
    shifted = shift(img, shift=(dy, dx), order=3, mode="constant", cval=0.0)
    rotated = np.rot90(shifted, 2)
    return shift(rotated, shift=(-dy, -dx), order=3, mode="constant", cval=0.0)


def _common_mask(img: np.ndarray, rotated: np.ndarray) -> np.ndarray:
    # Pixels that are non-zero in img and above 1e-3 * max(img) in the
    # rotated map. The threshold drops edge pixels that only pick up a small
    # smeared value from the sub-pixel shift.
    original_mask = img != 0
    if np.max(img) > 0:
        rotated_mask = rotated > 1e-3 * np.max(img)
    else:
        rotated_mask = rotated > 0
    return original_mask & rotated_mask


def compute_A(img: np.ndarray, cy: float, cx: float) -> tuple[float, np.ndarray, np.ndarray]:
    # \Sigma|I_0 - I_\phi| / \Sigma|I_0| over the common mask, rotating about (cy, cx).
    rotated = rotate_180(img, cy, cx)
    residual = img - rotated
    mask = _common_mask(img, rotated)

    denom = np.sum(np.abs(img[mask]))
    if denom == 0:
        return np.nan, rotated, residual
    return np.sum(np.abs(residual[mask])) / denom, rotated, residual


def grid_search_centre(img: np.ndarray, grid_points: int = GRID_POINTS,
                       grid_range: float = GRID_RANGE):
    # Return ((cy, cx), A_raw) for the candidate centre that gives the lowest A.
    # A_raw is NaN if A is undefined at every candidate centre.
    array_centre = (img.shape[0] - 1) / 2.0
    offsets = np.linspace(-grid_range, grid_range, grid_points)

    best_A = np.inf
    best_centre = (array_centre, array_centre)
    for dy in offsets:
        for dx in offsets:
            cy, cx = array_centre + dy, array_centre + dx
            A, _, _ = compute_A(img, cy, cx)
            if A < best_A:
                best_A = A
                best_centre = (cy, cx)
    if not np.isfinite(best_A):
        best_A = np.nan
    return best_centre, best_A


def background_asymmetry(error: np.ndarray, cy: float, cx: float, mask: np.ndarray,
                         denom: float, catid: int, n_draws: int = N_NOISE_DRAWS) -> float:
    # A_bg: median over n_draws of \Sigma|B_0 - B_\phi| / \Sigma|I_0|, where each
    # noise image B_0 is drawn fresh from Normal(0, error).
    # `mask` and `denom` are the real galaxy's common mask and \Sigma|I_0| at its
    # best-fit centre, so the noise is measured over exactly the same pixels.
    rng = np.random.default_rng((NOISE_SEED, catid))
    scale = np.where(error > 0, error, 0.0)

    numerators = []
    for _ in range(n_draws):  # one fresh noise image per pass
        noise = rng.normal(loc=0.0, scale=scale)
        residual = noise - rotate_180(noise, cy, cx)
        numerators.append(float(np.sum(np.abs(residual[mask]))))
    return float(np.median(numerators)) / denom


def process_one(halpha_path: Path) -> dict | None:
    """Compute all asymmetry columns for one galaxy. Returns None if the map is empty."""
    galaxy_id = halpha_path.name.replace("_Halpha_masked.fits", "")
    catid = int(galaxy_id.split("_")[0])

    flux_cube, err_cube = read_masked_halpha(str(halpha_path))
    img = np.nan_to_num(flux_cube[HALPHA_LAYER], nan=0.0, posinf=0.0, neginf=0.0)
    error = np.nan_to_num(err_cube[HALPHA_LAYER], nan=0.0, posinf=0.0, neginf=0.0)
    if not np.any(img != 0):
        return None

    # First term: A at the best-fit centre from the grid search
    (best_cy, best_cx), A_raw = grid_search_centre(img)

    # Common mask and \Sigma|I_0| at the best-fit centre
    rotated = rotate_180(img, best_cy, best_cx)
    mask = _common_mask(img, rotated)
    denom = float(np.sum(np.abs(img[mask])))

    # Background term and final A
    if denom:
        A_bg = background_asymmetry(error, best_cy, best_cx, mask, denom, catid)
        A = max(A_raw - A_bg, 0.0)
    else:
        A_bg = A = np.nan

    return {
        "CATID": catid,
        "galaxy_id": galaxy_id,
        "A_raw": round(A_raw, 6),
        "A_bg": round(A_bg, 6),
        "A": round(A, 6),
        "best_cy": round(best_cy, 3),
        "best_cx": round(best_cx, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catid", type=int, default=None,
                        help="Compute and print A for a single CATID (does not write the CSV).")
    args = parser.parse_args()

    if args.catid is not None:
        matches = sorted(MASKED_HALPHA_DIR.glob(f"{args.catid}_*_Halpha_masked.fits"))
        if not matches:
            raise FileNotFoundError(f"No masked Halpha FITS for CATID {args.catid}. "
                                    "Run 02_continuum_mask.py first.")
        result = process_one(matches[0])
        if result is None:
            print(f"CATID {args.catid}: empty Halpha map.")
            return
        print(f"CATID {args.catid}: A_raw={result['A_raw']:.4f}  A_bg={result['A_bg']:.4f}  "
              f"A={result['A']:.4f}")
        return

    halpha_files = sorted(MASKED_HALPHA_DIR.glob("*_Halpha_masked.fits"))
    if not halpha_files:
        raise FileNotFoundError(f"No masked Halpha FITS found in {MASKED_HALPHA_DIR}. "
                                "Run 02_continuum_mask.py first.")
    print(f"Found {len(halpha_files)} masked Halpha files.")

    rows = []
    n_empty = 0
    for halpha_path in tqdm(halpha_files, desc="Asymmetry"):
        result = process_one(halpha_path)
        if result is None:
            n_empty += 1
        else:
            rows.append(result)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("\n--- Summary ---")
    print(f"  Galaxies processed : {len(df)}")
    print(f"  Empty Halpha maps  : {n_empty}")
    print(f"  Wrote              : {OUT_CSV}")


if __name__ == "__main__":
    main()
