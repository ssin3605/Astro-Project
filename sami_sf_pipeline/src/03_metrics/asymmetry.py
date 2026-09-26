#!/usr/bin/env python3
"""
03_metrics/asymmetry.py
=========================
CAS Asymmetry index (A) for the Hα emission-line flux map, following
Conselice (2003), eq. 1:

    A = min_centre( sum(|I0 - I180|) / sum(|I0|) )

with a grid-search over rotation centre (100 candidates) to find the
minimum. AUDIT_REPORT.md section 3.4 confirmed this core ratio and the
grid-search minimisation are methodologically consistent with the
literature, but flagged one documented deviation: no background-asymmetry
term (A_bg) was subtracted, unlike Conselice's full definition. This is the
same class of gap Smoothness (S) had before it was fixed via B_S (see
smoothness.py's docstring and compute_B_S) - the fix here follows the same
recipe (each galaxy's own per-pixel error map, no blank-sky region needed),
adapted for A's own rotate-and-differance formula instead of smoothing.
Three columns are now output instead of one:

    A_no_sub = today's A_min, unchanged (kept for comparison/backward
               compatibility).
    A_BS     = median, over N_DRAWS_A_BS white-noise realisations drawn
               from the galaxy's own error map (noise_i ~ Normal(0, err),
               clipped via apply_error_subtraction - same treatment as the
               real I_clean), of the rotate-and-differance numerator
               (sum|noise_clean - rotate180(noise_clean)|) evaluated over
               the REAL galaxy's own common-mask footprint at its own
               best-fit centre (found on the real signal in step 5 below -
               no separate grid search on noise, mirrors B_S reusing the
               real sigma_px rather than re-deriving one from noise). Uses
               the REAL footprint rather than a mask freshly derived from
               each noise draw's own sparse post-clip pixels - validated in
               ../../../asymmetry/scripts/kernel_testing/03_background_subtraction_test.py
               to avoid a ~3x-too-large small-number-statistics artifact the
               naive version produces, and to sidestep degenerate all-zero-
               draw cases entirely (the numerator is always a well-defined,
               non-negative number over this fixed, non-empty mask).
    A        = max((real_numerator - A_BS) / real_denom, 0) - the new
               default/production value. See compute_A_BS for the formula.

IMPORTANT CAVEAT, established in asymmetry/scripts/kernel_testing/
01_noise_floor.py and 02_grid_stability.py *before* this fix was built:
the resolution-tracking problem in A_min (n_res outperforming re_kpc in
the sf_ml ML models - see ASYMMETRY_HANDOFF.md section 2) looks primarily
like a grid-search-minimum SELECTION-EFFECT artifact (A_range across the
100 candidate centres strongly anti-correlates with n_res; A_mean/A_median
computed from that SAME underlying grid show ~zero n_res correlation,
unlike A_min's rho~+0.30) - NOT primarily a noise-floor contamination
problem the way it was for S (pure noise explained only R^2=0.006 of
A_min's variance here, vs 94% for S pre-fix). This A_BS correction is
added for completeness with Conselice's formal definition and is validated
to correctly neutralise noise on held-out draws, but it does NOT
meaningfully change A_min's resolution correlation (confirmed in
03_background_subtraction_test.py: rho with n_res barely moves, +0.302 ->
+0.300) - that selection-effect problem is separate and still open.

Pipeline per galaxy:
    1. Load flux + error from the continuum-masked Halpha FITS (summed
       Gaussian slice, HDU0/HDU1 index 1)
    2. Error-subtract and clip: I_clean = clip(flux - error, 0)
    3. Rotate 180 deg about a candidate centre (shift -> rot90 -> shift back)
    4. Grid-search candidate centres within +/-2 px of the geometric centre;
       A_no_sub = the minimum sum(|I_clean - I_180|) / sum(|I_clean|) found,
       over the intersection of the original and rotated non-zero footprints
    5. A_BS, A as described above, at the best-fit centre found in step 4

Input : data/masked/halpha/
Output: outputs/csv/asymmetry.csv (CATID, A_geom, A_no_sub, A_BS, A,
        delta_A, best_cy, best_cx)

Usage
-----
    python src/03_metrics/asymmetry.py                # full sample
    python src/03_metrics/asymmetry.py --catid 7139    # single galaxy

Note: computing A_BS adds N_DRAWS_A_BS=50 extra rotate/shift operations per
galaxy - a modest, occasional cost for a batch script (~30s for the ~700
in_full_sam sample in kernel_testing validation), not optimised further here.
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

# A_BS (background-asymmetry) settings - see module docstring and
# asymmetry/scripts/kernel_testing/03_background_subtraction_test.py for
# the validation this was derived from.
N_DRAWS_A_BS = 50  # noise realisations used to estimate A_BS per galaxy;
                   # validated - the ~20% median relative draw-to-draw
                   # scatter on A_BS itself translates to <=1.6% absolute
                   # impact on A for 95% of galaxies (small correction term)
A_BS_SEED = 44     # fixed base seed - combined with CATID below so every
                   # galaxy gets its own reproducible, independent stream
                   # (same result whether run in batch or via --catid),
                   # matching smoothness.py's BS_SEED pattern


def apply_error_subtraction(flux: np.ndarray, error: np.ndarray) -> np.ndarray:
    """I_clean = clip(flux - error, 0). Down-weights low-S/N periphery pixels."""
    return np.clip(flux - error, 0, None)


def _common_mask(img: np.ndarray, rotated: np.ndarray) -> np.ndarray:
    """
    Intersection of the original non-zero footprint and a thresholded
    rotated footprint (rotated > 1e-3 * max(img), falling back to
    rotated > 0 if img is all zero) - extracted from compute_A so
    compute_A_BS can reuse the REAL galaxy's own mask for noise draws
    rather than deriving a fresh, much sparser mask from each noise
    realisation (see compute_A_BS docstring).
    """
    original_mask = img != 0
    rotated_mask = rotated > 1e-3 * np.max(img) if np.max(img) > 0 else rotated > 0
    return original_mask & rotated_mask


def compute_A(img: np.ndarray, cy: float, cx: float) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Asymmetry A from rotating img 180 deg about the point (cy, cx).

    Shifts (cy, cx) onto the array midpoint, rotates exactly with np.rot90
    (no interpolation error), then shifts back. Uses the intersection of the
    original non-zero footprint and a thresholded rotated footprint (see
    _common_mask) so edge spaxels smeared by the sub-pixel shift don't
    spuriously inflate A - this threshold is what individual_asymmetry_03.py
    / asymmetry_04.py actually used in production, refined from an earlier
    `rotated > 0` cut in individual_asymmetry_02.py.
    """
    array_centre = (img.shape[0] - 1) / 2.0
    dy = array_centre - cy
    dx = array_centre - cx

    shifted = shift(img, shift=(dy, dx), order=3, mode="constant", cval=0.0)
    rotated = np.rot90(shifted, 2)
    rotated = shift(rotated, shift=(-dy, -dx), order=3, mode="constant", cval=0.0)

    common_mask = _common_mask(img, rotated)
    residual = img - rotated
    denom = np.sum(np.abs(img[common_mask]))

    if denom == 0:
        return np.nan, rotated, residual

    A = np.sum(np.abs(residual[common_mask])) / denom
    return A, rotated, residual


def compute_A_BS(err: np.ndarray, cy: float, cx: float, common_mask: np.ndarray,
                  catid: int, n_draws: int = N_DRAWS_A_BS) -> float:
    """
    Median, over n_draws white-noise realisations drawn from this galaxy's
    own error map, of the A-numerator (sum|noise_clean -
    rotate180(noise_clean)|) evaluated over the REAL galaxy's own
    common-mask footprint at its own best-fit centre - the A-statistic
    analog of smoothness.py's compute_B_S.

    Uses the REAL galaxy's common_mask (from its own I_clean/rotation at
    this centre), NOT a mask derived from each noise draw's own sparse
    post-clip footprint - validated in
    asymmetry/scripts/kernel_testing/03_background_subtraction_test.py to
    avoid a ~3x-too-large small-number-statistics artifact the naive
    version produces, and to sidestep degenerate all-zero-draw cases
    entirely (the numerator is always a well-defined, non-negative number
    over this fixed, non-empty mask, regardless of how sparse a given
    noise draw's own footprint is).

    Seeded per-galaxy (A_BS_SEED, catid) so this is exactly reproducible
    whether the full sample or a single --catid is run.
    """
    rng = np.random.default_rng((A_BS_SEED, catid))
    scale = np.where(err > 0, err, 0.0)
    numerators = []
    for _ in range(n_draws):
        noise = rng.normal(loc=0.0, scale=scale)
        I_noise = apply_error_subtraction(noise, err)
        _, _, residual_noise = compute_A(I_noise, cy, cx)
        numerators.append(float(np.sum(np.abs(residual_noise[common_mask]))))
    return float(np.median(numerators))


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
    best_centre, A_no_sub = grid_search_centre(img)
    best_cy, best_cx = best_centre

    # Background-asymmetry correction (A_BS) - see compute_A_BS docstring
    # and the module docstring's IMPORTANT CAVEAT for the full rationale.
    # Reuses the best-fit centre and its own common-mask footprint, both
    # already established on the real signal above - no separate search.
    _, rotated_best, residual_best = compute_A(img, best_cy, best_cx)
    common_mask = _common_mask(img, rotated_best)
    real_numerator = float(np.sum(np.abs(residual_best[common_mask])))
    real_denom = float(np.sum(np.abs(img[common_mask])))
    A_BS = compute_A_BS(error, best_cy, best_cx, common_mask, catid)
    A_corrected = max((real_numerator - A_BS) / real_denom, 0.0) if real_denom else np.nan

    return {
        "CATID": catid,
        "galaxy_id": galaxy_id,
        "A_geom": round(A_geom, 6),
        "A_no_sub": round(A_no_sub, 6),
        "A_BS": round(A_BS, 6) if not np.isnan(A_BS) else A_BS,
        "A": round(A_corrected, 6) if not np.isnan(A_corrected) else A_corrected,
        "delta_A": round(A_geom - A_no_sub, 6),
        "best_cy": round(best_cy, 3),
        "best_cx": round(best_cx, 3),
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
        print(f"CATID {args.catid}: A_geom={result['A_geom']:.4f}  A_no_sub={result['A_no_sub']:.4f}  "
              f"A_BS={result['A_BS']:.4f}  A={result['A']:.4f}  delta_A={result['delta_A']:.4f}")
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
