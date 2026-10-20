#!/usr/bin/env python3
"""
03_metrics/smoothness.py
========================
CAS Smoothness / Clumpiness (S) of the Halpha emission-line flux map,
following Conselice (2003):

    S = [ sum(I - I_s) - sum(B - B_s) ] / sum(I)

where I is the masked Halpha flux map, I_s is I smoothed with a Gaussian
kernel, and B / B_s is a noise image and its smoothed version (the
background term B_S). Only positive residuals are kept: pixels where
I - I_s < 0 (or B - B_s < 0) are set to zero before summing.

Steps per galaxy:
    1. Load the masked Halpha flux and error maps (summed-flux slice).
       The footprint is the set of pixels inside the mask (flux != 0).
    2. Smooth the full map with a Gaussian of sigma = KERNEL_SCALE * r_e.
    3. Keep only positive residuals, max(I - I_s, 0), and sum them over
       the footprint. Dividing by sum(I) over the footprint gives S_no_B_S_sub.
    4. Background term B_S: make N_DRAWS_BS noise images, each pixel drawn
       from Normal(0, that pixel's error). Smooth each with the same kernel
       and sum max(B - B_s, 0) over the same footprint. B_S is the median
       over all draws.
    5. S = (sum max(I - I_s, 0) - B_S) / sum(I).

r_e (arcsec) comes from outputs/csv/concentration.csv: the SF value 're'
is used where available, otherwise the composite value 're_com'. Galaxies
with r_e > 50" (failed fits) are excluded. Pixel scale is 0.5"/px.

Input : data/masked/halpha/<CATID>_<IFU>_Halpha_masked.fits
        outputs/csv/concentration.csv
Output: outputs/csv/smoothness.csv
        CATID, galaxy_id, S_no_B_S_sub, B_S, S, kernel_scale, sigma_arcsec,
        sigma_px, re_used, re_source

    S_no_B_S_sub  S with no background term
    B_S           background term (numerator units, not a ratio)
    S             final smoothness, background-corrected

Usage
-----
    python src/03_metrics/smoothness.py                 # full sample
    python src/03_metrics/smoothness.py --catid 23117   # single galaxy (prints only)
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

HALPHA_LAYER = 1      # summed-flux slice, matching gini.py and asymmetry.py
KERNEL_SCALE = 0.35   # Gaussian sigma as a fraction of r_e
PLATE_SCALE = 0.5     # arcsec per pixel
RE_MAX_ARCSEC = 50.0  # r_e above this is treated as a failed fit
N_DRAWS_BS = 50       # number of noise images used for B_S
BS_SEED = 42          # combined with CATID so each galaxy's noise is reproducible


def load_re_lookup(concentration_csv: Path) -> pd.DataFrame:
    """CATID -> r_e (arcsec): SF 're' preferred, composite 're_com' as fallback, r_e > 50" dropped."""
    df = pd.read_csv(concentration_csv)
    df["CATID"] = df["CATID"].astype(str)

    df["re_used"] = df["re"]
    df["re_source"] = "re_sf"
    use_com = df["re_used"].isna() & df["re_com"].notna()
    df.loc[use_com, "re_used"] = df.loc[use_com, "re_com"]
    df.loc[use_com, "re_source"] = "re_com"

    n_total = len(df)
    bad_re = df["re_used"] > RE_MAX_ARCSEC
    if bad_re.any():
        print(f"  Excluding {bad_re.sum()} galaxies with unphysical r_e > {RE_MAX_ARCSEC}\"")
    df = df[~bad_re]

    print(f"  r_e lookup: {len(df)} / {n_total} galaxies with valid r_e")
    return df.set_index("CATID")


def positive_residual(img: np.ndarray, sigma_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Smooth img and return (smoothed image, max(img - smoothed, 0))."""
    smoothed = gaussian_filter(img, sigma=sigma_px)
    return smoothed, np.clip(img - smoothed, 0, None)


def compute_B_S(error: np.ndarray, sigma_px: float, footprint: np.ndarray,
                catid: int, n_draws: int = N_DRAWS_BS) -> float:
    """
    Background term: median over n_draws of sum(max(B - B_s, 0)) over the
    footprint, where each noise image B has pixels drawn from Normal(0, error).
    """
    rng = np.random.default_rng((BS_SEED, catid))
    scale = np.where(error > 0, error, 0.0)

    sums = []
    for _ in range(n_draws):
        noise = rng.normal(loc=0.0, scale=scale)
        _, residual = positive_residual(noise, sigma_px)
        sums.append(float(np.sum(residual[footprint])))
    return float(np.median(sums))


def process_one(halpha_path: Path, re_arcsec: float, catid: int) -> dict | None:
    """Compute all smoothness columns for one galaxy. Returns None if the map is empty."""
    flux_cube, err_cube = read_masked_halpha(str(halpha_path))
    img = np.nan_to_num(flux_cube[HALPHA_LAYER], nan=0.0, posinf=0.0, neginf=0.0)
    error = np.nan_to_num(err_cube[HALPHA_LAYER], nan=0.0, posinf=0.0, neginf=0.0)

    footprint = img != 0
    denom = float(np.sum(img[footprint]))
    if denom == 0:
        return None

    sigma_px = KERNEL_SCALE * re_arcsec / PLATE_SCALE

    # Real numerator: positive residuals inside the original footprint
    _, residual = positive_residual(img, sigma_px)
    numerator = float(np.sum(residual[footprint]))

    # Background-corrected S
    B_S = compute_B_S(error, sigma_px, footprint, catid)

    return {
        "S_no_B_S_sub": numerator / denom,
        "B_S": B_S,
        "S": (numerator - B_S) / denom,
        "sigma_arcsec": KERNEL_SCALE * re_arcsec,
        "sigma_px": sigma_px,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catid", type=int, default=None,
                        help="Compute and print S for a single CATID (does not write the CSV).")
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
            raise FileNotFoundError(f"No masked Halpha FITS for CATID {args.catid}. "
                                    "Run 02_continuum_mask.py first.")
        row = re_lookup.loc[catid_str]
        result = process_one(matches[0], row["re_used"], args.catid)
        if result is None:
            print(f"CATID {args.catid}: empty Halpha map.")
            return
        print(f"CATID {args.catid}: S_no_B_S_sub={result['S_no_B_S_sub']:.4f}  B_S={result['B_S']:.4f}  "
              f"S={result['S']:.4f}  sigma={result['sigma_px']:.2f}px  (r_e source: {row['re_source']})")
        return

    halpha_files = sorted(MASKED_HALPHA_DIR.glob("*_Halpha_masked.fits"))
    if not halpha_files:
        raise FileNotFoundError(f"No masked Halpha FITS found in {MASKED_HALPHA_DIR}. "
                                "Run 02_continuum_mask.py first.")
    print(f"Found {len(halpha_files)} masked Halpha files.")

    rows = []
    skipped = {"no_re": 0, "empty_map": 0}
    for halpha_path in tqdm(halpha_files, desc="Smoothness"):
        galaxy_id = halpha_path.name.replace("_Halpha_masked.fits", "")
        catid_str = galaxy_id.split("_")[0]

        if catid_str not in re_lookup.index:
            skipped["no_re"] += 1
            continue
        re_row = re_lookup.loc[catid_str]
        if isinstance(re_row, pd.DataFrame):  # duplicate CATID in concentration.csv
            re_row = re_row.iloc[0]
        if pd.isna(re_row["re_used"]):
            skipped["no_re"] += 1
            continue

        result = process_one(halpha_path, re_row["re_used"], int(catid_str))
        if result is None:
            skipped["empty_map"] += 1
            continue

        rows.append({
            "CATID": int(catid_str),
            "galaxy_id": galaxy_id,
            "S_no_B_S_sub": round(result["S_no_B_S_sub"], 6),
            "B_S": round(result["B_S"], 6),
            "S": round(result["S"], 6),
            "kernel_scale": KERNEL_SCALE,
            "sigma_arcsec": round(result["sigma_arcsec"], 4),
            "sigma_px": round(result["sigma_px"], 4),
            "re_used": re_row["re_used"],
            "re_source": re_row["re_source"],
        })

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("\n--- Summary ---")
    print(f"  Galaxies processed : {len(df)}")
    print(f"  No valid r_e       : {skipped['no_re']}")
    print(f"  Empty Halpha maps  : {skipped['empty_map']}")
    print(f"  Wrote              : {OUT_CSV}")


if __name__ == "__main__":
    main()
