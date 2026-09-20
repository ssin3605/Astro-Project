#!/usr/bin/env python3
"""
03_metrics/smoothness.py
==========================
Clumpiness (S) parameter for the Hα emission-line flux map, following the
methodology carried over unchanged from smoothness_04.py (audited). This
script computes the same "partial S" as the old pipeline, UNCHANGED, in
the existing 'S' column:

    S = sum(|I_clean - I_clean_smoothed|) / sum(|I_clean|)

As of the kernel_testing/ investigation (smoothness/scripts/kernel_testing/,
see its README.md), 'S' was found to be dominated by a resolution/kernel-
geometry noise artifact (02_noise_floor.py: a quantity built from pure
white noise, containing zero real flux information, explained 94% of S's
variance across the science sample). 08_background_subtraction_test.py
validated a fix - the missing background-smoothness term B_S from
Conselice (2003)'s full definition - on held-out noise draws never used to
fit the correction (R^2 of corrected-S-on-noise dropped 0.94 -> 0.12, and
held-out pure-noise pushed through the corrected formula clustered at
S~0, as it should). That correction is now added here as TWO NEW columns,
B_S and S, and the original value with no subtraction is called 'S_no_B_S_sub':

    B_S         = median, over N_DRAWS_BS white-noise realisations drawn
                  from the galaxy's own error map (noise_i ~ Normal(0, err),
                  same formula as I_clean: noise_i - err), of
                  sum(|I_noise - I_noise_smoothed|) at this galaxy's own
                  sigma_px - i.e. how rough THIS kernel makes pure noise
                  look, in the same units as the real residual sum. This
                  needs no blank-sky region (SAMI hexabundle pointings
                  don't reliably have one) - the galaxy's own per-pixel
                  error map is a better, galaxy-specific noise model than
                  a generic sky patch would be.
    S           = max((sum(residual) - B_S) / sum(|I_clean|), 0)
                  (clipped at 0 - standard convention: a negative value
                  means the real residual didn't exceed what pure noise
                  alone would produce at this kernel, i.e. not resolved
                  above the noise floor. 0/705 in_full_sam galaxies hit
                  this during validation.)

Note: B_S corrects the noise/kernel-geometry artifact specifically. It
does NOT remove the separate, genuine resolution/information limit
demonstrated in 03_redshift_degradation.py (real structure that coarser
sampling genuinely cannot recover, independent of noise) - S_corrected
still correlates with resolution proxies at close to the same level 'S'
does, and that residual correlation is expected, not a bug. Downstream
use (e.g. an ML "key drivers" step) should still control for a resolution
proxy (n_res = re_used/seeing) explicitly rather than relying on
S_corrected alone to have solved that.

This script still keeps the OTHER two documented simplifications relative
to the full Conselice (2003) S statistic: the central pixel region is not
excluded before summing the residual (deferred - see the kernel_testing
README's T4-first ordering: your own AGN handling operates on the
integrated SFR budget via Wang's pure_SF/composite_SF classification, not
on this spatially-resolved Hα map, so central-pixel contamination is still
an open question, pending a diagnostic), and the conventional x10 scaling
some CAS papers apply to S is not used (kept consistent with
smoothness_04.py throughout).

Pipeline per galaxy:
    1. Load flux + error from the continuum-masked Halpha FITS (summed
       Gaussian slice, HDU0/HDU1 index 1)
    2. Error-subtract, NOT clipped: I_clean = flux - error (negative values
       are retained - the shared continuum mask already zero-fills outside
       the galaxy footprint, so this is left exactly as smoothness_04.py had
       it; see AUDIT_REPORT.md section 3.3 for why this differs from
       asymmetry.py's clipped subtraction. kernel_testing/
       07_negative_clipping_test.py subsequently tested whether clipping/
       SN-masking this step reduces the noise-floor problem above, and
       found it does not - the shared continuum mask already keeps median
       Halpha S/N ~46 within its footprint, so there is almost no low-S/N
       population left to clip; the noise-floor problem is a property of
       the smoothing-and-ratio construction itself, not of retained bad
       pixels, which is why B_S rather than clipping is the fix applied
       here)
    3. Smooth I_clean with a Gaussian kernel: sigma = KERNEL_SCALE * r_e (px)
    4. residual = |I_clean - I_clean_smoothed|
    5. S = sum(residual) / sum(|I_clean|)
    6. B_S, S_corrected as described above

Kernel scale and plate scale are unchanged from smoothness_04.py:
    KERNEL_SCALE = 0.35 (fraction of r_e)
    PLATE_SCALE  = 0.5 arcsec/px (hardcoded; verified via kernel_testing -
                   CDELT1 read directly from all 2530 masked Halpha FITS
                   headers gives exactly 0.5"/px for every one, zero
                   deviation, so this is now a confirmed property of the
                   SAMI drizzled-cube grid, not an unverified assumption)

r_e is read from the Concentration table (03_metrics/concentration.py must
be run first), with the same SF-preferred / composite-fallback logic and the
same >50" unphysical-r_e exclusion as smoothness_04.py.

Input : data/masked/halpha/, outputs/csv/concentration.csv
Output: outputs/csv/smoothness.csv (CATID, S, B_S, S_no_B_S_sub, kernel_scale,
        sigma_arcsec, sigma_px, n_res, re_used, re_source)

Usage
-----
    python src/03_metrics/smoothness.py                # full sample
    python src/03_metrics/smoothness.py --catid 23117   # single galaxy

Note: computing B_S adds N_DRAWS_BS=50 extra Gaussian-filter passes per
galaxy (~6-7 minutes for the full ~2260-galaxy sample on typical hardware,
vs seconds before) - a one-time/occasional cost for a batch script, not
optimised further here.
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
PLATE_SCALE = 0.5     # arcsec/px - hardcoded, verified across the full sample (see module docstring)
RE_MAX_ARCSEC = 50.0  # excludes unphysical r_e (failed Sersic fits)

# B_S (background-smoothness) settings - see module docstring and
# smoothness/scripts/kernel_testing/08_background_subtraction_test.py for
# the validation this was derived from.
N_DRAWS_BS = 50   # noise realisations used to estimate B_S per galaxy;
                  # validated as stable (~4% median relative draw-to-draw
                  # scatter) at this count in 08_background_subtraction_test.py
BS_SEED = 42      # fixed base seed - combined with CATID below so every
                  # galaxy gets its own reproducible, independent stream
                  # (same result whether run in batch or via --catid)


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


def compute_B_S(err: np.ndarray, sigma_px: float, catid: int, n_draws: int = N_DRAWS_BS) -> float:
    """
    Median, over n_draws white-noise realisations drawn from this galaxy's
    own error map, of sum(|I_noise - I_noise_smoothed|) at sigma_px - the
    background-smoothness term from Conselice (2003), estimated from the
    galaxy's own per-pixel error map rather than a blank-sky region (see
    module docstring). Validated in
    smoothness/scripts/kernel_testing/08_background_subtraction_test.py.

    Seeded per-galaxy (BS_SEED, catid) so this is exactly reproducible
    whether the full sample or a single --catid is run.
    """
    rng = np.random.default_rng((BS_SEED, catid))
    scale = np.where(err > 0, err, 0.0)
    numerators = []
    for _ in range(n_draws):
        noise = rng.normal(loc=0.0, scale=scale)
        I_noise = noise - err
        smoothed = gaussian_filter(I_noise, sigma=sigma_px)
        numerators.append(float(np.nansum(np.abs(I_noise - smoothed))))
    return float(np.median(numerators))


def galaxy_id_from_masked_halpha_name(filename: str) -> str:
    return filename.replace("_Halpha_masked.fits", "")


def process_one(halpha_path: Path, re_arcsec: float, kernel_scale: float, plate_scale: float,
                 catid: int | None = None) -> dict:
    flux_cube, err_cube = read_masked_halpha(str(halpha_path))
    flux = flux_cube[HALPHA_LAYER]
    err = err_cube[HALPHA_LAYER]

    r_e_px = re_arcsec / plate_scale
    sigma_px = kernel_scale * r_e_px

    flux_clean = subtract_error(flux, err)
    flux_smoothed = gaussian_filter(flux_clean, sigma=sigma_px)
    residual = np.abs(flux_clean - flux_smoothed)
    S = compute_S(flux_clean, residual)

    result = {"S": S, "sigma_arcsec": kernel_scale * re_arcsec, "sigma_px": round(sigma_px, 4)}

    if catid is not None:
        denom = np.nansum(np.abs(flux_clean))
        B_S = compute_B_S(err, sigma_px, catid)
        S_corrected = max((np.nansum(residual) - B_S) / denom, 0.0) if denom else np.nan
        result["B_S"] = B_S
        result["S_corrected"] = S_corrected

    return result


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
        result = process_one(matches[0], row["re_used"], KERNEL_SCALE, PLATE_SCALE, catid=args.catid)
        print(f"CATID {args.catid}: S={result['S']:.4f}  B_S={result['B_S']:.4f}  "
              f"S_corrected={result['S_corrected']:.4f}  sigma={result['sigma_px']:.2f}px  "
              f"(r_e source: {row['re_source']})")
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

        result = process_one(halpha_path, re_row["re_used"], KERNEL_SCALE, PLATE_SCALE, catid=int(catid_str))
        if np.isnan(result["S"]):
            failures["nan_S"] += 1
            continue

        rows.append({
            "CATID": int(catid_str),
            "galaxy_id": galaxy_id,
            "S_no_B_S_sub": round(result["S"], 6),
            "B_S": round(result["B_S"], 6),
            "S": round(result["S_corrected"], 6),
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
