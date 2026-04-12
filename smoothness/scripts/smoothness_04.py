"""
smoothness_04.py
----------------
Full sample iteration of the Clumpiness (S) parameter for all SAMI Halpha
masked FITS files, following the methodology established in
smoothness_test_03.py.

Pipeline per galaxy:
    1. Load flux and error maps (summed Gaussian slices, HDU 0 and 1)
    2. Subtract error from flux: I_clean = I - err
    3. Smooth I_clean with Gaussian kernel: sigma = 0.35 * r_e (pixels)
    4. Compute residual: |I_clean - I_clean_smoothed|
    5. Compute partial S: sum(|I_clean - I_clean_smoothed|) / sum(|I_clean|)

Kernel scale: 0.35 * r_e 
"""

import os
import glob
import re
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import gaussian_filter

# =============================================================================
# CONFIG
# =============================================================================

HALPHA_DIR    = "gini/gini_outputs/tables/masked_halpha_maps"
CSV_PATH      = "concentration/outputs/tables/concentration_results.csv"
OUT_DIR       = "smoothness/outputs/tables"
OUT_FILENAME  = "smoothness_results.csv"

KERNEL_SCALE  = 0.35      # fraction of r_e used as Gaussian sigma
PLATE_SCALE   = 0.5       # arcsec/px (confirmed from FITS header)

# Columns to carry through from the catalogue into the output CSV
CARRY_COLS = [
    'spectral_class',
    'environment',
    'galclass',
    're',
    'ellip',
    'z',
    'g_i',
    'Mstar',
    'seeing',
    'seeing_re',
    'source_file',
    'conflict_flag',
]

os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# LOAD CATALOGUE
# =============================================================================

def load_catalogue(csv_path):
    """
    Load concentration_results.csv and prepare r_e column with SF priority.

    Galaxies with unphysical r_e (> 50 arcsec, e.g. failed Sersic fits)
    are flagged and excluded.
    """
    df = pd.read_csv(csv_path)
    df['CATID'] = df['CATID'].astype(str)

    # SF r_e priority, composite fallback
    df['re_used']    = df['re']
    df['re_source']  = 're_sf'
    fallback_mask    = df['re_used'].isna() & df['re_com'].notna()
    df.loc[fallback_mask, 're_used']   = df.loc[fallback_mask, 're_com']
    df.loc[fallback_mask, 're_source'] = 're_com'

    n_total = len(df)

    # Remove unphysical r_e values (failed Sersic fits — e.g. galaxy 620092)
    RE_MAX = 50.0   # arcseconds
    bad_re = df['re_used'] > RE_MAX
    if bad_re.sum() > 0:
        print(f"Removing {bad_re.sum()} galaxies with unphysical r_e > {RE_MAX}\":")
        print(df.loc[bad_re, ['CATID', 're_used', 'source_file']].to_string())
    df = df[~bad_re]

    print(f"\nCatalogue loaded: {len(df)} / {n_total} galaxies with valid r_e")
    df = df.set_index('CATID')
    return df


# =============================================================================
# LOAD SINGLE GALAXY
# =============================================================================

def load_halpha(fits_path):
    """
    Load Halpha flux and error maps from a SAMI masked FITS file.
    """
    with fits.open(fits_path) as hdul:
        flux = hdul[0].data[1, :, :]
        err  = hdul[1].data[1, :, :]
    return flux, err


# =============================================================================
# PIPELINE FUNCTIONS
# =============================================================================

def subtract_error(flux, err):
    """Subtract per-pixel error from flux: I_clean = I - err."""
    return flux - err


def conselice_smooth(flux, sigma_px):
    """Smooth flux map with Gaussian kernel of width sigma_px."""
    return gaussian_filter(flux, sigma=sigma_px)


def compute_residual(flux, flux_smoothed):
    """Absolute residual map: |I - I_S|."""
    return np.abs(flux - flux_smoothed)


def compute_S(flux_clean, residual):
    """
    Partial S: sum(|I_clean - I_clean_S|) / sum(|I_clean|)
    """
    denom = np.nansum(np.abs(flux_clean))
    if denom == 0:
        return np.nan
    return float(np.nansum(residual) / denom)


# =============================================================================
# PROCESS SINGLE GALAXY
# =============================================================================

def process_galaxy(fits_path, cat_row, kernel_scale, plate_scale):
    """
    Run the full smoothness pipeline for a single galaxy.
    """
    # Load
    flux, err = load_halpha(fits_path)

    # r_e in pixels
    r_e_arcsec = cat_row['re_used']
    r_e_px     = r_e_arcsec / plate_scale
    sigma_px   = kernel_scale * r_e_px

    # Pipeline: subtract error -> smooth -> residual -> S
    flux_clean    = subtract_error(flux, err)
    flux_smoothed = conselice_smooth(flux_clean, sigma_px)
    residual      = compute_residual(flux_clean, flux_smoothed)
    S             = compute_S(flux_clean, residual)

    return S


# =============================================================================
# EXTRACT CATID FROM FILENAME
# =============================================================================

def catid_from_filename(fits_path):
    """
    Extract the CATID from a filename of the form:
        <CATID>_<IFU>_Halpha_masked.fits
    """
    basename = os.path.basename(fits_path)
    match    = re.match(r'^(\d+)_', basename)
    if match is None:
        raise ValueError(f"Cannot extract CATID from filename: {basename}")
    return match.group(1)


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Load catalogue
    catalogue = load_catalogue(CSV_PATH)

    # Find all masked Halpha FITS files
    pattern    = os.path.join(HALPHA_DIR, "*_Halpha_masked.fits")
    fits_files = sorted(glob.glob(pattern))
    print(f"\nFound {len(fits_files)} Halpha masked FITS files in {HALPHA_DIR}")

    if len(fits_files) == 0:
        raise FileNotFoundError(
            f"No FITS files found matching pattern: {pattern}\n"
            "Check HALPHA_DIR in the CONFIG block."
        )

    # Iterate over all FITS files
    results  = []
    failures = []

    for i, fits_path in enumerate(fits_files):
        # Extract CATID from filename
        try:
            catid = catid_from_filename(fits_path)
        except ValueError as e:
            failures.append({'fits_path': fits_path, 'reason': str(e)})
            #print(f"  [SKIP] {os.path.basename(fits_path)}: {e}")
            continue

        # Progress indicator
        #print(f"  [{i+1}/{len(fits_files)}] CATID {catid} ...", end=" ")

        # Check galaxy is in catalogue
        if catid not in catalogue.index:
            reason = "CATID not found in catalogue"
            failures.append({'catid': catid, 'fits_path': fits_path,
                             'reason': reason})
            #print(f"SKIP — {reason}")
            continue

        cat_row = catalogue.loc[catid]

        # Check r_e is valid
        if pd.isna(cat_row['re_used']):
            reason = "r_e is NaN (no SF or composite value available)"
            failures.append({'catid': catid, 'fits_path': fits_path,
                             'reason': reason})
            #print(f"SKIP — {reason}")
            continue

        # Run pipeline
        try:
            S = process_galaxy(fits_path, cat_row, KERNEL_SCALE, PLATE_SCALE)
        except Exception as e:
            reason = f"Pipeline error: {e}"
            failures.append({'catid': catid, 'fits_path': fits_path,
                             'reason': reason})
            #print(f"SKIP — {reason}")
            continue

        if np.isnan(S):
            reason = "S is NaN (total flux is zero after error subtraction)"
            failures.append({'catid': catid, 'fits_path': fits_path,
                             'reason': reason})
            #print(f"SKIP — {reason}")
            continue

        # Collect result
        row = {'CATID': catid, 'S': round(S, 6)}
        for col in CARRY_COLS:
            row[col] = cat_row[col] if col in cat_row.index else np.nan
        row['re_source']    = cat_row['re_source']
        row['kernel_scale'] = KERNEL_SCALE
        row['sigma_arcsec'] = round(KERNEL_SCALE * cat_row['re_used'], 4)
        row['sigma_px']     = round(KERNEL_SCALE * cat_row['re_used'] / PLATE_SCALE, 4)

        results.append(row)
        #print(f"S = {S:.4f}")

    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================

    results_df = pd.DataFrame(results)
    out_path   = os.path.join(OUT_DIR, OUT_FILENAME)
    results_df.to_csv(out_path, index=False)

    # ==========================================================================
    # SUMMARY
    # ==========================================================================

    print(f"\n{'='*60}")
    print(f"SMOOTHNESS CALCULATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total FITS files found : {len(fits_files)}")
    print(f"  Successfully processed : {len(results)}")
    print(f"  Failed / skipped       : {len(failures)}")
    print(f"  Results saved to       : {os.path.abspath(out_path)}")

    if failures:
        print(f"\nFailed galaxies:")
        for f in failures:
            catid    = f.get('catid', 'unknown')
            basename = os.path.basename(f.get('fits_path', ''))
            reason   = f.get('reason', '')
            print(f"    CATID {catid:>10}  |  {basename:<45}  |  {reason}")

    print(f"\nOutput columns: CATID, S, kernel_scale, sigma_arcsec, sigma_px, "
          f"re_source, {', '.join(CARRY_COLS)}")


if __name__ == "__main__":
    main()