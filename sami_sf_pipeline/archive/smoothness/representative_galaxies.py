"""
Selects 4 representative SF galaxies from smoothness_results.csv that span
the parameter space of log(Mstar), log(sSFR), and S.
"""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

# =============================================================================
# CONFIG
# =============================================================================

SMOOTHNESS_CSV = 'smoothness/outputs/tables/smoothness_results.csv'
FITS_CATALOG   = 'data/catalogs/AR2022_SAMI_DR3_Mstar_SFR_GSWL_rescaled.fits'
HALPHA_DIR     = 'gini/gini_outputs/tables/masked_halpha_maps'

# =============================================================================
# LOG CONVERSION
# =============================================================================

def to_log(series, name):
    """Convert to log10 if raw (median > 100), else use as-is."""
    valid = series.dropna()
    if valid.empty:
        print(f'  {name}: no valid values — skipping')
        return series
    if valid.median() > 100:
        print(f'  {name}: raw values detected — applying log10')
        return np.log10(series.replace(0, np.nan))
    else:
        print(f'  {name}: already in log form — using as-is')
        return series

# =============================================================================
# LOAD AND PREPARE
# =============================================================================

def load_and_prepare(smoothness_csv, fits_catalog):
    """
    Load smoothness results, merge SFR, compute log columns.
    Restricts to SF-classified galaxies with valid Mstar, SFR, and S.
    """
    # Smoothness catalogue
    df = pd.read_csv(smoothness_csv)
    df['CATID'] = df['CATID'].astype(int)

    # SFR from FITS catalogue
    with fits.open(fits_catalog) as hdul:
        fits_df = Table(hdul[1].data).to_pandas()
    fits_df.columns = [c.upper() for c in fits_df.columns]
    fits_df['CATID'] = fits_df['CATID'].astype(int)

    SFR_COL = next((c for c in fits_df.columns if 'SFR' in c), None)
    if SFR_COL is None:
        raise KeyError(f'No SFR column found. Columns: {list(fits_df.columns)}')

    fits_slim = fits_df[['CATID', SFR_COL]].rename(columns={SFR_COL: 'sfr_raw'})
    df = df.merge(fits_slim, on='CATID', how='left')

    # Log columns
    #print('\nChecking log/raw status:')
    df['log_mstar'] = to_log(df['Mstar'],   'Mstar')
    df['log_sfr']   = to_log(df['sfr_raw'], 'SFR')
    df['log_ssfr']  = df['log_sfr'] - df['log_mstar']

    # Restrict to SF only with all required parameters valid
    df_sf = df[
        (df['spectral_class'] == 'SF') &
        df['log_mstar'].notna() &
        df['log_ssfr'].notna() &
        df['S'].notna()
    ].copy()

    #print(f'\n  Total galaxies in smoothness CSV : {len(df)}')
    #print(f'  SF galaxies with valid params    : {len(df_sf)}')

    return df_sf

# =============================================================================
# STRATIFIED SELECTION
# =============================================================================

def select_representatives(df_sf):
    """
    Divide SF sample into 2x2 grid (high/low Mstar x high/low sSFR) using
    sample medians as boundaries. Within each quadrant select the galaxy
    whose S is closest to the quadrant median S.
    """
    mstar_med = df_sf['log_mstar'].median()
    ssfr_med  = df_sf['log_ssfr'].median()

    #print(f'\n  Mstar median (log) : {mstar_med:.3f}')
    #print(f'  sSFR  median (log) : {ssfr_med:.3f}')

    quadrants = {
        'Low  Mstar  |  Low  sSFR' : (df_sf['log_mstar'] <= mstar_med) & (df_sf['log_ssfr'] <= ssfr_med),
        'Low  Mstar  |  High sSFR' : (df_sf['log_mstar'] <= mstar_med) & (df_sf['log_ssfr'] >  ssfr_med),
        'High Mstar  |  Low  sSFR' : (df_sf['log_mstar'] >  mstar_med) & (df_sf['log_ssfr'] <= ssfr_med),
        'High Mstar  |  High sSFR' : (df_sf['log_mstar'] >  mstar_med) & (df_sf['log_ssfr'] >  ssfr_med),
    }

    selected = []
    for label, mask in quadrants.items():
        quadrant = df_sf[mask].copy()
        if quadrant.empty:
            print(f'\n  WARNING: No galaxies found in quadrant "{label}" — skipping')
            continue

        # Select galaxy closest to median S in this quadrant
        s_median  = quadrant['S'].median()
        closest   = (quadrant['S'] - s_median).abs().idxmin()
        galaxy    = quadrant.loc[closest]

        selected.append({
            'quadrant'  : label,
            'CATID'     : int(galaxy['CATID']),
            'log_mstar' : round(galaxy['log_mstar'], 3),
            'log_ssfr'  : round(galaxy['log_ssfr'],  3),
            'log_sfr'   : round(galaxy['log_sfr'],   3),
            'S'         : round(galaxy['S'],          4),
            're'        : round(galaxy['re'],          3),
            'environment': galaxy['environment'],
            'source_file': galaxy['source_file'],
            'n_quadrant' : len(quadrant),
            's_median_quadrant': round(s_median, 4),
        })

    return selected

# =============================================================================
# PRINT RESULTS
# =============================================================================

def print_results(selected, halpha_dir):
    """Print selected galaxy parameters to terminal in a readable format."""
    divider = '=' * 68

    print(f'\n{divider}')
    print(f'  SELECTED REPRESENTATIVE SF GALAXIES')
    print(f'{divider}')

    for i, g in enumerate(selected):
        fits_pattern = os.path.join(halpha_dir, f"{g['CATID']}_*_Halpha_masked.fits")
        import glob
        fits_matches = glob.glob(fits_pattern)
        fits_file    = os.path.basename(fits_matches[0]) if fits_matches else 'NOT FOUND'

        print(f'\n  Galaxy {i+1} of {len(selected)} — {g["quadrant"]}')
        print(f'  {"─"*50}')
        print(f'  CATID          : {g["CATID"]}')
        print(f'  log(M*)        : {g["log_mstar"]}')
        print(f'  log(sSFR)      : {g["log_ssfr"]}')
        print(f'  log(SFR)       : {g["log_sfr"]}')
        print(f'  S              : {g["S"]}')
        #print(f'  r_e (arcsec)   : {g["re"]}')
        print(f'  Environment    : {g["environment"]}')
        #print(f'  Source file    : {g["source_file"]}')
        #print(f'  Quadrant N     : {g["n_quadrant"]} galaxies')
        #print(f'  Quadrant med S : {g["s_median_quadrant"]}')
        #print(f'  FITS file      : {fits_file}')

    print(f'\n{divider}')
    print(f'  CATID SUMMARY')
    print(f'  {"─"*50}')
    for g in selected:
        print(f'  {g["CATID"]}   '
              f'log(M*)={g["log_mstar"]}   '
              f'log(sSFR)={g["log_ssfr"]}   '
              f'S={g["S"]}')
    print(f'{divider}\n')

# =============================================================================
# MAIN
# =============================================================================

def main():
    df_sf    = load_and_prepare(SMOOTHNESS_CSV, FITS_CATALOG)
    selected = select_representatives(df_sf)
    print_results(selected, HALPHA_DIR)

if __name__ == '__main__':
    main()