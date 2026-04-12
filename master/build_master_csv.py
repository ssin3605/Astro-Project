"""
build_master.py
---------------
Builds the master summary CSV collating results from all four morphological
statistics analyses (Gini, Concentration, Asymmetry, Smoothness) for the
SAMI sample.

Output: /Users/sophiesinclair/Desktop/sami/master/results.csv

Columns in final CSV:
    CATID, gini_all_halpha, CIndex, A_min, S,
    spectral_class, environment, galclass, re, ellip, z, g_i,
    Mstar, SFR, log_SFR, sSFR, log_sSFR,
    seeing, seeing_re, source_file, conflict_flag

Data sources:
    - Gini        : gini/gini_outputs/tables/compare_gini.csv
    - Concentration: concentration/outputs/tables/concentration_results.csv
    - Asymmetry   : asymmetry/outputs/tables/asymmetry_results.csv
    - Smoothness  : smoothness/outputs/Tables/smoothness_results.csv
    - SFR         : data/catalogs/AR2022_SAMI_DR3_Mstar_SFR_GSWL_rescaled.fits

Priority for shared columns (spectral_class, environment, galclass, re,
ellip, z, g_i, Mstar, seeing, seeing_re, source_file, conflict_flag):
    concentration_results.csv takes priority.
    Mstar: SF value first, composite (Mstar_com) as fallback if NaN.
"""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR        = '/Users/sophiesinclair/Desktop/sami'

GINI_CSV        = os.path.join(BASE_DIR, 'gini/gini_outputs/tables/compare_gini.csv')
CONC_CSV        = os.path.join(BASE_DIR, 'concentration/outputs/tables/concentration_results.csv')
ASYM_CSV        = os.path.join(BASE_DIR, 'asymmetry/outputs/tables/asymmetry_results.csv')
SMOOTH_CSV      = os.path.join(BASE_DIR, 'smoothness/outputs/Tables/smoothness_results.csv')
FITS_CATALOG    = os.path.join(BASE_DIR, 'data/catalogs/AR2022_SAMI_DR3_Mstar_SFR_GSWL_rescaled.fits')

OUT_DIR         = os.path.join(BASE_DIR, 'master')
OUT_PATH        = os.path.join(OUT_DIR, 'results.csv')

os.makedirs(OUT_DIR, exist_ok=True)

# Shared columns sourced from concentration_results.csv (priority source)
SHARED_COLS = [
    'spectral_class',
    'environment',
    'galclass',
    're',
    'ellip',
    'z',
    'g_i',
    'seeing',
    'seeing_re',
    'source_file',
    'conflict_flag',
]

# =============================================================================
# LOAD AND NORMALISE CATID
# =============================================================================

def load_csv(path, catid_col):
    """
    Load a CSV, rename catid_col to CATID, and cast CATID to int.
    Strips whitespace from string CATID values before casting.
    """
    df = pd.read_csv(path)
    if catid_col not in df.columns:
        raise KeyError(
            f"Expected CATID column '{catid_col}' not found in {path}.\n"
            f"Available columns: {list(df.columns)}"
        )
    df = df.rename(columns={catid_col: 'CATID'})
    df['CATID'] = df['CATID'].astype(str).str.strip().astype(int)
    print(f"  Loaded {len(df):>5} rows : {os.path.basename(path)}")
    return df


# =============================================================================
# LOAD ALL CATALOGUES
# =============================================================================

print('Loading catalogues...')
gini   = load_csv(GINI_CSV,   'catid')
conc   = load_csv(CONC_CSV,   'CATID')
asym   = load_csv(ASYM_CSV,   'catid')
smooth = load_csv(SMOOTH_CSV, 'CATID')

# SFR from FITS catalogue
print(f"  Loading FITS : {os.path.basename(FITS_CATALOG)}")
with fits.open(FITS_CATALOG) as hdul:
    fits_df = Table(hdul[1].data).to_pandas()
fits_df.columns = [c.upper() for c in fits_df.columns]
fits_df['CATID'] = fits_df['CATID'].astype(str).str.strip().astype(int)

SFR_COL = next((c for c in fits_df.columns if 'SFR' in c), None)
if SFR_COL is None:
    raise KeyError(f'No SFR column found. Columns: {list(fits_df.columns)}')
print(f"  Using SFR column: {SFR_COL}")
sfr_slim = fits_df[['CATID', SFR_COL]].rename(columns={SFR_COL: 'SFR_raw'})

# =============================================================================
# BUILD MSTAR FROM CONCENTRATION (SF priority, composite fallback)
# =============================================================================

print('\nBuilding Mstar (SF priority, composite fallback)...')
conc['Mstar'] = conc['Mstar']   # SF value
fallback_mask = conc['Mstar'].isna() & conc['Mstar_com'].notna()
conc.loc[fallback_mask, 'Mstar'] = conc.loc[fallback_mask, 'Mstar_com']
n_fallback = fallback_mask.sum()
print(f"  {n_fallback} galaxies used composite Mstar as fallback")

# =============================================================================
# EXTRACT RELEVANT COLUMNS FROM EACH CATALOGUE
# =============================================================================

# Gini: just the metric column needed (shared cols come from concentration)
gini_slim = gini[['CATID', 'gini_all_halpha']].copy()

# Concentration: metric + all shared columns + Mstar
conc_slim = conc[['CATID', 'CIndex', 'Mstar'] + SHARED_COLS].copy()

# Asymmetry: just the metric column needed
asym_slim = asym[['CATID', 'A_min']].copy()

# Smoothness: just the metric column needed
smooth_slim = smooth[['CATID', 'S']].copy()

# =============================================================================
# MERGE
# =============================================================================

print('\nMerging catalogues...')

# Start from concentration as the anchor (priority source for shared cols)
master = conc_slim.copy()
print(f"  Anchor (concentration) : {len(master)} galaxies")

# Merge Gini
master = master.merge(gini_slim, on='CATID', how='left')
n_gini = master['gini_all_halpha'].notna().sum()
print(f"  Gini matched           : {n_gini} / {len(master)}")

# Merge Asymmetry
master = master.merge(asym_slim, on='CATID', how='left')
n_asym = master['A_min'].notna().sum()
print(f"  Asymmetry matched      : {n_asym} / {len(master)}")

# Merge Smoothness
master = master.merge(smooth_slim, on='CATID', how='left')
n_smooth = master['S'].notna().sum()
print(f"  Smoothness matched     : {n_smooth} / {len(master)}")

# Merge SFR
master = master.merge(sfr_slim, on='CATID', how='left')
n_sfr = master['SFR_raw'].notna().sum()
print(f"  SFR matched            : {n_sfr} / {len(master)}")

# =============================================================================
# COMPUTE SFR / sSFR (RAW AND LOG)
# =============================================================================

print('\nComputing SFR and sSFR columns...')

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

# Log Mstar for sSFR computation
log_mstar = to_log(master['Mstar'], 'Mstar')
log_sfr   = to_log(master['SFR_raw'], 'SFR')

# Raw SFR and sSFR
# If SFR_raw is already logged: convert back to linear for raw columns
if master['SFR_raw'].dropna().median() <= 100:
    # already logged
    master['SFR']    = 10 ** master['SFR_raw']
    master['log_SFR'] = master['SFR_raw']
else:
    master['SFR']    = master['SFR_raw']
    master['log_SFR'] = np.log10(master['SFR_raw'].replace(0, np.nan))

# Mstar linear (for sSFR denominator)
if master['Mstar'].dropna().median() <= 100:
    # already logged
    mstar_linear = 10 ** master['Mstar']
else:
    mstar_linear = master['Mstar']

master['sSFR']    = master['SFR'] / mstar_linear
master['log_sSFR'] = np.log10(master['sSFR'].replace(0, np.nan))

# Drop intermediate raw SFR column
master = master.drop(columns=['SFR_raw'])

# =============================================================================
# FINAL COLUMN ORDER
# =============================================================================

FINAL_COLS = [
    'CATID',
    'gini_all_halpha',
    'CIndex',
    'A_min',
    'S',
    'spectral_class',
    'environment',
    'galclass',
    're',
    'ellip',
    'z',
    'g_i',
    'Mstar',
    'SFR',
    'log_SFR',
    'sSFR',
    'log_sSFR',
    'seeing',
    'seeing_re',
    'source_file',
    'conflict_flag',
]

# Check all expected columns are present
missing = [c for c in FINAL_COLS if c not in master.columns]
if missing:
    raise KeyError(f'Missing expected columns in master: {missing}')

master = master[FINAL_COLS]

# =============================================================================
# SAVE
# =============================================================================

master.to_csv(OUT_PATH, index=False)

# =============================================================================
# SUMMARY
# =============================================================================

print(f'\n{"="*55}')
print(f'  MASTER CSV BUILT SUCCESSFULLY')
print(f'{"="*55}')
print(f'  Total galaxies         : {len(master)}')
print(f'  Gini values present    : {master["gini_all_halpha"].notna().sum()}')
print(f'  CIndex values present  : {master["CIndex"].notna().sum()}')
print(f'  A_min values present   : {master["A_min"].notna().sum()}')
print(f'  S values present       : {master["S"].notna().sum()}')
print(f'  SFR values present     : {master["SFR"].notna().sum()}')
print(f'  All 4 metrics present  : '
      f'{master[["gini_all_halpha","CIndex","A_min","S"]].notna().all(axis=1).sum()}')
print(f'  Saved to               : {OUT_PATH}')
print(f'{"="*55}\n')


if __name__ == '__main__':
    pass