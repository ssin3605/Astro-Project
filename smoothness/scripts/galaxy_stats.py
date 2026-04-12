"""
Prints key parameters for a given CATID to the terminal.

Outputs: CATID, log(M*), log(sSFR), log(SFR), G/A/C/S, Environment
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

# CONFIGURATION

CATID          = 617665   # <-- change this to look up a different galaxy

# Call whichever one is needed at the time
SMOOTHNESS_CSV = 'smoothness/outputs/tables/smoothness_results.csv'
FITS_CATALOG   = 'data/catalogs/AR2022_SAMI_DR3_Mstar_SFR_GSWL_rescaled.fits'

# LOG CONVERSION

def to_log(series, name):
    valid = series.dropna()
    if valid.empty:
        return series
    if valid.median() > 100:
        return np.log10(series.replace(0, np.nan))
    return series

# LOAD DATA

# Relevant catalogue
smooth = pd.read_csv(SMOOTHNESS_CSV)
smooth['CATID'] = smooth['CATID'].astype(int)

# SFR from FITS catalogue
with fits.open(FITS_CATALOG) as hdul:
    fits_df = Table(hdul[1].data).to_pandas()
fits_df.columns = [c.upper() for c in fits_df.columns]
fits_df['CATID'] = fits_df['CATID'].astype(int)

SFR_COL  = next((c for c in fits_df.columns if 'SFR' in c), None)
if SFR_COL is None:
    raise KeyError(f'No SFR column found. Columns: {list(fits_df.columns)}')

fits_slim = fits_df[['CATID', SFR_COL]].rename(columns={SFR_COL: 'sfr_raw'})

# Merge
df = smooth.merge(fits_slim, on='CATID', how='left')
df['log_mstar'] = to_log(df['Mstar'],   'Mstar')
df['log_sfr']   = to_log(df['sfr_raw'], 'SFR')
df['log_ssfr']  = df['log_sfr'] - df['log_mstar']

# LOOKUP AND PRINT

row = df[df['CATID'] == CATID]

divider = '=' * 40

if row.empty:
    print(f'\n{divider}')
    print(f'  CATID {CATID} not found in catalogue.')
    print(f'{divider}\n')
else:
    r = row.iloc[0]
    print(f'\n{divider}')
    print(f'  CATID       : {int(r["CATID"])}')
    print(f'  log(M*)     : {r["log_mstar"]:.3f}')
    print(f'  log(sSFR)   : {r["log_ssfr"]:.3f}')
    print(f'  log(SFR)    : {r["log_sfr"]:.3f}')
    print(f'  S           : {r["S"]:.4f}')
    print(f'  Environment : {r["environment"]}')
    print(f'{divider}\n')