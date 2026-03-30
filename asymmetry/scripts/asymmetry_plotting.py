"""
Asymmetry Plotting Script
==========================
Produces three plots:
  1. log(M*) vs log(sSFR) coloured by A_min
  2. log(M*) vs log(SFR)  coloured by A_min
  3. Gini (Hα) vs A_min   coloured by spectral class (SF / Composite)

M* is taken directly from the classified asymmetry CSV (Mstar_txt).
SFR is pulled from the FITS catalogue and merged on CATID.
Gini values are pulled from compare_gini.csv and merged on CATID.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table

# ── Configuration ──────────────────────────────────────────────────────────────

ASYMMETRY_CSV   = 'asymmetry/outputs/tables/asymmetry_results.csv'
COMPARE_GINI_CSV = 'gini/gini_outputs/tables/compare_gini.csv'
FITS_CATALOG    = 'data/catalogs/AR2022_SAMI_DR3_Mstar_SFR_GSWL_rescaled.fits'
OUTDIR          = 'asymmetry/outputs/figures'

os.makedirs(OUTDIR, exist_ok=True)

# Spectral class colours used across all plots
CLASS_COLOURS = {
    'SF'           : '#2196F3',  # blue
    'Composite'    : '#FF9800',  # orange
    'Unclassified' : '#9E9E9E',  # grey
}

# ── Load asymmetry catalogue ───────────────────────────────────────────────────

print('Loading asymmetry catalogue...')
asym = pd.read_csv(ASYMMETRY_CSV)
asym['catid'] = asym['catid'].astype(int)

# ── Pull SFR from FITS catalogue ───────────────────────────────────────────────

print('Loading FITS catalogue...')
with fits.open(FITS_CATALOG) as hdul:
    fits_df = Table(hdul[1].data).to_pandas()

fits_df.columns = [c.upper() for c in fits_df.columns]

catid_col = next((c for c in fits_df.columns if 'CATID' in c), None)
if catid_col is None:
    raise KeyError(f'No CATID column found in FITS catalogue. '
                   f'Available columns: {list(fits_df.columns)}')
fits_df = fits_df.rename(columns={catid_col: 'CATID'})
fits_df['CATID'] = fits_df['CATID'].astype(int)
print(f'Asymmetry columns: {list(asym.columns)}')

# Auto-detect SFR column
SFR_COL = next(
    (c for c in fits_df.columns if 'SFR' in c),
    None
)
if SFR_COL is None:
    raise KeyError(f'No SFR column found. Available columns: {list(fits_df.columns)}')
print(f'Using SFR column: {SFR_COL}')

fits_slim = fits_df[['CATID', SFR_COL]].rename(columns={SFR_COL: 'sfr_raw'})

# ── Merge SFR into asymmetry catalogue ────────────────────────────────────────

df = asym.merge(fits_slim, left_on='catid', right_on='CATID', how='left').drop(columns='CATID')
print(f'Asymmetry rows     : {len(asym)}')
print(f'Matched to FITS    : {df["sfr_raw"].notna().sum()}')

# ── Compute log columns ────────────────────────────────────────────────────────

def to_log(series, name):
    """Convert to log10 if values appear to be raw (median > 100), else use as-is."""
    valid = series.dropna()
    if valid.empty:
        return series
    if valid.median() > 100:
        print(f'  {name}: raw values detected — applying log10')
        return np.log10(series.replace(0, np.nan))
    else:
        print(f'  {name}: already in log form — using as-is')
        return series

print('\nChecking log/raw status:')
df['log_mstar'] = to_log(df['Mstar_txt'], 'M*')
df['log_sfr']   = to_log(df['sfr_raw'],   'SFR')
df['log_ssfr']  = df['log_sfr'] - df['log_mstar']

# ── Load compare_gini and merge ────────────────────────────────────────────────

print('\nLoading compare_gini catalogue...')
gini_df   = pd.read_csv(COMPARE_GINI_CSV)
gini_catid = next((c for c in gini_df.columns if c.lower() == 'catid'), None)
if gini_catid is None:
    raise KeyError(f'No CATID column in compare_gini. Columns: {list(gini_df.columns)}')

gini_df[gini_catid] = gini_df[gini_catid].astype(int)
gini_slim = gini_df[[gini_catid, 'gini_all_halpha']].rename(
    columns={gini_catid: 'CATID', 'gini_all_halpha': 'gini_compare'}
)

df = df.merge(gini_slim, left_on='catid', right_on='CATID', how='left').drop(columns='CATID')
print(f'Matched to Gini CSV: {df["gini_compare"].notna().sum()}')

# ── Plot 1: log(M*) vs log(sSFR) coloured by A_min ────────────────────────────

def plot_mstar_vs_y_coloured_A(x, y, A, xlabel, ylabel, title, outpath):
    valid = x.notna() & y.notna() & A.notna()
    vmax  = np.nanpercentile(A[valid], 95)

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        x[valid], y[valid], c=A[valid],
        cmap='plasma', vmin=0, vmax=vmax,
        s=20, alpha=0.8, linewidths=0
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r'$A_\mathrm{min}$', fontsize=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title,   fontsize=11)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved → {outpath}')

plot_mstar_vs_y_coloured_A(
    x       = df['log_mstar'],
    y       = df['log_ssfr'],
    A       = df['A_min'],
    xlabel  = r'$\log(M_* / M_\odot)$',
    ylabel  = r'$\log(\mathrm{sSFR} / \mathrm{yr}^{-1})$',
    title   = r'$\log(M_*)$ vs $\log(\mathrm{sSFR})$ — coloured by $A_\mathrm{min}$',
    outpath = os.path.join(OUTDIR, 'asymmetry_mstar_vs_ssfr.png'),
)

# ── Plot 2: log(M*) vs log(SFR) coloured by A_min ─────────────────────────────

plot_mstar_vs_y_coloured_A(
    x       = df['log_mstar'],
    y       = df['log_sfr'],
    A       = df['A_min'],
    xlabel  = r'$\log(M_* / M_\odot)$',
    ylabel  = r'$\log(\mathrm{SFR} / M_\odot \, \mathrm{yr}^{-1})$',
    title   = r'$\log(M_*)$ vs $\log(\mathrm{SFR})$ — coloured by $A_\mathrm{min}$',
    outpath = os.path.join(OUTDIR, 'asymmetry_mstar_vs_sfr.png'),
)

# ── Plot 3: Gini vs A_min coloured by spectral class ──────────────────────────

valid = df['gini_compare'].notna() & df['A_min'].notna() & df['spectral_class'].notna()
plot_df = df[valid].copy()

fig, ax = plt.subplots(figsize=(7, 5))

for spec_class, colour in CLASS_COLOURS.items():
    subset = plot_df[plot_df['spectral_class'] == spec_class]
    if subset.empty:
        continue
    ax.scatter(
        subset['gini_compare'], subset['A_min'],
        c=colour, label=spec_class,
        s=20, alpha=0.8, linewidths=0
    )

ax.set_xlabel(r'Gini coefficient $G$',        fontsize=11)
ax.set_ylabel(r'Asymmetry $A_\mathrm{min}$',  fontsize=11)
ax.set_title('Gini vs Asymmetry — by spectral class', fontsize=11)
ax.legend(title='Spectral class', fontsize=9, title_fontsize=9)
plt.tight_layout()
outpath3 = os.path.join(OUTDIR, 'gini_vs_asymmetry.png')
plt.savefig(outpath3, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved → {outpath3}')