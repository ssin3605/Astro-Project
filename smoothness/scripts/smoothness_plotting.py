"""
Plots produced:
    1. log(M*) vs log(sSFR) coloured by S
    2. log(M*) vs log(SFR)  coloured by S
    3. Gini vs S coloured by spectral class

Data sources:
    - S, Mstar, spectral_class : smoothness_results.csv
    - SFR                      : AR2022_SAMI_DR3_Mstar_SFR_GSWL_rescaled.fits
    - Gini                     : compare_gini.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table

# =============================================================================
# CONFIGURATION
# =============================================================================

SMOOTHNESS_CSV   = 'smoothness/outputs/tables/smoothness_results.csv'
COMPARE_GINI_CSV = 'gini/gini_outputs/tables/compare_gini.csv'
FITS_CATALOG     = 'data/catalogs/AR2022_SAMI_DR3_Mstar_SFR_GSWL_rescaled.fits'
OUTDIR           = 'smoothness/outputs/figures'

os.makedirs(OUTDIR, exist_ok=True)

# Spectral class colours
CLASS_COLOURS = {
    'SF'          : '#2196F3',   # blue
    'composite'   : '#FF9800',   # orange
    'Unclassified': '#9E9E9E',   # grey
}

highlight_ids = [9008500868, 422933, 105420, 39365]  # 4 selected catids
highlight_markers = ["o", "s", "^", "D"]         # circle, square, triangle, diamond

# =============================================================================
# LOAD SMOOTHNESS CATALOGUE
# =============================================================================

print('Loading smoothness catalogue...')
smooth = pd.read_csv(SMOOTHNESS_CSV)
smooth['CATID'] = smooth['CATID'].astype(int)
print(f'  Rows loaded: {len(smooth)}')
print(f'  Columns: {list(smooth.columns)}')

# =============================================================================
# LOAD AND MERGE SFR FROM FITS CATALOGUE
# =============================================================================

print('\nLoading FITS catalogue...')
with fits.open(FITS_CATALOG) as hdul:
    fits_df = Table(hdul[1].data).to_pandas()

fits_df.columns = [c.upper() for c in fits_df.columns]

# Confirm CATID column exists
if 'CATID' not in fits_df.columns:
    raise KeyError(
        f'No CATID column found in FITS catalogue. '
        f'Available columns: {list(fits_df.columns)}'
    )
fits_df['CATID'] = fits_df['CATID'].astype(int)

# Auto-detect SFR column
SFR_COL = next((c for c in fits_df.columns if 'SFR' in c), None)
if SFR_COL is None:
    raise KeyError(
        f'No SFR column found. Available columns: {list(fits_df.columns)}'
    )
print(f'  Using SFR column: {SFR_COL}')

fits_slim = fits_df[['CATID', SFR_COL]].rename(columns={SFR_COL: 'sfr_raw'})

# Merge SFR into smoothness catalogue — CATID retained throughout
df = smooth.merge(fits_slim, on='CATID', how='left')
print(f'  Smoothness rows    : {len(smooth)}')
print(f'  Matched to FITS    : {df["sfr_raw"].notna().sum()}')

# =============================================================================
# COMPUTE LOG COLUMNS
# =============================================================================

def to_log(series, name):
    """Convert to log10 if values appear to be raw (median > 100), else use as-is."""
    valid = series.dropna()
    if valid.empty:
        print(f'  {name}: no valid values — skipping')
        return series
    if valid.median() > 100:
        print(f'  {name}: raw values detected (median = {valid.median():.2f}) '
              f'— applying log10')
        return np.log10(series.replace(0, np.nan))
    else:
        print(f'  {name}: already in log form (median = {valid.median():.2f}) '
              f'— using as-is')
        return series

print('\nChecking log/raw status:')
df['log_mstar'] = to_log(df['Mstar'],    'Mstar')
df['log_sfr']   = to_log(df['sfr_raw'],  'SFR')
df['log_ssfr']  = df['log_sfr'] - df['log_mstar']

# =============================================================================
# LOAD AND MERGE GINI CATALOGUE
# =============================================================================

print('\nLoading compare_gini catalogue...')
gini_df = pd.read_csv(COMPARE_GINI_CSV)
print(f'  Columns: {list(gini_df.columns)}')

# Confirm CATID and gini column exist
if 'catid' not in gini_df.columns:
    raise KeyError(
        f'No catid column in compare_gini. '
        f'Available columns: {list(gini_df.columns)}'
    )
if 'gini_all_halpha' not in gini_df.columns:
    raise KeyError(
        f'No gini_all_halpha column in compare_gini. '
        f'Available columns: {list(gini_df.columns)}'
    )

gini_df = gini_df.rename(columns={'catid': 'CATID'})
gini_df['CATID'] = gini_df['CATID'].astype(int)
gini_slim = gini_df[['CATID', 'gini_all_halpha']].rename(
    columns={'gini_all_halpha': 'gini_compare'}
)

# Merge Gini — CATID still retained
df = df.merge(gini_slim, on='CATID', how='left')
print(f'  Matched to Gini CSV: {df["gini_compare"].notna().sum()}')

# Report spectral classes present — useful for catching capitalisation mismatches
print(f'\nSpectral classes in data: {df["spectral_class"].unique()}')

# =============================================================================
# PLOT 1 & 2: log(M*) vs log(sSFR or SFR) coloured by S
# =============================================================================

def plot_mstar_vs_y_coloured_S(x, y, S, xlabel, ylabel, title, outpath,
                                highlight_ids=None, highlight_markers=None,
                                annotate=True):
    valid = x.notna() & y.notna() & S.notna()
    vmax  = np.nanpercentile(S[valid], 95)

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        x[valid], y[valid], c=S[valid],
        cmap='plasma', vmin=0, vmax=vmax,
        s=20, alpha=0.8, linewidths=0
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r'Clumpiness $S$', fontsize=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title,   fontsize=11)

    if highlight_ids is not None and len(highlight_ids) > 0:
        if highlight_markers is None:
            highlight_markers = ["o", "s", "^", "D"]

        for i, catid in enumerate(highlight_ids):
            sel = df[df["CATID"] == catid]
            if sel.empty:
                print(f"  WARNING: CATID {catid} not found — skipping highlight")
                continue

            m     = highlight_markers[i % len(highlight_markers)]
            r_idx = sel.index[0]

            ax.scatter(
                x[r_idx], y[r_idx],
                s=180, marker=m,
                facecolors='none', edgecolors='k',
                linewidths=2.0, zorder=5,
                label=str(catid)
            )

            if annotate:
                ax.annotate(
                    str(int(catid)),
                    (x[r_idx], y[r_idx]),
                    textcoords='offset points',
                    xytext=(6, 6),
                    fontsize=9, weight='bold', color='k'
                )

        ax.legend(title='Highlighted CATID', frameon=True, loc='best',
                  fontsize=8, title_fontsize=8)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved → {outpath}')

plot_mstar_vs_y_coloured_S(
    x       = df['log_mstar'],
    y       = df['log_ssfr'],
    S       = df['S'],
    xlabel  = r'$\log(M_* / M_\odot)$',
    ylabel  = r'$\log(\mathrm{sSFR} / \mathrm{yr}^{-1})$',
    title   = r'$\log(M_*)$ vs $\log(\mathrm{sSFR})$ — coloured by $S$',
    outpath = os.path.join(OUTDIR, 'smoothness_mstar_vs_ssfr.png'),
    highlight_ids=highlight_ids,
    highlight_markers=highlight_markers,
    annotate=True
)

plot_mstar_vs_y_coloured_S(
    x       = df['log_mstar'],
    y       = df['log_sfr'],
    S       = df['S'],
    xlabel  = r'$\log(M_* / M_\odot)$',
    ylabel  = r'$\log(\mathrm{SFR} / M_\odot \, \mathrm{yr}^{-1})$',
    title   = r'$\log(M_*)$ vs $\log(\mathrm{SFR})$ — coloured by $S$',
    outpath = os.path.join(OUTDIR, 'smoothness_mstar_vs_sfr.png'),
    highlight_ids=highlight_ids,
    highlight_markers=highlight_markers,
    annotate=True
)

# =============================================================================
# PLOT 3: Gini vs S coloured by spectral class
# =============================================================================

valid   = df['gini_compare'].notna() & df['S'].notna() & df['spectral_class'].notna()
plot_df = df[valid].copy()

fig, ax = plt.subplots(figsize=(7, 5))

for spec_class, colour in CLASS_COLOURS.items():
    subset = plot_df[plot_df['spectral_class'] == spec_class]
    if subset.empty:
        continue
    ax.scatter(
        subset['gini_compare'], subset['S'],
        c=colour, label=spec_class,
        s=20, alpha=0.8, linewidths=0
    )

ax.set_xlabel(r'Gini coefficient $G$', fontsize=11)
ax.set_ylabel(r'Clumpiness $S$',       fontsize=11)
ax.set_title('Gini vs Smoothness — by spectral class', fontsize=11)
ax.legend(title='Spectral class', fontsize=9, title_fontsize=9)
plt.tight_layout()
outpath3 = os.path.join(OUTDIR, 'gini_vs_smoothness.png')
plt.savefig(outpath3, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved → {outpath3}')