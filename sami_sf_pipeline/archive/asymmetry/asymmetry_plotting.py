"""
Asymmetry Plotting Script
==========================
Produces three plots:
  1. log(M*) vs log(sSFR) coloured by A_min
  2. log(M*) vs log(SFR)  coloured by A_min
  3. Gini (Hα) vs A_min   coloured by spectral class (SF / Composite)

M* is taken directly from the classified asymmetry CSV (Mstar_txt).
SFR is pulled from the FITS catalogue and merged on catid.
Gini values are pulled from compare_gini.csv and merged on catid.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table

# ── Configuration ──────────────────────────────────────────────────────────────

ASYMMETRY_CSV    = 'asymmetry/outputs/tables/asymmetry_results.csv'
COMPARE_GINI_CSV = 'gini/gini_outputs/tables/compare_gini.csv'
FITS_CATALOG     = 'data/catalogs/AR2022_SAMI_DR3_Mstar_SFR_GSWL_rescaled.fits'
OUTDIR           = 'asymmetry/outputs/figures'

os.makedirs(OUTDIR, exist_ok=True)

CLASS_COLOURS = {
    'SF'           : '#2196F3',  # blue
    'Composite'    : '#FF9800',  # orange
    'Unclassified' : '#9E9E9E',  # grey
}

highlight_ids     = [9008500868, 422933, 105420, 39365]
highlight_markers = ['o', 's', '^', 'D']

# ── Load asymmetry catalogue ───────────────────────────────────────────────────

print('Loading asymmetry catalogue...')
asym = pd.read_csv(ASYMMETRY_CSV)
asym['catid'] = asym['catid'].astype(int)
print(f'  Columns: {list(asym.columns)}')

# ── Pull SFR from FITS catalogue ───────────────────────────────────────────────

print('\nLoading FITS catalogue...')
with fits.open(FITS_CATALOG) as hdul:
    fits_df = Table(hdul[1].data).to_pandas()

fits_df.columns = [c.upper() for c in fits_df.columns]

catid_col = next((c for c in fits_df.columns if 'CATID' in c), None)
if catid_col is None:
    raise KeyError(f'No CATID column in FITS catalogue. Columns: {list(fits_df.columns)}')

fits_df = fits_df.rename(columns={catid_col: 'catid'})
fits_df['catid'] = fits_df['catid'].astype(int)

SFR_COL = next((c for c in fits_df.columns if 'SFR' in c), None)
if SFR_COL is None:
    raise KeyError(f'No SFR column found. Columns: {list(fits_df.columns)}')
print(f'  Using SFR column: {SFR_COL}')

fits_slim = fits_df[['catid', SFR_COL]].rename(columns={SFR_COL: 'sfr_raw'})

# ── Merge SFR ─────────────────────────────────────────────────────────────────

df = asym.merge(fits_slim, on='catid', how='left')
print(f'  Asymmetry rows  : {len(asym)}')
print(f'  Matched to FITS : {df["sfr_raw"].notna().sum()}')

# ── Compute log columns ────────────────────────────────────────────────────────

def to_log(series, name):
    """Convert to log10 if values appear to be raw (median > 100), else use as-is."""
    valid = series.dropna()
    if valid.empty:
        return series
    if valid.median() > 100:
        print(f'  {name}: raw values — applying log10')
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
gini_df = pd.read_csv(COMPARE_GINI_CSV)

gini_catid_col = next((c for c in gini_df.columns if c.lower() == 'catid'), None)
if gini_catid_col is None:
    raise KeyError(f'No CATID column in compare_gini. Columns: {list(gini_df.columns)}')

gini_df = gini_df.rename(columns={gini_catid_col: 'catid'})
gini_df['catid'] = gini_df['catid'].astype(int)
gini_slim = gini_df[['catid', 'gini_all_halpha']].rename(
    columns={'gini_all_halpha': 'gini_compare'}
)

df = df.merge(gini_slim, on='catid', how='left')
print(f'  Matched to Gini CSV: {df["gini_compare"].notna().sum()}')

# ── Plotting helpers ───────────────────────────────────────────────────────────

def add_highlights(ax, x, y, df, highlight_ids, highlight_markers, annotate):
    """Overlay highlighted galaxies on an existing axis."""
    for i, catid in enumerate(highlight_ids):
        sel = df[df['catid'] == catid]
        if sel.empty:
            print(f'  WARNING: catid {catid} not found — skipping highlight')
            continue
        r_idx = sel.index[0]
        if pd.isna(x[r_idx]) or pd.isna(y[r_idx]):
            print(f'  WARNING: catid {catid} has NaN coordinates — skipping highlight')
            continue
        marker = highlight_markers[i % len(highlight_markers)]
        ax.scatter(x[r_idx], y[r_idx], s=180, marker=marker,
                   facecolors='none', edgecolors='k', linewidths=2.0,
                   zorder=5, label=str(catid))
        if annotate:
            ax.annotate(str(int(catid)), (x[r_idx], y[r_idx]),
                        textcoords='offset points', xytext=(6, 6),
                        fontsize=9, weight='bold', color='k')
    ax.legend(title='Highlighted CATID', frameon=True, loc='best',
              fontsize=8, title_fontsize=8)


def plot_mstar_vs_y_coloured_A(x, y, A, xlabel, ylabel, title, outpath,
                                highlight_ids=None, highlight_markers=None,
                                annotate=True):
    """Scatter plot of x vs y with A_min as the colour axis."""
    valid = x.notna() & y.notna() & A.notna()
    vmax  = np.nanpercentile(A[valid], 95)

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(x[valid], y[valid], c=A[valid],
                    cmap='plasma', vmin=0, vmax=vmax,
                    s=20, alpha=0.8, linewidths=0)
    plt.colorbar(sc, ax=ax).set_label(r'$A_\mathrm{min}$', fontsize=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title,   fontsize=11)

    if highlight_ids:
        add_highlights(ax, x, y, df, highlight_ids, highlight_markers, annotate)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved → {outpath}')

# ── Plot 1: log(M*) vs log(sSFR) coloured by A_min ────────────────────────────

plot_mstar_vs_y_coloured_A(
    x       = df['log_mstar'],
    y       = df['log_ssfr'],
    A       = df['A_min'],
    xlabel  = r'$\log(M_* / M_\odot)$',
    ylabel  = r'$\log(\mathrm{sSFR} / \mathrm{yr}^{-1})$',
    title   = r'$\log(M_*)$ vs $\log(\mathrm{sSFR})$ — coloured by $A_\mathrm{min}$',
    outpath = os.path.join(OUTDIR, 'asymmetry_mstar_vs_ssfr.png'),
    highlight_ids=highlight_ids, highlight_markers=highlight_markers, annotate=True,
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
    highlight_ids=highlight_ids, highlight_markers=highlight_markers, annotate=True,
)

# ── Plot 3: Gini vs A_min coloured by spectral class ──────────────────────────

valid   = df['gini_compare'].notna() & df['A_min'].notna() & df['spectral_class'].notna()
plot_df = df[valid].copy()

fig, ax = plt.subplots(figsize=(7, 5))

for spec_class, colour in CLASS_COLOURS.items():
    subset = plot_df[plot_df['spectral_class'] == spec_class]
    if subset.empty:
        continue
    ax.scatter(subset['gini_compare'], subset['A_min'],
               c=colour, label=spec_class, s=20, alpha=0.8, linewidths=0)

ax.set_xlabel(r'Gini coefficient $G$',       fontsize=11)
ax.set_ylabel(r'Asymmetry $A_\mathrm{min}$', fontsize=11)
ax.set_title('Gini vs Asymmetry — by spectral class', fontsize=11)
ax.legend(title='Spectral class', fontsize=9, title_fontsize=9)
plt.tight_layout()
outpath3 = os.path.join(OUTDIR, 'gini_vs_asymmetry.png')
plt.savefig(outpath3, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved → {outpath3}')