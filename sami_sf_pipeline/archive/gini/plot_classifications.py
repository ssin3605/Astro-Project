import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

## Use to create a 2x2 panel of gini, log(SFR), log(sSFR) and log(M*) analysis with Wang's classifcations

# ============================================================
# CONFIGURATION
# ============================================================

HALPHA_CSV    = 'gini/gini_outputs/tables/Halpha_gini.csv'
CONTINUUM_CSV = 'gini/gini_outputs/tables/continuum_gini.csv'
FIGURE_DIR    = Path('gini/gini_outputs/figures')
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Colourmap and Gini scale — shared across all panels
CMAP     = 'magma'
GINI_MIN = 0.0
GINI_MAX = 1.0

# Point styling
POINT_SIZE    = 18
POINT_ALPHA   = 0.6
UNCLASS_COLOR = '#aaaaaa'
UNCLASS_ALPHA = 0.3
UNCLASS_SIZE  = 12

# Spectral classes to show as columns (order matters — left to right)
SPEC_CLASSES = ['SF', 'Composite']

# ============================================================
# CORE PANEL FUNCTION
# ============================================================

def plot_gini_panel(ax, df, x_col, y_col, gini_col,
                    title, xlabel, ylabel,
                    show_unclassified=True):
    """
    Plot a single Gini scatter panel with:
      - Unclassified galaxies as grey background points
      - Classified galaxies coloured by Gini value
      - Individual colourbar attached to the axis
    Returns the scatter object for colourbar construction.
    """
    norm = mcolors.Normalize(vmin=GINI_MIN, vmax=GINI_MAX)

    # --- Layer 1: Unclassified galaxies (grey, behind) ---
    if show_unclassified:
        unclass = df[df['spectral_class'] == 'Unclassified']
        if len(unclass) > 0:
            ax.scatter(
                unclass[x_col], unclass[y_col],
                c=UNCLASS_COLOR,
                s=UNCLASS_SIZE,
                alpha=UNCLASS_ALPHA,
                linewidths=0,
                zorder=1,
                label=f'Unclassified (n={len(unclass)})'
            )

    # --- Layer 2: Classified galaxies (coloured by Gini) ---
    classified = df[df['spectral_class'] != 'Unclassified']
    sc = ax.scatter(
        classified[x_col], classified[y_col],
        c=classified[gini_col],
        cmap=CMAP,
        norm=norm,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        linewidths=0,
        zorder=2
    )

    # --- Colourbar ---
    cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label('Gini', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # --- Labels and title ---
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title,   fontsize=11, fontweight='bold')
    ax.tick_params(labelsize=9)

    # --- Galaxy count annotation ---
    n_classified = len(classified)
    ax.annotate(
        f'n = {n_classified}',
        xy=(0.03, 0.96),
        xycoords='axes fraction',
        fontsize=8,
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='none')
    )

    return sc

# ============================================================
# 2x2 FIGURE BUILDER
# ============================================================

def build_2x2_figure(csv_path, emission_label):
    """
    Build a 2x2 panel figure for one emission type.

    Layout:
      Row 0: Gini vs log(SFR) | y=logSFR,  x=logMstar
      Row 1: Gini vs log(sSFR)| y=logsSFR, x=logMstar
      Col 0: SF galaxies
      Col 1: Composite galaxies
    """

    df = pd.read_csv(csv_path)

    # Verify required columns exist
    required = ['catid', 'logSFR', 'logMstar', 'logsSFR',
                'gini_all', 'spectral_class']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns in {csv_path}: {missing}')

    # Remove rows with NaN or -inf in any plotted column
    plot_cols = ['logSFR', 'logMstar', 'logsSFR', 'gini_all']
    df_clean = df.replace(-np.inf, np.nan).dropna(subset=plot_cols).copy()

    print(f'\n  {emission_label}: {len(df_clean)} / {len(df)} rows after '
          f'dropping NaN in plot columns')
    print(f'  spectral_class breakdown:')
    print(df_clean['spectral_class'].value_counts().to_string())

    # ---- Figure setup ----
    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 9),
        constrained_layout=True
    )

    fig.suptitle(
        f'{emission_label} Gini — SF vs Composite\n'
        f'(n = {len(df_clean)} galaxies)',
        fontsize=13,
        fontweight='bold',
        y=1.02
    )

    # ---- Row and column definitions ----
    rows = [
        ('logSFR',  r'$\log(\mathrm{SFR})\ [M_\odot\ \mathrm{yr}^{-1}]$', 'Plot A'),
        ('logsSFR', r'$\log(\mathrm{sSFR})\ [\mathrm{yr}^{-1}]$',          'Plot B'),
    ]

    xlabel = r'$\log(M_*)\ [M_\odot]$'

    # ---- Plot each panel ----
    for row_idx, (y_col, ylabel, row_label) in enumerate(rows):
        for col_idx, spec_class in enumerate(SPEC_CLASSES):

            ax = axes[row_idx, col_idx]

            panel_df = df_clean[
                df_clean['spectral_class'].isin([spec_class, 'Unclassified'])
            ].copy()

            plot_gini_panel(
                ax       = ax,
                df       = panel_df,
                x_col    = 'logMstar',
                y_col    = y_col,
                gini_col = 'gini_all',
                title    = f'{row_label} — {spec_class}',
                xlabel   = xlabel,
                ylabel   = ylabel,
            )

    # ---- Save ----
    for fmt in ['pdf', 'png']:
        out_path = FIGURE_DIR / f'{emission_label}_gini_2x2.{fmt}'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'  Saved: {out_path}')

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':

    for emission_label, csv_path in [
        ('Halpha',    HALPHA_CSV),
        ('Continuum', CONTINUUM_CSV),
    ]:
        print(f'\n{"="*55}')
        print(f'  Building figures for: {emission_label}')
        print(f'{"="*55}')

        build_2x2_figure(csv_path, emission_label)

    print('\nDone.')