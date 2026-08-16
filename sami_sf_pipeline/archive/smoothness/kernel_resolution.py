"""
smoothness_psf_kernel.py
------------------------
Visualises the relationship between the r_e-dependent smoothing kernel sigma
and the per-galaxy PSF FWHM across the full sample, for three kernel scale
factors (0.25, 0.35, 0.50 * r_e).

This addresses the question: for galaxies of different physical sizes, is the
chosen smoothing kernel safely above the PSF resolution floor, or does it fall
below it for smaller/more distant galaxies?

Galaxies with missing seeing measurements are plotted as hollow markers so
they remain visible in the distribution without being silently dropped.

Points are colour-coded by SFR.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

CSV_PATH      = "concentration/outputs/tables/concentration_results.csv"
OUT_DIR       = "smoothness/outputs/figures"

# Kernel scales to plot (as fractions of r_e)
KERNEL_SCALES = [0.25, 0.35, 0.50]

# Plate scale (arcsec/px) — used only if you want sigma in pixels on a twin axis
# Confirmed from FITS header in smoothness_test_02.py
PLATE_SCALE   = 0.5   # arcsec/px

# PSF FWHM -> sigma conversion: sigma = FWHM / (2 * sqrt(2 * ln(2)))
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

# Plotting
CMAP          = "plasma"   # colourmap for SFR
ALPHA_FILLED  = 0.7        # opacity for galaxies with valid seeing
ALPHA_HOLLOW  = 0.9        # opacity for galaxies with missing seeing
POINT_SIZE    = 18

RE_MAX_PERC  = 98    # clip x-axis at this percentile of r_e
SIGMA_MAX_PERC = 98  # clip y-axis similarly

os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

def load_catalogue(csv_path):
    """
    Load concentration_results.csv and compute per-kernel sigma values.

    For each kernel scale ks, the smoothing sigma in arcseconds is:
        sigma_arcsec = ks * r_e

    The PSF sigma in arcseconds is derived from the seeing FWHM:
        psf_sigma = seeing * FWHM_TO_SIGMA

    Galaxies with missing seeing are flagged separately.

    Returns
    -------
    df_valid   : DataFrame, galaxies with valid seeing measurements
    df_missing : DataFrame, galaxies with missing seeing measurements
    """
    df = pd.read_csv(csv_path)
    df['CATID'] = df['CATID'].astype(str)

    # Drop rows with missing r_e (cannot compute kernel at all)
    n_before = len(df)
    df = df.dropna(subset=['re'])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} galaxies with missing r_e.")

    # Flag and remove unphysical r_e values (failed profile fits)
    RE_MAX_PHYSICAL = 50.0   # arcseconds
    n_before = len(df)
    df_flagged = df[df['re'] > RE_MAX_PHYSICAL]
    if len(df_flagged) > 0:
        print(f"\nRemoving {len(df_flagged)} galaxies with unphysical r_e > {RE_MAX_PHYSICAL}\":")
        print(df_flagged[['CATID', 're', 'seeing_re', 'source_file']].to_string())
    df = df[df['re'] <= RE_MAX_PHYSICAL]
    print(f"Retained {len(df)} / {n_before} galaxies after r_e cut.")

    # Compute kernel sigma in arcseconds for each scale
    for ks in KERNEL_SCALES:
        col = f"sigma_{ks}".replace(".", "p")   # e.g. sigma_0p25
        df[col] = ks * df['re']

    for ks, r_crit in zip([0.25, 0.35, 0.50], [3.46, 2.47, 1.73]):
        n_limited = (df['re'] < r_crit).sum()
        pct = 100 * n_limited / len(df)
        print(f"ks={ks}: {n_limited} galaxies ({pct:.1f}%) below r_e_crit = {r_crit}\"")

    # Compute PSF sigma in arcseconds from seeing FWHM
    df['psf_sigma_arcsec'] = df['seeing'] * FWHM_TO_SIGMA

    # Split into valid and missing seeing
    df_valid   = df[df['seeing'].notna()].copy()
    df_missing = df[df['seeing'].isna()].copy()

    print(f"\nCatalogue loaded: {len(df)} galaxies with valid r_e")
    print(f"  With seeing measurement   : {len(df_valid)}")
    print(f"  Missing seeing measurement: {len(df_missing)}")

    return df_valid, df_missing


# =============================================================================
# PLOTTING
# =============================================================================

def make_psf_kernel_plot(df_valid, df_missing):
    """
    Three-panel figure, one panel per kernel scale.

    Each panel shows:
        x-axis : r_e in arcseconds (galaxy size)
        y-axis : smoothing kernel sigma in arcseconds
        Diagonal line : the kernel scale itself (sigma = ks * r_e)
        Horizontal band: PSF sigma range across the sample (median +/- std)
        Per-galaxy PSF sigma: plotted as small grey ticks on y-axis? No —
            instead a shaded band shows the PSF sigma distribution, and a
            dashed line shows the median PSF sigma.
        Points: colour-coded by SFR, hollow if seeing is missing.

    A galaxy is "safely resolved" if its kernel sigma > its PSF sigma.
    The diagonal line (kernel = ks * r_e) intersects the median PSF sigma
    line at a critical r_e below which galaxies become resolution-limited.
    """

    # SFR colour normalisation across full sample (valid + missing combined)
    all_sfr = pd.concat([df_valid['SFR'], df_missing['SFR']]).dropna()
    sfr_min = np.percentile(all_sfr, 2)
    sfr_max = np.percentile(all_sfr, 98)
    norm    = Normalize(vmin=sfr_min, vmax=sfr_max)
    sm      = ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])

    # PSF sigma statistics from valid sample
    psf_median = df_valid['psf_sigma_arcsec'].median()
    psf_std    = df_valid['psf_sigma_arcsec'].std()
    psf_min    = df_valid['psf_sigma_arcsec'].min()
    psf_max    = df_valid['psf_sigma_arcsec'].max()

    print(f"\nPSF sigma stats (arcsec):")
    print(f"  Median : {psf_median:.3f}")
    print(f"  Std    : {psf_std:.3f}")
    print(f"  Range  : {psf_min:.3f} – {psf_max:.3f}")

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(
        "Smoothing kernel σ vs galaxy size ($r_e$) relative to PSF\n"
        "Colour = SFR  |  Hollow markers = missing seeing measurement  |  "
        "Shaded band = PSF σ range across sample",
        fontsize=11, fontweight='bold'
    )

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32)

    # x-axis range: span the full r_e range with some padding
    re_all  = pd.concat([df_valid['re'], df_missing['re']])
    re_min  = 0
    re_max  = np.percentile(re_all, RE_MAX_PERC) + 1.0
    re_line = np.linspace(re_min, re_max, 300)   # for drawing the kernel line

    axes = []
    for i, ks in enumerate(KERNEL_SCALES):
        sigma_col = f"sigma_{ks}".replace(".", "p")

        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)

        # --- Shaded PSF sigma band (full range) ---
        ax.axhspan(psf_min, psf_max, color='dodgerblue', alpha=0.10,
                   label='PSF σ range')

        # --- Median PSF sigma line ---
        ax.axhline(psf_median, color='dodgerblue', lw=1.5, ls='--',
                   label=f'Median PSF σ = {psf_median:.2f}\"')

        # --- Kernel scale line: sigma = ks * r_e ---
        ax.plot(re_line, ks * re_line, color='black', lw=1.2, ls='-',
                label=f'σ = {ks} × $r_e$')

        # --- Critical r_e: where kernel = median PSF sigma ---
        # ks * r_e_crit = psf_median  =>  r_e_crit = psf_median / ks
        r_e_crit = psf_median / ks
        ax.axvline(r_e_crit, color='red', lw=1.0, ls=':',
                   label=f'$r_{{e,crit}}$ = {r_e_crit:.2f}\"')

        # --- Valid galaxies: filled markers, colour = SFR ---
        sfr_valid = df_valid['SFR'].values
        colors_valid = [
            sm.to_rgba(s) if not np.isnan(s) else 'grey'
            for s in sfr_valid
        ]
        ax.scatter(
            df_valid['re'], df_valid[sigma_col],
            c=colors_valid, s=POINT_SIZE, alpha=ALPHA_FILLED,
            edgecolors='none', zorder=3
        )

        # --- Missing seeing: hollow markers ---
        if len(df_missing) > 0:
            sfr_missing = df_missing['SFR'].values
            colors_missing = [
                sm.to_rgba(s) if not np.isnan(s) else 'grey'
                for s in sfr_missing
            ]
            ax.scatter(
                df_missing['re'], df_missing[sigma_col],
                c='none', s=POINT_SIZE * 1.5, alpha=ALPHA_HOLLOW,
                edgecolors=colors_missing, linewidths=0.8,
                zorder=3, label='Missing seeing'
            )

        # --- Annotations ---
        ax.set_xlabel("$r_e$ (arcsec)", fontsize=10)
        if i == 0:
            ax.set_ylabel("Smoothing kernel σ (arcsec)", fontsize=10)
        ax.set_title(f"Kernel scale = {ks} × $r_e$", fontsize=10,
                     fontweight='bold')
        ax.set_xlim(re_min, re_max)
        ax.set_ylim(0, re_max * ks * 1.2)   # re_max is now already clipped
        ax.legend(fontsize=7.5, loc='upper left')

        # Shade the resolution-limited region (below median PSF sigma)
        ax.fill_between(re_line, 0, ks * re_line,
                        where=(ks * re_line < psf_median),
                        color='red', alpha=0.07,
                        label='_nolegend_')

        # Secondary y-axis in pixels
        ax2 = ax.twinx()
        y1, y2 = ax.get_ylim()
        ax2.set_ylim(y1 / PLATE_SCALE, y2 / PLATE_SCALE)
        ax2.set_ylabel("σ (pixels)" if i == len(KERNEL_SCALES) - 1 else "",
                       fontsize=9)
        ax2.tick_params(labelsize=8)

    # Shared colorbar
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical',
                        fraction=0.02, pad=0.08)
    cbar.set_label("SFR ($M_\\odot$ yr$^{-1}$)", fontsize=10)

    out_path = os.path.join(OUT_DIR, "smoothness_psf_kernel_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {os.path.abspath(out_path)}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    df_valid, df_missing = load_catalogue(CSV_PATH)
    make_psf_kernel_plot(df_valid, df_missing)


if __name__ == "__main__":
    main()