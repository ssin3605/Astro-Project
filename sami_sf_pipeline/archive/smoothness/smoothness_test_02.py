"""
smoothness_test_02.py
---------------------
Second iteration of the Clumpiness (S) parameter script for SAMI Halpha maps.

Changes from smoothness_test_01.py:
  - Smoothing kernel is now tied to the galaxy's effective radius (r_e) sourced
    from concentration_results.csv
  - Iterates over four kernel scale factors (0.10, 0.25, 0.35, 0.50 * r_e)
    to explore the effect of kernel choice on the residual map and S value
  - r_e converted from arcseconds to pixels using the plate scale read from
    the FITS header (CDELT1), avoiding hardcoded instrument parameters
  - Relevant galaxy properties (SFR, Mstar, environment) printed in plot title
    for physical context when interpreting results

Still omitting:
  - Background subtraction (B_S)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits
import pandas as pd
from scipy.ndimage import gaussian_filter

# =============================================================================
# CONFIGURATION
# =============================================================================

GALAXY_ID   = "23117"    # CATID 
FITS_PATH   = "gini/gini_outputs/tables/masked_halpha_maps/23117_A_Halpha_masked.fits"
CSV_PATH    = "concentration/outputs/tables/concentration_results.csv"
OUT_DIR     = "smoothness/outputs/figures"

# Kernel scale factors to iterate over (as fractions of r_e)
KERNEL_SCALES = [0.10, 0.25, 0.35, 0.50]

# Plotting
CMAP      = "inferno"
VMIN_PERC = 2
VMAX_PERC = 98

os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================

def load_halpha(fits_path):
    """
    Load Halpha flux and error maps from a SAMI masked FITS file.

    Also reads CDELT1 from the header to determine the plate scale
    (arcseconds per pixel), handling both degree and arcsecond storage
    conventions.
    """
    with fits.open(fits_path) as hdul:
        header     = hdul[0].header
        flux       = hdul[0].data[1, :, :]
        err        = hdul[1].data[1, :, :]

        # Read plate scale from header
        cdelt = abs(header.get('CDELT1', None))
        if cdelt is None:
            raise KeyError("CDELT1 not found in FITS header — cannot convert r_e to pixels.")
        # CDELT1 is often stored in degrees; convert to arcseconds if so
        if cdelt < 0.01:
            plate_scale = cdelt * 3600.0   # degrees -> arcseconds per pixel
        else:
            plate_scale = cdelt            # already in arcseconds per pixel

    print(f"Loaded: {fits_path}")
    print(f"  Flux shape : {flux.shape}")
    print(f"  Plate scale: {plate_scale:.4f} arcsec/px")
    return flux, err, plate_scale


def load_galaxy_params(csv_path, galaxy_id):
    """
    Load r_e and other physical parameters for galaxy from
    concentration_results.csv.
    """
    df = pd.read_csv(csv_path)

    # Ensure CATID is string for matching
    df['CATID'] = df['CATID'].astype(str)
    galaxy_id   = str(galaxy_id)

    row = df[df['CATID'] == galaxy_id]
    if row.empty:
        raise ValueError(f"Galaxy {galaxy_id} not found in {csv_path}.")
    if len(row) > 1:
        print(f"Warning: {len(row)} rows found for CATID {galaxy_id} — using first.")
    row = row.iloc[0]

    params = {
        're'            : float(row['re']),
        'seeing_re'     : float(row['seeing_re']),
        'SFR'           : row['SFR'],
        'Mstar'         : row['Mstar'],
        'environment'   : row['environment'],
        'spectral_class': row['spectral_class'],
        'ha_EW'         : row['ha_EW'],
        'CIndex'        : row['CIndex'],
    }

    print(f"\nGalaxy {galaxy_id} parameters from catalogue:")
    print(f"  r_e          = {params['re']:.3f} arcsec")
    print(f"  seeing r_e   = {params['seeing_re']:.3f} arcsec")
    print(f"  SFR          = {params['SFR']}")
    print(f"  log Mstar    = {params['Mstar']}")
    print(f"  Environment  = {params['environment']}")
    print(f"  Spectral class = {params['spectral_class']}")
    return params

# =============================================================================
# SMOOTHING
# =============================================================================

def conselice_smooth(flux, sigma_px):
    """
    Smooth a flux map with a Gaussian kernel of width sigma_px.
    sigma_px is pre-computed as kernel_scale * r_e_pixels.
    """
    return gaussian_filter(flux, sigma=sigma_px)


# =============================================================================
# RESIDUAL AND PARTIAL S
# =============================================================================

def compute_residual(flux, flux_smoothed):
    """Absolute residual map: |I - I_S|"""
    return np.abs(flux - flux_smoothed)


def compute_S_partial(flux, residual):
    """
    Partial S: sum(|I - I_S|) / sum(|I|)
    """
    denom = np.nansum(np.abs(flux))
    if denom == 0:
        raise ValueError("Total flux is zero — check input map.")
    return np.nansum(residual) / denom

# =============================================================================
# PLOTTING
# =============================================================================

def make_plot(flux, results, params, r_e_px, galaxy_id):
    """
    Multi-panel diagnostic figure iterating over kernel scale factors.

    Layout: one row per kernel scale, three columns:
        col 0: smoothed flux map (I_S)
        col 1: residual map |I - I_S|
        col 2: text summary panel (sigma, S value, physical scale)

    The original flux map is shown once at the top as a reference.
    """
    n_scales = len(results)

    # Figure: 1 reference row + n_scales rows, 3 columns
    fig = plt.figure(figsize=(14, 4 * (n_scales + 1)))
    fig.suptitle(
        f"Smoothness Kernel Exploration — CATID {galaxy_id}\n"
        f"$r_e$ = {params['re']:.2f}\" = {r_e_px:.1f} px  |  "
        f"SFR = {params['SFR']}  |  "
        f"log $M_*$ = {params['Mstar']}  |  "
        f"Environment = {params['environment']}  |  "
        f"Class = {params['spectral_class']}",
        fontsize=11, fontweight='bold'
    )

    gs = gridspec.GridSpec(
        n_scales + 1, 3, figure=fig, hspace=0.45, wspace=0.35
    )

    # Shared colour scaling from original flux
    finite = flux[np.isfinite(flux)]
    vmin = np.percentile(finite, VMIN_PERC)
    vmax = np.percentile(finite, VMAX_PERC)

    # --- Row 0: original flux map (reference) ---
    ax_orig = fig.add_subplot(gs[0, :])
    im_orig = ax_orig.imshow(flux, origin='lower', cmap='inferno',
                             vmin=vmin, vmax=vmax)
    ax_orig.set_title("Original Hα flux  ($I$)  — reference")
    ax_orig.set_xlabel("x (px)"); ax_orig.set_ylabel("y (px)")
    plt.colorbar(im_orig, ax=ax_orig, fraction=0.015, pad=0.04, label="Flux")

    # --- One row per kernel scale ---
    for i, res in enumerate(results):
        row = i + 1
        ks       = res['kernel_scale']
        sigma_px = res['sigma_px']
        smoothed = res['flux_smoothed']
        residual = res['residual']
        S        = res['S_partial']

        # Col 0: smoothed flux
        ax_s = fig.add_subplot(gs[row, 0])
        ax_s.imshow(smoothed, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
        ax_s.set_title(f"Smoothed $I_S$  (σ = {sigma_px:.2f} px)")
        ax_s.set_xlabel("x (px)"); ax_s.set_ylabel("y (px)")

        # Col 1: residual map
        ax_r = fig.add_subplot(gs[row, 1])
        res_finite = residual[np.isfinite(residual)]
        res_vmax   = np.percentile(res_finite, VMAX_PERC)
        ax_r.imshow(residual, origin='lower', cmap='magma', vmin=0, vmax=res_vmax)
        ax_r.set_title(f"Residual $|I - I_S|$")
        ax_r.set_xlabel("x (px)")

        # Col 2: summary text
        ax_t = fig.add_subplot(gs[row, 2])
        ax_t.axis('off')
        summary = (
            f"Kernel scale:  {ks} × $r_e$\n"
            f"σ (pixels):    {sigma_px:.2f} px\n"
            f"σ (arcsec):    {sigma_px * (params['re'] / r_e_px):.2f}\"\n\n"
            f"S (partial):   {S:.4f}\n\n"
            f"Note: no background\nsubtraction or central\nexclusion applied"
        )
        ax_t.text(0.05, 0.95, summary, transform=ax_t.transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightgrey', alpha=0.4))
        ax_t.set_title(f"Scale = {ks} × $r_e$", fontsize=10)

    out_path = os.path.join(OUT_DIR, f"smoothness_kernel_options.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {os.path.abspath(out_path)}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Load Halpha map and plate scale
    flux, err, plate_scale = load_halpha(FITS_PATH)

    # Load galaxy parameters from catalogue
    params = load_galaxy_params(CSV_PATH, GALAXY_ID)

    # Convert r_e from arcseconds to pixels
    r_e_px = params['re'] / plate_scale
    print(f"\n  r_e in pixels: {params['re']:.3f} arcsec / {plate_scale:.4f} arcsec/px = {r_e_px:.2f} px")

    # Iterate over kernel scale factors
    results = []
    print(f"\nComputing S for {len(KERNEL_SCALES)} kernel scales...")
    for ks in KERNEL_SCALES:
        sigma_px     = ks * r_e_px
        flux_smoothed = conselice_smooth(flux, sigma_px)
        residual     = compute_residual(flux, flux_smoothed)
        S            = compute_S_partial(flux, residual)

        print(f"  scale = {ks} * r_e  ->  sigma = {sigma_px:.2f} px  ->  S = {S:.4f}")

        results.append({
            'kernel_scale' : ks,
            'sigma_px'     : sigma_px,
            'flux_smoothed': flux_smoothed,
            'residual'     : residual,
            'S_partial'    : S,
        })

    # Plot
    make_plot(flux, results, params, r_e_px, GALAXY_ID)


if __name__ == "__main__":
    main()