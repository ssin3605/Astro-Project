"""
smoothness_test_03.py
---------------------
Third iteration of the Clumpiness (S) parameter script for SAMI Halpha maps.

Improvements over smoothness_test_02.py:
  - Error subtraction: the error map (err) is subtracted from the flux map
    before smoothing. This removes the noise contribution from the
    residual map before S is computed.
  - Pipeline order:
        1. Load flux and error (summed Gaussian slices)
        2. Subtract error from flux -> I_clean = I - err
        3. Smooth I_clean with Conselice kernel (sigma = ks * r_e)
        4. Compute residual |I_clean - I_clean_smoothed|
        5. Compute partial S (no central exclusion yet)
  - Existing SAMI mask (S/N 3, NaNs and negatives set to 0) is relied upon
    rather than applying additional SNR cuts — will revisit if needed
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits
import pandas as pd
from scipy.ndimage import gaussian_filter

# =============================================================================
# CONFIG
# =============================================================================

GALAXY_ID    = "23117"
FITS_PATH    = "gini/gini_outputs/tables/masked_halpha_maps/23117_A_Halpha_masked.fits"
CSV_PATH     = "concentration/outputs/tables/concentration_results.csv"
OUT_DIR      = "smoothness/outputs/figures"

# Kernel scale factors to iterate over (as fractions of r_e)
KERNEL_SCALES = [0.25, 0.35, 0.50]

# Plotting
CMAP         = "inferno"
VMIN_PERC    = 2
VMAX_PERC    = 98

os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================

def load_halpha(fits_path):
    """
    Load Halpha flux and error maps from a SAMI masked FITS file.
    """
    with fits.open(fits_path) as hdul:
        header      = hdul[0].header
        flux        = hdul[0].data[1, :, :]
        err         = hdul[1].data[1, :, :]
        cdelt       = abs(header.get('CDELT1', None))
        if cdelt is None:
            raise KeyError("CDELT1 not found in FITS header.")
        plate_scale = cdelt * 3600.0 if cdelt < 0.01 else cdelt

    print(f"Loaded: {fits_path}")
    print(f"  Flux shape : {flux.shape}")
    print(f"  Plate scale: {plate_scale:.4f} arcsec/px")
    return flux, err, plate_scale


def load_galaxy_params(csv_path, galaxy_id):
    """
    Load r_e and key physical parameters from
    concentration_results.csv
    """
    df = pd.read_csv(csv_path)
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

    print(f"\nGalaxy {galaxy_id} parameters:")
    print(f"  r_e           = {params['re']:.3f} arcsec")
    print(f"  SFR           = {params['SFR']}")
    print(f"  log Mstar     = {params['Mstar']}")
    print(f"  Environment   = {params['environment']}")
    print(f"  Spectral class= {params['spectral_class']}")
    return params


# =============================================================================
# ERROR SUBTRACTION
# =============================================================================

def subtract_error(flux, err):
    """
    Subtract the per-pixel error map from the flux map 

    The existing SAMI mask (S/N 3, NaNs and negatives set to 0) is relied
    upon — no additional SNR masking is applied here.

    I_clean = I - err
    """
    flux_clean = flux - err

    # Report how many pixels were affected (changed sign or reduced to <=0)
    n_total    = flux_clean.size
    n_negative = np.sum(flux_clean < 0)
    n_zero     = np.sum(flux == 0)   # pixels already masked in original
    print(f"\nError subtraction:")
    print(f"  Pixels already masked (flux = 0) : {n_zero} / {n_total}")
    print(f"  Pixels negative after subtraction: {n_negative} / {n_total}")
    print(f"  (Negative pixels remain as-is — existing mask handles these)")
    return flux_clean


# =============================================================================
# SMOOTHING
# =============================================================================

def conselice_smooth(flux, sigma_px):
    """
    Smooth a flux map with a Gaussian kernel of width sigma_px.
    sigma_px should be pre-computed as kernel_scale * r_e_pixels.
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

def make_plot(flux, flux_clean, results, params, r_e_px, galaxy_id):
    """
    Diagnostic figure showing the effect of error subtraction and kernel
    scale on the residual map and S value.

    Layout:
        Row 0 : original flux (left) and error-subtracted flux (right)
                side by side — to visualise the effect of error subtraction
        Rows 1+: one row per kernel scale —
                 col 0: smoothed I_clean
                 col 1: residual |I_clean - I_clean_smoothed|
                 col 2: text summary
    """
    n_scales = len(results)
    fig = plt.figure(figsize=(14, 4 * (n_scales + 1)))
    fig.suptitle(
        f"Smoothness — CATID {galaxy_id}  |  Error subtraction applied\n"
        f"$r_e$ = {params['re']:.2f}\"= {r_e_px:.1f} px  |  "
        f"SFR = {params['SFR']:.3f}  |  "
        f"log $M_*$ = {params['Mstar']}  |  "
        f"Environment = {params['environment']}  |  "
        f"Class = {params['spectral_class']}",
        fontsize=11, fontweight='bold'
    )

    gs = gridspec.GridSpec(n_scales + 1, 3, figure=fig,
                           hspace=0.45, wspace=0.35)

    # Shared colour scaling from original flux
    finite = flux[flux != 0]   # exclude masked pixels from scaling
    vmin   = np.percentile(finite, VMIN_PERC)
    vmax   = np.percentile(finite, VMAX_PERC)

    # --- Row 0: original flux and error-subtracted flux side by side ---
    ax_orig = fig.add_subplot(gs[0, 0])
    im_orig = ax_orig.imshow(flux, origin='lower', cmap=CMAP,
                             vmin=vmin, vmax=vmax)
    ax_orig.set_title("Original Hα flux  ($I$)")
    ax_orig.set_xlabel("x (px)"); ax_orig.set_ylabel("y (px)")
    plt.colorbar(im_orig, ax=ax_orig, fraction=0.046, label="Flux")

    ax_clean = fig.add_subplot(gs[0, 1])
    im_clean = ax_clean.imshow(flux_clean, origin='lower', cmap=CMAP,
                               vmin=vmin, vmax=vmax)
    ax_clean.set_title("Error-subtracted flux  ($I - \\epsilon$)")
    ax_clean.set_xlabel("x (px)")
    plt.colorbar(im_clean, ax=ax_clean, fraction=0.046, label="Flux")

    # Difference map: how much did error subtraction change the flux?
    ax_diff = fig.add_subplot(gs[0, 2])
    diff    = flux_clean - flux   # should be negative everywhere (= -err)
    im_diff = ax_diff.imshow(diff, origin='lower', cmap='coolwarm')
    ax_diff.set_title("Difference  $(I - \\epsilon) - I$\n(= $-\\epsilon$, shows error magnitude)")
    ax_diff.set_xlabel("x (px)")
    plt.colorbar(im_diff, ax=ax_diff, fraction=0.046, label="$\\Delta$ Flux")

    # --- One row per kernel scale ---
    for i, res in enumerate(results):
        row      = i + 1
        ks       = res['kernel_scale']
        sigma_px = res['sigma_px']
        smoothed = res['flux_smoothed']
        residual = res['residual']
        S        = res['S_partial']

        # Col 0: smoothed error-subtracted flux
        ax_s = fig.add_subplot(gs[row, 0])
        ax_s.imshow(smoothed, origin='lower', cmap=CMAP, vmin=vmin, vmax=vmax)
        ax_s.set_title(f"Smoothed $(I-\\epsilon)_S$  (σ = {sigma_px:.2f} px)")
        ax_s.set_xlabel("x (px)"); ax_s.set_ylabel("y (px)")

        # Col 1: residual map
        ax_r = fig.add_subplot(gs[row, 1])
        res_finite = residual[residual != 0]
        res_vmax   = np.percentile(res_finite, VMAX_PERC) if len(res_finite) > 0 else 1
        ax_r.imshow(residual, origin='lower', cmap='magma',
                    vmin=0, vmax=res_vmax)
        ax_r.set_title(f"Residual  $|(I-\\epsilon) - (I-\\epsilon)_S|$")
        ax_r.set_xlabel("x (px)")

        # Col 2: summary text
        ax_t = fig.add_subplot(gs[row, 2])
        ax_t.axis('off')
        summary = (
            f"Kernel scale:  {ks} × $r_e$\n"
            f"σ (pixels):    {sigma_px:.2f} px\n"
            f"σ (arcsec):    {sigma_px * (params['re'] / r_e_px):.2f}\"\n\n"
            f"S (partial):   {S:.4f}\n\n"
            f"No central exclusion\napplied yet"
        )
        ax_t.text(0.05, 0.95, summary, transform=ax_t.transAxes,
                  fontsize=10, verticalalignment='top',
                  fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightgrey',
                            alpha=0.4))
        ax_t.set_title(f"Scale = {ks} × $r_e$", fontsize=10)

    out_path = os.path.join(
        OUT_DIR, f"smoothness_error_subtracted_{galaxy_id}.png"
    )
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {os.path.abspath(out_path)}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Load
    flux, err, plate_scale = load_halpha(FITS_PATH)
    params = load_galaxy_params(CSV_PATH, GALAXY_ID)

    # Convert r_e to pixels
    r_e_px = params['re'] / plate_scale
    print(f"\n  r_e in pixels: {params['re']:.3f}\" / "
          f"{plate_scale:.4f} arcsec/px = {r_e_px:.2f} px")

    # Error subtraction: I_clean = I - err
    flux_clean = subtract_error(flux, err)

    # Iterate over kernel scales
    results = []
    print(f"\nComputing S for {len(KERNEL_SCALES)} kernel scales...")
    for ks in KERNEL_SCALES:
        sigma_px      = ks * r_e_px
        flux_smoothed = conselice_smooth(flux_clean, sigma_px)
        residual      = compute_residual(flux_clean, flux_smoothed)
        S             = compute_S_partial(flux_clean, residual)

        print(f"  scale = {ks} * r_e  ->  "
              f"sigma = {sigma_px:.2f} px  ->  S = {S:.4f}")

        results.append({
            'kernel_scale' : ks,
            'sigma_px'     : sigma_px,
            'flux_smoothed': flux_smoothed,
            'residual'     : residual,
            'S_partial'    : S,
        })

    # Plot
    make_plot(flux, flux_clean, results, params, r_e_px, GALAXY_ID)


if __name__ == "__main__":
    main()