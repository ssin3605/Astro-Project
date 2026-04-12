"""
clumpiness_start.py
-------------------
Starting point script for computing the Clumpiness (S) parameter on a SAMI
Halpha flux map, following Conselice (2003).

For this initial version I am:
  - Applying Conselice's smoothing kernel (Gaussian, sigma = 0.2 * 1.5 * R_Pet)
  - No background subtraction (B_S)
  - Not excluding central pixels

However, these are all decisions I will need to make/refine
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits
from scipy.ndimage import gaussian_filter


FITS_PATH    = "gini/gini_outputs/tables/masked_halpha_maps/7139_A_HALPHA_masked.fits"   # Path to your Halpha FITS file
EXT          = 0          # FITS extension containing the flux map
PETROSIAN_R  = 20         # Have just hardcoded for now but need to refine 
OUT_DIR = "smoothness/outputs/figures"

# Conselice (2003) kernel scale factor: sigma = KERNEL_SCALE * 1.5 * R_Pet
# Conselice uses 0.2 (but this is what I will fine tune if I decide on a different scale factor)
KERNEL_SCALE = 0.2

# Plotting options
CMAP         = "inferno"  # Colourmap for flux maps
VMIN_PERC    = 2          # Lower percentile for colour scaling
VMAX_PERC    = 98         # Upper percentile for colour scaling

os.makedirs(OUT_DIR, exist_ok=True)

## Bringing in data

def load_halpha(fits_path):
    with fits.open(fits_path) as hdul:
        header = hdul[0].header
        flux = hdul[0].data[1, :, :]   # summed Gaussian flux
        err  = hdul[1].data[1, :, :]   # summed Gaussian error
    #print(f"Loaded: {fits_path}")
    #print(f"  Flux  — shape: {flux.shape}, min: {np.nanmin(flux):.4f}, max: {np.nanmax(flux):.4f}")
    #print(f"  Error — shape: {err.shape},  min: {np.nanmin(err):.4f},  max: {np.nanmax(err):.4f}")
    return flux, err, header


## Petrosian Radius

def get_petrosian_radius(r_pet_pixels):
    # Returns the Petrosian radius, prompting at runtime if not set at the start
    if r_pet_pixels is None:
        val = input("Enter Petrosian radius in pixels (R_Pet): ")
        return float(val)
    return float(r_pet_pixels)


## Smoothing

def conselice_smooth(flux, r_pet, kernel_scale=0.2):
    # Smoothing a flux map using the Conselice (2003) Gaussian kernel.
        # sigma = kernel_scale * 1.5 * R_Pet
    sigma_px = kernel_scale * 1.5 * r_pet
    print(f"Kernel sigma = {kernel_scale} * 1.5 * {r_pet:.1f} px = {sigma_px:.2f} px")
    flux_smoothed = gaussian_filter(flux, sigma=sigma_px)
    return flux_smoothed, sigma_px


## Residual mapping

def compute_residual(flux, flux_smoothed):
    # Computes the absolute residual map: |I - I_S|
    return np.abs(flux - flux_smoothed)


## Clumpiness (currently partial with no background or central exclusion)

def compute_S_partial(flux, residual):
    # Compute a partial S value: sum(|I - I_S|) / sum(|I|)
    # This is the Conselice numerator/denominator ratio WITHOUT:
      # - background subtraction (B_S)
      # - central pixel exclusion
    # These omissions MUST be addressed
    denom = np.nansum(np.abs(flux))
    if denom == 0:
        raise ValueError("Total flux is zero — check your input map.")
    S = np.nansum(residual) / denom
    return S


## Plotting

def make_plot(flux, flux_smoothed, residual, sigma_px, r_pet, S_partial):
    # Three-panel diagnostic plot with original map, smoothed map, and absolute residual map

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(
        f"Clumpiness (S) — Starting Point\n"
        f"R_Pet = {r_pet:.1f} px  |  kernel σ = {sigma_px:.2f} px  |  "
        f"S (no bg, no central excl.) = {S_partial:.4f}",
        fontsize=13, fontweight='bold'
    )

    gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.35, wspace=0.35)

    # --- Shared colour scaling for flux panels ---
    finite = flux[np.isfinite(flux)]
    vmin = np.percentile(finite, VMIN_PERC)
    vmax = np.percentile(finite, VMAX_PERC)

    # 1. Original flux
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(flux, origin='lower', cmap=CMAP, vmin=vmin, vmax=vmax)
    ax1.set_title("Original Hα flux  ($I$)")
    ax1.set_xlabel("x (px)"); ax1.set_ylabel("y (px)")
    plt.colorbar(im1, ax=ax1, fraction=0.046, label="Flux")

    # 2. Smoothed flux
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(flux_smoothed, origin='lower', cmap=CMAP, vmin=vmin, vmax=vmax)
    ax2.set_title(f"Smoothed Hα flux  ($I_S$)\n$\\sigma$ = {sigma_px:.2f} px")
    ax2.set_xlabel("x (px)")
    plt.colorbar(im2, ax=ax2, fraction=0.046, label="Flux")

    # 3. Residual map
    ax3 = fig.add_subplot(gs[0, 2])
    res_finite = residual[np.isfinite(residual)]
    res_vmax = np.percentile(res_finite, VMAX_PERC)
    im3 = ax3.imshow(residual, origin='lower', cmap='magma', vmin=0, vmax=res_vmax)
    ax3.set_title("Residual  $|I - I_S|$\n(high-freq. structure)")
    ax3.set_xlabel("x (px)")
    plt.colorbar(im3, ax=ax3, fraction=0.046, label="Flux")

    out_path = os.path.join(OUT_DIR, "clumpiness_diagnostic.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {os.path.abspath(out_path)}")
    plt.close()


## Main

def main():
    # Load
    flux, err, header = load_halpha(FITS_PATH)

    # Petrosian radius
    r_pet = get_petrosian_radius(PETROSIAN_R)

    # Smooth using Conselice kernel
    flux_smoothed, sigma_px = conselice_smooth(flux, r_pet, kernel_scale=KERNEL_SCALE)

    # Residual map
    residual = compute_residual(flux, flux_smoothed)

    # Partial S (no background, no central exclusion)
    S_partial = compute_S_partial(flux, residual)
    print(f"\nPartial S (no background subtraction, no central exclusion): {S_partial:.4f}")
    print("NOTE: This value is not the final Conselice S — it is a diagnostic starting point.")
    print("      Next steps: decide on background region, central exclusion radius, and kernel validation.")

    # Plot
    make_plot(flux, flux_smoothed, residual, sigma_px, r_pet, S_partial)


if __name__ == "__main__":
    main()