"""
Computes the CAS Asymmetry index (A) for the Hα emission-line flux map
from a SAMI data cube.

Cube shape: (50, 50, 2)

Rotation centre: the exact geometric centre of a 50 × 50 grid lies
between the 4 central spaxels (indices 24 & 25 in each axis)

Asymmetry definition (Conselice 2003, eq. 1):
    A = ( Σ |I₀ - I₁₈₀| / Σ |I₀| )
where:
  I₀   = original flux map (within a circular aperture)
  I₁₈₀ = flux map rotated 180° about the galaxy centre
"""

## Used just to take Halpha map, flip 180, subtract, calculate A, and plot the maps --> no iteration through possible centres 

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

## REMEMBER, also change the name of the map being saved if using this to look at different flux maps

fits_path = "gini/gini_outputs/tables/masked_halpha_maps/7139_A_HALPHA_masked.fits"
outdir = "asymmetry/outputs/figures"
os.makedirs(outdir, exist_ok=True)


def extract_flux_map(data, layer=0):
    """
    Extract a 2D Halpha flux map from FITS data.

    Should be laid out to handle:
      - 2D array:            (50, 50)
      - 3D array, last axis: (50, 50, 2)
      - 3D array, first axis:(2, 50, 50)

    """
    data = np.asarray(data)

    if data.ndim == 2:
        return data.astype(float)

    if data.ndim == 3:
        # Case 1: (50, 50, 2)
        if data.shape[-1] == 2:
            return data[:, :, layer].astype(float)

        # Case 2: (2, 50, 50)
        if data.shape[0] == 2:
            return data[layer, :, :].astype(float)

    raise ValueError(f"Unexpected FITS data shape: {data.shape}")


def rotate_180_about_centre(img):
    """
    Rotate a 2D image by 180 degrees about its geometric centre.

    For an even-sized image like 50x50, this corresponds to rotation
    about the point between the 4 central pixels.
    """
    return np.rot90(img, 2)


def compute_asymmetry(img):
    """
    Compute asymmetry:
        A = sum(|I - I180|) / sum(|I|)

    Assumes masked/background-invalid pixels are already set to 0.
    """
    rotated = rotate_180_about_centre(img)
    residual = img - rotated

    denom = np.sum(np.abs(img))
    if denom == 0:
        return np.nan, rotated, residual

    A = np.sum(np.abs(residual)) / denom
    return A, rotated, residual


def plot_asymmetry_products(img, rotated, residual, A, outpath):
    """
    Save a 3-panel plot:
      1. Original image
      2. 180-degree rotated image
      3. Residual image (original - rotated)
    """
    vmax = np.nanpercentile(img, 99)
    rmax = np.nanpercentile(np.abs(residual), 99)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(img, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
    axes[0].set_title("Original Hα flux map")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(rotated, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
    axes[1].set_title("Rotated 180°")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(
        residual,
        origin="lower",
        cmap="viridis",
        vmin=-rmax,
        vmax=rmax
    )
    axes[2].set_title(f"Residual: Original - Rotated\nA = {A:.4f}")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


# Main

def main():
    with fits.open(fits_path) as hdul:
        data = hdul[0].data

    img = extract_flux_map(data, layer=0)

    # Safety: convert NaNs/infs to 0 even though you said masked maps already do this
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    A, rotated, residual = compute_asymmetry(img)

    print(f"Asymmetry index A for {os.path.basename(fits_path)} = {A:.6f}")

    plot_path = os.path.join(outdir, "7139_A_Halpha_asymmetry_residuals.png")
    plot_asymmetry_products(img, rotated, residual, A, plot_path)

    print(f"Saved residual plot to: {plot_path}")


if __name__ == "__main__":
    main()