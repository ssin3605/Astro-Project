"""
Computes the CAS Asymmetry index (A) for a SAMI Hα flux map, with a
grid search over candidate rotation centres to find the minimum A.

Cube shape: (50, 50, 2)

Rotation centre: by default the geometric centre (24.5, 24.5), but a
10x10 grid search within ±2 pixels is used to find the true minimum.

Error subtraction (Conselice 2003 background correction approach):
    I_clean = clip(flux - error, 0)
    Pixel-wise subtraction of the HDU 1 error map from the HDU 0 flux map.
    Negative values clipped to zero (unphysical flux). This suppresses
    low-SNR spaxels at the galaxy periphery more than the bright core,
    down-weighting regions where asymmetry measurements are least reliable.

Rotation method: np.rot90 for the exact centre (no interpolation error),
with scipy.ndimage.shift used to translate candidate centres onto the
array midpoint before rotating.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import shift

# ── Configuration ──────────────────────────────────────────────────────────────

fits_path = "gini/gini_outputs/tables/masked_halpha_maps/7139_A_HALPHA_masked.fits"
outdir    = "asymmetry/outputs/figures"
galaxy_id = "7139"

GRID_POINTS = 10   # grid resolution per axis (10x10 = 100 candidate centres)
GRID_RANGE  = 2.0  # search ± this many pixels from (24.5, 24.5)

os.makedirs(outdir, exist_ok=True)

# ── Data loading ───────────────────────────────────────────────────────────────

def extract_flux_and_error(fits_path, layer=0):
    """
    Load the 2D Hα flux map (HDU 0) and error map (HDU 1) from a SAMI FITS file.

    Handles cube shapes:
      - 2D array:             (50, 50)
      - 3D, channels last:    (50, 50, 2)
      - 3D, channels first:   (2, 50, 50)

    Returns
    -------
    flux : 2D float array
    error: 2D float array (same shape as flux)
    """
    with fits.open(fits_path) as hdul:
        flux_raw  = np.asarray(hdul[0].data)
        error_raw = np.asarray(hdul[1].data)

    def extract_layer(data, layer):
        data = data.astype(float)
        if data.ndim == 2:
            return data
        if data.ndim == 3:
            if data.shape[-1] == 2:   # (50, 50, 2)
                return data[:, :, layer]
            if data.shape[0] == 2:    # (2, 50, 50)
                return data[layer]
        raise ValueError(f"Unexpected FITS data shape: {data.shape}")

    flux  = extract_layer(flux_raw,  layer)
    error = extract_layer(error_raw, layer)
    return flux, error


def apply_error_subtraction(flux, error):
    """
    Subtract the per-spaxel error map from the flux map and clip to zero.

    I_clean = clip(flux - error, 0)

    Negative values after subtraction are unphysical and are set to zero.
    This down-weights low-SNR spaxels at the galaxy periphery relative to
    the bright central regions where the measurement is most reliable.
    """
    return np.clip(flux - error, 0, None)

# ── Asymmetry calculation ──────────────────────────────────────────────────────

def compute_A(img, cy, cx):
    """
    Compute asymmetry A by rotating img 180° about the point (cy, cx).

    Steps:
      1. Shift so (cy, cx) lands on the array midpoint (24.5, 24.5)
      2. Rotate 180° exactly using np.rot90 (no interpolation error)
      3. Shift back to original coordinate frame
      4. Use intersection of original and rotated footprints to avoid
         penalising edge spaxels displaced by the shift
    """
    array_centre = (img.shape[0] - 1) / 2.0  # 24.5

    dy = array_centre - cy
    dx = array_centre - cx

    shifted = shift(img,     shift=(dy, dx),    order=3, mode='constant', cval=0.0)
    rotated = np.rot90(shifted, 2)
    rotated = shift(rotated, shift=(-dy, -dx),  order=3, mode='constant', cval=0.0)

    original_mask = img     != 0
    rotated_mask  = rotated  > 1e-3 * np.max(img) if np.max(img) > 0 else rotated > 0
    common_mask   = original_mask & rotated_mask

    residual = img - rotated
    denom    = np.sum(np.abs(img[common_mask]))

    if denom == 0:
        return np.nan, rotated, residual

    A = np.sum(np.abs(residual[common_mask])) / denom
    return A, rotated, residual

# ── Grid search ────────────────────────────────────────────────────────────────

def grid_search_centre(img, grid_points=GRID_POINTS, grid_range=GRID_RANGE):
    """
    Search a grid of candidate centres within ±grid_range pixels of (24.5, 24.5).
    Returns the centre that minimises A, the full A grid, and the offset axes.
    """
    array_centre = (img.shape[0] - 1) / 2.0
    offsets      = np.linspace(-grid_range, grid_range, grid_points)
    A_grid       = np.full((grid_points, grid_points), np.nan)

    best_A      = np.inf
    best_centre = (array_centre, array_centre)

    for i, dy in enumerate(offsets):
        for j, dx in enumerate(offsets):
            cy = array_centre + dy
            cx = array_centre + dx
            A, _, _ = compute_A(img, cy, cx)
            A_grid[i, j] = A
            if A < best_A:
                best_A      = A
                best_centre = (cy, cx)

    return best_centre, best_A, A_grid, offsets

# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_asymmetry(flux_raw, img, rotated, residual, A_geom, A_min, best_centre, outpath):
    """
    Five-panel figure:
      1. Raw flux map (before error subtraction)
      2. Error-subtracted flux map I_clean
      3. 180° rotated I_clean (at best centre)
      4. Residual |I_clean - I_clean_180|
      5. Best centre vs geometric centre overlay
    """
    vmax_raw  = np.nanpercentile(flux_raw[flux_raw > 0], 99) if (flux_raw > 0).any() else 1
    vmax_clean = np.nanpercentile(img[img > 0], 99)          if (img > 0).any() else 1
    rmax       = np.nanpercentile(np.abs(residual), 99)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

    panels = [
        (flux_raw,         "Raw flux  $I_0$",               "viridis", 0, vmax_raw),
        (img,              "Error-subtracted  $I_{{clean}}$","viridis", 0, vmax_clean),
        (rotated,          "Rotated 180°  $I_{{180}}$",      "viridis", 0, vmax_clean),
        (np.abs(residual), "$|I_{{clean}} - I_{{180}}|$",    "inferno", 0, rmax),
    ]

    for ax, (data, title, cmap, vmin, vmax_) in zip(axes[:4], panels):
        im = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax_)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (spaxel)")
        ax.set_ylabel("y (spaxel)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 5: centre overlay on error-subtracted map
    im4 = axes[4].imshow(img, origin="lower", cmap="viridis", vmin=0, vmax=vmax_clean)
    axes[4].plot(best_centre[1], best_centre[0],
                 '+', color='red', ms=12, mew=2, label='Best centre')
    axes[4].plot(24.5, 24.5,
                 'x', color='white', ms=10, mew=1.5, label='Geometric centre')
    axes[4].set_title("Centre comparison", fontsize=10)
    axes[4].set_xlabel("x (spaxel)")
    axes[4].set_ylabel("y (spaxel)")
    axes[4].legend(fontsize=8)
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Galaxy {galaxy_id} — Hα Asymmetry  |  "
        f"$A_{{geom}}$ = {A_geom:.4f}   "
        f"$A_{{min}}$ = {A_min:.4f}   "
        f"$\\Delta A$ = {A_geom - A_min:.4f}",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved asymmetry plot → {outpath}")


def plot_A_grid(A_grid, offsets, best_centre, outpath):
    """Heatmap of A values across the grid search, with best centre marked."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(A_grid, origin="lower", cmap="viridis",
                   extent=[offsets[0], offsets[-1], offsets[0], offsets[-1]])
    plt.colorbar(im, ax=ax, label="Asymmetry  $A$")

    best_dy = best_centre[0] - 24.5
    best_dx = best_centre[1] - 24.5
    ax.plot(best_dx, best_dy, 'r+', ms=12, mew=2, label=f'Min A = {A_grid.min():.4f}')
    ax.plot(0, 0, 'wx', ms=10, mew=1.5, label='Geometric centre')

    ax.set_xlabel("Δx (pixels)")
    ax.set_ylabel("Δy (pixels)")
    ax.set_title(f"Galaxy {galaxy_id} — A over centre grid", fontsize=11)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved grid search plot → {outpath}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # 1. Load flux and error maps
    flux, error = extract_flux_and_error(fits_path)
    flux  = np.nan_to_num(flux,  nan=0.0, posinf=0.0, neginf=0.0)
    error = np.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. Subtract error from flux, clip negatives to zero
    img = apply_error_subtraction(flux, error)

    # 3. Asymmetry at default geometric centre
    A_geom, _, _ = compute_A(img, cy=24.5, cx=24.5)

    # 4. Grid search for minimum A
    best_centre, A_min, A_grid, offsets = grid_search_centre(img)

    print(f"\nGalaxy:                             {galaxy_id}")
    print(f"A at geometric centre (24.5, 24.5): {A_geom:.6f}")
    print(f"A at best centre {best_centre}:     {A_min:.6f}")
    print(f"ΔA (correction):                    {A_geom - A_min:.6f}\n")

    # 5. Final maps at best centre for plotting
    A_min, rotated, residual = compute_A(img, *best_centre)

    # 6. Save plots
    plot_asymmetry(flux, img, rotated, residual, A_geom, A_min, best_centre,
                   outpath=os.path.join(outdir, f"{galaxy_id}_Halpha_centre_fit.png"))

    plot_A_grid(A_grid, offsets, best_centre,
                outpath=os.path.join(outdir, f"{galaxy_id}_asymmetry_03_grid_search.png"))


if __name__ == "__main__":
    main()