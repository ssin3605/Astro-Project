"""
Hα Asymmetry Analysis — Conselice (2003)
=========================================
Computes the CAS Asymmetry index (A) for a SAMI Hα flux map, with a
grid search over candidate rotation centres to find the minimum A.

Cube shape: (50, 50, 2)

Rotation centre: by default the geometric centre (24.5, 24.5), but a
10x10 grid search within ±2 pixels is used to find the true minimum.

Asymmetry definition (Conselice 2003):
    A = Σ|I₀ - I₁₈₀| / Σ|I₀|

Rotation method: np.rot90 for the exact centre (no interpolation error),
with scipy.ndimage.shift used to translate candidate centres onto the
array midpoint before rotating.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import shift

# ── Configuration

fits_path = "gini/gini_outputs/tables/masked_halpha_maps/7139_A_HALPHA_masked.fits"
outdir = "asymmetry/outputs/figures"
galaxy_id   = "7139"

GRID_POINTS = 10    # grid resolution per axis (10x10 = 100 candidate centres)
GRID_RANGE  = 2.0   # search ± this many pixels from (24.5, 24.5)

os.makedirs(outdir, exist_ok=True)

# ── Data loading to work for any HDU structure 
def extract_flux_map(fits_path, layer=0):
    """Load and return a 2D Hα flux map from a SAMI FITS cube."""
    with fits.open(fits_path) as hdul:
        data = np.asarray(hdul[0].data)

    if data.ndim == 2:
        return data.astype(float)
    
    if data.ndim == 3:
        if data.shape[-1] == 2:   # (50, 50, 2)
            return data[:, :, layer].astype(float)
        
        if data.shape[0] == 2:    # (2, 50, 50)
            return data[layer].astype(float)

    raise ValueError(f"Unexpected FITS data shape: {data.shape}")

# ── Asymmetry calculation 

def compute_A(img, cy, cx):
    """
    Compute asymmetry A by rotating img 180° about the point (cy, cx).

      1. Physically shift the image so (cy, cx) lands where the array midpoint (24.5, 24.5) used to be. 
         np.rot90 is now effectively rotating about (cy, cx) in the original coordinate system. 
      2. Rotate 180° exactly using np.rot90 (no interpolation error)
      3. Then once the rotation is done it shifts back to the original coordinate frame. 
         So these three steps are just a way of tricking np.rot90 into rotating about an arbitrary point.
      4. Apply original galaxy mask to suppress boundary interpolation artefacts, as scipy.ndimage.shift
         does not know or care about your mask. This can result in the edges of the galaxy footprint 
         blending real flux values with the surrounding zeros. This smears a small amount of flux into 
         spaxels that were zero before the shift. So need to re-set them to zero. 
    """
    array_centre = (img.shape[0] - 1) / 2.0  # 24.5 for my purposes

    dy = array_centre - cy
    dx = array_centre - cx

    # Shift → rotate → shift back
    shifted  = shift(img,  shift=(dy, dx),   order=3, mode='constant', cval=0.0)
    rotated  = np.rot90(shifted, 2)
    rotated  = shift(rotated, shift=(-dy, -dx), order=3, mode='constant', cval=0.0)

    # Use intersection of original and rotated footprints
    # This avoids penalising edge spaxels that fall outside one map due to the shift
    original_mask = img != 0
    rotated_mask  = rotated > 0
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

def plot_asymmetry(img, rotated, residual, A_geom, A_min, best_centre, outpath):
    """
    Four-panel figure:
      1. Original Hα flux map
      2. 180° rotated map (at best centre)
      3. Residual map |I₀ - I₁₈₀|
      4. Best centre marked on original
    """
    vmax = np.nanpercentile(img[img > 0], 99) if (img > 0).any() else 1
    rmax = np.nanpercentile(np.abs(residual), 99)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    panels = [
        (img,              "Original  $I_0$",           "viridis",  0,     vmax),
        (rotated,          "Rotated 180°  $I_{180}$",   "viridis",  0,     vmax),
        (np.abs(residual), "$|I_0 - I_{180}|$",         "inferno",  0,     rmax),
    ]

    for ax, (data, title, cmap, vmin, vmax_) in zip(axes[:3], panels):
        im = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax_)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (spaxel)")
        ax.set_ylabel("y (spaxel)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 4: best centre overlay
    im3 = axes[3].imshow(img, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
    axes[3].plot(best_centre[1], best_centre[0],
                 '+', color='red', ms=12, mew=2, label='Best centre')
    axes[3].plot(24.5, 24.5,
                 'x', color='white', ms=10, mew=1.5, label='Geometric centre')
    axes[3].set_title("Centre comparison", fontsize=10)
    axes[3].set_xlabel("x (spaxel)")
    axes[3].set_ylabel("y (spaxel)")
    axes[3].legend(fontsize=8)
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

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
    """
    Heatmap of A values across the grid search, with best centre marked.
    """
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
    # Load
    img = extract_flux_map(fits_path)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    # Asymmetry at default geometric centre
    A_geom, _, _ = compute_A(img, cy=24.5, cx=24.5)

    # Grid search for minimum A
    best_centre, A_min, A_grid, offsets = grid_search_centre(img)

    print(f"\nGalaxy:              {galaxy_id}")
    print(f"A at geometric centre (24.5, 24.5): {A_geom:.6f}")
    print(f"A at best centre {best_centre}:     {A_min:.6f}")
    print(f"ΔA (correction):                    {A_geom - A_min:.6f}\n")

    # Final maps at best centre
    A_min, rotated, residual = compute_A(img, *best_centre)

    # Save plots
    plot_asymmetry(img, rotated, residual, A_geom, A_min, best_centre,
                   outpath=os.path.join(outdir, f"{galaxy_id}_halpha_asymmetry.png"))

    plot_A_grid(A_grid, offsets, best_centre,
                outpath=os.path.join(outdir, f"{galaxy_id}_A_grid_search.png"))


if __name__ == "__main__":
    main()