"""
Plots the Halpha flux sum layer (HDU 0, slice 1) for 4 selected representative
galaxies side by side in a single figure.

Colour scaling is shared across all 4 panels using the global percentile range
across all four flux maps combined, so brightness is directly comparable
between galaxies.

Edit GALAXY_IDS in the CONFIG block to change which galaxies are plotted.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits

# =============================================================================
# CONFIG
# =============================================================================

GALAXY_IDS  = [422933, 105420, 9008500868, 39365]   # 4 selected CATIDs
HALPHA_DIR  = 'gini/gini_outputs/tables/masked_halpha_maps'
OUT_DIR     = 'smoothness/outputs/figures/individual_maps'
OUT_FILE    = 'representative_s_galaxies_flux.png'

CMAP        = 'inferno'
VMIN_PERC   = 2    # lower percentile for shared colour scaling
VMAX_PERC   = 98   # upper percentile for shared colour scaling

os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# FIND FITS FILES
# =============================================================================

def find_fits(halpha_dir, catid):
    """
    Locate the masked Halpha FITS file for a given CATID.
    """
    pattern = os.path.join(halpha_dir, f"{catid}_*_Halpha_masked.fits")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No masked Halpha FITS found for CATID {catid} "
            f"in {halpha_dir}"
        )
    if len(matches) > 1:
        print(f"  Warning: multiple FITS found for CATID {catid} "
              f"— using {os.path.basename(matches[0])}")
    return matches[0]


# =============================================================================
# LOAD FLUX MAPS
# =============================================================================

def load_flux(fits_path):
    """
    Load the Halpha flux sum layer from a masked FITS file.

    HDU 0, slice [1] = summed Gaussian flux map (50x50).
    """
    with fits.open(fits_path) as hdul:
        flux = hdul[0].data[1, :, :]
    label = os.path.basename(fits_path).replace('_Halpha_masked.fits', '')
    return flux, label


# =============================================================================
# COMPUTE SHARED COLOUR SCALE
# =============================================================================

def shared_colour_scale(flux_maps, vmin_perc=2, vmax_perc=98):
    """
    Compute shared vmin and vmax from the combined pixel distribution
    across all flux maps, using percentile clipping to avoid outliers
    dominating the colour scale.

    Only non-zero pixels are included (zero = masked in SAMI convention).
    """
    all_pixels = np.concatenate([
        f[f != 0].ravel() for f in flux_maps if np.any(f != 0)
    ])
    vmin = np.percentile(all_pixels, vmin_perc)
    vmax = np.percentile(all_pixels, vmax_perc)
    print(f"\n  Shared colour scale:")
    print(f"    vmin ({vmin_perc}th percentile) : {vmin:.4f}")
    print(f"    vmax ({vmax_perc}th percentile) : {vmax:.4f}")
    return vmin, vmax


# =============================================================================
# PLOT
# =============================================================================

def make_plot(galaxy_ids, flux_maps, labels, vmin, vmax):
    """
    Four-panel figure with one galaxy per panel, shared colour scale.
    A single shared colorbar is placed to the right of all panels.
    """
    n       = len(galaxy_ids)
    fig     = plt.figure(figsize=(4 * n + 1, 5))
    fig.suptitle(
        "Hα flux maps — selected representative SF galaxies\n"
        f"Colour scale: shared global {VMIN_PERC}th–{VMAX_PERC}th percentile",
        fontsize=12, fontweight='bold'
    )

    # GridSpec: n columns for maps, 1 narrow column for colorbar
    gs  = gridspec.GridSpec(
        1, n + 1,
        figure=fig,
        width_ratios=[1] * n + [0.05],
        wspace=0.08
    )

    axes = []
    im   = None

    for i, (catid, flux, label) in enumerate(
        zip(galaxy_ids, flux_maps, labels)
    ):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)

        im = ax.imshow(
            flux, origin='lower', cmap=CMAP,
            vmin=vmin, vmax=vmax
        )

        ax.set_title(f"CATID {catid}", fontsize=10, fontweight='bold')
        ax.set_xlabel("x (px)", fontsize=9)
        if i == 0:
            ax.set_ylabel("y (px)", fontsize=9)
        else:
            ax.set_yticklabels([])

        ax.tick_params(labelsize=8)

    # Shared colorbar in the extra column
    cax  = fig.add_subplot(gs[0, n])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Hα flux (summed Gaussian)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    out_path = os.path.join(OUT_DIR, OUT_FILE)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {os.path.abspath(out_path)}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Find and load all flux maps
    flux_maps = []
    labels    = []

    print("Loading flux maps:")
    for catid in GALAXY_IDS:
        try:
            fits_path    = find_fits(HALPHA_DIR, catid)
            flux, label  = load_flux(fits_path)
            flux_maps.append(flux)
            labels.append(label)
            print(f"  CATID {catid}: {os.path.basename(fits_path)} "
                  f"| flux range {np.nanmin(flux):.4f} – {np.nanmax(flux):.4f}")
        except FileNotFoundError as e:
            print(f"  SKIP — {e}")

    if len(flux_maps) == 0:
        raise RuntimeError("No flux maps loaded — check HALPHA_DIR and GALAXY_IDS.")

    if len(flux_maps) < len(GALAXY_IDS):
        print(f"\n  Warning: only {len(flux_maps)} of {len(GALAXY_IDS)} "
              f"galaxies loaded successfully — plotting available maps only.")

    # Shared colour scaling
    vmin, vmax = shared_colour_scale(flux_maps, VMIN_PERC, VMAX_PERC)

    # Plot
    make_plot(GALAXY_IDS[:len(flux_maps)], flux_maps, labels, vmin, vmax)


if __name__ == '__main__':
    main()