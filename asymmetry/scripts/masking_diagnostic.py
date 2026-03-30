# masking_diagnostic.py
# One-off diagnostic to validate masking strategy used in compute_A function of individual_asymmetry_02.
# Designed to run this on a representative galaxy, then set aside.

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from individual_asymmetry_02 import extract_flux_map, grid_search_centre, outdir, galaxy_id

FITS_PATH = "gini/gini_outputs/tables/masked_halpha_maps/7139_A_HALPHA_masked.fits"


def diagnose_masking(img, cy, cx):
    """
    Visualises the three masking strategies and their effect on A.
    Top row:    the three binary masks
    Bottom row: |residual| under each mask, with disputed spaxels in red
    """
    array_centre = (img.shape[0] - 1) / 2.0
    dy = array_centre - cy
    dx = array_centre - cx

    shifted = shift(img,   shift=(dy, dx),    order=3, mode='constant', cval=0.0)
    rotated = np.rot90(shifted, 2)
    rotated = shift(rotated, shift=(-dy, -dx), order=3, mode='constant', cval=0.0)

    original_mask = img    != 0
    rotated_mask  = rotated > 1e-6
    common_mask   = original_mask & rotated_mask

    residual = img - rotated

    def A_from_mask(mask):
        denom = np.sum(np.abs(img[mask]))
        return np.sum(np.abs(residual[mask])) / denom if denom > 0 else np.nan

    A_original = A_from_mask(original_mask)
    A_rotated  = A_from_mask(rotated_mask)
    A_common   = A_from_mask(common_mask)

    print(f"A (original mask): {A_original:.4f}")
    print(f"A (rotated mask):  {A_rotated:.4f}")
    print(f"A (common mask):   {A_common:.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    mask_panels = [
        (original_mask, f"Original mask only\nA = {A_original:.4f}"),
        (rotated_mask,  f"Rotated mask only\nA = {A_rotated:.4f}"),
        (common_mask,   f"Intersection (common)\nA = {A_common:.4f}"),
    ]

    # Top row: binary masks
    for ax, (mask, title) in zip(axes[0], mask_panels):
        ax.imshow(mask.astype(float), origin="lower", cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (spaxel)")
        ax.set_ylabel("y (spaxel)")

    # Bottom row: residuals under each mask, red overlay on disputed spaxels
    diff_region = original_mask & ~common_mask
    for ax, (mask, title) in zip(axes[1], mask_panels):
        masked_residual = np.where(mask, np.abs(residual), 0)
        rmax = np.nanpercentile(masked_residual[mask], 99) if mask.any() else 1
        im = ax.imshow(masked_residual, origin="lower", cmap="inferno",
                       vmin=0, vmax=rmax)
        ax.set_title(f"|Residual| — {title.split(chr(10))[0]}", fontsize=10)
        ax.set_xlabel("x (spaxel)")
        ax.set_ylabel("y (spaxel)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        overlay = np.zeros((*img.shape, 4))
        overlay[diff_region] = [1, 0, 0, 0.4]
        ax.imshow(overlay, origin="lower")

    fig.suptitle(
        f"Galaxy {galaxy_id} — Masking diagnostic at centre ({cy:.2f}, {cx:.2f})\n"
        f"Red = spaxels in original mask but not rotated mask (boundary artefact zone)",
        fontsize=11
    )
    plt.tight_layout()
    outpath = os.path.join(outdir, f"{galaxy_id}_masking_diagnostic.png")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved → {outpath}")


if __name__ == "__main__":
    img = extract_flux_map(FITS_PATH)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    best_centre, _, _, _ = grid_search_centre(img)
    diagnose_masking(img, *best_centre)