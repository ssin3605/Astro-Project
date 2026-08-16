#!/usr/bin/env python3
"""
thesis_galaxies.py
===================
Thesis-ready 2x3 comparison figure for three galaxies spanning the star
formation main sequence (SFMS):
    Row 1: optical (GAMA) cutout images
    Row 2: Hα flux maps (raw, 1-comp), matching 01_flux_map.py exactly -
           grey background for no-data pixels + an r_e ellipse overlay.

Optical images
--------------
The PNGs in OPTICAL_IMAGES were saved from an external image viewer and
still have RA/Dec axes, tick labels and a compass arrow baked into the
raster (they're not separate plot elements we can restyle). Rather than
show three panels with mismatched coordinate ranges/gridlines next to the
pixel-axis Hα row, the axis margin is auto-cropped off (see
crop_axes_margin) and the panel is shown with no axes at all - the
in-image compass arrow is kept for orientation, and the magenta
segmentation contour is left untouched since it's scientifically relevant,
not a plot decoration.

Usage
-----
    python notebooks/thesis_galaxies.py
    python notebooks/thesis_galaxies.py --show

Output: outputs/figures/thesis_galaxy_comparison.pdf and .png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from scipy import ndimage

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from src.utils.io import read_sami_halpha_layers  # noqa: E402
from src.utils.plotting import plot_map  # noqa: E402
from src.utils.halpha_map import BACKGROUND_COLOR, build_re_ellipse, find_halpha_fits  # noqa: E402

FIGURE_DIR = PIPELINE_ROOT / "outputs" / "figures"
OPTICAL_IMAGES = {
    "above": FIGURE_DIR / "optical_above_ms.png",
    "on":    FIGURE_DIR / "optical_on_ms.png",
    "below": FIGURE_DIR / "optical_below_ms.png",
}

# Order matches the SFMS position, left to right
GALAXIES = [
    {"key": "above", "catid": 77967,  "label": "Above MS", "delta_ms": "+1.29"},
    {"key": "on",    "catid": 14812,  "label": "On MS",    "delta_ms": "+0.05"},
    {"key": "below", "catid": 107528, "label": "Below MS", "delta_ms": "−4.19"},
]

HALPHA_LAYER = "sum"

OUT_STEM = FIGURE_DIR / "thesis_galaxy_comparison"


def crop_axes_margin(img: np.ndarray) -> np.ndarray:
    """
    Crop the RA/Dec axis margin (tick labels, spines, axis labels) off a
    saved viewer PNG, keeping only the plotted image content.

    The margin is plain white background with sparse, small, disconnected
    black text/tick marks; the image content is one large, dense,
    contiguous non-white region. Taking the bounding box of the largest
    connected non-white component reliably separates the two without
    hardcoding pixel offsets that would only be valid for one image size.
    """
    gray = img.mean(axis=2)
    nonwhite = gray < 250
    labeled, n_components = ndimage.label(nonwhite, structure=np.ones((3, 3)))
    sizes = ndimage.sum(nonwhite, labeled, range(1, n_components + 1))
    biggest = np.argmax(sizes) + 1
    ys, xs = np.where(labeled == biggest)
    return img[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def plot_optical_panel(ax, img_path: Path, catid: int):
    img = np.array(Image.open(img_path).convert("RGB"))
    cropped = crop_axes_margin(img)
    ax.imshow(cropped, origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"CATID {catid}", fontsize=10, pad=4)


def plot_halpha_panel(ax, catid: int, delta_ms: str, label: str):
    fits_path = find_halpha_fits(catid)
    flux_layers, err_layers, header, meta = read_sami_halpha_layers(str(fits_path))
    flux_map = flux_layers[HALPHA_LAYER]

    ellipse = build_re_ellipse(catid, header)

    im = plot_map(
        flux_map,
        ax=ax,
        background_color=BACKGROUND_COLOR,
        ellipse=ellipse,
    )
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Flux", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title(f"{label} ($\\Delta$ms = {delta_ms})", fontsize=9, pad=4)
    ax.set_xlabel("X pixel", fontsize=8)
    ax.tick_params(labelsize=7)


def build_figure():
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.15, wspace=0.35)

    axes_opt = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axes_ha = [fig.add_subplot(gs[1, i]) for i in range(3)]

    for i, gal in enumerate(GALAXIES):
        plot_optical_panel(axes_opt[i], OPTICAL_IMAGES[gal["key"]], gal["catid"])
        plot_halpha_panel(axes_ha[i], gal["catid"], gal["delta_ms"], gal["label"])

    axes_opt[0].set_ylabel("Optical", fontsize=10, fontweight="bold", labelpad=8)
    axes_ha[0].set_ylabel("H$\\alpha$ flux", fontsize=10, fontweight="bold")

    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", action="store_true", help="Display the figure interactively instead of only saving it.")
    args = parser.parse_args()

    fig = build_figure()

    if args.show:
        plt.show()
    else:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{OUT_STEM}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(f"{OUT_STEM}.png", dpi=300, bbox_inches="tight")
        print(f"Saved -> {OUT_STEM}.pdf")
        print(f"Saved -> {OUT_STEM}.png")


if __name__ == "__main__":
    main()
