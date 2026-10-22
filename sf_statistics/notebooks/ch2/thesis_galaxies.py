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
    python sf_statistics/notebooks/ch2/thesis_galaxies.py
    python sf_statistics/notebooks/ch2/thesis_galaxies.py --show

Input : data/optical/optical_{above,on,below}_ms.png
        data/fits/sami_dr3_Halpha_maps/<CATID>_<IFU>_Halpha_default_1-comp.fits
Output: outputs/ch2/figures/thesis_galaxy_comparison.pdf and .png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from PIL import Image
from scipy import ndimage

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_ROOT))

from src.utils.io import read_sami_halpha_layers  # noqa: E402
from src.utils.plotting import plot_map  # noqa: E402
from src.utils.halpha_map import BACKGROUND_COLOR, build_re_ellipse, find_halpha_fits  # noqa: E402
from src.utils.paths import MASTER_CSV, OPTICAL_DIR, OUTPUTS_DIR  # noqa: E402

FIGURE_DIR = OUTPUTS_DIR / "ch2" / "figures"
OPTICAL_IMAGES = {
    "above": OPTICAL_DIR / "optical_above_ms.png",
    "on":    OPTICAL_DIR / "optical_on_ms.png",
    "below": OPTICAL_DIR / "optical_below_ms.png",
}

# Order matches the SFMS position, left to right
# delta_ms for each galaxy is read from master.csv (added by 05_sfms.py)
GALAXIES = [
    {"key": "above", "catid": 77967,  "label": "Above MS"},
    {"key": "on",    "catid": 14812,  "label": "On MS"},
    {"key": "below", "catid": 611002, "label": "Below MS"},
]

HALPHA_LAYER = "sum"

OUT_STEM = FIGURE_DIR / "thesis_galaxy_comparison"
FLUX_LABEL = r"H$\alpha$ flux [$10^{-16}$ erg s$^{-1}$ cm$^{-2}$ spaxel$^{-1}$]"


def load_delta_ms(catids: list[int]) -> dict[int, float]:
    """Return {CATID: delta_ms} from master.csv for the given galaxies."""
    master = pd.read_csv(MASTER_CSV, usecols=["CATID", "delta_ms"])
    lookup = master.set_index("CATID")["delta_ms"]
    missing = [c for c in catids if c not in lookup.index or pd.isna(lookup[c])]
    if missing:
        raise KeyError(f"No delta_ms in {MASTER_CSV} for CATID(s) {missing}. Run 05_sfms.py first.")
    return {c: float(lookup[c]) for c in catids}


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


def plot_optical_panel(ax, img_path: Path, catid: int, delta_ms: float, label: str):
    img = np.array(Image.open(img_path).convert("RGB"))
    cropped = crop_axes_margin(img)
    ax.imshow(cropped, origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"CATID {catid}\n{label} ($\\Delta$ms = {delta_ms:+.2f})", fontsize=9, pad=4)

def plot_halpha_panel(ax, catid: int):
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
    if catid == GALAXIES[-1]["catid"]:
        cbar.set_label(f"Hα Flux", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlim(3, 48); ax.set_ylim(3, 48)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")


def build_figure():
    fig = plt.figure(figsize=(7, 4.3))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.08, wspace=0.25)

    axes_opt = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axes_ha = [fig.add_subplot(gs[1, i]) for i in range(3)]

    delta_ms = load_delta_ms([gal["catid"] for gal in GALAXIES])
    for i, gal in enumerate(GALAXIES):
        plot_optical_panel(axes_opt[i], OPTICAL_IMAGES[gal["key"]], gal["catid"],
                   delta_ms[gal["catid"]], gal["label"])
        plot_halpha_panel(axes_ha[i], gal["catid"])

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
