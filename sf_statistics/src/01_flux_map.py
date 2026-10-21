#!/usr/bin/env python3
"""
Load the raw (unmasked) Halpha FITS cube for a given galaxy CATID and plot
its flux map.

Usage
-----
    python sf_statistics/src/01_flux_map.py --catid 14812

Plots the summed-flux slice of the cube.

CATID naming: SAMI Halpha filenames are "<CATID>_<IFU>_Halpha_default_1-comp.fits".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from src.utils.io import read_sami_halpha_layers
from src.utils.plotting import plot_map
from src.utils.halpha_map import BACKGROUND_COLOR, build_re_ellipse, find_halpha_fits
from src.utils.paths import OUTPUTS_DIR

OUT_FIGURE_DIR = OUTPUTS_DIR / "ch2" / "figures"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catid", type=int, required=True, help="SAMI CATID to plot")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively instead of only saving it.")
    parser.add_argument("--outdir", type=Path, default=OUT_FIGURE_DIR, help="Directory to save the figure into.")
    args = parser.parse_args()

    fits_path = find_halpha_fits(args.catid)
    flux_layers, err_layers, header, meta = read_sami_halpha_layers(str(fits_path))

    flux_map = flux_layers["sum"]
    print(f"Loaded: {fits_path.name}")
    print(f"  Shape: {flux_map.shape}")
    print(f"  Flux range (finite pixels): {np.nanmin(flux_map):.4g} - {np.nanmax(flux_map):.4g}")

    ellipse = build_re_ellipse(args.catid, header)

    args.outdir.mkdir(parents=True, exist_ok=True)
    outfile = args.outdir / f"raw_Halpha_{args.catid}_map.png"

    plot_map(
        flux_map,
        title=f"CATID {args.catid} - Hα flux",
        outfile=None if args.show else str(outfile),
        background_color=BACKGROUND_COLOR,
        ellipse=ellipse,
    )

    if not args.show:
        print(f"Saved -> {outfile}")


if __name__ == "__main__":
    main()
