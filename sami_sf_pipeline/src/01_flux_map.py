#!/usr/bin/env python3
"""
01_flux_map.py
===============
Load the raw (unmasked) Halpha FITS cube for a given galaxy CATID and plot
its flux map.

This is a raw-data sanity-check step: it reads straight from data/raw/, so
what it shows is what the pipeline started from, before any continuum
masking (see 02_continuum_mask.py) or metric calculation (see 03_metrics/).

Usage
-----
    python src/01_flux_map.py --catid 7139
    python src/01_flux_map.py --catid 7139 --layer fit --show

CATID matching: SAMI Halpha filenames are "<CATID>_<IFU>_Halpha_default_1-comp.fits".
If a galaxy has multiple IFU observations, the first match (sorted) is used
and a warning is printed.

Call using: python sami_sf_pipeline/src/01_flux_map.py --catid {x} --show
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

OUT_FIGURE_DIR = PIPELINE_ROOT / "outputs" / "figures" / "flux_maps"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catid", type=int, required=True, help="SAMI CATID to plot")
    parser.add_argument(
        "--layer", choices=["fit", "sum"], default="sum",
        help="Which Halpha slice to plot: Gaussian fit (fit) or summed flux (sum). Default: sum.",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot interactively instead of only saving it.")
    parser.add_argument("--outdir", type=Path, default=OUT_FIGURE_DIR, help="Directory to save the figure into.")
    args = parser.parse_args()

    fits_path = find_halpha_fits(args.catid)
    flux_layers, err_layers, header, meta = read_sami_halpha_layers(str(fits_path))

    flux_map = flux_layers[args.layer]
    print(f"Loaded: {fits_path.name}")
    print(f"  Layer: {args.layer}   shape: {flux_map.shape}")
    print(f"  Flux range (finite pixels): {np.nanmin(flux_map):.4g} - {np.nanmax(flux_map):.4g}")

    ellipse = build_re_ellipse(args.catid, header)

    args.outdir.mkdir(parents=True, exist_ok=True)
    outfile = args.outdir / f"{args.catid}_Halpha_{args.layer}_raw.png"

    plot_map(
        flux_map,
        title=f"CATID {args.catid} - Hα flux ({args.layer})",
        outfile=None if args.show else str(outfile),
        background_color=BACKGROUND_COLOR,
        ellipse=ellipse,
    )

    if not args.show:
        print(f"Saved -> {outfile}")


if __name__ == "__main__":
    main()
