"""
Shared helpers for locating raw SAMI Halpha cubes and building the r_e
ellipse overlay for their flux maps.

Factored out of 01_flux_map.py so other scripts (e.g.
notebooks/thesis_galaxies.py) can reuse the exact same lookup/geometry
logic rather than re-implementing it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils.wang import load_size_lookup

RAW_HALPHA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "sami_dr3_Halpha_maps"

PLATE_SCALE = 0.5           # arcsec/px, matches 03_metrics/smoothness.py convention
BACKGROUND_COLOR = "0.6"    # grey fill for no-data (NaN) pixels, distinct from the pale-yellow low end of YlOrRd


def find_halpha_fits(catid: int, raw_dir: Path = RAW_HALPHA_DIR) -> Path:
    """Locate the raw 1-comp Halpha FITS file for a CATID."""
    matches = sorted(raw_dir.glob(f"{catid}_*_Halpha_default_1-comp.fits"))
    if not matches:
        raise FileNotFoundError(
            f"No raw Halpha FITS found for CATID {catid} in {raw_dir}"
        )
    if len(matches) > 1:
        print(f"  Warning: {len(matches)} files matched CATID {catid} - using {matches[0].name}")
    return matches[0]


def build_re_ellipse(catid: int, header) -> dict | None:
    """
    Build the matplotlib Ellipse kwargs for a galaxy's r_e, from PA and
    ellip in the SAMI_Cindex_data catalogues (SF-preferred, Composite
    fallback - see load_size_lookup). Returns None (with a warning) if the
    galaxy or any of its size fields aren't found.

    Semi-major axis a = r_e (px); semi-minor axis b = a * (1 - ellip).
    PA is the standard astronomical position angle (degrees, North through
    East). The raw Halpha cubes have CDELT1 < 0 / CDELT2 > 0 (RA---TAN /
    DEC--TAN, no rotation term), i.e. North is up and East is left in the
    plotted image, so PA=0 (North) is vertical and increasing PA rotates
    counterclockwise - hence angle = PA + 90 in matplotlib's from-+x-axis
    convention.
    """
    lookup = load_size_lookup()
    if catid not in lookup.index:
        print(f"  Warning: no size entry (r_e/PA/ellip) found for CATID {catid} in SAMI_Cindex_data - plotting without r_e ellipse")
        return None

    row = lookup.loc[catid]
    re_arcsec, pa_deg, ellip = row["re"], row["PA"], row["ellip"]
    if np.isnan(re_arcsec) or np.isnan(pa_deg) or np.isnan(ellip):
        print(f"  Warning: incomplete size data for CATID {catid} (re={re_arcsec}, PA={pa_deg}, ellip={ellip}) - plotting without r_e ellipse")
        return None

    a_px = re_arcsec / PLATE_SCALE
    b_px = a_px * (1 - ellip)
    angle_deg = pa_deg + 90

    cx = header["CRPIX1"] - 1  # FITS 1-indexed -> 0-indexed pixel coords
    cy = header["CRPIX2"] - 1

    return {"xy": (cx, cy), "width": 2 * a_px, "height": 2 * b_px, "angle": angle_deg}
