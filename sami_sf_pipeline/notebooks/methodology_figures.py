#!/usr/bin/env python3
"""
methodology_figures.py
========================
"How is A / S calculated?" 3-panel figures: masked Halpha map -> rotated
(A) / smoothed (S) -> residual, using the pipeline's own grid-search
rotation (src/03_metrics/asymmetry.py) and Gaussian-smoothing (src/03_
metrics/smoothness.py) logic exactly, so the numbers on these figures
match outputs/csv/asymmetry.csv and smoothness.csv.

Both panels use the base statistic before its noise/background
correction (A_no_sub / S_no_B_S_sub) - B_S/A_BS are medians over 50
random noise-map realisations and have no single visual instance to
show in a 3-panel figure; the reported A_no_sub/S_no_B_S_sub value is
printed on the figure for reference.

Styling matches src/utils/plotting.py::plot_map (same colormap/background
convention as 01_flux_map.py), extended to 3 shared-axes panels with one
shared color scale across all three (per the requested style: makes clear
just how small the residual is relative to the input flux).

Usage
-----
    python notebooks/methodology_figures.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from src.utils.io import read_masked_halpha  # noqa: E402
from src.utils.plotting import plot_map  # noqa: E402
from src.utils.halpha_map import BACKGROUND_COLOR  # noqa: E402

# "03_metrics" isn't a valid dotted-import identifier (leading digit) -
# load it the same way importlib.import_module handles it elsewhere.
asymmetry = importlib.import_module("src.03_metrics.asymmetry")
smoothness = importlib.import_module("src.03_metrics.smoothness")

MASKED_HALPHA_DIR = PIPELINE_ROOT / "data" / "masked" / "halpha"
MASTER_CSV = PIPELINE_ROOT / "outputs" / "csv" / "master_final.csv"
OUT_DIR = PIPELINE_ROOT / "outputs" / "figures" / "methodology"

A_CATID = 517868
S_CATID = 71146


def _find_masked(catid: int) -> Path:
    matches = sorted(MASKED_HALPHA_DIR.glob(f"{catid}_*_Halpha_masked.fits"))
    if not matches:
        raise FileNotFoundError(f"No masked Halpha FITS for CATID {catid}.")
    return matches[0]


def _nan_where(arr: np.ndarray, background_mask: np.ndarray) -> np.ndarray:
    """Copy of arr with background_mask pixels set to NaN, for display only."""
    out = arr.copy()
    out[background_mask] = np.nan
    return out


def _crop_to_content(panels: list[np.ndarray], margin: int = 5) -> list[np.ndarray]:
    """
    Crop every panel (already NaN-background) to the bounding box of finite
    pixels across ALL panels combined, + margin px on each side (clipped to
    the array edge). Display-only - identical crop applied to every panel
    so pixel (i, j) still means the same sky position in each.
    """
    any_finite = np.zeros_like(panels[0], dtype=bool)
    for p in panels:
        any_finite |= np.isfinite(p)
    ys, xs = np.where(any_finite)
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, panels[0].shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, panels[0].shape[1])
    return [p[y0:y1, x0:x1] for p in panels]


def build_asymmetry_figure(catid: int = A_CATID):
    fits_path = _find_masked(catid)
    flux_cube, err_cube = read_masked_halpha(str(fits_path))
    flux = np.nan_to_num(flux_cube[asymmetry.HALPHA_LAYER], nan=0.0, posinf=0.0, neginf=0.0)

    img = flux  # no error subtraction, matching asymmetry.py
    (best_cy, best_cx), A_no_sub = asymmetry.grid_search_centre(img)
    A_value, rotated, residual_signed = asymmetry.compute_A(img, best_cy, best_cx)
    assert np.isclose(A_value, A_no_sub)

    common_mask = asymmetry._common_mask(img, rotated)
    residual_abs = np.abs(residual_signed)

    # Display-only background masks (do not affect the computed statistic above)
    original_bg = img == 0
    rotated_bg = ~(rotated > 1e-3 * np.max(img))
    residual_bg = ~common_mask

    panel1 = _nan_where(img, original_bg)
    panel2 = _nan_where(rotated, rotated_bg)
    panel3 = _nan_where(residual_abs, residual_bg)
    panel1, panel2, panel3 = _crop_to_content([panel1, panel2, panel3])

    vmax = np.nanmax([np.nanmax(panel1), np.nanmax(panel2), np.nanmax(panel3)])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    titles = [
        "1. Masked H$\\alpha$ map ($I$)",
        f"2. Rotated 180° about best-fit\ncenter (cy={best_cy:.1f}, cx={best_cx:.1f})",
        "3. Residual $|I - I_{\\rm rot}|$\n(summed over common footprint $\\to$ A)",
    ]
    for ax, panel, title in zip(axes, (panel1, panel2, panel3), titles):
        plot_map(panel, ax=ax, title=title, vmin=0, vmax=vmax, background_color=BACKGROUND_COLOR)

    fig.suptitle(f"CATID {catid} — Asymmetry (A = {A_no_sub:.3f})", fontsize=13, y=1.02)
    im = axes[0].images[0]
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Flux")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outfile = OUT_DIR / f"{catid}_asymmetry_method.png"
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"CATID {catid}: A_no_sub={A_no_sub:.4f}  best_centre=({best_cy:.2f},{best_cx:.2f})")
    print(f"Saved -> {outfile}")


def build_smoothness_figure(catid: int = S_CATID):
    fits_path = _find_masked(catid)
    flux_cube, err_cube = read_masked_halpha(str(fits_path))
    flux = np.nan_to_num(flux_cube[smoothness.HALPHA_LAYER], nan=0.0, posinf=0.0, neginf=0.0)

    master = pd.read_csv(MASTER_CSV)
    row = master.loc[master["CATID"] == catid].iloc[0]
    re_arcsec = float(row["re_used"])
    sigma_px = smoothness.KERNEL_SCALE * re_arcsec / smoothness.PLATE_SCALE

    # Same calculation as smoothness.py: positive residuals summed over the footprint
    footprint = flux != 0
    flux_smoothed, residual = smoothness.positive_residual(flux, sigma_px)
    S_value = np.sum(residual[footprint]) / np.sum(flux[footprint])

    # All 3 panels are shown over the original footprint only
    footprint_bg = ~footprint

    panel1 = _nan_where(flux, footprint_bg)
    panel2 = _nan_where(flux_smoothed, footprint_bg)
    panel3 = _nan_where(residual, footprint_bg)
    panel1, panel2, panel3 = _crop_to_content([panel1, panel2, panel3])

    vmax = np.nanmax([np.nanmax(panel1), np.nanmax(panel2), np.nanmax(panel3)])
    vmin = np.nanmin([np.nanmin(panel1), np.nanmin(panel2)])  # residual >= 0, let panel1/2 set the floor

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    titles = [
        "1. Masked H$\\alpha$ map ($I$)",
        f"2. Gaussian-smoothed\n($\\sigma$ = {sigma_px:.2f} px = 0.35 $r_e$)",
        "3. Positive residual max($I - I_{\\rm smooth}$, 0)\n(summed over footprint $\\to$ S)",
    ]
    for ax, panel, title in zip(axes, (panel1, panel2, panel3), titles):
        plot_map(panel, ax=ax, title=title, vmin=vmin, vmax=vmax, background_color=BACKGROUND_COLOR)

    fig.suptitle(f"CATID {catid} — Smoothness/Clumpiness (S = {S_value:.3f})", fontsize=13, y=1.02)
    im = axes[0].images[0]
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Flux")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outfile = OUT_DIR / f"{catid}_smoothness_method.png"
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"CATID {catid}: S_no_B_S_sub={S_value:.4f}  sigma_px={sigma_px:.3f}  re_used={re_arcsec}\"")
    print(f"Saved -> {outfile}")


def main():
    build_asymmetry_figure()
    build_smoothness_figure()


if __name__ == "__main__":
    main()
