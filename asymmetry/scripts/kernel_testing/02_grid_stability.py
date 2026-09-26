#!/usr/bin/env python3
"""
02_grid_stability.py
======================
Tests whether A_min's correlation with resolution (n_res) is a selection/
order-statistic artifact of taking the MINIMUM over the 100-candidate
centre grid search, rather than (only, or at all) a missing background-
asymmetry term.

Motivation: 01_noise_floor.py found that noise pushed through compute_A at
a galaxy's already-best-fit centre barely predicts A_min or its n_res
correlation (R^2=0.006, vs S's B_S-analog R^2=0.94) - the opposite of what
a direct B_S-style port assumes. One candidate explanation: A_min is the
minimum of 100 candidate-centre evaluations computed on the real (noisy)
signal itself. For a poorly-resolved galaxy, those 100 candidates are
noisier/more spread out (fewer independent pixels -> shakier per-candidate
alignment), so the *minimum* of 100 noisier values is pulled down further
than the minimum of 100 tighter values would be - a selection effect that
tracks resolution without needing any separate background/noise term.

Method: replays the *exact* production grid_search_centre logic (same
GRID_POINTS/GRID_RANGE, same compute_A calls, on the real I_clean - no new
randomness at all) for every in_full_sam galaxy, but keeps the full array
of 100 A values instead of discarding all but the minimum. Compares
A_min (=today's production value) against A_mean and A_median computed
from that *same* underlying grid, and checks whether mean/median show a
weaker n_res correlation than min on identical underlying data.

Output: outputs/02_grid_stability/grid_stability_results.csv,
        outputs/02_grid_stability/grid_stability_diagnostics.png,
        printed correlation summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c

sys.path.insert(0, str(c.PIPELINE_ROOT))
from src.utils.io import read_masked_halpha  # noqa: E402

OUT = c.ensure_outdir("02_grid_stability")


def grid_search_full(prod, img, grid_points, grid_range):
    """Exact replay of production's grid_search_centre(), but returns the
    full array of 100 A values instead of only the minimum."""
    array_centre = (img.shape[0] - 1) / 2.0
    offsets = np.linspace(-grid_range, grid_range, grid_points)

    values = np.full((grid_points, grid_points), np.nan)
    for i, dy in enumerate(offsets):
        for j, dx in enumerate(offsets):
            cy = array_centre + dy
            cx = array_centre + dx
            A, _, _ = prod.compute_A(img, cy, cx)
            values[i, j] = A
    return values


def main():
    c.print_header("Grid stability: A_min vs A_mean/A_median on the SAME 100-candidate grid")

    prod = c.import_production_asymmetry()
    df = c.load_full_sam()
    print(f"Sample: in_full_sam, n={len(df)}")

    rows = []
    n_missing = 0
    n_mismatch = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Grid stability"):
        catid = int(row["CATID"])
        halpha_path = c.find_halpha_path(catid)
        if halpha_path is None:
            n_missing += 1
            continue

        flux_cube, err_cube = read_masked_halpha(str(halpha_path))
        flux = np.nan_to_num(flux_cube[prod.HALPHA_LAYER], nan=0.0, posinf=0.0, neginf=0.0)
        error = np.nan_to_num(err_cube[prod.HALPHA_LAYER], nan=0.0, posinf=0.0, neginf=0.0)

        img = prod.apply_error_subtraction(flux, error)
        if np.sum(img != 0) == 0:
            continue

        grid = grid_search_full(prod, img, prod.GRID_POINTS, prod.GRID_RANGE)
        flat = grid[~np.isnan(grid)]
        if flat.size == 0:
            continue

        a_min_recomputed = float(np.min(flat))
        # Sanity check against production's own stored A_min for this galaxy.
        if not np.isclose(a_min_recomputed, row["A_min"], rtol=1e-4, atol=1e-4):
            n_mismatch += 1

        rows.append({
            "CATID": catid,
            "A_min": a_min_recomputed,               # matches production
            "A_min_stored": row["A_min"],             # from asymmetry.csv, for the sanity check
            "A_mean": float(np.mean(flat)),
            "A_median": float(np.median(flat)),
            "A_std": float(np.std(flat)),
            "A_max": float(np.max(flat)),
            "A_range": float(np.max(flat) - np.min(flat)),
            "n_grid_vals": int(flat.size),
            "n_res": row["n_res"],
            "seeing": row["seeing"],
            "re_used": row["re_used"],
            "z": row["z"],
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT / "grid_stability_results.csv", index=False)

    print(f"\nMissing FITS (skipped)     : {n_missing}")
    print(f"Galaxies processed         : {len(result_df)}")
    print(f"A_min sanity mismatches    : {n_mismatch} / {len(result_df)} "
          f"(recomputed grid-min vs asymmetry.csv's stored A_min; should be ~0)")

    print(f"\nA_min    : median={result_df['A_min'].median():.4f}")
    print(f"A_mean   : median={result_df['A_mean'].median():.4f}")
    print(f"A_median : median={result_df['A_median'].median():.4f}")
    print(f"A_range  : median={result_df['A_range'].median():.4f}  "
          f"(spread across the 100 candidate centres - the quantity the selection-effect hypothesis says should track n_res)")

    print("\n--- Does A_range (grid spread) itself track resolution, as the selection-effect hypothesis predicts? ---")
    for driver in ["n_res", "seeing", "re_used", "z"]:
        rho, p = spearmanr(result_df["A_range"], result_df[driver])
        print(f"  A_range vs {driver:10s} rho={rho:+.3f} (p={p:.1e})")

    print("\n--- n_res correlation: A_min vs A_mean vs A_median (SAME underlying 100-value grid per galaxy) ---")
    for stat in ["A_min", "A_mean", "A_median"]:
        for driver in ["n_res", "seeing", "re_used", "z"]:
            rho, p = spearmanr(result_df[stat], result_df[driver])
            print(f"  {stat:10s} vs {driver:10s} rho={rho:+.3f} (p={p:.1e})")
        print()

    print("--- Rank agreement between A_min and A_mean/A_median (do they still order galaxies similarly?) ---")
    for stat in ["A_mean", "A_median"]:
        rho, p = spearmanr(result_df["A_min"], result_df[stat])
        print(f"  A_min vs {stat}: rho={rho:+.3f} (p={p:.1e})")

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.scatter(result_df["n_res"], result_df["A_range"], s=10, alpha=0.5, color="firebrick")
    ax.set_xlabel(r"$n_{res} = r_e / \mathrm{seeing}$")
    ax.set_ylabel(r"$A_{range}$ (max$-$min over 100 candidates)")
    ax.set_title("Grid spread vs resolution")

    ax = axes[1]
    ax.scatter(result_df["n_res"], result_df["A_min"], s=10, alpha=0.4, color="firebrick", label="A_min (production)")
    ax.scatter(result_df["n_res"], result_df["A_mean"], s=10, alpha=0.4, color="steelblue", label="A_mean")
    ax.set_xlabel(r"$n_{res}$")
    ax.set_ylabel("A")
    ax.legend(fontsize=8)
    ax.set_title("A_min vs A_mean against resolution")

    ax = axes[2]
    ax.scatter(result_df["A_min"], result_df["A_mean"], s=10, alpha=0.5, color="seagreen")
    lims = [result_df["A_min"].min(), max(result_df["A_min"].max(), result_df["A_mean"].max())]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
    ax.set_xlabel("A_min")
    ax.set_ylabel("A_mean")
    ax.set_title("Rank agreement: A_min vs A_mean")

    fig.suptitle(f"Grid stability (in_full_sam, n={len(result_df)}, 100 candidates/galaxy)")
    fig.tight_layout()
    fig.savefig(OUT / "grid_stability_diagnostics.png", dpi=150)
    plt.close(fig)

    print(f"\nWrote: {OUT / 'grid_stability_results.csv'}")
    print(f"Wrote: {OUT / 'grid_stability_diagnostics.png'}")


if __name__ == "__main__":
    main()
