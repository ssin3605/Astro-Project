#!/usr/bin/env python3
"""
05_kernel_scale_sensitivity.py
=================================
T5 - is 0.35 x r_e special, or would the same scientific conclusions hold
across the range explored in the original smoothness_test_02.py (0.10,
0.25, 0.35, 0.50)? This is the piece of the "why 0.35?" defence that
doesn't rely on the Conselice-heritage argument (see README.md) - it
demonstrates the choice isn't load-bearing.

For each kernel scale in KERNEL_SCALES, on in_full_sam:
    1. Recompute S for every galaxy at that scale (via the actual
       production compute_S()/gaussian_filter, not a re-implementation).
    2. Resolution-safety fraction: what fraction of the sample has
       sigma_kernel > 2 x sigma_PSF at that scale (the floor used in the
       original diagnostic conversation).
    3. Rank stability: Spearman rho of S(scale) against S(0.35) across
       galaxies - do galaxies keep approximately the same relative
       clumpiness ranking regardless of the exact scale? (This is what
       actually matters for an ML feature - it cares about relative
       ordering/values across galaxies, not the absolute number.)
    4. Driver-correlation stability: re-run the raw Spearman rho(S, driver)
       table from 04_resolution_matched_validation.py at each scale, to
       see whether which drivers "matter" depends on the exact kernel
       choice.

Output: outputs/05_kernel_scale_sensitivity/kernel_scale_S_values.csv,
        outputs/05_kernel_scale_sensitivity/driver_rho_by_scale.csv,
        outputs/05_kernel_scale_sensitivity/sensitivity_plots.png,
        printed summary.
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

OUT = c.ensure_outdir("05_kernel_scale_sensitivity")

KERNEL_SCALES = [0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
PRODUCTION_SCALE = 0.35
PSF_SAFETY_MULTIPLE = 2.0


def main():
    c.print_header("T5: Kernel-scale sensitivity")

    prod = c.import_production_smoothness()
    df = c.load_full_sam()
    print(f"Sample: in_full_sam, n={len(df)}")
    print(f"Kernel scales tested: {KERNEL_SCALES}")

    wide_rows = {"CATID": df["CATID"].values}
    safety_summary = []

    for ks in tqdm(KERNEL_SCALES, desc="Kernel scales"):
        S_vals, sigma_over_psf_vals = [], []
        for _, row in df.iterrows():
            catid = int(row["CATID"])
            halpha_path = c.find_halpha_path(catid)
            if halpha_path is None:
                S_vals.append(np.nan)
                sigma_over_psf_vals.append(np.nan)
                continue
            result = prod.process_one(halpha_path, row["re_used"], ks, prod.PLATE_SCALE)
            S_vals.append(result["S"])
            sigma_over_psf_vals.append(result["sigma_arcsec"] / row["psf_sigma_arcsec"])

        wide_rows[f"S_{ks}"] = S_vals
        wide_rows[f"sigma_over_psf_{ks}"] = sigma_over_psf_vals

        safe_frac = np.mean(np.array(sigma_over_psf_vals) > PSF_SAFETY_MULTIPLE)
        safety_summary.append({"kernel_scale": ks, "resolution_safe_fraction": safe_frac})

    wide = pd.DataFrame(wide_rows)
    wide.to_csv(OUT / "kernel_scale_S_values.csv", index=False)

    safety_df = pd.DataFrame(safety_summary)
    print(f"\n--- Resolution-safety fraction (sigma_kernel > {PSF_SAFETY_MULTIPLE}x sigma_PSF) by scale ---")
    print(safety_df.to_string(index=False))

    print(f"\n--- Rank stability: Spearman rho(S(scale), S({PRODUCTION_SCALE})) ---")
    rank_stability = []
    ref = wide[f"S_{PRODUCTION_SCALE}"]
    for ks in KERNEL_SCALES:
        rho, p = spearmanr(wide[f"S_{ks}"], ref, nan_policy="omit")
        rank_stability.append({"kernel_scale": ks, "rho_vs_production": rho})
        print(f"  scale={ks:.2f}: rho={rho:.4f}")
    rank_df = pd.DataFrame(rank_stability)

    print("\n--- Driver-correlation stability across kernel scale ---")
    driver_rows = []
    for ks in KERNEL_SCALES:
        tmp = df.copy()
        tmp["S"] = wide[f"S_{ks}"].values
        tab = c.spearman_table(tmp, "S", c.CANDIDATE_DRIVERS)
        tab["kernel_scale"] = ks
        driver_rows.append(tab[["kernel_scale", "column", "label", "rho", "p"]])
    driver_by_scale = pd.concat(driver_rows, ignore_index=True)
    driver_by_scale.to_csv(OUT / "driver_rho_by_scale.csv", index=False)

    pivot = driver_by_scale.pivot(index="column", columns="kernel_scale", values="rho")
    pivot = pivot.reindex(pivot[PRODUCTION_SCALE].abs().sort_values(ascending=False).index)
    print(pivot.round(3).to_string())

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.plot(safety_df["kernel_scale"], safety_df["resolution_safe_fraction"], "o-", color="firebrick")
    ax.axvline(PRODUCTION_SCALE, color="grey", ls="--", lw=1, label="production (0.35)")
    ax.set_xlabel("kernel scale (x r_e)")
    ax.set_ylabel(f"fraction with sigma_kernel > {PSF_SAFETY_MULTIPLE}x sigma_PSF")
    ax.set_title("Resolution-safety vs kernel scale")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(rank_df["kernel_scale"], rank_df["rho_vs_production"], "o-", color="steelblue")
    ax.axvline(PRODUCTION_SCALE, color="grey", ls="--", lw=1, label="production (0.35)")
    ax.set_ylim(0.5, 1.02)
    ax.set_xlabel("kernel scale (x r_e)")
    ax.set_ylabel(f"Spearman rho of S(scale) vs S({PRODUCTION_SCALE})")
    ax.set_title("Rank stability of S vs kernel scale")
    ax.legend(fontsize=8)

    ax = axes[2]
    for col in pivot.index:
        ax.plot(pivot.columns, pivot.loc[col], "o-", label=col, alpha=0.8, lw=1.5, markersize=4)
    ax.axvline(PRODUCTION_SCALE, color="grey", ls="--", lw=1)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("kernel scale (x r_e)")
    ax.set_ylabel("Spearman rho(S, driver)")
    ax.set_title("Driver-correlation stability vs kernel scale")
    ax.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.0, 0.5))

    fig.tight_layout()
    fig.savefig(OUT / "sensitivity_plots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote: {OUT / 'kernel_scale_S_values.csv'}")
    print(f"Wrote: {OUT / 'driver_rho_by_scale.csv'}")
    print(f"Wrote: {OUT / 'sensitivity_plots.png'}")


if __name__ == "__main__":
    main()
