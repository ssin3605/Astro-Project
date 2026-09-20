#!/usr/bin/env python3
"""
06_fixed_kpc_comparison.py
=============================
T3 - the direct, data-driven answer to "wouldn't a kernel defined in kpc
be more physical?" 01_unit_invariance.py already showed that r_e-in-kpc
vs r_e-in-arcsec is not actually a choice (they're algebraically and
numerically identical once you go through the pixel grid). The real
alternative worth testing is a genuinely different definition: a FIXED
physical kernel width (e.g. sigma = 1 kpc for every galaxy, not 0.35 x
r_e), which does NOT cancel against z, since it isn't tied to each
galaxy's own angular size.

For each fixed physical scale in FIXED_KPC_SCALES, on in_full_sam:
    1. sigma_arcsec = X_kpc / kpc_per_arcsec(z); sigma_px = sigma_arcsec /
       PLATE_SCALE - this SHRINKS in pixels at higher z (opposite of the
       production kernel, whose pixel size is z-independent - see
       README.md), so it is expected to become resolution-limited for
       exactly the distant/small galaxies that 0.35 x r_e already
       struggles with, and worse for close/large ones it doesn't.
    2. Recompute S_kpc via the same production formula.
    3. Compare, against the production S (0.35 x r_e):
       - correlation with n_res / sigma_over_psf (is a fixed-kpc kernel
         MORE or LESS resolution-confounded than the self-similar one?)
       - resolution-safety fraction (sigma_kernel > 2x sigma_PSF)
       - the same driver-correlation table as 04_resolution_matched_
         validation.py, to see if S_kpc's associations with candidate
         drivers are more or less resolution-contaminated.

Output: outputs/06_fixed_kpc_comparison/fixed_kpc_S_values.csv,
        outputs/06_fixed_kpc_comparison/fixed_kpc_comparison.png,
        printed summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c

sys.path.insert(0, str(c.PIPELINE_ROOT))
from src.utils.io import read_masked_halpha  # noqa: E402

OUT = c.ensure_outdir("06_fixed_kpc_comparison")

FIXED_KPC_SCALES = [0.5, 1.0, 2.0]
PSF_SAFETY_MULTIPLE = 2.0


def compute_S(I_clean: np.ndarray, sigma_px: float) -> float:
    smoothed = gaussian_filter(I_clean, sigma=sigma_px)
    residual = np.abs(I_clean - smoothed)
    denom = np.nansum(np.abs(I_clean))
    return float(np.nansum(residual) / denom) if denom != 0 else np.nan


def main():
    c.print_header("T3: Fixed-physical-scale (kpc) kernel vs self-similar (0.35 x r_e) kernel")

    prod = c.import_production_smoothness()
    df = c.load_full_sam()
    print(f"Sample: in_full_sam, n={len(df)}")

    results = {"CATID": df["CATID"].values}
    safety_rows = []

    for x_kpc in FIXED_KPC_SCALES:
        S_vals, sigma_over_psf_vals = [], []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"sigma={x_kpc}kpc"):
            catid = int(row["CATID"])
            halpha_path = c.find_halpha_path(catid)
            if halpha_path is None:
                S_vals.append(np.nan)
                sigma_over_psf_vals.append(np.nan)
                continue

            flux_cube, err_cube = read_masked_halpha(str(halpha_path))
            flux = flux_cube[prod.HALPHA_LAYER]
            err = err_cube[prod.HALPHA_LAYER]

            sigma_arcsec = x_kpc / row["kpc_per_as"]
            sigma_px = sigma_arcsec / prod.PLATE_SCALE

            S = compute_S(flux - err, sigma_px)
            S_vals.append(S)
            sigma_over_psf_vals.append(sigma_arcsec / row["psf_sigma_arcsec"])

        results[f"S_kpc_{x_kpc}"] = S_vals
        results[f"sigma_over_psf_kpc_{x_kpc}"] = sigma_over_psf_vals
        safe_frac = np.nanmean(np.array(sigma_over_psf_vals) > PSF_SAFETY_MULTIPLE)
        safety_rows.append({"scale": f"{x_kpc} kpc (fixed)", "resolution_safe_fraction": safe_frac})

    results_df = pd.DataFrame(results)
    results_df["S_production"] = df["S"].values
    results_df["n_res"] = df["n_res"].values
    results_df["sigma_over_psf_production"] = df["sigma_over_psf"].values
    results_df.to_csv(OUT / "fixed_kpc_S_values.csv", index=False)

    safety_rows.append({
        "scale": "0.35 x r_e (production)",
        "resolution_safe_fraction": np.mean(df["sigma_over_psf"] > PSF_SAFETY_MULTIPLE),
    })
    safety_df = pd.DataFrame(safety_rows)

    print(f"\n--- Resolution-safety fraction (sigma_kernel > {PSF_SAFETY_MULTIPLE}x sigma_PSF) ---")
    print(safety_df.to_string(index=False))

    print("\n--- Correlation with resolution proxy n_res: fixed-kpc kernels vs production ---")
    rho_prod, _ = spearmanr(results_df["S_production"], results_df["n_res"], nan_policy="omit")
    print(f"  S_production (0.35 x r_e)  vs n_res:  rho={rho_prod:+.3f}")
    for x_kpc in FIXED_KPC_SCALES:
        rho, p = spearmanr(results_df[f"S_kpc_{x_kpc}"], results_df["n_res"], nan_policy="omit")
        print(f"  S_kpc ({x_kpc} kpc fixed)     vs n_res:  rho={rho:+.3f}")

    print("\n--- Driver-correlation table: production S vs fixed-kpc S (1.0 kpc) ---")
    tmp = df.copy()
    prod_table = c.spearman_table(tmp, "S", c.CANDIDATE_DRIVERS)
    tmp_kpc = df.copy()
    tmp_kpc["S"] = results_df["S_kpc_1.0"].values
    kpc_table = c.spearman_table(tmp_kpc, "S", c.CANDIDATE_DRIVERS)
    compare = prod_table[["column", "rho"]].rename(columns={"rho": "rho_production_re"}).merge(
        kpc_table[["column", "rho"]].rename(columns={"rho": "rho_fixed_1kpc"}), on="column")
    compare = compare.sort_values("rho_production_re", key=lambda s: s.abs(), ascending=False)
    print(compare.to_string(index=False))
    compare.to_csv(OUT / "driver_comparison_production_vs_fixed_kpc.csv", index=False)

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.bar(safety_df["scale"], safety_df["resolution_safe_fraction"],
           color=["salmon"] * len(FIXED_KPC_SCALES) + ["steelblue"])
    ax.set_ylabel(f"fraction with sigma_kernel > {PSF_SAFETY_MULTIPLE}x sigma_PSF")
    ax.set_title("Resolution safety: fixed-kpc vs self-similar kernel")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1]
    ax.scatter(results_df["n_res"], results_df["S_production"], s=8, alpha=0.4, label="0.35 x r_e (production)", color="steelblue")
    ax.scatter(results_df["n_res"], results_df["S_kpc_1.0"], s=8, alpha=0.4, label="1.0 kpc (fixed)", color="firebrick")
    ax.set_xlabel(r"$n_{res}$")
    ax.set_ylabel("S")
    ax.set_title("S vs resolution: self-similar vs fixed-kpc kernel")
    ax.legend(fontsize=8)

    ax = axes[2]
    x = np.arange(len(compare))
    width = 0.35
    ax.bar(x - width/2, compare["rho_production_re"], width, label="0.35 x r_e (production)", color="steelblue")
    ax.bar(x + width/2, compare["rho_fixed_1kpc"], width, label="1.0 kpc (fixed)", color="firebrick")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(compare["column"], rotation=40, ha="right")
    ax.set_ylabel("Spearman rho with S")
    ax.set_title("Driver correlations: self-similar vs fixed-kpc kernel")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "fixed_kpc_comparison.png", dpi=150)
    plt.close(fig)

    print(f"\nWrote: {OUT / 'fixed_kpc_S_values.csv'}")
    print(f"Wrote: {OUT / 'driver_comparison_production_vs_fixed_kpc.csv'}")
    print(f"Wrote: {OUT / 'fixed_kpc_comparison.png'}")


if __name__ == "__main__":
    main()
