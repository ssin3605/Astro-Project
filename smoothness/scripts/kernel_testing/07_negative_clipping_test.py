#!/usr/bin/env python3
"""
07_negative_clipping_test.py
===============================
T7 - does clipping/masking low-S/N pixels out of I_clean actually reduce
the noise-domination result found in 02_noise_floor.py (a white-noise-only
S explains ~94% of the variance in the real S)?

Context: the shared continuum mask (src/utils/masking.py) selects pixels on
CONTINUUM S/N >= 3, which says nothing about the Hα LINE S/N at that same
pixel - a pixel can easily be continuum-detected but Hα-faint/noisy (Hα
emission is typically far more spatially concentrated than the stellar
continuum). Production's I_clean = flux - err (unclipped) therefore
includes every one of those low-Hα-S/N footprint pixels in the S sum, with
no down-weighting - unlike Asymmetry (which clips negative post-
subtraction pixels to zero) or Gini (which re-applies its own hard S/N
cut). This script tests three variants of I_clean, each pushed through the
*same* real-vs-noise comparison 02_noise_floor.py used:

    (a) production  : I_clean = flux - err                     (unclipped)
    (b) clipped      : I_clean = max(flux - err, 0)
    (c) SN-masked    : I_clean = (flux - err) where flux/err >= SN_THRESH,
                        else 0  (SN_THRESH=3.0, matching the continuum
                        mask's own sn_cut convention in masking.py)

For each variant, on in_full_sam:
    1. Recompute S_actual.
    2. Recompute S_noise (median of N_DRAWS white-noise draws from the real
       error map, same variant applied to the noise draw).
    3. Report R² / Spearman rho of S_actual ~ S_noise (the headline T6
       metric) - if a variant meaningfully reduces this, that variant is
       evidence-based justification to change the production formula.
    4. Report how much each variant changes galaxy *rankings* relative to
       production (Spearman rho of S_actual_variant vs S_actual_production)
       - a fix that also scrambles the whole sample's ranking is a bigger
       change than one that mostly re-scales it.
    5. Diagnostic: fraction of footprint pixels with Hα S/N < 3 (i.e. how
       many pixels are continuum-selected but Hα-noisy, to quantify the
       mechanism above).

Output: outputs/07_negative_clipping_test/clipping_variant_results.csv,
        outputs/07_negative_clipping_test/clipping_comparison.png,
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

OUT = c.ensure_outdir("07_negative_clipping_test")
N_DRAWS = 20
SEED = 42
SN_THRESH = 3.0  # matches src/utils/masking.py::make_continuum_mask's sn_cut default

VARIANTS = ["production", "clipped", "sn_masked"]


def make_I_clean(flux: np.ndarray, err: np.ndarray, variant: str) -> np.ndarray:
    if variant == "production":
        return flux - err
    if variant == "clipped":
        return np.clip(flux - err, 0, None)
    if variant == "sn_masked":
        with np.errstate(divide="ignore", invalid="ignore"):
            sn = np.where(err > 0, flux / err, 0.0)
        keep = sn >= SN_THRESH
        return np.where(keep, flux - err, 0.0)
    raise ValueError(variant)


def compute_S(I_clean: np.ndarray, sigma_px: float) -> float:
    smoothed = gaussian_filter(I_clean, sigma=sigma_px)
    residual = np.abs(I_clean - smoothed)
    denom = np.nansum(np.abs(I_clean))
    return float(np.nansum(residual) / denom) if denom != 0 else np.nan


def main():
    c.print_header("T7: Does clipping/SN-masking I_clean reduce noise domination?")

    prod = c.import_production_smoothness()
    df = c.load_full_sam()
    print(f"Sample: in_full_sam, n={len(df)}")
    print(f"SN_THRESH for sn_masked variant: {SN_THRESH} (matches masking.py's continuum sn_cut)")

    rng = np.random.default_rng(SEED)
    rows = []
    frac_low_sn_footprint = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Clipping variants"):
        catid = int(row["CATID"])
        halpha_path = c.find_halpha_path(catid)
        if halpha_path is None:
            continue

        flux_cube, err_cube = read_masked_halpha(str(halpha_path))
        flux = flux_cube[prod.HALPHA_LAYER]
        err = err_cube[prod.HALPHA_LAYER]
        sigma_px = float(row["sigma_px"])

        footprint = err > 0
        if footprint.sum() > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                sn_footprint = flux[footprint] / err[footprint]
            frac_low_sn_footprint.append(np.mean(sn_footprint < SN_THRESH))

        entry = {"CATID": catid, "n_res": row["n_res"], "z": row["z"]}

        for variant in VARIANTS:
            I_clean_actual = make_I_clean(flux, err, variant)
            S_actual = compute_S(I_clean_actual, sigma_px)

            s_noise_draws = []
            for _ in range(N_DRAWS):
                noise = rng.normal(loc=0.0, scale=np.where(err > 0, err, 0.0))
                # noise stands in for "flux" (true signal = 0), passed through
                # make_I_clean exactly as the real flux map is - so for variant
                # "production" this reproduces 02_noise_floor.py's I_clean_noise
                # = noise - err exactly.
                I_clean_noise = make_I_clean(noise, err, variant)
                s = compute_S(I_clean_noise, sigma_px)
                if not np.isnan(s):
                    s_noise_draws.append(s)

            entry[f"S_actual_{variant}"] = S_actual
            entry[f"S_noise_{variant}"] = np.median(s_noise_draws) if s_noise_draws else np.nan

        rows.append(entry)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT / "clipping_variant_results.csv", index=False)

    print(f"\nGalaxies processed: {len(result_df)}")
    print(f"Median fraction of footprint pixels with Halpha S/N < {SN_THRESH}: "
          f"{np.median(frac_low_sn_footprint)*100:.1f}%  "
          f"(these are continuum-selected but Halpha-noisy pixels - the mechanism this test targets)")

    print(f"\n{'variant':12s}  {'median S_actual':>16s}  {'rho(S_actual,S_noise)':>22s}  "
          f"{'R^2':>8s}  {'rho vs production S':>20s}")
    prod_S = result_df["S_actual_production"]
    for variant in VARIANTS:
        Sa = result_df[f"S_actual_{variant}"]
        Sn = result_df[f"S_noise_{variant}"]
        valid = Sa.notna() & Sn.notna()
        rho, p = spearmanr(Sa[valid], Sn[valid])
        b, a = np.polyfit(Sn[valid], Sa[valid], 1)
        pred = a + b * Sn[valid]
        r2 = 1 - np.sum((Sa[valid] - pred) ** 2) / np.sum((Sa[valid] - Sa[valid].mean()) ** 2)
        rho_vs_prod, _ = spearmanr(Sa[valid], prod_S[valid])
        print(f"{variant:12s}  {Sa.median():16.4f}  {rho:22.3f}  {r2:8.3f}  {rho_vs_prod:20.3f}")

    print("\n--- Driver-correlation check: does the fix preserve/strengthen the real drivers? ---")
    for variant in VARIANTS:
        tmp = df.copy()
        tmp["S"] = result_df[f"S_actual_{variant}"].values
        tab = c.spearman_table(tmp, "S", {"log_SFR": "", "log_Mstar": "", "delta_ms": ""})
        vals = ", ".join(f"{r.column}={r.rho:+.3f}" for r in tab.itertuples())
        print(f"  {variant:12s}: {vals}")

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {"production": "grey", "clipped": "steelblue", "sn_masked": "firebrick"}

    ax = axes[0]
    for variant in VARIANTS:
        Sa = result_df[f"S_actual_{variant}"]
        Sn = result_df[f"S_noise_{variant}"]
        ax.scatter(Sn, Sa, s=8, alpha=0.35, color=colors[variant], label=variant)
    ax.set_xlabel(r"$S_{noise}$")
    ax.set_ylabel(r"$S_{actual}$")
    ax.set_title("S vs its noise floor, by I_clean variant")
    ax.legend(fontsize=8)

    ax = axes[1]
    r2_vals = []
    for variant in VARIANTS:
        Sa = result_df[f"S_actual_{variant}"]
        Sn = result_df[f"S_noise_{variant}"]
        valid = Sa.notna() & Sn.notna()
        b, a = np.polyfit(Sn[valid], Sa[valid], 1)
        pred = a + b * Sn[valid]
        r2 = 1 - np.sum((Sa[valid] - pred) ** 2) / np.sum((Sa[valid] - Sa[valid].mean()) ** 2)
        r2_vals.append(r2)
    ax.bar(VARIANTS, r2_vals, color=[colors[v] for v in VARIANTS])
    ax.set_ylabel(r"$R^2$ of $S_{actual} \sim S_{noise}$")
    ax.set_title("Variance explained by pure noise\n(lower = less noise-dominated)")
    ax.set_ylim(0, 1)

    ax = axes[2]
    ax.hist(frac_low_sn_footprint, bins=30, color="seagreen", edgecolor="white")
    ax.axvline(np.median(frac_low_sn_footprint), color="red", ls="--", lw=1,
               label=f"median={np.median(frac_low_sn_footprint):.2f}")
    ax.set_xlabel(f"fraction of footprint pixels with H$\\alpha$ S/N < {SN_THRESH}")
    ax.set_ylabel("N galaxies")
    ax.set_title("Continuum-selected but Halpha-noisy pixels")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "clipping_comparison.png", dpi=150)
    plt.close(fig)

    print(f"\nWrote: {OUT / 'clipping_variant_results.csv'}")
    print(f"Wrote: {OUT / 'clipping_comparison.png'}")


if __name__ == "__main__":
    main()
