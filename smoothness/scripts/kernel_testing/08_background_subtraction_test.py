#!/usr/bin/env python3
"""
08_background_subtraction_test.py
====================================
T8 - implements and validates the missing B_S (background-smoothness) term
from Conselice's S statistic, using each galaxy's own per-pixel error map
as the noise model - no blank-sky region needed, addressing exactly the
practical constraint (no clean background patch in a SAMI hexabundle
pointing) that led to using I_clean = flux - err as the only noise
handling in the first place.

This is NOT the same correction as I_clean = flux - err (see this
script's companion 07_negative_clipping_test.py, which showed clipping/
masking barely changes anything because the masked footprint has almost no
low-S/N pixels to begin with - median Halpha S/N ~46 within the shared
mask). flux - err is a per-pixel bias correction applied BEFORE smoothing.
B_S is a post-smoothing ROUGHNESS measurement: how rough does the SAME
kernel make a noise-only field look, measured in the same units as the
galaxy's own residual sum, then subtracted from it before the final
division. 02_noise_floor.py showed this roughness is large and tracks
resolution the same way the real S does (R^2=0.94) - B_S is the
mathematically-correct way to remove it.

Formula (matches Conselice 2003's structure, keeping this pipeline's
existing choice to omit the customary x10 scaling for consistency with
smoothness_04.py/03_metrics/smoothness.py):

    B_S     = median over draws of sum(|noise_i - err|) after smoothing,
              where noise_i ~ Normal(0, err) - i.e. the NUMERATOR only,
              not divided into a ratio (that distinction matters - see
              02_noise_floor.py's own NOTE on why the ratio isn't the
              right quantity to subtract).
    S_corrected = (sum(|I_clean - I_clean_smoothed|) - B_S) / sum(|I_clean|)

Two independent noise-draw sets per galaxy, to avoid circularity:
    - N_DRAWS_ESTIMATE draws -> used ONLY to estimate B_S itself
    - N_DRAWS_VALIDATE draws (fresh, never used for the estimate) -> used
      ONLY to check whether S_corrected, applied to pure noise the
      correction was never fitted to, now clusters near 0 and has lost its
      correlation with resolution proxies. This is the real test of
      whether the correction works, not just a self-consistency check.

Validation criteria:
    1. Does S_corrected(pure noise, held-out draws) cluster near 0, and is
       its correlation with n_res/z much weaker than S_noise's was in
       02_noise_floor.py?
    2. Does S_corrected still rank-correlate sensibly with production S
       (it should change things, especially for poorly-resolved galaxies,
       but not scramble the sample)?
    3. Do the real drivers (log_SFR, log_Mstar, delta_ms) survive or
       strengthen?
    4. How many galaxies go negative before clipping, and how stable is
       the B_S estimate itself (convergence / draw-to-draw scatter)?

Output: outputs/08_background_subtraction_test/background_subtraction_results.csv,
        outputs/08_background_subtraction_test/bs_validation_plots.png,
        printed summary + a go/no-go recommendation.
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

OUT = c.ensure_outdir("08_background_subtraction_test")
N_DRAWS_ESTIMATE = 50   # draws used to estimate B_S itself
N_DRAWS_VALIDATE = 20   # independent, held-out draws used only to validate the correction
N_DRAWS_CONVERGENCE = 200  # extra draws for a subsample, to check B_S estimate stability
SEED = 42


def residual_and_denom(I_clean: np.ndarray, sigma_px: float):
    smoothed = gaussian_filter(I_clean, sigma=sigma_px)
    residual = np.abs(I_clean - smoothed)
    numerator = float(np.nansum(residual))
    denom = float(np.nansum(np.abs(I_clean)))
    return numerator, denom


def main():
    c.print_header("T8: Background-smoothness term (B_S) - implementation and validation")

    prod = c.import_production_smoothness()
    df = c.load_full_sam()
    print(f"Sample: in_full_sam, n={len(df)}")
    print(f"B_S estimated from {N_DRAWS_ESTIMATE} draws; validated on {N_DRAWS_VALIDATE} independent held-out draws")

    rng = np.random.default_rng(SEED)
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Background subtraction"):
        catid = int(row["CATID"])
        halpha_path = c.find_halpha_path(catid)
        if halpha_path is None:
            continue

        flux_cube, err_cube = read_masked_halpha(str(halpha_path))
        flux = flux_cube[prod.HALPHA_LAYER]
        err = err_cube[prod.HALPHA_LAYER]
        sigma_px = float(row["sigma_px"])
        scale = np.where(err > 0, err, 0.0)

        # --- Real data ---
        I_clean_real = flux - err
        num_real, denom_real = residual_and_denom(I_clean_real, sigma_px)
        S_production = num_real / denom_real if denom_real else np.nan

        # --- B_S estimate (N_DRAWS_ESTIMATE draws, held apart from validation) ---
        est_numerators = []
        for _ in range(N_DRAWS_ESTIMATE):
            noise = rng.normal(loc=0.0, scale=scale)
            num_n, _ = residual_and_denom(noise - err, sigma_px)
            est_numerators.append(num_n)
        B_S = float(np.median(est_numerators))
        B_S_std = float(np.std(est_numerators))

        S_corrected_raw = (num_real - B_S) / denom_real if denom_real else np.nan
        S_corrected = max(S_corrected_raw, 0.0)

        # --- Held-out validation draws: apply the SAME correction (fixed B_S
        #     from the estimate set) to fresh noise never used to fit it ---
        val_S_noise_corrected = []
        for _ in range(N_DRAWS_VALIDATE):
            noise = rng.normal(loc=0.0, scale=scale)
            num_v, denom_v = residual_and_denom(noise - err, sigma_px)
            if denom_v:
                val_S_noise_corrected.append(max((num_v - B_S) / denom_v, 0.0))

        rows.append({
            "CATID": catid, "n_res": row["n_res"], "z": row["z"],
            "S_production": S_production,
            "B_S": B_S, "B_S_std": B_S_std, "B_S_relative_std": B_S_std / B_S if B_S else np.nan,
            "S_corrected_raw": S_corrected_raw, "S_corrected": S_corrected,
            "frac_negative_before_clip": S_corrected_raw < 0,
            "S_noise_corrected_median": np.median(val_S_noise_corrected) if val_S_noise_corrected else np.nan,
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT / "background_subtraction_results.csv", index=False)

    print(f"\nGalaxies processed: {len(result_df)}")
    print(f"B_S estimate stability: median relative draw-to-draw std = "
          f"{result_df['B_S_relative_std'].median()*100:.1f}%  "
          f"(std of the {N_DRAWS_ESTIMATE}-draw numerator distribution / its median)")
    n_neg = result_df["frac_negative_before_clip"].sum()
    print(f"Galaxies with S_corrected_raw < 0 before clipping: {n_neg} / {len(result_df)} "
          f"({100*n_neg/len(result_df):.1f}%) - these are clipped to S_corrected=0 "
          f"(standard convention: real residual didn't exceed the noise-only floor at this kernel)")

    print("\n--- Criterion 1: does the corrected formula neutralise pure noise on HELD-OUT draws? ---")
    print(f"  S_production median : {result_df['S_production'].median():.4f}")
    print(f"  S_corrected median   : {result_df['S_corrected'].median():.4f}")
    print(f"  S_noise_corrected (held-out draws, median per galaxy): "
          f"median={result_df['S_noise_corrected_median'].median():.4f}, "
          f"mean={result_df['S_noise_corrected_median'].mean():.4f}")
    for col in ["n_res", "z"]:
        rho_before, _ = spearmanr(result_df["S_production"], result_df[col])
        rho_after_signal, _ = spearmanr(result_df["S_corrected"], result_df[col])
        rho_after_noise, _ = spearmanr(result_df["S_noise_corrected_median"], result_df[col])
        print(f"  rho(., {col:5s}):  S_production={rho_before:+.3f}   S_corrected={rho_after_signal:+.3f}   "
              f"S_noise_corrected(held-out)={rho_after_noise:+.3f}")

    b, a = np.polyfit(result_df["S_noise_corrected_median"].fillna(0), result_df["S_corrected"], 1)
    pred = a + b * result_df["S_noise_corrected_median"].fillna(0)
    r2_after = 1 - np.sum((result_df["S_corrected"] - pred) ** 2) / np.sum((result_df["S_corrected"] - result_df["S_corrected"].mean()) ** 2)
    rho_after, p_after = spearmanr(result_df["S_corrected"], result_df["S_noise_corrected_median"])
    print(f"\n  R^2 of S_corrected ~ S_noise_corrected(held-out): {r2_after:.3f}  (was 0.94 before correction, from 02_noise_floor.py)")
    print(f"  Spearman rho: {rho_after:+.3f} (p={p_after:.1e})")

    print("\n--- Criterion 2: rank relationship to production S ---")
    rho_rank, _ = spearmanr(result_df["S_corrected"], result_df["S_production"])
    print(f"  Spearman rho(S_corrected, S_production) = {rho_rank:.3f}")

    print("\n--- Criterion 3: do the real drivers survive? ---")
    for label, S_col in [("S_production", "S_production"), ("S_corrected", "S_corrected")]:
        tmp = df.copy()
        tmp["S"] = result_df[S_col].values
        tab = c.spearman_table(tmp, "S", {"log_SFR": "", "log_Mstar": "", "delta_ms": ""})
        vals = ", ".join(f"{r.column}={r.rho:+.3f}" for r in tab.itertuples())
        print(f"  {label:14s}: {vals}")

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.scatter(df["n_res"], result_df["S_production"], s=8, alpha=0.3, color="grey", label="S_production")
    ax.scatter(df["n_res"], result_df["S_corrected"], s=8, alpha=0.3, color="seagreen", label="S_corrected")
    ax.set_xlabel(r"$n_{res}$"); ax.set_ylabel("S")
    ax.set_title("S vs resolution: before/after B_S correction")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(result_df["S_noise_corrected_median"].dropna(), bins=40, color="firebrick", alpha=0.7,
             label="S_noise_corrected\n(held-out draws)")
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("S computed on pure noise, after B_S correction")
    ax.set_ylabel("N galaxies")
    ax.set_title("Does corrected S neutralise pure noise?")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.scatter(result_df["S_production"], result_df["S_corrected"], s=8, alpha=0.4, color="steelblue")
    lim = [0, result_df["S_production"].max() * 1.05]
    ax.plot(lim, lim, "k--", lw=1, label="1:1")
    ax.set_xlabel("S_production"); ax.set_ylabel("S_corrected")
    ax.set_title(f"Rank relationship (rho={rho_rank:.2f})")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "bs_validation_plots.png", dpi=150)
    plt.close(fig)

    print(f"\nWrote: {OUT / 'background_subtraction_results.csv'}")
    print(f"Wrote: {OUT / 'bs_validation_plots.png'}")

    R2_BASELINE = 0.94  # from 02_noise_floor.py, uncorrected S_actual ~ S_noise
    print("\n=== RECOMMENDATION ===")
    print(f"  R^2 before correction (02_noise_floor.py baseline): {R2_BASELINE:.2f}")
    print(f"  R^2 after correction (this script, held-out draws) : {r2_after:.2f}")
    if r2_after < 0.3 and abs(rho_after) < 0.4:
        print("  PASS: held-out noise no longer explains a large fraction of S_corrected's variance.")
        print("  B_S correction looks effective - reasonable to port into production, pending your review.")
    elif r2_after < R2_BASELINE - 0.2:
        print("  PARTIAL: noise-explained variance dropped substantially but is not eliminated.")
        print("  Worth reviewing the plots before deciding whether to port to production.")
    else:
        print("  NO-GO: correction did not meaningfully reduce noise dependence - do not port as-is.")


if __name__ == "__main__":
    main()
