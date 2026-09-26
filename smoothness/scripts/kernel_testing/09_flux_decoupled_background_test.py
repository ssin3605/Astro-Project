#!/usr/bin/env python3
"""
09_flux_decoupled_background_test.py
=======================================
T9 - tests a refinement of the B_S background-smoothness correction
(08_background_subtraction_test.py) motivated by a supervisor review:
the per-pixel error map `err` is not a clean, source-independent
"background" noise level - within a galaxy's own footprint it correlates
strongly with local flux (Spearman rho=0.905, brightest-quartile/faintest-
quartile mean-err ratio ~4.5x, measured directly on CATID 278760 in this
sample), consistent with Poisson (shot) noise from the real Halpha source
itself dominating err in bright regions. Since B_S is built by simulating
noise ~ Normal(0, err) and measuring how rough the production kernel makes
it look, that noise is itself spatially patterned to track real flux - so
B_S risks absorbing some of the real source's own structure (specifically
in bright, clumpy regions) rather than purely generic background/
instrumental noise, which is what Conselice's original background term is
supposed to represent.

Fix tested here: for each galaxy, decompose the noise budget via a linear
fit err_i^2 ~ a*flux_i + b across its own footprint (standard Poisson-plus-
floor variance model: `a*flux` = source-coupled Poisson term, `b` = flux-
independent floor - sky, read noise, fibre-coverage effects). Use
sqrt(max(b, 0)) - NOT err - as the noise-draw scale when estimating B_S, so
the simulated background noise is deliberately decoupled from the real
source's own brightness structure.

Explicitly NOT addressed here (per the same review): SAMI cubes are
drizzle-resampled from hexabundle fibres onto a square grid, which can
introduce real fibre-coverage-driven spatial variation in `b` itself (e.g.
edge spaxels covered by fewer fibres). This script assumes a SINGLE b per
galaxy (uniform floor) - a stated simplification, not a claim that the
instrumental floor is really spatially flat. No fibre-coverage/exposure map
is available in this repo's data products (checked: the raw Halpha maps in
data/raw/sami_dr3_Halpha_maps/ carry only PRIMARY + HALPHA_ERR) to do
better without pulling new SAMI DR3 data products.

Validation - two held-out tests per galaxy, not one, because the usual
single criterion ("does corrected S on pure noise cluster near 0?") means
something different once the Poisson term is deliberately excluded:

  (a) Held-out noise ~ Normal(0, sqrt(b)) (the model's OWN noise
      assumption) -> should cluster near 0. Self-consistency check that
      the new B_S_fd correctly neutralises the noise it's actually built
      from.
  (b) Held-out noise ~ Normal(0, err) (the FULL real per-pixel error
      budget, same as 08's test, i.e. what genuine "no real structure"
      data really looks like) -> will NOT fully cluster at 0 under this
      correction, by design - the residual here is an estimate of how
      much of the original noise-floor artifact was Poisson/source-
      coupled specifically, deliberately left uncorrected because it's
      argued to be part of the real measurement of a real bright region,
      not generic background.

Also reports whether the correction actually targets what it claims to:
correlation of the fractional B_S change (old MC B_S -> new flux-decoupled
B_S_fd) against each galaxy's fitted Poisson slope `a` - galaxies with a
stronger flux-noise coupling should see a proportionally larger reduction
in B_S if the fix is doing what it's supposed to.

Output: outputs/09_flux_decoupled_background_test/flux_decoupled_results.csv,
        outputs/09_flux_decoupled_background_test/fd_validation_plots.png,
        printed summary + comparison against 08's production B_S.
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

OUT = c.ensure_outdir("09_flux_decoupled_background_test")
N_DRAWS_ESTIMATE = 50    # draws used to estimate B_S_fd itself (matches 08)
N_DRAWS_VALIDATE = 20    # held-out draws per validation branch (matches 08)
SEED = 43                # distinct from 08's SEED=42, independent draw stream


def residual_and_denom(I_clean: np.ndarray, sigma_px: float):
    smoothed = gaussian_filter(I_clean, sigma=sigma_px)
    residual = np.abs(I_clean - smoothed)
    numerator = float(np.nansum(residual))
    denom = float(np.nansum(np.abs(I_clean)))
    return numerator, denom


def fit_poisson_floor(flux: np.ndarray, err: np.ndarray, valid: np.ndarray):
    """
    OLS fit of err_i^2 ~ a*flux_i + b over valid footprint pixels.
    Returns (a, b, b_clipped, n_px). b_clipped = max(b, 0.0).
    """
    x = flux[valid]
    y = err[valid] ** 2
    n = len(x)
    if n < 10:
        return np.nan, np.nan, np.nan, n
    a, b = np.polyfit(x, y, 1)
    b_clipped = max(b, 0.0)
    return float(a), float(b), float(b_clipped), n


def main():
    c.print_header("T9: Flux-decoupled background-smoothness term (B_S_fd) - implementation and validation")

    prod = c.import_production_smoothness()
    df = c.load_full_sam()
    print(f"Sample: in_full_sam, n={len(df)}")
    print(f"B_S_fd estimated from {N_DRAWS_ESTIMATE} draws; validated on {N_DRAWS_VALIDATE} held-out draws per branch")

    rng = np.random.default_rng(SEED)
    rows = []
    skipped_fit = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Flux-decoupled background"):
        catid = int(row["CATID"])
        halpha_path = c.find_halpha_path(catid)
        if halpha_path is None:
            continue

        flux_cube, err_cube = read_masked_halpha(str(halpha_path))
        flux = flux_cube[prod.HALPHA_LAYER]
        err = err_cube[prod.HALPHA_LAYER]
        sigma_px = float(row["sigma_px"])
        valid = err > 0

        a_fit, b_raw, b_clipped, n_px = fit_poisson_floor(flux, err, valid)
        if np.isnan(b_clipped):
            skipped_fit += 1
            continue
        sigma_floor = np.sqrt(b_clipped)
        scale_err = np.where(valid, err, 0.0)
        scale_floor = np.where(valid, sigma_floor, 0.0)

        # --- Real data (identical to production) ---
        I_clean_real = flux - err
        num_real, denom_real = residual_and_denom(I_clean_real, sigma_px)
        S_production = num_real / denom_real if denom_real else np.nan

        # --- B_S_fd estimate: noise scaled by the decontaminated floor, not err ---
        est_numerators = []
        for _ in range(N_DRAWS_ESTIMATE):
            noise = rng.normal(loc=0.0, scale=scale_floor)
            num_n, _ = residual_and_denom(noise - scale_floor, sigma_px)  # mirrors production's I_clean = flux - err
            est_numerators.append(num_n)
        B_S_fd = float(np.median(est_numerators))
        B_S_fd_std = float(np.std(est_numerators))

        S_corrected_fd_raw = (num_real - B_S_fd) / denom_real if denom_real else np.nan
        S_corrected_fd = max(S_corrected_fd_raw, 0.0)

        # --- Validation branch (a): held-out noise from the model's OWN assumption ---
        val_floor_noise_corrected = []
        for _ in range(N_DRAWS_VALIDATE):
            noise = rng.normal(loc=0.0, scale=scale_floor)
            num_v, denom_v = residual_and_denom(noise - scale_floor, sigma_px)
            if denom_v:
                val_floor_noise_corrected.append(max((num_v - B_S_fd) / denom_v, 0.0))

        # --- Validation branch (b): held-out noise from the FULL real error budget ---
        val_full_noise_corrected = []
        for _ in range(N_DRAWS_VALIDATE):
            noise = rng.normal(loc=0.0, scale=scale_err)
            I_noise = noise - err
            num_v, denom_v = residual_and_denom(I_noise, sigma_px)
            if denom_v:
                val_full_noise_corrected.append(max((num_v - B_S_fd) / denom_v, 0.0))

        rows.append({
            "CATID": catid, "n_res": row["n_res"], "z": row["z"],
            "S_production": S_production,
            "a_poisson_slope": a_fit, "b_floor_raw": b_raw, "b_floor_clipped": b_clipped,
            "n_px": n_px, "max_flux": float(np.nanmax(flux[valid])) if valid.any() else np.nan,
            "B_S_fd": B_S_fd, "B_S_fd_std": B_S_fd_std,
            "S_corrected_fd_raw": S_corrected_fd_raw, "S_corrected_fd": S_corrected_fd,
            "frac_negative_before_clip": S_corrected_fd_raw < 0,
            "S_floor_noise_corrected_median": np.median(val_floor_noise_corrected) if val_floor_noise_corrected else np.nan,
            "S_full_noise_corrected_median": np.median(val_full_noise_corrected) if val_full_noise_corrected else np.nan,
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT / "flux_decoupled_results.csv", index=False)
    print(f"\nGalaxies processed: {len(result_df)}  (skipped, fit failed/too few px: {skipped_fit})")

    # --- Load 08's production B_S for direct comparison ---
    prod_bs_path = c.OUTPUTS_DIR / "08_background_subtraction_test" / "background_subtraction_results.csv"
    have_prod_bs = prod_bs_path.exists()
    if have_prod_bs:
        prod_bs = pd.read_csv(prod_bs_path)[["CATID", "B_S", "S_corrected"]].rename(
            columns={"B_S": "B_S_original", "S_corrected": "S_corrected_original"})
        result_df = result_df.merge(prod_bs, on="CATID", how="left")
    else:
        print("\n(08_background_subtraction_test.py has not been run in this environment - "
              "skipping direct old-vs-new B_S comparison; run it first for the full picture.)")

    n_neg = result_df["frac_negative_before_clip"].sum()
    print(f"Galaxies with S_corrected_fd_raw < 0 before clipping: {n_neg} / {len(result_df)} "
          f"({100*n_neg/len(result_df):.1f}%)")
    print(f"Poisson slope 'a' (err^2 ~ a*flux + b): median={result_df['a_poisson_slope'].median():.4g}, "
          f"fraction with a>0: {(result_df['a_poisson_slope']>0).mean()*100:.1f}%")
    n_b_clipped = (result_df["b_floor_raw"] < 0).sum()
    print(f"Galaxies where fitted b (floor) was negative before clipping to 0: {n_b_clipped} / {len(result_df)} "
          f"({100*n_b_clipped/len(result_df):.1f}%)")

    print("\n--- Validation (a): does B_S_fd neutralise noise drawn from its OWN model (floor only)? ---")
    print(f"  S_floor_noise_corrected (held-out): median={result_df['S_floor_noise_corrected_median'].median():.4f}, "
          f"mean={result_df['S_floor_noise_corrected_median'].mean():.4f}  (want: near 0)")

    print("\n--- Validation (b): residual when tested against the FULL real error budget (not fully correctable by design) ---")
    print(f"  S_full_noise_corrected (held-out): median={result_df['S_full_noise_corrected_median'].median():.4f}, "
          f"mean={result_df['S_full_noise_corrected_median'].mean():.4f}")
    for col in ["n_res", "z"]:
        rho_full, _ = spearmanr(result_df["S_full_noise_corrected_median"], result_df[col])
        print(f"  rho(S_full_noise_corrected, {col}) = {rho_full:+.3f}  "
              f"(this is the part of the original resolution-tracking noise artifact NOT removed by "
              f"design - compare to 08's pre-correction S_noise rho for the same column)")

    if have_prod_bs:
        print("\n--- Comparison: original MC B_S vs new flux-decoupled B_S_fd ---")
        valid_cmp = result_df.dropna(subset=["B_S_original", "B_S_fd"])
        frac_reduction = 1 - valid_cmp["B_S_fd"] / valid_cmp["B_S_original"]
        print(f"  n compared: {len(valid_cmp)}")
        print(f"  B_S_original median: {valid_cmp['B_S_original'].median():.4f}")
        print(f"  B_S_fd median      : {valid_cmp['B_S_fd'].median():.4f}")
        print(f"  Median fractional reduction (B_S_original -> B_S_fd): {frac_reduction.median()*100:.1f}%")

        rho_target, p_target = spearmanr(frac_reduction, valid_cmp["a_poisson_slope"])
        print(f"\n  Does the reduction track the fitted Poisson slope 'a' (evidence the fix targets what it claims)?")
        print(f"  Spearman rho(fractional B_S reduction, a_poisson_slope) = {rho_target:+.3f} (p={p_target:.1e})")

        rho_s, _ = spearmanr(result_df["S_corrected_fd"].dropna(),
                              result_df.loc[result_df["S_corrected_fd"].notna(), "S_corrected_original"])
        print(f"\n  Spearman rho(S_corrected_fd, S_corrected_original) = {rho_s:.3f}  (rank stability check)")

        print("\n--- Do the real drivers survive under S_corrected_fd? ---")
        for label, S_col in [("S_corrected_original", "S_corrected_original"), ("S_corrected_fd", "S_corrected_fd")]:
            df_no_s = df.drop(columns=["S"]) if "S" in df.columns else df
            tmp = df_no_s.merge(result_df[["CATID", S_col]], on="CATID", how="inner")
            tmp = tmp.rename(columns={S_col: "S"})
            tab = c.spearman_table(tmp, "S", {"log_SFR": "", "log_Mstar": "", "delta_ms": ""})
            vals = ", ".join(f"{r.column}={r.rho:+.3f}" for r in tab.itertuples())
            print(f"  {label:22s}: {vals}")

        # --- Plots ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        ax = axes[0]
        ax.scatter(valid_cmp["B_S_original"], valid_cmp["B_S_fd"], s=8, alpha=0.4, color="steelblue")
        lim = [0, valid_cmp["B_S_original"].max() * 1.05]
        ax.plot(lim, lim, "k--", lw=1, label="1:1")
        ax.set_xlabel("B_S (original, full err)"); ax.set_ylabel("B_S_fd (flux-decoupled floor)")
        ax.set_title("Background term: before vs after decoupling")
        ax.legend(fontsize=8)

        ax = axes[1]
        ax.scatter(valid_cmp["a_poisson_slope"], frac_reduction, s=8, alpha=0.4, color="firebrick")
        ax.set_xlabel("Fitted Poisson slope a (err^2 ~ a*flux + b)")
        ax.set_ylabel("Fractional reduction in B_S")
        ax.set_title(f"Does the fix target flux-coupled noise?\n(rho={rho_target:.2f})")

        ax = axes[2]
        ax.scatter(result_df["S_corrected_original"] if "S_corrected_original" in result_df else [],
                    result_df["S_corrected_fd"], s=8, alpha=0.4, color="seagreen")
        ax.set_xlabel("S_corrected (original B_S)"); ax.set_ylabel("S_corrected_fd (flux-decoupled)")
        ax.set_title(f"Rank relationship (rho={rho_s:.2f})")

        fig.tight_layout()
        fig.savefig(OUT / "fd_validation_plots.png", dpi=150)
        plt.close(fig)
        print(f"\nWrote: {OUT / 'fd_validation_plots.png'}")

    print(f"\nWrote: {OUT / 'flux_decoupled_results.csv'}")


if __name__ == "__main__":
    main()
