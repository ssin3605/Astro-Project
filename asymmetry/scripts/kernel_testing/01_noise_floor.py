#!/usr/bin/env python3
"""
01_noise_floor.py
===================
Baseline test for the missing A_bg term flagged in AUDIT_REPORT.md section
3.4 and sami_sf_pipeline/ASYMMETRY_HANDOFF.md - the A-statistic analog of
smoothness/scripts/kernel_testing/02_noise_floor.py's T6 test for S.

Establishes A's own "before" number: how much of A_min's variance across
the science sample is explained by a quantity built from pure white noise,
containing zero real flux information, pushed through the *identical*
production rotate-and-differance formula. Also runs the clipped vs
unclipped noise-draw variants side by side (open design question #2 in
ASYMMETRY_HANDOFF.md section 3) to check the clipped variant doesn't
degenerate before committing to it for the real A_BS fix.

Method, per galaxy (in_full_sam only):
    1. Load the real error map (err) from the masked Halpha FITS - the same
       error map production's I_clean = clip(flux - err, 0) uses.
    2. Reuse this galaxy's own best-fit rotation centre (best_cy, best_cx)
       from asymmetry.csv, already found by production's grid search on the
       real signal - no separate grid search on noise (mirrors B_S reusing
       the real sigma_px rather than re-deriving one from noise).
    3. Draw a pure-noise realisation: noise ~ Normal(0, err) at every pixel.
    4. Run it through the *actual* production compute_A(), at the fixed
       best-fit centre, in two variants:
         - clipped:   I_noise = apply_error_subtraction(noise, err)
                      (matches production's own I_clean construction)
         - unclipped: I_noise = noise - err
    5. Repeat N_DRAWS times per galaxy; keep the median as that galaxy's
       noise-floor A_noise (per variant), and the std as draw-to-draw
       scatter.

Output: outputs/01_noise_floor/noise_floor_results.csv,
        outputs/01_noise_floor/noise_floor_diagnostics.png,
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

OUT = c.ensure_outdir("01_noise_floor")
N_DRAWS = 20
SEED = 42


def main():
    c.print_header("Baseline: noise floor of A vs resolution")

    prod = c.import_production_asymmetry()
    df = c.load_full_sam()
    print(f"Sample: in_full_sam, n={len(df)}")

    rng = np.random.default_rng(SEED)
    rows = []
    n_missing = 0
    n_all_zero_clipped_draws = 0
    n_all_zero_clipped_galaxies = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Noise floor"):
        catid = int(row["CATID"])
        halpha_path = c.find_halpha_path(catid)
        if halpha_path is None:
            n_missing += 1
            continue

        _, err_cube = read_masked_halpha(str(halpha_path))
        err = np.nan_to_num(err_cube[prod.HALPHA_LAYER], nan=0.0, posinf=0.0, neginf=0.0)

        best_cy, best_cx = float(row["best_cy"]), float(row["best_cx"])
        scale = np.where(err > 0, err, 0.0)

        a_clipped_draws, a_unclipped_draws = [], []
        galaxy_had_all_zero_clipped_draw = False

        for _ in range(N_DRAWS):
            noise = rng.normal(loc=0.0, scale=scale)

            I_clipped = prod.apply_error_subtraction(noise, err)
            if np.sum(I_clipped != 0) == 0:
                n_all_zero_clipped_draws += 1
                galaxy_had_all_zero_clipped_draw = True
            else:
                A_clipped, _, _ = prod.compute_A(I_clipped, best_cy, best_cx)
                if not np.isnan(A_clipped):
                    a_clipped_draws.append(A_clipped)

            I_unclipped = noise - err
            A_unclipped, _, _ = prod.compute_A(I_unclipped, best_cy, best_cx)
            if not np.isnan(A_unclipped):
                a_unclipped_draws.append(A_unclipped)

        if galaxy_had_all_zero_clipped_draw:
            n_all_zero_clipped_galaxies += 1

        if not a_clipped_draws or not a_unclipped_draws:
            continue

        a_clipped_draws = np.array(a_clipped_draws)
        a_unclipped_draws = np.array(a_unclipped_draws)
        rows.append({
            "CATID": catid,
            "A_actual": row["A_min"],
            "A_noise_clipped_median": np.median(a_clipped_draws),
            "A_noise_clipped_std": np.std(a_clipped_draws),
            "n_clipped_draws": len(a_clipped_draws),
            "A_noise_unclipped_median": np.median(a_unclipped_draws),
            "A_noise_unclipped_std": np.std(a_unclipped_draws),
            "n_unclipped_draws": len(a_unclipped_draws),
            "n_res": row["n_res"],
            "seeing": row["seeing"],
            "re_used": row["re_used"],
            "z": row["z"],
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT / "noise_floor_results.csv", index=False)

    print(f"\nMissing FITS (skipped)              : {n_missing}")
    print(f"Galaxies processed                  : {len(result_df)}")
    print(f"Total clipped draws (across all gal) : {len(result_df) * N_DRAWS}")
    print(f"All-zero-after-clip draws            : {n_all_zero_clipped_draws} "
          f"({100 * n_all_zero_clipped_draws / (len(result_df) * N_DRAWS + 1e-9):.2f}%)")
    print(f"Galaxies with >=1 all-zero clipped draw: {n_all_zero_clipped_galaxies} / {len(result_df)}  "
          f"(design question #2 in ASYMMETRY_HANDOFF.md: does clipping degenerate?)")

    print(f"\nA_actual                : median={result_df['A_actual'].median():.4f}")
    print(f"A_noise_clipped_median  : median={result_df['A_noise_clipped_median'].median():.4f}")
    print(f"A_noise_unclipped_median: median={result_df['A_noise_unclipped_median'].median():.4f}")

    for variant, col in [("CLIPPED", "A_noise_clipped_median"), ("UNCLIPPED", "A_noise_unclipped_median")]:
        print(f"\n--- {variant}: does this noise-only reference scale track the same things A_actual does? ---")
        for driver in ["n_res", "seeing", "re_used", "z"]:
            rho_noise, p_noise = spearmanr(result_df[col], result_df[driver])
            rho_actual, p_actual = spearmanr(result_df["A_actual"], result_df[driver])
            print(f"  {driver:10s}  A_noise vs {driver:10s} rho={rho_noise:+.3f} (p={p_noise:.1e})   "
                  f"|  A_actual vs {driver:10s} rho={rho_actual:+.3f} (p={p_actual:.1e})")

        rho_cross, p_cross = spearmanr(result_df["A_actual"], result_df[col])
        b, a = np.polyfit(result_df[col], result_df["A_actual"], 1)
        pred = a + b * result_df[col]
        r2_linear = 1 - np.sum((result_df["A_actual"] - pred) ** 2) / np.sum((result_df["A_actual"] - result_df["A_actual"].mean()) ** 2)
        print(f"\n  A_actual vs {col}: Spearman rho={rho_cross:+.3f} (p={p_cross:.1e}),  rho^2={rho_cross**2:.3f}")
        print(f"  Linear R^2 of A_actual ~ {col}: {r2_linear:.3f}")

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.scatter(result_df["n_res"], result_df["A_noise_clipped_median"], s=10, alpha=0.5, color="firebrick")
    ax.set_xlabel(r"$n_{res} = r_e / \mathrm{seeing}$")
    ax.set_ylabel(r"$A_{noise}$ (clipped, median of %d draws)" % N_DRAWS)
    ax.set_title("Noise-only A vs resolution")

    ax = axes[1]
    ax.scatter(result_df["A_noise_clipped_median"], result_df["A_actual"], s=10, alpha=0.5, color="steelblue")
    b, a = np.polyfit(result_df["A_noise_clipped_median"], result_df["A_actual"], 1)
    xline = np.linspace(result_df["A_noise_clipped_median"].min(), result_df["A_noise_clipped_median"].max(), 50)
    ax.plot(xline, a + b * xline, "r-", lw=1.5, label="best fit")
    ax.set_xlabel(r"$A_{noise}$ (clipped, white-noise reference)")
    ax.set_ylabel(r"$A_{actual}$ (A_min)")
    ax.legend(fontsize=8)
    ax.set_title("Measured A vs its noise-only reference")

    ax = axes[2]
    ax.hist(result_df["A_noise_clipped_median"] / result_df["A_actual"], bins=40, color="seagreen", edgecolor="white")
    ax.set_xlabel(r"$A_{noise} / A_{actual}$ (reference scale, not a literal noise fraction)")
    ax.set_ylabel("N galaxies")
    ax.set_title("Noise-only reference scale relative to measured A")

    fig.suptitle(f"Baseline: noise floor of A (in_full_sam, n={len(result_df)}, {N_DRAWS} draws/galaxy)")
    fig.tight_layout()
    fig.savefig(OUT / "noise_floor_diagnostics.png", dpi=150)
    plt.close(fig)

    print(f"\nWrote: {OUT / 'noise_floor_results.csv'}")
    print(f"Wrote: {OUT / 'noise_floor_diagnostics.png'}")


if __name__ == "__main__":
    main()
