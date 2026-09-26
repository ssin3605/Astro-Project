#!/usr/bin/env python3
"""
03_background_subtraction_test.py
====================================
Implements and validates A_BS (the missing background-asymmetry term from
Conselice 2003), for completeness with the literature - following the exact
recipe validated for Smoothness's B_S in
../../../smoothness/scripts/kernel_testing/08_background_subtraction_test.py,
adapted for Asymmetry's own formula (rotate-and-differance instead of
smoothing) and resolving the four open design questions left in
sami_sf_pipeline/ASYMMETRY_HANDOFF.md section 3.

IMPORTANT CONTEXT (see 01_noise_floor.py, 02_grid_stability.py): the
resolution-tracking problem in A_min (n_res beating re_kpc in the sf_ml ML
models, per ASYMMETRY_HANDOFF.md section 2) was already diagnosed as
primarily a grid-search-minimum SELECTION-EFFECT artifact (A_range strongly
anti-correlates with n_res at rho=-0.32; A_mean/A_median computed from the
SAME underlying 100-candidate grid show ~zero n_res correlation, unlike
A_min's rho=+0.30), not a noise-floor contamination problem the way it was
for S (noise-only R^2=0.006 for A vs 0.94 for S, pre-correction). This
script is built for COMPLETENESS with Conselice's formal definition, not
because it's expected to resolve that resolution confound - don't read a
"pass" below as having solved that separate, larger, still-open problem.

Design choices resolved here (ASYMMETRY_HANDOFF.md section 3's four open
questions):

  1. Centre for noise draws: reuse each galaxy's already-found best_cy/
     best_cx from asymmetry.csv (production's own grid search on the REAL
     signal) - no separate grid search on noise.
  2. Clip vs unclipped: clipped, via the real apply_error_subtraction(noise,
     err) - matches production's own I_clean construction.
  3. Footprint for noise draws: uses the REAL galaxy's own common-mask
     footprint (precomputed once from the real I_clean and its rotation),
     NOT a mask freshly derived from each noise draw's own sparse, post-clip
     nonzero pixels - see git history / conversation notes for why (avoids a
     ~3x-too-large small-number-statistics artifact, and sidesteps the
     degenerate-all-zero-draw problem entirely).
  4. Draw count: N_DRAWS_ESTIMATE=50 / N_DRAWS_VALIDATE=20. Checked: the
     headline "20% relative scatter" on A_BS is inflated by galaxies where
     A_BS itself is near-zero; the ABSOLUTE uncertainty this contributes to
     A_corrected is small in practice (median 0.2%, 95th pct 1.6% of a
     typical A_corrected value) - 50 draws is adequate.

RNG seeding: each galaxy gets its own independent stream via
np.random.default_rng((SEED, catid)) - NOT a single generator advanced
sequentially across the batch loop. This guarantees a galaxy gets the
IDENTICAL A_BS/A_corrected whether processed as part of the full batch or
via --catid in isolation, matching smoothness.py's compute_B_S convention
exactly (its own docstring: "same result whether run in batch or via
--catid").

Formula (mirrors compute_B_S / S's correction, ported to A's own
rotate-and-differance construction):

    A_BS  = median, over N_DRAWS_ESTIMATE white-noise realisations drawn
            from the galaxy's own error map (noise ~ Normal(0, err), then
            clipped via apply_error_subtraction), of
            sum(|noise_clean - rotate180(noise_clean)|) evaluated over the
            REAL galaxy's own common-mask footprint at its own best-fit
            centre.
    A_corrected = max((real_numerator - A_BS) / real_denom, 0)
    A_no_sub    = today's A_min, unchanged (renamed for the output only).

Two independent noise-draw sets per galaxy (estimate vs held-out
validation), mirroring 08_background_subtraction_test.py, to avoid
circularity.

Usage
-----
    python3 03_background_subtraction_test.py                # full in_full_sam batch
    python3 03_background_subtraction_test.py --catid 143148  # single galaxy + saved arrays/plot

Output (batch): outputs/03_background_subtraction_test/background_subtraction_results.csv,
                outputs/03_background_subtraction_test/bs_validation_plots.png
Output (--catid): outputs/03_background_subtraction_test/single_galaxy_<CATID>/arrays.npz,
                  outputs/03_background_subtraction_test/single_galaxy_<CATID>/diagnostic.png
"""

from __future__ import annotations

import argparse
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

OUT = c.ensure_outdir("03_background_subtraction_test")
N_DRAWS_ESTIMATE = 50
N_DRAWS_VALIDATE = 20
SEED = 44  # distinct from smoothness's 42/43


def compute_for_galaxy(prod, catid, halpha_path, best_cy, best_cx,
                        n_draws_estimate=N_DRAWS_ESTIMATE, n_draws_validate=N_DRAWS_VALIDATE,
                        return_arrays=False):
    """
    Full A_BS computation for one galaxy - the single source of truth used by
    both the batch loop and --catid mode below, so there's no risk of the two
    entry points silently drifting apart.

    Seeded independently per galaxy via (SEED, catid) - see module docstring.
    Returns (result_dict, arrays_dict_or_None). result_dict is None if the
    galaxy has an empty footprint after cleaning (mirrors production's own
    empty-map guard in asymmetry.py::process_one).
    """
    rng = np.random.default_rng((SEED, catid))

    flux_cube, err_cube = read_masked_halpha(str(halpha_path))
    flux = np.nan_to_num(flux_cube[prod.HALPHA_LAYER], nan=0.0, posinf=0.0, neginf=0.0)
    err = np.nan_to_num(err_cube[prod.HALPHA_LAYER], nan=0.0, posinf=0.0, neginf=0.0)
    scale = np.where(err > 0, err, 0.0)

    I_clean_real = prod.apply_error_subtraction(flux, err)
    if np.sum(I_clean_real != 0) == 0:
        return None, None

    A_check, rotated_real, residual_real = prod.compute_A(I_clean_real, best_cy, best_cx)
    original_mask = I_clean_real != 0
    rotated_mask = rotated_real > 1e-3 * np.max(I_clean_real) if np.max(I_clean_real) > 0 else rotated_real > 0
    common_mask = original_mask & rotated_mask
    real_numerator = float(np.sum(np.abs(residual_real[common_mask])))
    real_denom = float(np.sum(np.abs(I_clean_real[common_mask])))
    if real_denom == 0 or common_mask.sum() == 0:
        return None, None

    example_noise_arrays = None
    est_numerators = []
    for i in range(n_draws_estimate):
        noise = rng.normal(loc=0.0, scale=scale)
        I_noise = prod.apply_error_subtraction(noise, err)
        _, rotated_noise, residual_noise = prod.compute_A(I_noise, best_cy, best_cx)
        est_numerators.append(float(np.sum(np.abs(residual_noise[common_mask]))))
        if return_arrays and i == 0:  # keep the first draw as a representative example
            example_noise_arrays = {
                "I_noise_example": I_noise,
                "rotated_noise_example": rotated_noise,
                "residual_noise_example": residual_noise,
            }
    A_BS = float(np.median(est_numerators))
    A_BS_std = float(np.std(est_numerators))

    A_corrected_raw = (real_numerator - A_BS) / real_denom
    A_corrected = max(A_corrected_raw, 0.0)

    val_noise_corrected = []
    for _ in range(n_draws_validate):
        noise = rng.normal(loc=0.0, scale=scale)
        I_noise = prod.apply_error_subtraction(noise, err)
        _, _, residual_noise = prod.compute_A(I_noise, best_cy, best_cx)
        num_v = float(np.sum(np.abs(residual_noise[common_mask])))
        val_noise_corrected.append(max((num_v - A_BS) / real_denom, 0.0))

    result = {
        "CATID": catid,
        "A_check_recompute": A_check,
        "A_BS": A_BS, "A_BS_std": A_BS_std,
        "A_BS_relative_std": A_BS_std / A_BS if A_BS else np.nan,
        "A_corrected_raw": A_corrected_raw, "A_corrected": A_corrected,
        "frac_negative_before_clip": A_corrected_raw < 0,
        "A_noise_corrected_median_heldout": np.median(val_noise_corrected) if val_noise_corrected else np.nan,
        "real_numerator": real_numerator, "real_denom": real_denom,
    }

    arrays = None
    if return_arrays:
        arrays = {
            "I_clean_real": I_clean_real, "rotated_real": rotated_real, "residual_real": residual_real,
            "common_mask": common_mask,
            **(example_noise_arrays or {}),
        }
    return result, arrays


def run_single_catid(catid: int):
    """--catid mode: one galaxy, printed numbers, saved arrays + explainer plot."""
    prod = c.import_production_asymmetry()
    df = c.load_full_sample()  # full merged catalogue, not restricted to in_full_sam
    df = df[df["CATID"] == catid]
    if df.empty:
        print(f"CATID {catid}: not found in the merged master/asymmetry catalogue.")
        return
    row = df.iloc[0]

    halpha_path = c.find_halpha_path(catid)
    if halpha_path is None:
        print(f"CATID {catid}: no masked Halpha FITS found.")
        return

    best_cy, best_cx = float(row["best_cy"]), float(row["best_cx"])
    result, arrays = compute_for_galaxy(prod, catid, halpha_path, best_cy, best_cx, return_arrays=True)
    if result is None:
        print(f"CATID {catid}: empty footprint after error subtraction.")
        return

    A_no_sub = float(row["A_min"])
    print(f"CATID {catid}:")
    print(f"  A_no_sub (production A_min)      = {A_no_sub:.4f}")
    print(f"  A_BS (median of {N_DRAWS_ESTIMATE} draws)      = {result['A_BS']:.4f}  "
          f"(relative std: {result['A_BS_relative_std']*100:.1f}%)")
    print(f"  A_corrected                       = {result['A_corrected']:.4f}")
    print(f"  best-fit centre (cy, cx)          = ({best_cy:.3f}, {best_cx:.3f})")
    print(f"  sanity check: recompute at that centre = {result['A_check_recompute']:.4f} "
          f"(stored A_min = {A_no_sub:.4f})")

    out_dir = c.ensure_outdir(f"03_background_subtraction_test/single_galaxy_{catid}")
    np.savez(
        out_dir / "arrays.npz",
        I_clean_real=arrays["I_clean_real"],
        rotated_real=arrays["rotated_real"],
        residual_real=arrays["residual_real"],
        common_mask=arrays["common_mask"],
        I_noise_example=arrays.get("I_noise_example"),
        rotated_noise_example=arrays.get("rotated_noise_example"),
        residual_noise_example=arrays.get("residual_noise_example"),
    )
    print(f"\nWrote: {out_dir / 'arrays.npz'}")

    # --- Explainer plot: real data (top row) vs one example noise draw (bottom row) ---
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    real_panels = [
        ("Original (I_clean)", arrays["I_clean_real"]),
        ("Flipped 180deg", arrays["rotated_real"]),
        ("|Residual| (mismatch)", np.abs(arrays["residual_real"])),
    ]
    vmax_real = max(np.max(np.abs(p[1])) for p in real_panels)
    for ax, (title, img) in zip(axes[0], real_panels):
        im = ax.imshow(img, cmap="inferno", vmin=0, vmax=vmax_real, origin="lower")
        ax.contour(arrays["common_mask"], levels=[0.5], colors="cyan", linewidths=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("Real galaxy", fontsize=11)

    if "I_noise_example" in arrays and arrays["I_noise_example"] is not None:
        noise_panels = [
            ("Example noise draw", arrays["I_noise_example"]),
            ("Flipped 180deg", arrays["rotated_noise_example"]),
            ("|Residual| (mismatch)", np.abs(arrays["residual_noise_example"])),
        ]
        vmax_noise = max(np.max(np.abs(p[1])) for p in noise_panels)
        for ax, (title, img) in zip(axes[1], noise_panels):
            ax.imshow(img, cmap="inferno", vmin=0, vmax=vmax_noise, origin="lower")
            ax.contour(arrays["common_mask"], levels=[0.5], colors="cyan", linewidths=0.8)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        axes[1, 0].set_ylabel("Example noise draw\n(used to estimate A_BS)", fontsize=11)

    fig.suptitle(f"CATID {catid}: A_no_sub={A_no_sub:.3f}, A_BS={result['A_BS']:.3f}, "
                 f"A_corrected={result['A_corrected']:.3f}\n(cyan contour = common-mask footprint "
                 f"used for both rows)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostic.png", dpi=150)
    plt.close(fig)
    print(f"Wrote: {out_dir / 'diagnostic.png'}")


def run_batch():
    c.print_header("A_BS: Background-asymmetry term - implementation and validation")

    prod = c.import_production_asymmetry()
    df = c.load_full_sam()
    print(f"Sample: in_full_sam, n={len(df)}")
    print(f"A_BS estimated from {N_DRAWS_ESTIMATE} draws; validated on {N_DRAWS_VALIDATE} independent held-out draws")
    print("NOTE: built for completeness with Conselice's formal definition - not expected to resolve")
    print("the separate grid-search-minimum selection-effect issue found in 02_grid_stability.py.\n")

    rows = []
    n_missing = 0
    n_empty_real = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="A background subtraction"):
        catid = int(row["CATID"])
        halpha_path = c.find_halpha_path(catid)
        if halpha_path is None:
            n_missing += 1
            continue

        best_cy, best_cx = float(row["best_cy"]), float(row["best_cx"])
        result, _ = compute_for_galaxy(prod, catid, halpha_path, best_cy, best_cx, return_arrays=False)
        if result is None:
            n_empty_real += 1
            continue

        result["n_res"] = row["n_res"]
        result["z"] = row["z"]
        result["A_no_sub"] = row["A_min"]
        rows.append(result)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT / "background_subtraction_results.csv", index=False)

    print(f"\nMissing FITS (skipped)          : {n_missing}")
    print(f"Empty real footprint (skipped)  : {n_empty_real}")
    print(f"Galaxies processed              : {len(result_df)}")

    n_mismatch = (~np.isclose(result_df["A_check_recompute"], result_df["A_no_sub"], rtol=1e-4, atol=1e-4)).sum()
    print(f"A recompute-at-best-centre vs stored A_min mismatches: {n_mismatch} / {len(result_df)} "
          f"(expected: small, from best_cy/best_cx being stored rounded to 3dp in asymmetry.csv)")

    print(f"\nA_BS relative draw-to-draw scatter: median={result_df['A_BS_relative_std'].median()*100:.1f}%")
    n_neg = result_df["frac_negative_before_clip"].sum()
    print(f"Galaxies with A_corrected_raw < 0 before clipping: {n_neg} / {len(result_df)} ({100*n_neg/len(result_df):.1f}%)")

    print("\n--- Criterion 1: does the corrected formula neutralise pure noise on HELD-OUT draws? ---")
    print(f"  A_no_sub median                    : {result_df['A_no_sub'].median():.4f}")
    print(f"  A_corrected median                  : {result_df['A_corrected'].median():.4f}")
    print(f"  A_noise_corrected (held-out) median  : {result_df['A_noise_corrected_median_heldout'].median():.4f}")
    for col in ["n_res", "z"]:
        rho_before, _ = spearmanr(result_df["A_no_sub"], result_df[col])
        rho_after_signal, _ = spearmanr(result_df["A_corrected"], result_df[col])
        rho_after_noise, _ = spearmanr(result_df["A_noise_corrected_median_heldout"], result_df[col])
        print(f"  rho(., {col:5s}):  A_no_sub={rho_before:+.3f}   A_corrected={rho_after_signal:+.3f}   "
              f"A_noise_corrected(held-out)={rho_after_noise:+.3f}")

    print("\n--- Criterion 2: rank relationship to A_no_sub ---")
    rho_rank, _ = spearmanr(result_df["A_corrected"], result_df["A_no_sub"])
    print(f"  Spearman rho(A_corrected, A_no_sub) = {rho_rank:.3f}")

    print("\n--- Criterion 3: do candidate drivers survive? ---")
    for label, A_col in [("A_no_sub", "A_no_sub"), ("A_corrected", "A_corrected")]:
        tmp = df.merge(result_df[["CATID", A_col]], on="CATID", how="inner")
        vals = []
        for driver in ["log_SFR", "log_Mstar", "delta_ms"]:
            sub = tmp[[A_col, driver]].apply(pd.to_numeric, errors="coerce").dropna()
            rho, p = spearmanr(sub[A_col], sub[driver])
            vals.append(f"{driver}={rho:+.3f}")
        print(f"  {label:12s}: {', '.join(vals)}")

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.scatter(result_df["n_res"], result_df["A_no_sub"], s=8, alpha=0.3, color="grey", label="A_no_sub")
    ax.scatter(result_df["n_res"], result_df["A_corrected"], s=8, alpha=0.3, color="seagreen", label="A_corrected")
    ax.set_xlabel(r"$n_{res}$"); ax.set_ylabel("A")
    ax.set_title("A vs resolution: before/after A_BS correction")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(result_df["A_noise_corrected_median_heldout"].dropna(), bins=40, color="firebrick", alpha=0.7,
             label="A_noise_corrected\n(held-out draws)")
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("A computed on pure noise, after A_BS correction")
    ax.set_ylabel("N galaxies")
    ax.set_title("Does corrected A neutralise pure noise?")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.scatter(result_df["A_no_sub"], result_df["A_corrected"], s=8, alpha=0.4, color="steelblue")
    lim = [0, result_df["A_no_sub"].max() * 1.05]
    ax.plot(lim, lim, "k--", lw=1, label="1:1")
    ax.set_xlabel("A_no_sub"); ax.set_ylabel("A_corrected")
    ax.set_title(f"Rank relationship (rho={rho_rank:.2f})")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "bs_validation_plots.png", dpi=150)
    plt.close(fig)

    print(f"\nWrote: {OUT / 'background_subtraction_results.csv'}")
    print(f"Wrote: {OUT / 'bs_validation_plots.png'}")

    print("\n=== NOTE ===")
    print("This validates A_BS as a formally-complete implementation of Conselice's background-")
    print("asymmetry term. It is NOT expected to (and, per 02_grid_stability.py, is unlikely to)")
    print("meaningfully reduce A_min's separate resolution/n_res correlation, which looks driven")
    print("by the grid-search-minimum selection effect instead - a different, unaddressed problem.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catid", type=int, default=None,
                         help="Compute A_BS for a single CATID and save arrays/plot for it (does not run the batch).")
    args = parser.parse_args()

    if args.catid is not None:
        run_single_catid(args.catid)
    else:
        run_batch()


if __name__ == "__main__":
    main()
