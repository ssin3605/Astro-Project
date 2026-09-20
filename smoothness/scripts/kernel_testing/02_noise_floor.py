#!/usr/bin/env python3
"""
02_noise_floor.py
===================
T6 - characterises the noise-only contribution to S, and tests whether it
is what's driving the S-vs-resolution correlation found in the initial
diagnostic (S vs n_res: Spearman rho = +0.52 on in_full_sam).

Method, per galaxy (in_full_sam only - this is about the actual science
sample, not the full catalogue):
    1. Load the real error map (err) from the masked Halpha FITS - this is
       the same error map production's I_clean = flux - err uses.
    2. Draw a pure-noise realisation: noise ~ Normal(0, err) at every pixel
       (err is already zero outside the shared continuum mask footprint, so
       noise is automatically zero there too - same footprint as production).
    3. Run it through the *identical* production pipeline, substituting the
       noise draw for flux: I_clean_noise = noise - err, smoothed with the
       galaxy's own production sigma_px, S_noise = sum(|I_clean_noise -
       smoothed|) / sum(|I_clean_noise|).
    4. Repeat N_DRAWS times per galaxy, keep the median as that galaxy's
       noise-floor S_noise, and the std as its draw-to-draw scatter.

This isolates a mechanism, not just a correlation: if S_noise (which by
construction contains zero real astrophysical clumpiness - it's smoothed
noise, nothing else) still correlates with n_res the same way the real S
does, that is direct evidence the real S-vs-resolution trend has a
noise/resolution origin, not a physical one.

Output: outputs/02_noise_floor/noise_floor_results.csv,
        outputs/02_noise_floor/noise_floor_diagnostics.png,
        printed correlation summary.
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

OUT = c.ensure_outdir("02_noise_floor")
N_DRAWS = 20
SEED = 42


def compute_S_from_arrays(I_clean: np.ndarray, sigma_px: float) -> float:
    """Identical formula to production's compute_S(), applied to an arbitrary I_clean."""
    smoothed = gaussian_filter(I_clean, sigma=sigma_px)
    residual = np.abs(I_clean - smoothed)
    denom = np.nansum(np.abs(I_clean))
    if denom == 0:
        return np.nan
    return float(np.nansum(residual) / denom)


def main():
    c.print_header("T6: Noise floor of S vs resolution")

    prod = c.import_production_smoothness()
    df = c.load_full_sam()
    print(f"Sample: in_full_sam, n={len(df)}")

    rng = np.random.default_rng(SEED)
    rows = []
    n_missing = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Noise floor"):
        catid = int(row["CATID"])
        halpha_path = c.find_halpha_path(catid)
        if halpha_path is None:
            n_missing += 1
            continue

        flux_cube, err_cube = read_masked_halpha(str(halpha_path))
        err = err_cube[prod.HALPHA_LAYER]

        sigma_px = float(row["sigma_px"])

        s_noise_draws = []
        for _ in range(N_DRAWS):
            noise = rng.normal(loc=0.0, scale=np.where(err > 0, err, 0.0))
            I_clean_noise = noise - err
            s = compute_S_from_arrays(I_clean_noise, sigma_px)
            if not np.isnan(s):
                s_noise_draws.append(s)

        if not s_noise_draws:
            continue

        s_noise_draws = np.array(s_noise_draws)
        rows.append({
            "CATID": catid,
            "S_actual": row["S"],
            "S_noise_median": np.median(s_noise_draws),
            "S_noise_std": np.std(s_noise_draws),
            "n_draws": len(s_noise_draws),
            "n_res": row["n_res"],
            "seeing": row["seeing"],
            "re_used": row["re_used"],
            "z": row["z"],
            "sigma_over_psf": row["sigma_over_psf"],
        })

    result_df = pd.DataFrame(rows)
    # NOTE on interpretation: this ratio is NOT a literal "fraction of S_actual
    # that is noise" decomposition. S_noise is computed from spatially
    # *uncorrelated* (white) per-pixel noise, which has a qualitatively
    # different correlation structure to real Halpha flux (which has genuine
    # spatial coherence from real HII regions/diffuse gas even in a clumpy
    # galaxy) - white noise is closer to the *maximum* achievable roughness
    # under this kernel, not an additive contamination term that sums with a
    # separate "real signal" S. So S_noise/S_actual > 1 for most galaxies does
    # NOT mean "noise exceeds signal in the real map" - it means real Halpha
    # flux is smoother than white noise (expected), and that S_noise is best
    # read as a per-galaxy *reference roughness scale* set purely by kernel
    # geometry (sigma_px vs mask footprint), containing zero real flux
    # information. What matters below is whether that artifact-only
    # reference scale tracks the same things S_actual does.
    result_df["ratio_noise_to_actual"] = result_df["S_noise_median"] / result_df["S_actual"]
    result_df.to_csv(OUT / "noise_floor_results.csv", index=False)

    print(f"\nMissing FITS (skipped): {n_missing}")
    print(f"Galaxies processed     : {len(result_df)}")
    print(f"\nS_actual      : median={result_df['S_actual'].median():.4f}")
    print(f"S_noise_median: median={result_df['S_noise_median'].median():.4f}  "
          f"(a per-galaxy roughness reference from kernel geometry alone - see NOTE in source)")

    print("\n--- Does this noise-only reference scale track the same things S_actual does? ---")
    for col in ["n_res", "seeing", "re_used", "z", "sigma_over_psf"]:
        rho_noise, p_noise = spearmanr(result_df["S_noise_median"], result_df[col])
        rho_actual, p_actual = spearmanr(result_df["S_actual"], result_df[col])
        print(f"  {col:16s}  S_noise vs {col:16s} rho={rho_noise:+.3f} (p={p_noise:.1e})   "
              f"|  S_actual vs {col:16s} rho={rho_actual:+.3f} (p={p_actual:.1e})")

    rho_cross, p_cross = spearmanr(result_df["S_actual"], result_df["S_noise_median"])
    b, a = np.polyfit(result_df["S_noise_median"], result_df["S_actual"], 1)
    pred = a + b * result_df["S_noise_median"]
    r2_linear = 1 - np.sum((result_df["S_actual"] - pred) ** 2) / np.sum((result_df["S_actual"] - result_df["S_actual"].mean()) ** 2)
    print(f"\n  S_actual vs S_noise_median: Spearman rho={rho_cross:+.3f} (p={p_cross:.1e}),  rho^2={rho_cross**2:.3f}")
    print(f"  Linear R^2 of S_actual ~ S_noise_median: {r2_linear:.3f}")
    print(f"  => A quantity computed from PURE WHITE NOISE, containing zero real flux")
    print(f"     information, explains {r2_linear*100:.0f}% of the variance in the measured S across the")
    print(f"     in_full_sam sample. This is the headline result of this test.")

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.scatter(result_df["n_res"], result_df["S_noise_median"], s=10, alpha=0.5, color="firebrick")
    ax.set_xlabel(r"$n_{res} = r_e / \mathrm{seeing}$")
    ax.set_ylabel(r"$S_{noise}$ (median of %d draws)" % N_DRAWS)
    ax.set_title("Noise-only S vs resolution")

    ax = axes[1]
    ax.scatter(result_df["S_noise_median"], result_df["S_actual"], s=10, alpha=0.5, color="steelblue")
    b, a = np.polyfit(result_df["S_noise_median"], result_df["S_actual"], 1)
    xline = np.linspace(result_df["S_noise_median"].min(), result_df["S_noise_median"].max(), 50)
    ax.plot(xline, a + b * xline, "r-", lw=1.5, label=f"best fit (R$^2$={r2_linear:.2f})")
    ax.set_xlabel(r"$S_{noise}$  (white-noise reference, zero real flux info)")
    ax.set_ylabel(r"$S_{actual}$")
    ax.legend(fontsize=8)
    ax.set_title(f"Measured S vs its noise-only reference\n(rho={rho_cross:.2f}, R$^2$={r2_linear:.2f})")

    ax = axes[2]
    ax.hist(result_df["ratio_noise_to_actual"], bins=40, color="seagreen", edgecolor="white")
    ax.set_xlabel(r"$S_{noise} / S_{actual}$  (reference scale, not a literal noise fraction - see NOTE)")
    ax.set_ylabel("N galaxies")
    ax.set_title("Noise-only reference scale relative to measured S")

    fig.suptitle(f"T6: Noise floor of S (in_full_sam, n={len(result_df)}, {N_DRAWS} draws/galaxy)")
    fig.tight_layout()
    fig.savefig(OUT / "noise_floor_diagnostics.png", dpi=150)
    plt.close(fig)

    print(f"\nWrote: {OUT / 'noise_floor_results.csv'}")
    print(f"Wrote: {OUT / 'noise_floor_diagnostics.png'}")


if __name__ == "__main__":
    main()
