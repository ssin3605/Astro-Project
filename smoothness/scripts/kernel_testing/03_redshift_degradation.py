#!/usr/bin/env python3
"""
03_redshift_degradation.py
============================
T2 - the core test. Takes real, well-resolved galaxies and simulates what
their Halpha map would look like if the *same physical galaxy* had been
observed at a series of higher redshifts with the *same instrument*, then
recomputes S at each simulated z. Because the underlying physical structure
is held fixed (only its apparent angular size changes), any drift in S
along the sequence is a direct, causal measurement of the resolution
artifact - not a correlation that could have some other origin.

Scope, deliberately: this isolates the *geometric* resolution effect only
(angular size shrinking relative to the fixed pixel grid and PSF). It does
NOT attempt to model cosmological surface-brightness dimming or the
increased photometric noise a real higher-z observation would have -
those are real, separate, compounding effects that would make things
*worse*, but modelling them accurately would need a surface-brightness/
exposure-time model this repo doesn't have. This is a legitimate,
statable scope limit, not an oversight - and it's a fair test *against*
this pipeline's own definition of S, because S is invariant to a uniform
flux rescaling (sum(|I|) and sum(residual) both scale linearly, so a
constant multiplicative flux change cancels exactly - the same cancellation
argument as 01_unit_invariance.py, just applied to amplitude instead of
r_e units), so this geometric-only version already isolates the ONE effect
that structurally cannot cancel out: how many resolution elements fit
across the galaxy.

Method, per anchor galaxy (real, well-resolved: top n_res, z_native < 0.06
so there's headroom to degrade upward within/beyond the survey's own z
range):
    1. Load its real flux/err maps at native resolution. These already
       carry the native PSF's blur (they're real observations).
    2. For a grid of target redshifts z_target >= z_native:
       - zoom_factor f = kpc_per_arcsec(z_native) / kpc_per_arcsec(z_target)
         (< 1: the galaxy shrinks on the sky as it's pushed further away,
         for a fixed physical size - see README.md for the derivation)
       - Spatially shrink flux & err by f (scipy.ndimage.zoom, order=1),
         re-pad onto the native 50x50 canvas - this is "the same galaxy,
         same instrument, further away", nothing about its real structure
         is touched, only how many pixels/PSF-widths span it.
       - Re-convolve flux with the NATIVE PSF sigma in pixels (fixed, not
         scaled by f). This is the physically essential step: the PSF is a
         property of the instrument/atmosphere, fixed in pixel space,
         independent of the target's distance - shrinking the galaxy
         without re-imposing the (now relatively larger) fixed-pixel PSF
         would under-blur it and understate the degradation. This skips
         the "deconvolve-then-reconvolve" refinement real artificial-
         redshifting pipelines (e.g. FERENGI, Barden+2008) use - PSF
         deconvolution is unstable on noisy SAMI maps and out of scope
         here - so the native PSF is applied a second time on top of the
         one already baked into the real map, a documented, deliberately
         conservative (mildly over-blurring) approximation rather than an
         oversight.
       - sigma_px_target = sigma_px_native * f (the kernel shrinks with the
         galaxy, exactly as production's r_e-in-arcsec-based kernel would).
       - Run the identical S formula.
    3. Record S_target / S_native as a function of z_target (and of the
       resulting n_res_target), per anchor galaxy.

Output: outputs/03_redshift_degradation/degradation_curves.csv,
        outputs/03_redshift_degradation/degradation_plot.png (per-galaxy
        curves + comparison to the real cross-sample S-vs-n_res relation),
        printed summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, gaussian_filter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c

sys.path.insert(0, str(c.PIPELINE_ROOT))
from src.utils.io import read_masked_halpha  # noqa: E402

OUT = c.ensure_outdir("03_redshift_degradation")

N_ANCHORS = 25
Z_NATIVE_MAX = 0.06     # anchors must start this low to leave headroom
Z_TARGET_MAX = 0.15     # push out beyond the sample's own max z (~0.107)
N_Z_STEPS = 14
MIN_ZOOM_FACTOR = 0.15  # guard against degenerate near-zero arrays


def zoom_shrink_and_repad(arr: np.ndarray, factor: float) -> np.ndarray:
    """Shrink arr by `factor` (<=1) and re-pad, centred, onto arr's own shape."""
    out_shape = arr.shape
    zoomed = zoom(np.nan_to_num(arr, nan=0.0), factor, order=1)
    out = np.zeros(out_shape, dtype=float)
    zh, zw = zoomed.shape
    oh, ow = out_shape
    y0 = max((oh - zh) // 2, 0)
    x0 = max((ow - zw) // 2, 0)
    yh = min(zh, oh - y0)
    xw = min(zw, ow - x0)
    out[y0:y0 + yh, x0:x0 + xw] = zoomed[:yh, :xw]
    return out


def compute_S(I_clean: np.ndarray, sigma_px: float) -> float:
    smoothed = gaussian_filter(I_clean, sigma=sigma_px)
    residual = np.abs(I_clean - smoothed)
    denom = np.nansum(np.abs(I_clean))
    return float(np.nansum(residual) / denom) if denom != 0 else np.nan


def select_anchors(df: pd.DataFrame) -> pd.DataFrame:
    cand = df[df["z"] < Z_NATIVE_MAX].copy()
    cand = cand.sort_values("n_res", ascending=False)
    return cand.head(N_ANCHORS)


def main():
    c.print_header("T2: Redshift-degradation (artificial redshifting) test")

    prod = c.import_production_smoothness()
    df = c.load_full_sam()
    anchors = select_anchors(df)
    print(f"Selected {len(anchors)} well-resolved, low-z anchor galaxies "
          f"(z_native < {Z_NATIVE_MAX}, ranked by n_res)")
    print(f"  n_res range of anchors: {anchors['n_res'].min():.2f} - {anchors['n_res'].max():.2f}")

    z_native_typical = anchors["z"].median()
    rows = []

    for _, row in tqdm(anchors.iterrows(), total=len(anchors), desc="Redshift degradation"):
        catid = int(row["CATID"])
        halpha_path = c.find_halpha_path(catid)
        if halpha_path is None:
            continue

        flux_cube, err_cube = read_masked_halpha(str(halpha_path))
        flux_native = flux_cube[prod.HALPHA_LAYER]
        err_native = err_cube[prod.HALPHA_LAYER]

        z_native = float(row["z"])
        sigma_px_native = float(row["sigma_px"])
        re_used = float(row["re_used"])
        seeing = float(row["seeing"])
        psf_sigma_px_native = (seeing * c.FWHM_TO_SIGMA) / prod.PLATE_SCALE  # fixed, not scaled by f

        S_native = compute_S(flux_native - err_native, sigma_px_native)

        z_targets = np.linspace(z_native, Z_TARGET_MAX, N_Z_STEPS)
        for z_target in z_targets:
            if z_target <= z_native:
                f = 1.0
            else:
                f = c.kpc_per_arcsec(z_native) / c.kpc_per_arcsec(z_target)
            if f < MIN_ZOOM_FACTOR:
                continue

            if f >= 0.999:
                flux_t, err_t = flux_native, err_native
            else:
                flux_t = zoom_shrink_and_repad(flux_native, f)
                err_t = zoom_shrink_and_repad(err_native, f)
                # Re-impose the fixed-in-pixels native PSF on the now-smaller
                # object - see module docstring step 2.
                flux_t = gaussian_filter(flux_t, sigma=psf_sigma_px_native)
                err_t = gaussian_filter(err_t, sigma=psf_sigma_px_native)

            sigma_px_t = sigma_px_native * f
            S_t = compute_S(flux_t - err_t, sigma_px_t)
            n_res_t = (re_used * f) / seeing

            rows.append({
                "CATID": catid, "z_native": z_native, "z_target": z_target,
                "zoom_factor": f, "sigma_px_native": sigma_px_native,
                "sigma_px_target": sigma_px_t, "n_res_native": row["n_res"],
                "n_res_target": n_res_t, "S_native": S_native, "S_target": S_t,
                "S_ratio": S_t / S_native if S_native else np.nan,
            })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT / "degradation_curves.csv", index=False)

    print(f"\nRows written: {len(result_df)}")
    print("\n--- Degradation summary (median S_ratio across anchors, binned by n_res_target) ---")
    bins = [0, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 100]
    result_df["n_res_bin"] = pd.cut(result_df["n_res_target"], bins)
    summary = result_df.groupby("n_res_bin", observed=True)["S_ratio"].agg(["median", "count"])
    print(summary)

    # Fraction of anchor-tracks showing monotonic decline
    declines = []
    for catid, g in result_df.groupby("CATID"):
        g = g.sort_values("z_target")
        declines.append(g["S_target"].iloc[-1] < g["S_target"].iloc[0])
    print(f"\nAnchors whose S declined from lowest to highest simulated z: "
          f"{sum(declines)} / {len(declines)} ({100*sum(declines)/len(declines):.0f}%)")

    # --- Plot: individual degradation tracks + real cross-sample relation ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for catid, g in result_df.groupby("CATID"):
        g = g.sort_values("n_res_target")
        ax.plot(g["n_res_target"], g["S_ratio"], alpha=0.35, lw=1, color="steelblue")
    med = result_df.groupby(pd.cut(result_df["n_res_target"], 20), observed=True).agg(
        n_res_mid=("n_res_target", "median"), S_ratio_med=("S_ratio", "median"))
    ax.plot(med["n_res_mid"], med["S_ratio_med"], color="black", lw=2.5, label="median across anchors")
    ax.axhline(1.0, color="red", ls="--", lw=1, label="S unchanged")
    ax.set_xlabel(r"simulated $n_{res}$ (= $r_e / $seeing at target z)")
    ax.set_ylabel(r"$S_{target} / S_{native}$")
    ax.set_title(f"T2: S degradation vs resolution\n({len(anchors)} anchor galaxies, artificially redshifted)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(df["n_res"], df["S"], s=8, alpha=0.35, color="grey", label="real in_full_sam galaxies")
    # overlay median degradation track, anchored (rescaled) to the real S axis for shape comparison
    for catid, g in result_df.groupby("CATID"):
        g = g.sort_values("n_res_target")
        ax.plot(g["n_res_target"], g["S_target"], alpha=0.5, lw=1, color="firebrick")
    ax.set_xlabel(r"$n_{res}$")
    ax.set_ylabel("S")
    ax.set_title("Simulated degradation tracks (red) overlaid on\nreal sample's own S vs n_res (grey)")
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT / "degradation_plot.png", dpi=150)
    plt.close(fig)

    print(f"\nWrote: {OUT / 'degradation_curves.csv'}")
    print(f"Wrote: {OUT / 'degradation_plot.png'}")


if __name__ == "__main__":
    main()
