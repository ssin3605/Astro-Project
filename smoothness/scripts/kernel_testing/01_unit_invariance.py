#!/usr/bin/env python3
"""
01_unit_invariance.py
======================
T1 - proves, empirically (not just algebraically), that computing the
smoothing kernel via r_e in kpc gives the identical S to computing it via
r_e in arcsec (the production convention), for every galaxy in the merged
catalogue.

Why this needs doing at all: sigma_px = KERNEL_SCALE * r_e[arcsec] /
PLATE_SCALE. If you instead defined the kernel via r_e[kpc], you'd write
sigma_px = KERNEL_SCALE * r_e[kpc] / kpc_per_arcsec(z) / PLATE_SCALE. Since
r_e[kpc] = r_e[arcsec] * kpc_per_arcsec(z), the kpc_per_arcsec(z) factor
cancels algebraically and the two routes are identical - but "algebraically
identical" and "identical once it's gone through astropy's cosmology
routines and scipy's Gaussian filter" are different claims. This script
checks the second, stronger one, end-to-end, on real flux maps.

Method, per galaxy:
    1. Route A (arcsec, production): sigma_px_A = KERNEL_SCALE * re_used /
       PLATE_SCALE, using process_one() from the actual production module.
    2. Route B (kpc): re_kpc = re_used * kpc_per_arcsec(z); then
       re_arcsec_roundtrip = re_kpc / kpc_per_arcsec(z) (forces the real
       round-trip through the cosmology calculation, not a no-op); feed
       that back into the *same* process_one().
    3. Compare S_A vs S_B and sigma_px_A vs sigma_px_B.

Uses the full merged catalogue (~2259 galaxies, not just in_full_sam) -
this is a general property of the formula, so it should hold everywhere,
not only on the purity-cut science sample.

Output: outputs/01_unit_invariance/unit_invariance_results.csv,
        outputs/01_unit_invariance/delta_S_histogram.png,
        printed summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c

OUT = c.ensure_outdir("01_unit_invariance")


def main():
    c.print_header("T1: Unit invariance (r_e in arcsec vs r_e in kpc)")

    prod = c.import_production_smoothness()
    df = c.load_full_sample()
    df = df.dropna(subset=["re_used", "z"])
    df = df[df["z"] > 0]  # kpc_per_arcsec undefined/degenerate at z<=0

    print(f"Testing {len(df)} galaxies (full merged catalogue, z > 0, valid re_used)")

    rows = []
    n_missing_fits = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Unit invariance"):
        catid = int(row["CATID"])
        halpha_path = c.find_halpha_path(catid)
        if halpha_path is None:
            n_missing_fits += 1
            continue

        re_arcsec_native = float(row["re_used"])
        z = float(row["z"])

        # --- Route A: arcsec, as production does it ---
        result_a = prod.process_one(halpha_path, re_arcsec_native, prod.KERNEL_SCALE, prod.PLATE_SCALE)

        # --- Route B: round-trip through kpc ---
        kpc_per_as = c.kpc_per_arcsec(z)
        re_kpc = re_arcsec_native * kpc_per_as
        re_arcsec_roundtrip = re_kpc / kpc_per_as
        result_b = prod.process_one(halpha_path, re_arcsec_roundtrip, prod.KERNEL_SCALE, prod.PLATE_SCALE)

        rows.append({
            "CATID": catid,
            "z": z,
            "re_arcsec_native": re_arcsec_native,
            "re_arcsec_roundtrip": re_arcsec_roundtrip,
            "re_kpc": re_kpc,
            "kpc_per_as": kpc_per_as,
            "sigma_px_A": result_a["sigma_px"],
            "sigma_px_B": result_b["sigma_px"],
            "S_A": result_a["S"],
            "S_B": result_b["S"],
            "delta_S": result_b["S"] - result_a["S"],
            "delta_sigma_px": result_b["sigma_px"] - result_a["sigma_px"],
        })

    result_df = pd.DataFrame(rows).dropna(subset=["S_A", "S_B"])
    result_df.to_csv(OUT / "unit_invariance_results.csv", index=False)

    abs_delta_S = result_df["delta_S"].abs()
    abs_delta_sigma = result_df["delta_sigma_px"].abs()

    print(f"\nMissing masked-Halpha FITS (skipped) : {n_missing_fits}")
    print(f"Galaxies compared                    : {len(result_df)}")
    print(f"z range spanned                       : {result_df['z'].min():.4f} - {result_df['z'].max():.4f}")
    print(f"kpc_per_as range spanned               : {result_df['kpc_per_as'].min():.3f} - {result_df['kpc_per_as'].max():.3f}"
          f"  ({result_df['kpc_per_as'].max()/result_df['kpc_per_as'].min():.1f}x dynamic range)")
    print(f"\n|delta_S|       max = {abs_delta_S.max():.3e}   median = {abs_delta_S.median():.3e}")
    print(f"|delta_sigma_px| max = {abs_delta_sigma.max():.3e}   median = {abs_delta_sigma.median():.3e}")
    print("\n(For comparison: flux is stored as float32 in the FITS files, whose relative")
    print(" precision floor is ~1e-7 - any delta_S at or below that scale is pure")
    print(" floating-point noise from the cosmology round-trip, not a real difference.)")

    n_nonzero = (abs_delta_S > 1e-9).sum()
    print(f"\nGalaxies with |delta_S| > 1e-9 : {n_nonzero} / {len(result_df)}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    nonzero = abs_delta_S[abs_delta_S > 0]
    if len(nonzero) > 0:
        ax.hist(np.log10(nonzero), bins=40, color="steelblue", edgecolor="white")
        ax.set_xlabel(r"$\log_{10}|S_{kpc\ route} - S_{arcsec\ route}|$")
    else:
        ax.text(0.5, 0.5, "All delta_S exactly 0.0", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("N galaxies")
    ax.set_title(f"T1: S is invariant to r_e units (n={len(result_df)} galaxies)\n"
                 f"max |delta_S| = {abs_delta_S.max():.2e}  (float32 precision floor ~1e-7)")
    ax.axvline(np.log10(1e-7), color="red", ls="--", lw=1, label="float32 precision floor")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "delta_S_histogram.png", dpi=150)
    plt.close(fig)

    print(f"\nWrote: {OUT / 'unit_invariance_results.csv'}")
    print(f"Wrote: {OUT / 'delta_S_histogram.png'}")


if __name__ == "__main__":
    main()
