#!/usr/bin/env python3
"""
04_resolution_matched_validation.py
======================================
T4 - the validation gate for anything downstream (the planned ML "key
drivers" step in particular). No ML model exists in this repo yet, so this
script does the next best, fully-implementable thing: for each candidate
"driver" of S already in master_final.csv, it compares:

    (a) raw Spearman rho(S, driver)             - on the full in_full_sam
        sample (n=705)
    (b) raw Spearman rho(S, driver)              - on a resolution-matched
        subsample (n_res restricted to a narrow band, so every galaxy in
        it is comparably well/poorly resolved - the resolution confound
        found in 02_noise_floor.py and 03_redshift_degradation.py cannot
        be driving a trend that only exists *between* resolution classes)
    (c) partial Spearman rho(S, driver | n_res)  - on the full sample,
        controlling for n_res directly (uses all 705 galaxies, higher
        power than (b), but relies on the linear-in-ranks partial-
        correlation model rather than physically removing the confound
        by construction the way (b) does - the two are complementary,
        not redundant)

A driver whose association with S survives (b) and (c) close to its raw
(a) value is not just riding on the resolution artifact. A driver whose
rho collapses, flips sign, or loses significance under (b)/(c) is a
candidate for exactly the false "discovery" the ML step would otherwise be
at risk of making. z is included deliberately as a negative control: it is
not a physical driver of star-formation morphology, so a large surviving
rho(S, z | n_res) would itself be a red flag about the method, not a
discovery.

Output: outputs/04_resolution_matched_validation/driver_correlation_table.csv,
        outputs/04_resolution_matched_validation/driver_comparison.png,
        printed summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c

OUT = c.ensure_outdir("04_resolution_matched_validation")

# Resolution-matched band: chosen to keep a substantial subsample while
# narrowing n_res enough that resolution differences within the band are
# small relative to the full sample's range (1.3 - 8.8).
N_RES_BAND = (2.5, 4.5)


def main():
    c.print_header("T4: Resolution-matched validation of candidate S drivers")

    df = c.load_full_sam()
    print(f"Full in_full_sam sample: n={len(df)}")
    print(f"n_res range: {df['n_res'].min():.2f} - {df['n_res'].max():.2f}  "
          f"(median {df['n_res'].median():.2f})")

    matched = df[(df["n_res"] >= N_RES_BAND[0]) & (df["n_res"] <= N_RES_BAND[1])].copy()
    print(f"\nResolution-matched subsample: n_res in {N_RES_BAND} -> n={len(matched)} "
          f"({100*len(matched)/len(df):.0f}% of full sample)")

    raw_full = c.spearman_table(df, "S", c.CANDIDATE_DRIVERS)
    raw_matched = c.spearman_table(matched, "S", c.CANDIDATE_DRIVERS)
    partial_full = c.spearman_table(df, "S", c.CANDIDATE_DRIVERS, control="n_res")

    table = raw_full[["column", "label", "n", "rho", "p"]].rename(
        columns={"n": "n_full", "rho": "rho_full", "p": "p_full"})
    table = table.merge(
        raw_matched[["column", "rho", "p", "n"]].rename(
            columns={"rho": "rho_matched", "p": "p_matched", "n": "n_matched"}),
        on="column")
    table = table.merge(
        partial_full[["column", "partial_rho"]].rename(columns={"partial_rho": "partial_rho_n_res"}),
        on="column")

    table["delta_raw_vs_partial"] = (table["rho_full"] - table["partial_rho_n_res"]).round(4)
    table["sign_flip_matched"] = np.sign(table["rho_full"]) != np.sign(table["rho_matched"])
    table["sign_flip_partial"] = np.sign(table["rho_full"]) != np.sign(table["partial_rho_n_res"])
    table["loses_significance_matched"] = (table["p_full"] < 0.05) & (table["p_matched"] >= 0.05)

    table = table.sort_values("rho_full", key=lambda s: s.abs(), ascending=False)
    table.to_csv(OUT / "driver_correlation_table.csv", index=False)

    pd.set_option("display.width", 140)
    print("\n" + table.to_string(index=False))

    print("\n--- Flags ---")
    flagged = table[table["sign_flip_matched"] | table["sign_flip_partial"] | table["loses_significance_matched"]]
    if len(flagged):
        print("Drivers whose association with S looks resolution-sensitive:")
        print(flagged[["column", "rho_full", "rho_matched", "partial_rho_n_res"]].to_string(index=False))
    else:
        print("No candidate driver flipped sign or lost significance under resolution control.")

    z_row = table[table["column"] == "z"].iloc[0]
    print(f"\nNegative control (z): rho_full={z_row['rho_full']:+.3f} -> "
          f"rho_matched={z_row['rho_matched']:+.3f}, partial_rho={z_row['partial_rho_n_res']:+.3f}")
    print("(z is not a physical driver of SF morphology - its raw correlation with S is expected to be")
    print(" largely a resolution artifact; watching how much of it survives resolution control calibrates")
    print(" how much to trust the *other* drivers' surviving correlations.)")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    order = table["column"].tolist()
    x = np.arange(len(order))
    width = 0.25
    ax.bar(x - width, table["rho_full"], width, label="raw (full sample)", color="grey")
    ax.bar(x, table["rho_matched"], width, label=f"raw (resolution-matched, n_res in {N_RES_BAND})", color="steelblue")
    ax.bar(x + width, table["partial_rho_n_res"], width, label="partial (controlling n_res, full sample)", color="firebrick")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=40, ha="right")
    ax.set_ylabel("Spearman rho with S")
    ax.set_title("T4: candidate driver correlations with S, before/after resolution control")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "driver_comparison.png", dpi=150)
    plt.close(fig)

    print(f"\nWrote: {OUT / 'driver_correlation_table.csv'}")
    print(f"Wrote: {OUT / 'driver_comparison.png'}")


if __name__ == "__main__":
    main()
