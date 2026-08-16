#!/usr/bin/env python3
"""
05_sfms.py
===========
Fits the star-forming main sequence (SFMS) ridge line - Lee et al. (2015)
turnover form, fit to the peak of the log(SFR) histogram per mass bin
(Tuntipong+2024 approach) - against the current master catalogue, and
computes each galaxy's vertical offset from it (delta_ms).

This was previously master/scripts/sfms_prep.py in the old repo, using
src/sfms.py's fitting logic. Neither was flagged as incorrect in
AUDIT_REPORT.md - SFMS fitting just wasn't part of the original 01-04
rebuild scope. The fitting logic itself (now src/utils/sfms.py) is ported
unchanged; only the driver is new, updated to read the current
outputs/csv/master_<date>.csv schema instead of the old
master_results.csv/G/C/A-less schema.

Note: because AR2022 was dropped from 04_build_master_csv.py (see
CHANGELOG.md), the galaxy sample here differs from what the old
sfms_prep.py fit against (3031 vs 3070 galaxies) - so refit parameters are
expected to differ slightly from the old master_sfms.csv even though the
fitting code is identical. This is a resample effect, not a code change.

SF selection for the fit itself (not the whole sample): finite log_SFR and
log_SFR > SF_SFR_FLOOR (default -2.0). This is deliberately looser than
spectral_class == 'pure_SF'/'composite_SF' - it's a rough pre-fit cut so the
ridge-line/histogram-peak method isn't swamped by clearly-quenched galaxies;
refine SF_SFR_FLOOR if the fit looks off for your sample.

delta_ms is computed for every galaxy with finite log_SFR, regardless of
whether it was used to anchor the fit.

Input : outputs/csv/master_<date>.csv (most recent by default)
Output: outputs/csv/master_sfms_<date>.csv       (master + delta_ms column)
        outputs/csv/sfms_fit_params_<date>.csv   (S0, a1, Mt + fit provenance)
        outputs/csv/sfms_ridge_points_<date>.csv (the per-mass-bin points the
                                                    fit was anchored on - for
                                                    overplotting the fitted
                                                    line's inputs)

Usage
-----
    python src/05_sfms.py
    python src/05_sfms.py --input outputs/csv/master_2026-08-03.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from src.utils.sfms import compute_delta_ms, fit_ridge_line  # noqa: E402

CSV_DIR = PIPELINE_ROOT / "outputs" / "csv"

# Matches exactly "master_YYYY-MM-DD.csv" - not master_sfms_*.csv (this
# script's own output) and not stray Finder-style " 2.csv" copies.
MASTER_CSV_PATTERN = re.compile(r"^master_\d{4}-\d{2}-\d{2}\.csv$")

# Rough pre-fit SF cut (see module docstring) - distinct from fit_ridge_line's
# own internal per-bin sfr_floor.
SF_SFR_FLOOR = -2.0

RUN_DATE = date.today().isoformat()


def find_latest_master_csv() -> Path:
    candidates = sorted(p for p in CSV_DIR.glob("master_*.csv") if MASTER_CSV_PATTERN.match(p.name))
    if not candidates:
        raise FileNotFoundError(
            f"No master_<date>.csv found in {CSV_DIR}. Run 04_build_master_csv.py first."
        )
    return candidates[-1]  # filenames sort chronologically (YYYY-MM-DD)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=None, help="Master CSV to fit against. Defaults to the most recent outputs/csv/master_<date>.csv.")
    parser.add_argument("--sf-sfr-floor", type=float, default=SF_SFR_FLOOR, help=f"log_SFR floor for the pre-fit SF selection (default {SF_SFR_FLOOR}).")
    args = parser.parse_args()

    input_path = args.input or find_latest_master_csv()
    print(f"Input master catalogue: {input_path}")

    df = pd.read_csv(input_path)
    if "log_Mstar" not in df.columns or "log_SFR" not in df.columns:
        raise KeyError(f"{input_path} must have log_Mstar and log_SFR columns. Found: {list(df.columns)}")

    log_mstar = df["log_Mstar"].to_numpy(dtype=float)
    log_sfr = df["log_SFR"].to_numpy(dtype=float)

    sf_mask = np.isfinite(log_sfr) & (log_sfr > args.sf_sfr_floor)
    print(f"  Galaxies in catalogue      : {len(df)}")
    print(f"  Used to anchor the fit     : {sf_mask.sum()}  (finite log_SFR > {args.sf_sfr_floor})")

    sfms_params, ridge_mass, ridge_sfr = fit_ridge_line(log_mstar[sf_mask], log_sfr[sf_mask])
    S0, a1, Mt = sfms_params

    print("\nSFMS fit complete:")
    print(f"  S0 = {S0:.4f}")
    print(f"  a1 = {a1:.4f}")
    print(f"  Mt = {Mt:.4f}")
    print(f"  Fit anchored on {len(ridge_mass)} mass bins")

    df["delta_ms"] = compute_delta_ms(log_mstar, log_sfr, sfms_params)
    print(f"\n  delta_ms computed for {df['delta_ms'].notna().sum()} of {len(df)} galaxies")

    CSV_DIR.mkdir(parents=True, exist_ok=True)

    master_out_path = CSV_DIR / f"master_sfms_{RUN_DATE}.csv"
    df.to_csv(master_out_path, index=False)

    params_out_path = CSV_DIR / f"sfms_fit_params_{RUN_DATE}.csv"
    pd.DataFrame([{
        "S0": S0, "a1": a1, "Mt": Mt,
        "n_ridge_bins": len(ridge_mass),
        "n_sf_galaxies": int(sf_mask.sum()),
        "sf_sfr_floor": args.sf_sfr_floor,
        "input_master_csv": input_path.name,
    }]).to_csv(params_out_path, index=False)

    ridge_out_path = CSV_DIR / f"sfms_ridge_points_{RUN_DATE}.csv"
    pd.DataFrame({"ridge_log_mstar": ridge_mass, "ridge_log_sfr": ridge_sfr}).to_csv(ridge_out_path, index=False)

    print("\n--- Summary ---")
    print(f"  Wrote (master + delta_ms) : {master_out_path}")
    print(f"  Wrote (fit params)        : {params_out_path}")
    print(f"  Wrote (ridge points)      : {ridge_out_path}")


if __name__ == "__main__":
    main()
