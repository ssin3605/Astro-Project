#!/usr/bin/env python3
"""
Fits the star-forming main sequence (SFMS) with the Lee et al. (2015) form,
and computes each galaxy's vertical offset from it (delta_ms).

The fit uses every galaxy in the master catalogue with a finite log_SFR.
Within src/utils/sfms.py::fit_ridge_line, only galaxies with
9 <= log_Mstar < 11 and log_SFR >= -1 are used to find the ridge line.

delta_ms is computed for every galaxy with finite log_SFR, regardless of
whether it was used to anchor the fit.

Input : outputs/pipeline/master.csv (from 04_build_master_csv.py)
Output: outputs/pipeline/master.csv            (updated in place: adds delta_ms)
        outputs/pipeline/sfms_fit_params.csv   (S0, a1, Mt, number of ridge
                                                bins and galaxies used)
        outputs/pipeline/sfms_ridge_points.csv (the per-mass-bin points the
                                                    fit was anchored on - for
                                                    overplotting the fitted
                                                    line's inputs)

Usage
    python src/05_sfms.py
    python src/05_sfms.py --input path/to/other_master.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from src.utils.paths import (  # noqa: E402
    MASTER_CSV,
    PIPELINE_DIR,
    SFMS_FIT_PARAMS_CSV,
    SFMS_RIDGE_POINTS_CSV,
)
from src.utils.sfms import compute_delta_ms, fit_ridge_line  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=MASTER_CSV, help="Master CSV to fit against. Defaults to outputs/pipeline/master.csv.")
    args = parser.parse_args()

    input_path = args.input
    if not input_path.exists():
        sys.exit(f"{input_path} not found. Run 04_build_master_csv.py first.")
    print(f"Input master catalogue: {input_path}")

    df = pd.read_csv(input_path)
    if "log_Mstar" not in df.columns or "log_SFR" not in df.columns:
        raise KeyError(f"{input_path} must have log_Mstar and log_SFR columns. Found: {list(df.columns)}")

    log_mstar = df["log_Mstar"].to_numpy(dtype=float)
    log_sfr = df["log_SFR"].to_numpy(dtype=float)

    sfms_params, ridge_mass, ridge_sfr, n_used = fit_ridge_line(log_mstar, log_sfr)
    S0, a1, Mt = sfms_params
    print(f"  Galaxies in catalogue      : {len(df)}")
    print(f"  With finite log_SFR        : {np.isfinite(log_sfr).sum()}")
    print(f"  Used to find the ridge     : {n_used}")

    print("\nSFMS fit complete:")
    print(f"  S0 = {S0:.4f}")
    print(f"  a1 = {a1:.4f}")
    print(f"  Mt = {Mt:.4f}")
    print(f"  Fit anchored on {len(ridge_mass)} mass bins")

    df["delta_ms"] = compute_delta_ms(log_mstar, log_sfr, sfms_params)
    print(f"\n  delta_ms computed for {df['delta_ms'].notna().sum()} of {len(df)} galaxies")

    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(MASTER_CSV, index=False)

    pd.DataFrame([{
        "S0": S0, "a1": a1, "Mt": Mt,
        "n_ridge_bins": len(ridge_mass),
        "n_sf_galaxies": n_used,
    }]).to_csv(SFMS_FIT_PARAMS_CSV, index=False)

    pd.DataFrame({"ridge_log_mstar": ridge_mass, "ridge_log_sfr": ridge_sfr}).to_csv(SFMS_RIDGE_POINTS_CSV, index=False)

    print("\n--- Summary ---")
    print(f"  Wrote (master + delta_ms) : {MASTER_CSV}")
    print(f"  Wrote (fit params)        : {SFMS_FIT_PARAMS_CSV}")
    print(f"  Wrote (ridge points)      : {SFMS_RIDGE_POINTS_CSV}")


if __name__ == "__main__":
    main()
