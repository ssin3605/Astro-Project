#!/usr/bin/env python3
"""
Builds the input table for the Chapter 4 ML analysis from the finished
master table.

Drops columns the ML notebooks don't use, expands group_status into boolean
flags (is_satellite, is_central, is_isolated, is_cluster), and renames two
columns:
    gama_pair_sep_kpc -> closest_companion_kpc
    re_used           -> re   (the r_e actually used, SF value or composite fallback)

is_cluster is True wherever group_status is empty (rows with no GAMA group
information, which have NaN BCGCATAID and log_MassA_1).

Input : outputs/pipeline/master.csv   (finished by sf_statistics/src/06_add_cosmic_web.py)
Output: outputs/ch4/csv/ml_data.csv

Usage
    python ml/src/prepare_ml_data.py
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "sf_statistics"))

from src.utils.paths import MASTER_CSV, ML_DATA_CSV  # noqa: E402

COLUMNS_TO_DROP = [
    "in_C", "in_G", "in_A", "in_S", "galclass", "galclass_exclude",
    "re_com", "re", "re_source", "Mstar", "SFR", "sSFR", "sfr_flag",
    "BCGCATAID", "MassA_1", "group_status", "gama_hubble_type", "gama_final_class",
    "5NN_flag", "cut_size", "cut_inclination", "cut_seeing",
    "cut_star_forming", "cut_not_pure_agn", "cut_ssfr",
]


def main():
    if not MASTER_CSV.exists():
        raise FileNotFoundError(f"{MASTER_CSV} not found - run the sf_statistics pipeline (01-06) first.")
    df = pd.read_csv(MASTER_CSV)

    df["is_satellite"] = df["group_status"] == "satellite"
    df["is_central"] = df["group_status"] == "central"
    df["is_isolated"] = df["group_status"] == "isolated"
    df["is_cluster"] = df["group_status"].isna()

    df = df.drop(columns=COLUMNS_TO_DROP)
    df = df.rename(columns={"gama_pair_sep_kpc": "closest_companion_kpc", "re_used": "re"})

    ML_DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ML_DATA_CSV, index=False)
    print(f"Wrote {len(df)} rows, {len(df.columns)} columns to {ML_DATA_CSV}")


if __name__ == "__main__":
    main()
