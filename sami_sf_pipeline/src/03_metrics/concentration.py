#!/usr/bin/env python3
"""
03_metrics/concentration.py
=============================
Builds the Concentration (C) table directly from Wang's external
SAMI_CIndex_data catalogue.

AUDIT_REPORT.md section 1/3.4 confirmed that Concentration is not computed
anywhere in this repo's own FITS/masking pipeline: CIndex is defined by
Wang et al. as log10(r50,Hα / r50,continuum), computed on raw (non
continuum-masked) LZIFU maps with their own elliptical-aperture and
BPT-dust-correction pipeline (see archive/README.md for the original
calculate_r50_example_*.py scripts). That upstream computation is out of
scope for this repo - what belongs here is taking Wang's four classification
tables and turning them into one clean, per-galaxy Concentration table. That
merge logic was audited and confirmed correct; this is a refactor for
clarity, not a change in behaviour.

Because Concentration is not derived from the shared continuum mask, its
sample selection (which galaxies have a valid C) is independent of Gini/
Asymmetry/Smoothness - this is expected and is surfaced downstream via the
in_C flag in 04_build_master_csv.py.

Input : data/catalogues/SAMI_Cindex_data/{clusterout_sf_hbsn3,gamaout_sf_hbsn3,
        cluout_com,gamaout_com}_v0.12.txt
Output: outputs/csv/concentration.csv

Usage
-----
    python src/03_metrics/concentration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_ROOT))

from src.utils.wang import load_all_wang_files  # noqa: E402

OUT_CSV = PIPELINE_ROOT / "outputs" / "csv" / "concentration.csv"

# Columns present in all four Wang files
SHARED_COLS = ["CATID", "re", "PA", "ellip", "z", "g_i", "Mstar", "galclass", "galclass_exclude", "seeing", "seeing_re"]

# SF files only - single CIndex value per galaxy
SF_EXTRA_COLS = ["CIndex", "totalHagas", "ha_r50", "cont_r50", "ha_EW", "SFR"]

# Composite files only - multiple CIndex variants (BPT diagnostic x aperture size)
COMPOSITE_CINDEX_COLS = [
    "CIndex_n2ha_s", "ha_EW_n_s", "SFR_n2ha_s",
    "CIndex_n2ha_m", "ha_EW_n_m", "SFR_n2ha_m",
    "CIndex_n2ha_l", "ha_EW_n_l", "SFR_n2ha_l",
    "CIndex_o3hb_s", "ha_EW_o_s", "SFR_o3hb_s",
    "CIndex_o3hb_m", "ha_EW_o_m", "SFR_o3hb_m",
    "CIndex_o3hb_l", "ha_EW_o_l", "SFR_o3hb_l",
    "CIndex_s2ha_s", "ha_EW_s_s", "SFR_s2ha_s",
    "CIndex_s2ha_m", "ha_EW_s_m", "SFR_s2ha_m",
    "CIndex_s2ha_l", "ha_EW_s_l", "SFR_s2ha_l",
]

# GAMA-only (present in all four Wang GAMA files - both SF_GAMA and
# Composite_GAMA - but absent from every Cluster file; there is no
# BCGCATAID-equivalent anywhere in Wang's SAMI_CIndex_data for Cluster
# galaxies).
GAMA_BCG_COLS = ["BCGCATAID", "MassA_1"]

META_COLS = ["spectral_class", "environment", "conflict_flag", "source_file"]


def build_concentration_table() -> pd.DataFrame:
    print("Loading Wang classification files...")
    files = load_all_wang_files()
    for label, df in files.items():
        print(f"  Loaded {label:<12} - {len(df):>4} rows")

    all_classified = pd.concat(
        [files["sf_gama"], files["sf_cluster"], files["com_gama"], files["com_cluster"]],
        ignore_index=True,
    )

    # NOTE (bug fix found during porting, not part of the original audit):
    # the old concentration_merging.py filtered all_classified down to
    # desired_cols *before* the SF/Composite split, but that filter omitted
    # GAMA_BCG_COLS (BCGCATAID, MassA_1 - present in gamaout_com_v0.12.txt),
    # so those two columns were silently dropped even though they exist in
    # the raw source data. concentration_results.csv (old repo) is missing
    # them as a result. Including GAMA_BCG_COLS here is a genuine fix,
    # not a behaviour change to anything the audit reviewed.
    desired_cols = SHARED_COLS + SF_EXTRA_COLS + COMPOSITE_CINDEX_COLS + GAMA_BCG_COLS + META_COLS
    available_cols = [c for c in desired_cols if c in all_classified.columns]
    all_classified = all_classified[available_cols].copy()

    # A galaxy can appear in both an SF file and its matching Composite file.
    # Flag this rather than silently dropping one - the merge below resolves
    # it (SF row is the base, composite CIndex columns are joined on), but the
    # flag preserves that the classification wasn't clean-cut.
    conflict_catids = (
        all_classified.groupby("CATID")["spectral_class"].nunique().pipe(lambda s: set(s[s > 1].index))
    )
    all_classified["conflict_flag"] = all_classified["CATID"].isin(conflict_catids)

    print(f"\n  Total classified entries : {len(all_classified)}")
    print(f"  Unique CATIDs            : {all_classified['CATID'].nunique()}")
    print(f"  SF + Composite conflicts : {len(conflict_catids)}")

    # GAMA_BCG_COLS is in both sf_cols and composite_cols now (bug fix - see
    # note above): a pure_SF-only GAMA galaxy keeps its own BCGCATAID/MassA_1
    # straight from the SF file, instead of only ever getting it via a
    # Composite-file match.
    sf_cols = SHARED_COLS + SF_EXTRA_COLS + GAMA_BCG_COLS
    composite_cols = SHARED_COLS + COMPOSITE_CINDEX_COLS + GAMA_BCG_COLS

    sf_rows = all_classified[all_classified["spectral_class"] == "SF"].copy()
    com_rows = all_classified[all_classified["spectral_class"] == "Composite"].copy()

    sf_keep = [c for c in sf_cols + META_COLS if c in sf_rows.columns]
    com_keep = [c for c in composite_cols if c in com_rows.columns]

    sf_rows = sf_rows[sf_keep].drop_duplicates(subset="CATID", keep="first")
    com_rows = com_rows[com_keep].drop_duplicates(subset="CATID", keep="first")

    # SF rows are the base; composite CIndex columns are left-joined on where
    # available. GAMA_BCG_COLS now exists on both sides - coalesce rather than
    # keep two near-duplicate columns (they're galaxy-level group properties,
    # not spectral-class-dependent, so SF and Composite agree when both have
    # a value; SF is preferred, Composite fills in where SF's was missing).
    merged = sf_rows.merge(com_rows, on="CATID", how="left", suffixes=("", "_com"))
    for col in GAMA_BCG_COLS:
        com_col = f"{col}_com"
        if com_col in merged.columns:
            merged[col] = merged[col].fillna(merged[com_col])
            merged = merged.drop(columns=[com_col])

    # Galaxies that only appear in a composite file (no SF counterpart) are appended separately.
    composite_only_catids = set(com_rows["CATID"]) - set(sf_rows["CATID"])
    if composite_only_catids:
        composite_only = all_classified[all_classified["CATID"].isin(composite_only_catids)].drop_duplicates(
            subset="CATID", keep="first"
        )
        merged = pd.concat([merged, composite_only], ignore_index=True)
        print(f"  Composite-only CATIDs appended : {len(composite_only_catids)}")

    merged = merged.reset_index(drop=True)

    print(f"\n  Rows after merge : {len(merged)}")
    print(f"  SF CATIDs (base) : {len(sf_rows)}")
    print(f"  Conflict-flagged : {merged['conflict_flag'].sum()}")

    return merged


def main():
    df = build_concentration_table()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"\n  Written : {OUT_CSV}")
    print(f"  Columns ({len(df.columns)}): {list(df.columns)}")


if __name__ == "__main__":
    main()
