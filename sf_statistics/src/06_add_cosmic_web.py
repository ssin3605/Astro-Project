#!/usr/bin/env python3
"""
Merge cosmic web distance features (Dfil, Dnode, Dvoid) from Stefania's
cosmic web FITS catalogue into the master table, matched by galaxy CATID.
This is the final pipeline step: it adds the three columns to master.csv in
place, so master.csv is the finished table used by every chapter.

- Every galaxy already in master.csv is kept, regardless of whether it
  has a cosmic web match.
- Galaxies with no match in the cosmic web FITS file get NaN in the three
  new columns (Dfil, Dnode, Dvoid) rather than being dropped.
- Any existing Dfil/Dnode/Dvoid columns are replaced, so re-running this
  step is safe.

Input : outputs/pipeline/master.csv            (after 05_sfms.py)
        data/fits/cosmic_web_metrics.fits
Output: outputs/pipeline/master.csv            (updated in place)

Usage
    python src/06_add_cosmic_web.py

    # If the join column is named differently in each file:
    python src/06_add_cosmic_web.py --master-key CATID --cosmic-web-key CATAID
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))
from src.utils.paths import COSMIC_WEB_FITS, MASTER_CSV  # noqa: E402

COSMIC_WEB_COLUMNS = ["Dfil", "Dnode", "Dvoid"]


def load_cosmic_web_table(fits_path: Path, key_col: str, value_cols: list[str]) -> pd.DataFrame:
    """Read the cosmic web FITS table and return just the key + value columns."""
    table = Table.read(fits_path)
    colnames = table.colnames

    missing = [c for c in [key_col] + value_cols if c not in colnames]
    if missing:
        raise KeyError(
            f"Column(s) {missing} not found in {fits_path.name}. "
            f"Available columns: {colnames}"
        )

    df = table[[key_col] + value_cols].to_pandas()
    df = df.rename(columns={key_col: "_cw_key"})

    # FITS ASCII key columns (e.g. CATID as '7A') come back as fixed-width
    # byte strings via Table.read - decode and strip before matching against
    # the master key.
    if df["_cw_key"].dtype == object:
        df["_cw_key"] = df["_cw_key"].apply(
            lambda v: v.decode() if isinstance(v, bytes) else v
        ).str.strip()

    # Guard against duplicate CATIDs in the cosmic web file - would silently
    # fan out the merge otherwise.
    dupes = df["_cw_key"][df["_cw_key"].duplicated()].unique()
    if len(dupes) > 0:
        print(
            f"  Warning: {len(dupes)} duplicate key(s) in cosmic web file "
            f"(e.g. {dupes[:5].tolist()}). Keeping first occurrence of each."
        )
        df = df.drop_duplicates(subset="_cw_key", keep="first")

    return df


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--master", type=Path, default=MASTER_CSV, help="Master CSV to add columns to (updated in place). Default: outputs/pipeline/master.csv.")
    parser.add_argument("--cosmic-web", type=Path, default=COSMIC_WEB_FITS, help="Cosmic web FITS file. Default: data/fits/cosmic_web_metrics.fits.")
    parser.add_argument("--master-key", default="CATID", help="Join column name in the master CSV. Default: CATID.")
    parser.add_argument("--cosmic-web-key", default="CATID", help="Join column name in the cosmic web FITS file. Default: CATID.")
    parser.add_argument(
        "--columns", nargs="+", default=COSMIC_WEB_COLUMNS,
        help=f"Cosmic web columns to bring across. Default: {COSMIC_WEB_COLUMNS}",
    )
    args = parser.parse_args()

    if not args.master.exists():
        sys.exit(f"Master CSV not found: {args.master}")
    if not args.cosmic_web.exists():
        sys.exit(f"Cosmic web FITS file not found: {args.cosmic_web}")

    master_df = pd.read_csv(args.master)
    print(f"Loaded master CSV: {args.master.name}  ({len(master_df)} galaxies)")

    # Replace, not duplicate, columns from an earlier run of this step.
    master_df = master_df.drop(columns=[c for c in args.columns if c in master_df.columns])

    if args.master_key not in master_df.columns:
        sys.exit(f"'{args.master_key}' not found in master CSV columns: {list(master_df.columns)}")

    cw_df = load_cosmic_web_table(args.cosmic_web, args.cosmic_web_key, args.columns)
    print(f"Loaded cosmic web file: {args.cosmic_web.name}  ({len(cw_df)} galaxies)")

    # Match keys as strings so an int64 master key (e.g. CATID) lines up with
    # a string key from the FITS file (or vice versa) without dtype errors.
    # The master CSV's own key column is left untouched in the output.
    master_df["_merge_key"] = master_df[args.master_key].astype(str).str.strip()
    cw_df["_merge_key"] = cw_df["_cw_key"].astype(str).str.strip()

    # Left merge: keep every master row, NaN where no cosmic web match exists.
    merged = master_df.merge(
        cw_df.drop(columns="_cw_key"),
        on="_merge_key",
        how="left",
    ).drop(columns="_merge_key")

    n_matched = merged[args.columns[0]].notna().sum()
    n_total = len(merged)
    print(f"Matched {n_matched} / {n_total} galaxies to cosmic web data.")
    print(f"  {n_total - n_matched} galaxies set to NaN for {args.columns} (no match).")

    merged.to_csv(args.master, index=False)
    print(f"Saved -> {args.master}")


if __name__ == "__main__":
    main()