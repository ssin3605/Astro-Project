"""
Shared loader for Wang's four SAMI_CIndex_data classification/SFR/Mstar
catalogue files.

Centralised here (rather than duplicated in 03_metrics/concentration.py and
04_build_master_csv.py, the way the old repo independently duplicated this
same parsing logic across src/gini_merging.py, src/concentration_merging.py,
and src/asymmetry_merging.py - one of those copies, gini_merging_new.py, had
drifted into a broken state; see archive/README.md) so there is exactly one
place that knows how to read these files.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

CATALOGUES_DIR = Path(__file__).resolve().parents[2] / "data" / "catalogues" / "SAMI_Cindex_data"

TXT_FILES = {
    "SF_Cluster": CATALOGUES_DIR / "clusterout_sf_hbsn3_v0.12.txt",
    "SF_GAMA": CATALOGUES_DIR / "gamaout_sf_hbsn3_v0.12.txt",
    "Composite_Cluster": CATALOGUES_DIR / "cluout_com_v0.12.txt",
    "Composite_GAMA": CATALOGUES_DIR / "gamaout_com_v0.12.txt",
}


def load_wang_txt(path: Path, source_label: str, spectral_class: str, environment: str) -> pd.DataFrame:
    """Load one whitespace-delimited Wang catalogue file, tagging its provenance."""
    with open(path) as f:
        lines = f.readlines()
    lines[0] = lines[0].lstrip("#").lstrip()

    df = pd.read_csv(io.StringIO("".join(lines)), sep=r"\s+", engine="python")

    catid_col = next((c for c in df.columns if c.upper() == "CATID"), None)
    if catid_col is None:
        raise ValueError(f"No CATID column found in {path}. Columns: {list(df.columns)}")
    df = df.rename(columns={catid_col: "CATID"})
    df["CATID"] = df["CATID"].astype(int)

    # Sentinel cleaning (bug found while building the SFMS explorer notebook -
    # this centralised loader was missing logic the old master/scripts/build_csv.py
    # had: SFR columns use exact 0 to mean "no valid detection", and Mstar/SFR
    # both use <=-999 as an explicit missing-data sentinel elsewhere in Wang's
    # catalogue. Left uncleaned, a sentinel-zero SFR silently becomes a fake
    # log10(SFR)=0 -> SFR=1 Msun/yr "measurement" shared by every affected
    # galaxy regardless of mass, which shows up as an obviously-too-tight
    # diagonal streak in log_Mstar vs log_sSFR space. Confirmed 445/1201
    # pure_SF and 747/1758 composite_SF galaxies were affected.
    sfr_cols = [c for c in df.columns if c.upper().startswith("SFR")]
    for c in sfr_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] == 0, c] = np.nan
        df.loc[df[c] <= -999, c] = np.nan
    if "Mstar" in df.columns:
        df["Mstar"] = pd.to_numeric(df["Mstar"], errors="coerce")
        df.loc[df["Mstar"] <= -999, "Mstar"] = np.nan

    df["source_file"] = source_label
    df["spectral_class"] = spectral_class
    df["environment"] = environment

    return df


def load_all_wang_files() -> dict[str, pd.DataFrame]:
    """Load all four files, tagged with (spectral_class, environment) as Wang defines them."""
    return {
        "sf_gama": load_wang_txt(TXT_FILES["SF_GAMA"], "SF_GAMA", "SF", "GAMA"),
        "sf_cluster": load_wang_txt(TXT_FILES["SF_Cluster"], "SF_Cluster", "SF", "Cluster"),
        "com_gama": load_wang_txt(TXT_FILES["Composite_GAMA"], "Composite_GAMA", "Composite", "GAMA"),
        "com_cluster": load_wang_txt(TXT_FILES["Composite_Cluster"], "Composite_Cluster", "Composite", "Cluster"),
    }


def load_size_lookup() -> pd.DataFrame:
    """
    CATID -> (re, PA, ellip) lookup for size/shape overlays (e.g. r_e ellipses
    drawn on flux maps), sourced directly from the four raw catalogue files
    rather than outputs/csv/concentration.csv - so scripts that run before
    the concentration step (e.g. 01_flux_map.py) don't depend on it existing.

    SF value preferred; Composite value used as a fallback for galaxies with
    no SF-classified entry - same priority as smoothness.py's re/re_com logic.
    """
    files = load_all_wang_files()
    cols = ["CATID", "re", "PA", "ellip"]

    sf = pd.concat([files["sf_gama"][cols], files["sf_cluster"][cols]], ignore_index=True)
    sf = sf.drop_duplicates(subset="CATID", keep="first")

    com = pd.concat([files["com_gama"][cols], files["com_cluster"][cols]], ignore_index=True)
    com = com.drop_duplicates(subset="CATID", keep="first")

    com_only = com[~com["CATID"].isin(sf["CATID"])]
    lookup = pd.concat([sf, com_only], ignore_index=True)

    return lookup.set_index("CATID")


def wang_membership_sets(files: dict[str, pd.DataFrame]) -> dict[str, set]:
    """
    Derive the pure_SF / composite_SF CATID sets exactly as build_csv.py did:
        pure_SF      = appears in an SF file but not in any Composite file
        composite_SF = appears in both an SF file and its Composite counterpart
    """
    sf_all = set(files["sf_gama"]["CATID"]) | set(files["sf_cluster"]["CATID"])
    com_all = set(files["com_gama"]["CATID"]) | set(files["com_cluster"]["CATID"])

    return {
        "sf_all": sf_all,
        "com_all": com_all,
        "pure_sf_ids": sf_all - com_all,
        "composite_sf_ids": sf_all & com_all,
    }
