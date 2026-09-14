#!/usr/bin/env python3
"""
04_build_master_csv.py
========================
Builds the presentation-ready master catalogue by merging:
    - Concentration (C) + Wang metadata (Mstar, re, ellip, z, environment,
      galclass, BCGCATAID, MassA_1, ...)      <- outputs/csv/concentration.csv
    - Gini (G)                                <- outputs/csv/gini.csv
    - Asymmetry (A)                           <- outputs/csv/asymmetry.csv
    - Smoothness (S)                          <- outputs/csv/smoothness.csv
    - GAMA morphology + pairs (environment context, kept per your review of
      AUDIT_REPORT.md - these are environment metadata, not an SFR/Mstar
      source, so they're out of scope of the "standardize on Wang + masked
      Halpha only" instruction)

Per your review of AUDIT_REPORT.md, AR2022_SAMI_DR3_Mstar_SFR_GSWL_rescaled
is DROPPED ENTIRELY - it is not read anywhere in this script. Every galaxy's
Mstar/SFR/spectral_class comes from Wang's own catalogue (via
concentration.csv); a galaxy with no Wang data gets NaN Mstar/SFR rather
than a silently-mixed-in AR2022 fallback value. The CATID universe is the
union of Concentration, Gini, Asymmetry, and Smoothness CATIDs only.

SFR resolution (confirmed against your point 3): a galaxy's Wang membership
determines the SFR source -
    pure_SF      (in an SF file, not in any Composite file) -> Wang's own SFR
    composite_SF (in both)                                  -> SFR_n2ha_s,
        the [NII]/Halpha BPT-isolated star-forming component, mitigating
        AGN/LINER contamination of the Halpha flux
    composite_SF galaxies with no valid SFR_n2ha_s -> NaN (not silently
        backfilled with the SF-file SFR, which would reintroduce AGN/LINER
        contamination)

RESOLVED (2026-08-17): the old master_results.csv contained `5NN`,
`5NN_err`, and `is_isolated` columns (5th-nearest-neighbour surface density
/ isolation flag) that no script in the old repo ever computed - only ever
*read* - and one notebook cell patched a corrupted value by hand
('Falsea' for CATID 7969), suggesting the column had drifted out of any
reproducible pipeline. Rather than reintroduce that unverified column,
`5NN`/`5NN_err`/`5NN_flag` are now sourced fresh from
`data/fits/surface_density.fits` (GAMA's own surface-density catalogue,
previously unused anywhere in this pipeline - see CHANGELOG.md), and
`is_isolated` is replaced by `group_status` (central/satellite/isolated/NaN),
derived directly from Wang's `BCGCATAID` via `classify_group_status()`
below - GAMA-only, since no Cluster file has a BCGCATAID equivalent.

Output: outputs/csv/master_<YYYY-MM-DD>.csv (dated per run, never overwritten)

Usage
-----
    python src/04_build_master_csv.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from src.utils.units import to_log_sfr  
from src.utils.wang import load_all_wang_files, wang_membership_sets  
from src.utils.wang_quality_cuts import CUT_COLUMNS, apply_quality_cuts  

CSV_DIR = PIPELINE_ROOT / "outputs" / "csv"
CONCENTRATION_CSV = CSV_DIR / "concentration.csv"
GINI_CSV = CSV_DIR / "gini.csv"
ASYMMETRY_CSV = CSV_DIR / "asymmetry.csv"
SMOOTHNESS_CSV = CSV_DIR / "smoothness.csv"

GAMA_MORPH_FITS = PIPELINE_ROOT / "data" / "catalogues" / "gama_morphology.fits"
GAMA_PAIRS_FITS = PIPELINE_ROOT / "data" / "catalogues" / "gama_pairs.fits"
SURFACE_DENSITY_FITS = PIPELINE_ROOT / "data" / "catalogues" / "surface_density.fits"

OUT_CSV = CSV_DIR / f"master_{date.today().isoformat()}.csv"

# SFR fallback policy for composite_SF galaxies with no valid SFR_n2ha_s.
# 'nan' matches the audited/confirmed-correct old build_csv.py default
# (methodologically safe: never silently reuses the AGN/LINER-contaminated
# SF-file SFR for a galaxy Wang itself flagged as composite).
COMPOSITE_SFR_FALLBACK = "nan"  # 'nan' | 'sf_file'

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
PAIR_VEL_CUT_KMS = 500.0
PAIR_SEP_100_KPC = 100.0
PAIR_SEP_50_KPC = 50.0


def load_fits_as_df(path: Path, label: str = "") -> pd.DataFrame:
    print(f"  Loading FITS: {path.name}  [{label}]")
    with fits.open(path) as hdul:
        df = Table(hdul[1].data).to_pandas()
    df.columns = [c.upper() for c in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            df[col] = df[col].str.decode("utf-8").str.strip()
        except AttributeError:
            pass
    return df


def arcsec_to_kpc(sep_arcsec: pd.Series, redshift: pd.Series) -> pd.Series:
    def convert_row(row):
        sep, z = row
        try:
            sep, z = float(sep), float(z)
        except (ValueError, TypeError):
            return np.nan
        if np.isnan(sep) or np.isnan(z) or z <= 0:
            return np.nan
        return COSMO.kpc_proper_per_arcmin(z).value / 60.0 * sep

    return pd.Series(list(map(convert_row, zip(sep_arcsec, redshift))), index=sep_arcsec.index)


def classify_group_status(bcgcataid: pd.Series, catid: pd.Series) -> pd.Series:
    """BCGCATAID == CATID -> central; BCGCATAID == 0 -> isolated; anything
    else non-null -> satellite. GAMA-only - BCGCATAID has no Cluster
    equivalent anywhere in Wang's SAMI_CIndex_data, so Cluster galaxies (and
    any GAMA galaxy Wang has no BCGCATAID for) get NaN, not a guessed
    default - "undetermined" is the honest answer, not "isolated".

    Built with explicit object-dtype assignment rather than np.select/
    np.where: mixing string choices with a np.nan default there lets numpy's
    dtype promotion pick a fixed-width unicode dtype, which silently
    stringifies the nan into the literal text "nan" - .notna() then wrongly
    returns True for it. Only survives unnoticed because pandas' CSV writer/
    reader round-trip happens to treat bare "nan" as a missing value again
    on read - anyone using the in-memory frame directly would get it wrong."""
    status = pd.Series(np.nan, index=catid.index, dtype=object)
    status[bcgcataid.notna()] = "satellite"
    status[bcgcataid == 0] = "isolated"
    status[bcgcataid == catid] = "central"
    return status


def make_dataset_tag(row) -> str:
    """Human-readable provenance tag. Simplified from the old AR2022-aware
    version now that Wang concentration.csv is the only metadata anchor."""
    if row["in_C"] and row["n_metrics"] == 3:
        return "full"
    if row["in_C"] and row["n_metrics"] >= 1:
        return "wang+morph"
    if row["in_C"]:
        return "wang_only"
    if row["n_metrics"] >= 1:
        return "morph_only"
    return "unclassified"


def main():
    print("=" * 60)
    print("STEP 1 - Loading metric tables")
    print("=" * 60)
    conc = pd.read_csv(CONCENTRATION_CSV)
    gini = pd.read_csv(GINI_CSV)
    asym = pd.read_csv(ASYMMETRY_CSV)
    smooth = pd.read_csv(SMOOTHNESS_CSV)
    for label, df in [("concentration", conc), ("gini", gini), ("asymmetry", asym), ("smoothness", smooth)]:
        print(f"  {label:<14} : {len(df)} rows")

    print("\n" + "=" * 60)
    print("STEP 2 - Loading Wang TXT membership (pure_SF / composite_SF)")
    print("=" * 60)
    wang_files = load_all_wang_files()
    membership = wang_membership_sets(wang_files)
    print(f"  pure_SF      : {len(membership['pure_sf_ids'])}")
    print(f"  composite_SF : {len(membership['composite_sf_ids'])}")

    print("\n" + "=" * 60)
    print("STEP 3 - Applying Wang (2022) Full_sam quality/purity cuts")
    print("=" * 60)
    conc = apply_quality_cuts(conc, membership)
    for col in CUT_COLUMNS:
        print(f"  {col:<18}: {int(conc[col].sum())} pass")
    print(f"  {'in_full_sam':<18}: {int(conc['in_full_sam'].sum())} pass (all six cuts)")

    print("\n" + "=" * 60)
    print("STEP 4 - Building CATID universe (Wang concentration + G/A/S only - no AR2022)")
    print("=" * 60)
    all_catids = pd.DataFrame({
        "CATID": pd.concat([conc["CATID"], gini["CATID"], asym["CATID"], smooth["CATID"]])
        .drop_duplicates()
        .reset_index(drop=True)
    })
    print(f"  Total unique CATIDs : {len(all_catids)}")

    print("\n" + "=" * 60)
    print("STEP 5 - Merging metric tables")
    print("=" * 60)
    master = all_catids.copy()

    conc_cols = [
        "CATID", "CIndex", "re", "re_com", "ellip", "z", "g_i", "Mstar", "galclass",
        "galclass_exclude", "environment", "seeing", "seeing_re", "source_file",
        "conflict_flag", "SFR", "SFR_n2ha_s", "BCGCATAID", "MassA_1",
        *CUT_COLUMNS, "in_full_sam",
    ]
    conc_present = [c for c in conc_cols if c in conc.columns]
    master = master.merge(conc[conc_present], on="CATID", how="left")
    master["in_C"] = master["CATID"].isin(set(conc["CATID"]))
    master = master.rename(columns={"CIndex": "C"})

    # No concentration entry -> never evaluated against Wang's cuts -> not in Full_sam.
    # Individual cut_* columns are left as NaN for these rows (honestly "not evaluated",
    # not "failed") - only the overall in_full_sam flag needs a hard boolean for filtering.
    master["in_full_sam"] = master["in_full_sam"].fillna(False).astype(bool)

    master = master.merge(gini[["CATID", "gini_halpha"]], on="CATID", how="left")
    master = master.merge(asym[["CATID", "A_min"]], on="CATID", how="left")
    master = master.merge(smooth[["CATID", "S"]], on="CATID", how="left")
    master = master.rename(columns={"gini_halpha": "G", "A_min": "A"})

    master["in_G"] = master["CATID"].isin(set(gini["CATID"]))
    master["in_A"] = master["CATID"].isin(set(asym["CATID"]))
    master["in_S"] = master["CATID"].isin(set(smooth["CATID"]))
    master["n_metrics"] = (
        master["G"].notna().astype(int) + master["A"].notna().astype(int) + master["S"].notna().astype(int)
    )

    print(f"  in_C : {master['in_C'].sum()}   in_G : {master['in_G'].sum()}   in_A : {master['in_A'].sum()}   in_S : {master['in_S'].sum()}")

    print("\n" + "=" * 60)
    print("STEP 6 - Assigning spectral_class from Wang TXT membership")
    print("=" * 60)

    def assign_spectral_class(catid):
        if catid in membership["composite_sf_ids"]:
            return "composite_SF"
        if catid in membership["pure_sf_ids"]:
            return "pure_SF"
        return np.nan

    master["spectral_class"] = master["CATID"].apply(assign_spectral_class)
    print(f"  pure_SF      : {(master['spectral_class'] == 'pure_SF').sum()}")
    print(f"  composite_SF : {(master['spectral_class'] == 'composite_SF').sum()}")
    print(f"  NaN (not in any Wang TXT) : {master['spectral_class'].isna().sum()}")

    print("\n" + "=" * 60)
    print("STEP 7 - Resolving SFR (Wang-only; no AR2022)")
    print(f'  Composite fallback policy: COMPOSITE_SFR_FALLBACK = "{COMPOSITE_SFR_FALLBACK}"')
    print("=" * 60)

    sfr_values, sfr_flags = [], []
    for sc, sf_val, n2ha_val in zip(master["spectral_class"], master.get("SFR"), master.get("SFR_n2ha_s")):
        if sc == "pure_SF":
            sfr, flag = (sf_val, "sf_file") if pd.notna(sf_val) else (np.nan, np.nan)
        elif sc == "composite_SF":
            if pd.notna(n2ha_val):
                sfr, flag = n2ha_val, "n2ha_s"
            elif COMPOSITE_SFR_FALLBACK == "sf_file" and pd.notna(sf_val):
                sfr, flag = sf_val, "sf_file"
            else:
                sfr, flag = np.nan, "nan_no_n2ha"
        else:
            sfr, flag = np.nan, np.nan
        sfr_values.append(sfr)
        sfr_flags.append(flag)

    master["SFR_raw"] = sfr_values
    master["sfr_flag"] = sfr_flags
    master = master.drop(columns=[c for c in ["SFR", "SFR_n2ha_s"] if c in master.columns])

    print(pd.Series(sfr_flags).value_counts(dropna=False).to_string())

    print("\n" + "=" * 60)
    print("STEP 8 - Computing log_SFR / log_Mstar / sSFR / log_sSFR")
    print("=" * 60)
    log_sfr_series, sfr_already_log = to_log_sfr(master["SFR_raw"])
    if sfr_already_log:
        master["log_SFR"] = master["SFR_raw"]
        master["SFR"] = 10 ** master["SFR_raw"]
    else:
        master["SFR"] = master["SFR_raw"]
        master["log_SFR"] = log_sfr_series
    master = master.drop(columns=["SFR_raw"])

    _, mstar_already_log = to_log_sfr(master["Mstar"])
    if mstar_already_log:
        master["log_Mstar"] = master["Mstar"]
        mstar_linear = 10 ** master["Mstar"]
    else:
        master["log_Mstar"] = np.log10(master["Mstar"].replace(0, np.nan))
        mstar_linear = master["Mstar"]

    master["sSFR"] = master["SFR"] / mstar_linear
    with np.errstate(invalid="ignore", divide="ignore"):
        master["log_sSFR"] = np.log10(master["sSFR"].replace(0, np.nan))

    if "MassA_1" in master.columns:
        with np.errstate(invalid="ignore", divide="ignore"):
            master["log_MassA_1"] = np.log10(master["MassA_1"].replace(0, np.nan))

    print("\n" + "=" * 60)
    print("STEP 9 - Joining GAMA morphology + pairs (kept - environment metadata, not an SFR/Mstar source)")
    print("=" * 60)

    if GAMA_MORPH_FITS.exists():
        gama_morph = load_fits_as_df(GAMA_MORPH_FITS, "GAMA Morphology")
        cataid_col = next((c for c in gama_morph.columns if "CATAID" in c or "CATID" in c), None)
        gama_morph = gama_morph.rename(columns={cataid_col: "CATID"})
        gama_morph["CATID"] = gama_morph["CATID"].astype(str).str.strip().astype(int)
        morph_slim = gama_morph[["CATID"] + [c for c in ["HUBBLE_TYPE", "FINAL_CLASS"] if c in gama_morph.columns]]
        morph_slim = morph_slim.rename(columns={"HUBBLE_TYPE": "gama_hubble_type", "FINAL_CLASS": "gama_final_class"})
        master = master.merge(morph_slim, on="CATID", how="left")
        print(f"  GAMA Morphology matched : {master['gama_hubble_type'].notna().sum() if 'gama_hubble_type' in master.columns else 0}")
    else:
        print(f"  WARNING: {GAMA_MORPH_FITS} not found - skipping")
        master["gama_hubble_type"] = np.nan
        master["gama_final_class"] = np.nan

    if GAMA_PAIRS_FITS.exists():
        gama_pairs = load_fits_as_df(GAMA_PAIRS_FITS, "GAMA Pairs")
        pairs_as_1 = gama_pairs[["CATAID1", "CATAID2", "SEPPROJANG", "SEPVEL"]].rename(columns={"CATAID1": "CATID", "CATAID2": "PARTNER"})
        pairs_as_2 = gama_pairs[["CATAID1", "CATAID2", "SEPPROJANG", "SEPVEL"]].rename(columns={"CATAID2": "CATID", "CATAID1": "PARTNER"})
        all_pairs = pd.concat([pairs_as_1, pairs_as_2], ignore_index=True)
        all_pairs["CATID"] = all_pairs["CATID"].astype(str).str.strip().astype(int)
        all_pairs["ABS_VEL"] = all_pairs["SEPVEL"].abs()

        confirmed = all_pairs[all_pairs["ABS_VEL"] <= PAIR_VEL_CUT_KMS].copy()
        z_lookup = master.set_index("CATID")["z"].dropna().to_dict()
        confirmed["z_gal"] = confirmed["CATID"].map(z_lookup)
        confirmed["SEP_KPC"] = arcsec_to_kpc(confirmed["SEPPROJANG"], confirmed["z_gal"])

        nearest = confirmed.sort_values("SEP_KPC").groupby("CATID", as_index=False).first()[["CATID", "SEP_KPC"]]
        nearest = nearest.rename(columns={"SEP_KPC": "gama_pair_sep_kpc"})
        master = master.merge(nearest, on="CATID", how="left")
        master["gama_in_pairs"] = master["gama_pair_sep_kpc"].notna()
        master["gama_close_pair_100kpc"] = master["gama_pair_sep_kpc"] <= PAIR_SEP_100_KPC
        master["gama_close_pair_50kpc"] = master["gama_pair_sep_kpc"] <= PAIR_SEP_50_KPC
        print(f"  In GAMA pairs        : {master['gama_in_pairs'].sum()}")
        print(f"  Close pair < 100 kpc : {master['gama_close_pair_100kpc'].sum()}")
        print(f"  Close pair <  50 kpc : {master['gama_close_pair_50kpc'].sum()}")
    else:
        print(f"  WARNING: {GAMA_PAIRS_FITS} not found - skipping")
        master["gama_pair_sep_kpc"] = np.nan
        master["gama_in_pairs"] = False
        master["gama_close_pair_100kpc"] = False
        master["gama_close_pair_50kpc"] = False

    print("\n" + "=" * 60)
    print("STEP 10 - Joining 5NN surface density (data/fits/surface_density.fits)")
    print("=" * 60)

    if SURFACE_DENSITY_FITS.exists():
        sd = load_fits_as_df(SURFACE_DENSITY_FITS, "Surface Density")
        sd_slim = sd[["CATID", "SURFACEDENSITY", "SURFACEDENSITY_ERR", "SURFACEDENSITYFLAG"]].rename(
            columns={"SURFACEDENSITY": "5NN", "SURFACEDENSITY_ERR": "5NN_err", "SURFACEDENSITYFLAG": "5NN_flag"}
        )
        master = master.merge(sd_slim, on="CATID", how="left")
        print(f"  5NN matched : {master['5NN'].notna().sum()}")
    else:
        print(f"  WARNING: {SURFACE_DENSITY_FITS} not found - skipping")
        master["5NN"] = np.nan
        master["5NN_err"] = np.nan
        master["5NN_flag"] = np.nan

    print("\n" + "=" * 60)
    print("STEP 11 - Deriving group_status (central/satellite/isolated) from BCGCATAID")
    print("=" * 60)
    master["group_status"] = classify_group_status(master["BCGCATAID"], master["CATID"])
    print(master["group_status"].value_counts(dropna=False).to_string())

    print("\n" + "=" * 60)
    print("STEP 12 - Dataset tags + final column order")
    print("=" * 60)
    master["dataset_tag"] = master.apply(make_dataset_tag, axis=1)
    print(master["dataset_tag"].value_counts().to_string())

    final_cols = [
        "CATID", "dataset_tag", "in_C", "in_G", "in_A", "in_S", "n_metrics",
        "C", "G", "A", "S",
        "spectral_class", "environment", "galclass", "galclass_exclude",
        "re", "re_com", "ellip", "z",
        "Mstar", "log_Mstar", "SFR", "log_SFR", "sSFR", "log_sSFR", "sfr_flag",
        "seeing", "seeing_re", "source_file", "conflict_flag",
        "BCGCATAID", "MassA_1", "log_MassA_1", "group_status",
        "gama_hubble_type", "gama_final_class",
        "gama_in_pairs", "gama_pair_sep_kpc", "gama_close_pair_100kpc", "gama_close_pair_50kpc",
        "5NN", "5NN_err", "5NN_flag",
        *CUT_COLUMNS, "in_full_sam",
    ]
    final_cols_present = [c for c in final_cols if c in master.columns]
    missing = [c for c in final_cols if c not in master.columns]
    if missing:
        print(f"  NOTE: expected columns absent (omitted): {missing}")

    master_out = master[final_cols_present]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    master_out.to_csv(OUT_CSV, index=False)

    print("\n" + "=" * 60)
    print("MASTER CSV BUILD COMPLETE")
    print("=" * 60)
    print(f"  Total galaxies         : {len(master_out)}")
    print(f"  All 3 morph metrics    : {master_out[['G','A','S']].notna().all(axis=1).sum()}")
    print(f"  C present              : {master_out['C'].notna().sum()}")
    print(f"  SFR present            : {master_out['SFR'].notna().sum()}")
    print(f"  in_full_sam            : {int(master_out['in_full_sam'].sum())}  (Wang 2022 Full_sam quality cuts, see wang_quality_cuts.py)")
    print(f"  5NN present            : {master_out['5NN'].notna().sum()}")
    print(f"  group_status present   : {master_out['group_status'].notna().sum()}  ({master_out['group_status'].value_counts(dropna=False).to_dict()})")
    print(f"  Written                : {OUT_CSV}")


if __name__ == "__main__":
    main()
