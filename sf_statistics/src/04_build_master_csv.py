#!/usr/bin/env python3
"""
Builds the master catalogue, one row per galaxy, by merging:
    - Concentration (C) + Wang metadata (Mstar, re, ellip, z, environment,
      galclass, BCGCATAID, MassA_1, ...)      <- outputs/ch3/csv/concentration.csv
    - Gini (G)                                <- outputs/ch3/csv/gini.csv
    - Asymmetry (A)                           <- outputs/ch3/csv/asymmetry.csv
    - Clumpiness (S)                          <- outputs/ch3/csv/clumpiness.csv

and adding:
    - spectral_class (pure_SF / composite_SF) from Wang's catalogue membership
    - SFR, sSFR and their log10 values
    - Wang et al. (2022) Full_sam quality cuts: one cut_* column per cut,
      plus in_full_sam (passes all six)
    - r_e in kpc (re_kpc) and r_e / seeing (n_res)
    - GAMA environment:
        gama_hubble_type, gama_final_class  <- data/fits/gama_morphology.fits
        gama_pair_sep_kpc                   <- data/fits/gama_pairs.fits
        5NN, 5NN_err, 5NN_flag              <- data/fits/surface_density.fits
        group_status                        <- Wang's BCGCATAID

Wang membership determines the SFR source:
    pure_SF      (in an SF file, not in any Composite file) -> Wang's SFR
    composite_SF (in both)                                   -> SFR_n2ha_s
    composite_SF with no valid SFR_n2ha_s                    -> NaN

Wang's catalogue stores Mstar and SFR as log10 values.

gama_pair_sep_kpc is the projected separation to the nearest GAMA pair
companion within |dv| <= 500 km/s, converted from the catalogue's kpc/h to
kpc with h = 0.7. NaN means no such companion.

group_status: BCGCATAID == CATID -> central, BCGCATAID == 0 -> isolated,
any other BCGCATAID -> satellite. GAMA only (Cluster galaxies have no
BCGCATAID), so Cluster galaxies get NaN.

Output: outputs/pipeline/master.csv (overwritten on each run)

Usage
    python src/04_build_master_csv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from src.utils.paths import (  
    ASYMMETRY_CSV,
    CLUMPINESS_CSV,
    CONCENTRATION_CSV,
    GAMA_MORPH_FITS,
    GAMA_PAIRS_FITS,
    GINI_CSV,
    MASTER_CSV,
    SURFACE_DENSITY_FITS,
)
from src.utils.wang import load_all_wang_files, wang_membership_sets  
from src.utils.wang_quality_cuts import CUT_COLUMNS, apply_quality_cuts 

OUT_CSV = MASTER_CSV

COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
PAIR_VEL_CUT_KMS = 500.0  # max |velocity separation| for a pair companion


def load_fits_as_df(path: Path, label: str) -> pd.DataFrame:
    """Read HDU 1 of a FITS table into a DataFrame with upper-case column names."""
    if not path.exists():
        raise FileNotFoundError(f"{label} catalogue not found: {path}")
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


def classify_group_status(bcgcataid: pd.Series, catid: pd.Series) -> pd.Series:
    """BCGCATAID == CATID -> central; BCGCATAID == 0 -> isolated; any other
    non-null value -> satellite; no BCGCATAID -> NaN.

    Built as an object Series so missing values stay NaN rather than the text 'nan'."""
    status = pd.Series(np.nan, index=catid.index, dtype=object)
    status[bcgcataid.notna()] = "satellite"
    status[bcgcataid == 0] = "isolated"
    status[bcgcataid == catid] = "central"
    return status


def main():
    print("=" * 60)
    print("STEP 1 - Loading metric tables")
    print("=" * 60)
    conc = pd.read_csv(CONCENTRATION_CSV)
    gini = pd.read_csv(GINI_CSV)
    asym = pd.read_csv(ASYMMETRY_CSV)
    clump = pd.read_csv(CLUMPINESS_CSV)
    for label, df in [("concentration", conc), ("gini", gini), ("asymmetry", asym), ("clumpiness", clump)]:
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
    print("STEP 4 - Building CATID list (union of C, G, A, S tables)")
    print("=" * 60)
    all_catids = pd.DataFrame({
        "CATID": pd.concat([conc["CATID"], gini["CATID"], asym["CATID"], clump["CATID"]])
        .drop_duplicates()
        .reset_index(drop=True)
    })
    print(f"  Total unique CATIDs : {len(all_catids)}")

    print("\n" + "=" * 60)
    print("STEP 5 - Merging metric tables")
    print("=" * 60)
    master = all_catids.copy()

    conc_cols = [
        "CATID", "CIndex", "re", "re_com", "ellip", "z", "Mstar", "galclass",
        "galclass_exclude", "environment", "seeing", "seeing_re",
        "SFR", "SFR_n2ha_s", "BCGCATAID", "MassA_1",
        *CUT_COLUMNS, "in_full_sam",
    ]
    master = master.merge(conc[conc_cols], on="CATID", how="left")
    master["in_C"] = master["CATID"].isin(set(conc["CATID"]))
    master = master.rename(columns={"CIndex": "C"})

    # No concentration entry -> never evaluated against Wang's cuts -> not in Full_sam.
    # Individual cut_* columns stay NaN for these rows ("not evaluated", not "failed").
    master["in_full_sam"] = master["in_full_sam"].fillna(False).astype(bool)

    master = master.merge(gini[["CATID", "gini_halpha"]], on="CATID", how="left")
    master = master.rename(columns={"gini_halpha": "G"})
    master = master.merge(asym[["CATID", "A"]], on="CATID", how="left")

    # re_used/re_source: the r_e clumpiness.py used to size its kernel (SF value,
    # composite fallback, r_e > 50" excluded). Kept alongside the catalogue re/re_com.
    master = master.merge(clump[["CATID", "S", "re_used", "re_source"]], on="CATID", how="left")

    # re_kpc: physical size; n_res: r_e in units of the seeing (resolution).
    kpc_per_arcsec = COSMO.kpc_proper_per_arcmin(master["z"].to_numpy(dtype=float)).value / 60.0
    master["re_kpc"] = master["re_used"] * kpc_per_arcsec
    master["n_res"] = master["re_used"] / master["seeing"]

    master["in_G"] = master["CATID"].isin(set(gini["CATID"]))
    master["in_A"] = master["CATID"].isin(set(asym["CATID"]))
    master["in_S"] = master["CATID"].isin(set(clump["CATID"]))

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
    print("STEP 7 - Resolving SFR from Wang's catalogue")
    print("=" * 60)

    log_sfr, sfr_flags = [], []
    for sc, sf_val, n2ha_val in zip(master["spectral_class"], master["SFR"], master["SFR_n2ha_s"]):
        if sc == "pure_SF":
            value, flag = (sf_val, "sf_file") if pd.notna(sf_val) else (np.nan, np.nan)
        elif sc == "composite_SF":
            value, flag = (n2ha_val, "n2ha_s") if pd.notna(n2ha_val) else (np.nan, "nan_no_n2ha")
        else:
            value, flag = np.nan, np.nan
        log_sfr.append(value)
        sfr_flags.append(flag)

    master = master.drop(columns=["SFR", "SFR_n2ha_s"])
    master["log_SFR"] = log_sfr
    master["sfr_flag"] = sfr_flags
    print(pd.Series(sfr_flags).value_counts(dropna=False).to_string())

    print("\n" + "=" * 60)
    print("STEP 8 - Computing SFR / log_Mstar / sSFR / log_sSFR")
    print("=" * 60)
    # Wang's Mstar and SFR are already log10.
    master["SFR"] = 10 ** master["log_SFR"]
    master["log_Mstar"] = master["Mstar"]
    master["sSFR"] = master["SFR"] / 10 ** master["log_Mstar"]
    with np.errstate(invalid="ignore", divide="ignore"):
        master["log_sSFR"] = np.log10(master["sSFR"].replace(0, np.nan))
        master["log_MassA_1"] = np.log10(master["MassA_1"].replace(0, np.nan))

    print("\n" + "=" * 60)
    print("STEP 9 - Joining GAMA morphology and nearest pair separation")
    print("=" * 60)

    gama_morph = load_fits_as_df(GAMA_MORPH_FITS, "GAMA Morphology")
    cataid_col = next(c for c in gama_morph.columns if "CATAID" in c or "CATID" in c)
    gama_morph = gama_morph.rename(columns={cataid_col: "CATID"})
    gama_morph["CATID"] = gama_morph["CATID"].astype(str).str.strip().astype(int)
    morph_slim = gama_morph[["CATID", "HUBBLE_TYPE", "FINAL_CLASS"]].rename(
        columns={"HUBBLE_TYPE": "gama_hubble_type", "FINAL_CLASS": "gama_final_class"}
    )
    master = master.merge(morph_slim, on="CATID", how="left")
    print(f"  GAMA Morphology matched : {master['gama_hubble_type'].notna().sum()}")

    # Each pair lists two galaxies; stack both orientations so every galaxy
    # sees all of its companions. SEPPROJANG is a physical separation in kpc/h.
    gama_pairs = load_fits_as_df(GAMA_PAIRS_FITS, "GAMA Pairs")
    pair_cols = ["CATAID1", "CATAID2", "SEPPROJANG", "SEPVEL"]
    pairs_as_1 = gama_pairs[pair_cols].rename(columns={"CATAID1": "CATID", "CATAID2": "PARTNER"})
    pairs_as_2 = gama_pairs[pair_cols].rename(columns={"CATAID2": "CATID", "CATAID1": "PARTNER"})
    all_pairs = pd.concat([pairs_as_1, pairs_as_2], ignore_index=True)
    all_pairs["CATID"] = all_pairs["CATID"].astype(str).str.strip().astype(int)

    confirmed = all_pairs[all_pairs["SEPVEL"].abs() <= PAIR_VEL_CUT_KMS].copy()
    confirmed["gama_pair_sep_kpc"] = confirmed["SEPPROJANG"] / COSMO.h
    nearest = confirmed.groupby("CATID", as_index=False)["gama_pair_sep_kpc"].min()
    master = master.merge(nearest, on="CATID", how="left")
    print(f"  Galaxies with a pair companion : {master['gama_pair_sep_kpc'].notna().sum()}")

    print("\n" + "=" * 60)
    print("STEP 10 - Joining 5NN surface density")
    print("=" * 60)
    sd = load_fits_as_df(SURFACE_DENSITY_FITS, "Surface Density")
    sd_slim = sd[["CATID", "SURFACEDENSITY", "SURFACEDENSITY_ERR", "SURFACEDENSITYFLAG"]].rename(
        columns={"SURFACEDENSITY": "5NN", "SURFACEDENSITY_ERR": "5NN_err", "SURFACEDENSITYFLAG": "5NN_flag"}
    )
    master = master.merge(sd_slim, on="CATID", how="left")
    print(f"  5NN matched : {master['5NN'].notna().sum()}")

    print("\n" + "=" * 60)
    print("STEP 11 - Deriving group_status (central/satellite/isolated) from BCGCATAID")
    print("=" * 60)
    master["group_status"] = classify_group_status(master["BCGCATAID"], master["CATID"])
    print(master["group_status"].value_counts(dropna=False).to_string())

    final_cols = [
        "CATID", "in_C", "in_G", "in_A", "in_S",
        "C", "G", "A", "S",
        "spectral_class", "environment", "galclass", "galclass_exclude",
        "re", "re_com", "re_used", "re_source", "re_kpc", "n_res", "ellip", "z",
        "Mstar", "log_Mstar", "SFR", "log_SFR", "sSFR", "log_sSFR", "sfr_flag",
        "seeing", "seeing_re",
        "BCGCATAID", "MassA_1", "log_MassA_1", "group_status",
        "gama_hubble_type", "gama_final_class", "gama_pair_sep_kpc",
        "5NN", "5NN_err", "5NN_flag",
        *CUT_COLUMNS, "in_full_sam",
    ]
    master_out = master[final_cols]

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
