"""
create_sf_ml_csv.py
--------------------
Reads master_results.csv and gama_pairs.fits (both in the same folder)
and produces star_formation_ml.csv.

Changes from previous version:
  - Includes both pure_SF and composite_SF galaxies (not pure_SF only)
  - Replaces gama_close_pair_50kpc / gama_close_pair_100kpc binary flags
    with a continuous closest_companion_kpc column derived from gama_pairs.fits
  - Adds ellip (ellipticity), z (redshift), re (half-light radius, arcsec)

Cosmology used for arcsec -> kpc conversion: FlatLambdaCDM, H0=70, Om0=0.3

Usage (run from the folder containing master_results.csv and gama_pairs.fits):
    python create_sf_ml_csv.py
"""

import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM

# ── Paths ─────────────────────────────────────────────────────────────────────
MASTER_PATH = "master/outputs/tables/master_results.csv"
PAIRS_PATH  = "data/fits/gama_pairs.fits"
OUTPUT_PATH = "master/outputs/tables/star_formation_ml_02.csv"

# ── Cosmology ─────────────────────────────────────────────────────────────────
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# ── 1. Load master catalogue ──────────────────────────────────────────────────
df = pd.read_csv(MASTER_PATH)
print(f"Loaded {MASTER_PATH}  ->  {len(df):,} galaxies, {df.shape[1]} columns")

# ── 2. Filter to pure_SF and composite_SF ─────────────────────────────────────
df = df[df["spectral_class"].isin(["pure_SF", "composite_SF"])].copy()
print(f"After spectral class filter        ->  {len(df):,} galaxies")
print(f"  pure_SF:      {(df['spectral_class'] == 'pure_SF').sum():,}")
print(f"  composite_SF: {(df['spectral_class'] == 'composite_SF').sum():,}")

# ── 3. Clean target metrics ───────────────────────────────────────────────────
# 3a. Replace C sentinel (-999) with NaN
n_sentinel = (df["C"] == -999).sum()
df["C"] = df["C"].replace(-999, np.nan)
print(f"C sentinel (-999) replaced         ->  {n_sentinel} values set to NaN")

# 3b. Replace inf/-inf in all metric columns with NaN
for col in ["G", "C", "A", "S"]:
    n_inf = np.isinf(df[col].fillna(0)).sum()
    if n_inf > 0:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        print(f"Inf values in {col} replaced       ->  {n_inf} values set to NaN")

# ── 3c. Cap re outliers — likely failed Sérsic fits ──────────────────────────
re_lo, re_hi = 0.5, 30.0
n_re_bad = ((df['re'] < re_lo) | (df['re'] > re_hi)).sum()
df.loc[(df['re'] < re_lo) | (df['re'] > re_hi), 're'] = np.nan
print(f"re outliers (< {re_lo} or > {re_hi} arcsec) -> {n_re_bad} values set to NaN")

# ── 4. Encode environment as binary (GAMA=0, Cluster=1) ──────────────────────
df["environment_enc"] = df["environment"].map({"GAMA": 0, "Cluster": 1})
n_env_nan = df["environment_enc"].isna().sum()
print(f"environment encoded                ->  GAMA=0, Cluster=1  ({n_env_nan} NaN)")

# ── 5. Encode spectral_class (pure_SF=0, composite_SF=1) ─────────────────────
# Drop the original string column and replace with integer encoding in-place
df["spectral_class"] = df["spectral_class"].map({"pure_SF": 0, "composite_SF": 1})
print(f"spectral_class encoded             ->  pure_SF=0, composite_SF=1")

# ── 6. Encode boolean columns as int 0/1 ─────────────────────────────────────
bool_map = {True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}
for col in ["is_isolated"]:
    df[col] = df[col].map(bool_map).astype("Int64")
print(f"Boolean -> int encoded             ->  ['is_isolated']")

# ── 7. Derive is_central from BCGCATAID ──────────────────────────────────────
df["is_central"] = np.where(
    df["BCGCATAID"].isna(),
    np.nan,
    (df["CATID"] == df["BCGCATAID"]).astype(float)
)
n_central   = (df["is_central"] == 1).sum()
n_satellite = (df["is_central"] == 0).sum()
n_cent_nan  = df["is_central"].isna().sum()
print(f"is_central derived                 ->  {n_central} centrals, "
      f"{n_satellite} satellites, {n_cent_nan} NaN")

# ── 8. Build closest_companion_kpc from gama_pairs.fits ──────────────────────
print(f"\nLoading {PAIRS_PATH} ...")
hdul = fits.open(PAIRS_PATH)
pdata = hdul[1].data

# Extract columns, converting FITS big-endian to native endian for pandas
cat1    = np.array(pdata["CATAID1"],    dtype=np.int64)
cat2    = np.array(pdata["CATAID2"],    dtype=np.int64)
sep_ang = np.array(pdata["SepProjAng"], dtype=np.float64)  # arcsec
zbar    = np.array(pdata["Zbar"],       dtype=np.float64)   # mean redshift of pair
hdul.close()

print(f"  Loaded {len(cat1):,} pair entries")

# Convert angular separation (arcsec) -> projected physical separation (kpc)
# using the mean redshift of each pair and a flat LCDM cosmology
print("  Converting SepProjAng (arcsec) to kpc using FlatLambdaCDM (H0=70, Om0=0.3) ...")
kpc_per_arcsec = np.array(
    [cosmo.kpc_proper_per_arcmin(float(z)).value / 60.0 for z in zbar]
)
sep_kpc = sep_ang * kpc_per_arcsec

# Build a per-galaxy lookup: every master galaxy that appears as CATAID1 or CATAID2
pairs_df = pd.DataFrame({"CATAID1": cat1, "CATAID2": cat2, "sep_kpc": sep_kpc})
master_ids = set(df["CATID"].tolist())

as1 = (pairs_df[pairs_df["CATAID1"].isin(master_ids)]
       [["CATAID1", "sep_kpc"]]
       .rename(columns={"CATAID1": "CATID"}))
as2 = (pairs_df[pairs_df["CATAID2"].isin(master_ids)]
       [["CATAID2", "sep_kpc"]]
       .rename(columns={"CATAID2": "CATID"}))
all_pairs = pd.concat([as1, as2], ignore_index=True)

# For each galaxy, keep only the closest companion (minimum sep_kpc)
closest = (all_pairs.groupby("CATID")["sep_kpc"]
           .min()
           .reset_index()
           .rename(columns={"sep_kpc": "closest_companion_kpc"}))

df = df.merge(closest, on="CATID", how="left")

n_with_pair    = df["closest_companion_kpc"].notna().sum()
n_without_pair = df["closest_companion_kpc"].isna().sum()
print(f"  closest_companion_kpc added      ->  {n_with_pair:,} galaxies with a pair, "
      f"{n_without_pair:,} NaN (no pair entry)")
print(f"  Distance range: {df['closest_companion_kpc'].min():.2f} - "
      f"{df['closest_companion_kpc'].max():.2f} kpc")

# ── 9. Select and order final columns ────────────────────────────────────────
FINAL_COLS = [
    # identifier
    "CATID",

    # targets
    "G", "C", "A", "S",

    # metric availability flags (applied per-model at runtime)
    "in_G", "in_C", "in_A", "in_S",

    # predictors — continuous
    "log_Mstar",
    "log_sSFR",
    "5NN",
    "log_MassA_1",           # halo mass (NaN for many; Model B only)
    "closest_companion_kpc", # continuous pair separation (NaN = no pair in catalogue)

    # predictors — categorical / binary
    "environment_enc",       # GAMA=0, Cluster=1
    "spectral_class",    # pure_SF=0, composite_SF=1
    "is_isolated",
    "is_central",

    # structural properties
    "ellip",                 # ellipticity
    "z",                     # spectroscopic redshift
    "re",                    # half-light radius (arcsec)

    # quality
    "seeing_re",             # PSF FWHM / re (resolution quality metric)
]

ml = df[FINAL_COLS].copy()
ml = ml.rename(columns={"environment_enc": "environment",
                         "spectral_class_enc": "spectral_class"})

# ── 10. Report ────────────────────────────────────────────────────────────────
print(f"\nFinal ML dataframe             ->  {len(ml):,} galaxies, {ml.shape[1]} columns")
print("\nMissing values per column:")
missing = ml.isnull().sum()
missing = missing[missing > 0]
for col, n in missing.items():
    print(f"  {col:<30} {n:>5}  ({100*n/len(ml):.1f}%)")

print("\nMetric sample sizes (in_M == True, no other cuts):")
for m in ["G", "C", "A", "S"]:
    n_flag  = ml[f"in_{m}"].sum()
    n_valid = ml[ml[f"in_{m}"] == True][m].notna().sum()
    print(f"  {m}: in_flag={n_flag:,}, non-NaN target={n_valid:,}")

print("\nspectral_class distribution in ML CSV:")
print(f"  pure_SF (0):      {(ml['spectral_class'] == 0).sum():,}")
print(f"  composite_SF (1): {(ml['spectral_class'] == 1).sum():,}")

# ── 11. Save ──────────────────────────────────────────────────────────────────
ml.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved -> {OUTPUT_PATH}")