"""
create_sf_ml_csv.py
--------------------
Reads master_results.csv and produces star_formation_ml.csv:
  - Pure SF galaxies only
  - Cleaned and encoded columns ready for Random Forest pipeline
  - No metric-specific filtering (in_G/C/A/S flags travel with the data
    and are applied at runtime per model)

Usage:
    python create_sf_ml_csv.py
"""

import pandas as pd
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
INPUT_PATH  = "master/outputs/tables/master_results.csv"
OUTPUT_PATH = "master/outputs/tables/star_formation_ml.csv"

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH)
print(f"Loaded master_results.csv  →  {len(df):,} galaxies, {df.shape[1]} columns")

# ── 1. Filter to pure SF ──────────────────────────────────────────────────────
df = df[df["spectral_class"] == "pure_SF"].copy()
print(f"After pure_SF filter       →  {len(df):,} galaxies")

# ── 2. Target metrics: replace C sentinel (-999) with NaN ────────────────────
n_sentinel = (df["C"] == -999).sum()
df["C"] = df["C"].replace(-999, np.nan)
print(f"C sentinel (-999) replaced →  {n_sentinel} values set to NaN")

# ── 3. Encode environment as binary (GAMA=0, Cluster=1) ─────────────────────
#    NaN environment left as NaN (no GAMA/Cluster assignment)
df["environment_enc"] = df["environment"].map({"GAMA": 0, "Cluster": 1})
n_env_nan = df["environment_enc"].isna().sum()
print(f"environment encoded        →  GAMA=0, Cluster=1  ({n_env_nan} NaN)")

# ── 4. Encode boolean columns as integers (True=1, False=0) ──────────────────
#    Handles both Python bool dtype and string 'True'/'False' (pandas reads
#    these as strings on some platforms/versions when the CSV was written with
#    Python bools that got serialised as text).
bool_cols = ["is_isolated", "gama_close_pair_50kpc"]
bool_map = {True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}
for col in bool_cols:
    df[col] = df[col].map(bool_map).astype(int)
print(f"Boolean → int encoded      →  {bool_cols}")

# ── 5. Derive is_central from BCGCATAID ──────────────────────────────────────
#    is_central = 1  if CATID == BCGCATAID  (galaxy is its own group's BCG)
#    is_central = 0  if CATID != BCGCATAID  (satellite)
#    is_central = NaN if BCGCATAID is NaN   (Cluster galaxies not in GAMA GGC)
df["is_central"] = np.where(
    df["BCGCATAID"].isna(),
    np.nan,
    (df["CATID"] == df["BCGCATAID"]).astype(float)
)
n_central  = (df["is_central"] == 1).sum()
n_satellite = (df["is_central"] == 0).sum()
n_cent_nan = df["is_central"].isna().sum()
print(f"is_central derived         →  {n_central} centrals, {n_satellite} satellites, {n_cent_nan} NaN (unmatched cluster)")

# ── 6. Select and order final columns ────────────────────────────────────────
#
#   Identifier
#     CATID
#
#   Target metrics
#     G, C, A, S
#
#   Metric availability flags (applied per-model at runtime)
#     in_G, in_C, in_A, in_S
#
#   Predictor features
#     log_Mstar         — stellar mass
#     log_sSFR          — specific star formation rate
#     5NN               — 5th nearest neighbour surface density (environment)
#     environment_enc   — GAMA(0) / Cluster(1)
#     is_isolated       — binary
#     gama_close_pair_50kpc — binary interaction flag
#     log_MassA_1       — halo mass (NaN for ~66%; used in Model B only)
#     is_central        — central(1) / satellite(0) / NaN
#
#   Quality
#     seeing_re         — PSF/half-light ratio (for optional seeing cut)

FINAL_COLS = [
    # identifier
    "CATID",
    # targets
    "G", "C", "A", "S",
    # availability flags
    "in_G", "in_C", "in_A", "in_S",
    # predictors — continuous
    "log_Mstar", "log_sSFR", "5NN", "log_MassA_1",
    # predictors — categorical/binary
    "environment_enc", "is_isolated", "gama_close_pair_50kpc", "is_central",
    # quality
    "seeing_re",
]

ml = df[FINAL_COLS].copy()

# ── 7. Rename for clarity ─────────────────────────────────────────────────────
ml = ml.rename(columns={"environment_enc": "environment"})

# ── 8. Report final state ────────────────────────────────────────────────────
print(f"\nFinal ML dataframe         →  {len(ml):,} galaxies, {ml.shape[1]} columns")
print("\nMissing values per column:")
missing = ml.isnull().sum()
missing = missing[missing > 0]
for col, n in missing.items():
    print(f"  {col:<30} {n:>4}  ({100*n/len(ml):.1f}%)")

print("\nMetric sample sizes (in_M == True, no other cuts):")
for m in ["G", "C", "A", "S"]:
    n = ml[f"in_{m}"].sum()
    n_valid = ml[ml[f"in_{m}"] == True][m].notna().sum()
    print(f"  {m}: in_flag={n}, non-NaN target={n_valid}")

# ── 9. Save ───────────────────────────────────────────────────────────────────
ml.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved → {OUTPUT_PATH}")