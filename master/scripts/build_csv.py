"""
Builds the master summary CSV collating results from Gini, Concentration, Asymmetry, 
Smoothness for SAMI, enriched with GAMA environmental data and per-galaxy dataset tags.

Dataset Priority Chain
----------------------
1. Wang Concentration catalogue  → primary anchor for shared metadata
2. AR2022 SAMI DR3 catalogue     → SFR/Mstar best estimates; fallback metadata
3. Gini / Asymmetry / Smoothness → morphology metrics only (no metadata priority)

Dataset Tag Logic
-----------------
dataset_tag (string, human-readable summary):
    'wang_conc'        : in Wang concentration catalogue only
    'ar2022_only'      : in AR2022 only (no morphology metrics at all)
    'morph_only'       : has ≥1 morphology metric but NOT in Wang or AR2022
    'wang+ar2022'      : in both Wang and AR2022
    'multi_source'     : in Wang + AR2022 + ≥1 other morphology catalogue
    'full'             : all four morphology metrics present

Boolean provenance flags (one per source):
    in_wang_conc, in_ar2022, in_gini, in_asym, in_smooth

n_metrics (int 0-4): count of morphology metrics present

GAMA Environmental Columns Added
---------------------------------
From GAMA Pairs (nearest confirmed pair):
    gama_in_pairs           : does it appear at all in gama_pairs.fits
    gama_close_pair_100kpc  : does it have a confirmed close pair within 100 kpc
    gama_close_pair_50kpc   : does it have a confirmed close pair within 50 kpc

From GAMA Morphology:
    gama_hubble_type        : numerical Hubble T-type (HUBBLE_TYPE)
    gama_final_class        : visual morphological class string (FINAL_CLASS)

Columns in Final CSV
--------------------
CATID,
dataset_tag, in_wang_conc, in_ar2022, in_gini, in_asym, in_smooth, n_metrics,
gini_all_halpha, CIndex, A_min, S,
spectral_class, environment, galclass, re, ellip, z, g_i,
Mstar, SFR, log_SFR, sSFR, log_sSFR,
seeing, seeing_re, source_file, conflict_flag,
gama_hubble_type, gama_final_class,
gama_in_pairs, gama_close_pair_100kpc, gama_close_pair_50kpc
"""

import os
import warnings
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM

warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# CONFIG — update paths as needed
# =============================================================================

BASE_DIR     = '/Users/sophiesinclair/Desktop/sami'

GINI_CSV     = os.path.join(BASE_DIR, 'gini/gini_outputs/tables/compare_gini.csv')
CONC_CSV     = os.path.join(BASE_DIR, 'concentration/outputs/tables/concentration_results.csv')
ASYM_CSV     = os.path.join(BASE_DIR, 'asymmetry/outputs/tables/asymmetry_results.csv')
SMOOTH_CSV   = os.path.join(BASE_DIR, 'smoothness/outputs/Tables/smoothness_results.csv')

AR2022_FITS  = os.path.join(BASE_DIR, 'data/catalogs/AR2022_SAMI_DR3_Mstar_SFR_GSWL_rescaled.fits')
GAMA_PAIRS_FITS = os.path.join(BASE_DIR, 'data/fits/gama_pairs.fits')       
GAMA_MORPH_FITS = os.path.join(BASE_DIR, 'data/fits/gama_morphology.fits') 

OUT_DIR      = os.path.join(BASE_DIR, 'master')
OUT_PATH     = os.path.join(OUT_DIR, 'outputs/tables/master_results.csv')

os.makedirs(OUT_DIR, exist_ok=True)

# Cosmology for arcsec → kpc conversion
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

# Pair velocity cut (km/s) — standard for interaction studies
PAIR_VEL_CUT_KMS = 500.0

# Pair projected separation cuts (arcsec — converted to kpc per galaxy redshift)
# Store raw arcsec separation and compute kpc at merge time using galaxy z.
# These kpc thresholds are applied to pre-computed kpc separations in GAMA.
# If SepProjAng is already in kpc in your FITS, set GAMA_SEP_IN_KPC = True.
GAMA_SEP_IN_KPC  = False   # set True if SepProjAng column is already kpc
PAIR_SEP_100_KPC = 100.0   # kpc threshold for close-pair count
PAIR_SEP_50_KPC  =  50.0   # kpc threshold for very-close-pair count

# Shared metadata columns — sourced from Wang concentration (priority 1)
SHARED_COLS = [
    'spectral_class', 'environment', 'galclass',
    're', 'ellip', 'z', 'g_i',
    'seeing', 'seeing_re', 'source_file', 'conflict_flag',
]

# =============================================================================
# HELPERS
# =============================================================================

def load_csv(path, catid_col, label=''):
    """Load CSV, rename catid_col → CATID, cast to int."""
    df = pd.read_csv(path)
    if catid_col not in df.columns:
        raise KeyError(
            f"[{label}] Expected CATID column '{catid_col}' not found.\n"
            f"  Available: {list(df.columns)}"
        )
    df = df.rename(columns={catid_col: 'CATID'})
    df['CATID'] = df['CATID'].astype(str).str.strip().astype(int)
    print(f"  {len(df):>5} rows  ←  {os.path.basename(path)}  [{label}]")
    return df


def load_fits_as_df(path, label=''):
    """Load FITS binary table into a DataFrame with upper-cased columns."""
    print(f"  Loading FITS: {os.path.basename(path)}  [{label}]")
    with fits.open(path) as hdul:
        df = Table(hdul[1].data).to_pandas()
    df.columns = [c.upper() for c in df.columns]
    # Decode byte strings that FITS sometimes produces
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = df[col].str.decode('utf-8').str.strip()
        except AttributeError:
            pass
    return df


def normalise_catid(df, col):
    """Normalise a CATID column in-place to int."""
    df[col] = df[col].astype(str).str.strip().astype(int)
    return df


def to_log_sfr(series):
    """
    Detect whether SFR values are raw linear or already log10.
    Returns (log_series, is_already_log).
    """
    valid = series.dropna()
    if valid.empty:
        return series, False
    if valid.median() > 100:
        return np.log10(series.replace(0, np.nan)), False   # raw → log
    else:
        return series, True                                  # already log

def arcsec_to_kpc(sep_arcsec, redshift):
    def convert_row(row):
        sep, z = row
        try:
            sep, z = float(sep), float(z)
        except (ValueError, TypeError):
            return np.nan
        if np.isnan(sep) or np.isnan(z) or z <= 0:
            return np.nan
        return COSMO.kpc_proper_per_arcmin(z).value / 60.0 * sep
    return pd.Series(
        list(map(convert_row, zip(sep_arcsec, redshift))),
        index=sep_arcsec.index
    )

def make_dataset_tag(row):
    """Assign a human-readable dataset_tag from boolean provenance flags."""
    wang  = row['in_wang_conc']
    ar    = row['in_ar2022']
    n     = row['n_metrics']

    if n == 4:
        return 'full'
    if wang and ar and n >= 2:
        return 'multi_source'
    if wang and ar:
        return 'wang+ar2022'
    if wang:
        return 'wang_conc'
    if ar and n == 0:
        return 'ar2022_only'
    if n >= 1:
        return 'morph_only'
    return 'ar2022_only'


# =============================================================================
# STEP 1 — LOAD MORPHOLOGY CATALOGUES
# =============================================================================

print('\n' + '='*60)
print('STEP 1 — Loading morphology catalogues')
print('='*60)

gini   = load_csv(GINI_CSV,   'CATID',  'Gini')
conc   = load_csv(CONC_CSV,   'CATID',  'Wang Concentration')
asym   = load_csv(ASYM_CSV,   'catid',  'Asymmetry')
smooth = load_csv(SMOOTH_CSV, 'CATID',  'Smoothness')

# =============================================================================
# STEP 2 — LOAD AR2022 FITS (SFR/Mstar)
# =============================================================================

print('\n' + '='*60)
print('STEP 2 — Loading AR2022 FITS catalogue')
print('='*60)

ar2022 = load_fits_as_df(AR2022_FITS, 'AR2022')
ar2022 = normalise_catid(ar2022, 'CATID')

# Identify best SFR and Mstar columns
sfr_col   = 'SFR_BEST'   if 'SFR_BEST'   in ar2022.columns else \
            next((c for c in ar2022.columns if 'SFR' in c), None)
mstar_col = 'MSTAR_BEST' if 'MSTAR_BEST' in ar2022.columns else \
            next((c for c in ar2022.columns if 'MSTAR' in c), None)

if sfr_col is None:
    raise KeyError(f'No SFR column found in AR2022. Columns: {list(ar2022.columns)}')
if mstar_col is None:
    raise KeyError(f'No Mstar column found in AR2022. Columns: {list(ar2022.columns)}')

print(f"  Using SFR column   : {sfr_col}")
print(f"  Using Mstar column : {mstar_col}")

ar2022_slim = ar2022[['CATID', sfr_col, mstar_col]].rename(
    columns={sfr_col: 'SFR_ar2022', mstar_col: 'Mstar_ar2022'}
)

# =============================================================================
# STEP 3 — LOAD GAMA CATALOGUES
# =============================================================================

print('\n' + '='*60)
print('STEP 3 — Loading GAMA catalogues')
print('='*60)

# --- GAMA Morphology ---
try:
    gama_morph = load_fits_as_df(GAMA_MORPH_FITS, 'GAMA Morphology')
    # Standardise CATAID column name
    cataid_col = next((c for c in gama_morph.columns if 'CATAID' in c or 'CATID' in c), None)
    if cataid_col is None:
        raise KeyError(f'No CATAID column found. Columns: {list(gama_morph.columns)}')
    gama_morph = gama_morph.rename(columns={cataid_col: 'CATAID'})
    gama_morph['CATAID'] = gama_morph['CATAID'].astype(str).str.strip().astype(int)

    morph_cols_wanted = ['CATAID', 'HUBBLE_TYPE', 'FINAL_CLASS']
    morph_cols_present = [c for c in morph_cols_wanted if c in gama_morph.columns]
    gama_morph_slim = gama_morph[morph_cols_present].rename(
        columns={'HUBBLE_TYPE': 'gama_hubble_type', 'FINAL_CLASS': 'gama_final_class'}
    )
    print(f"  GAMA Morphology: {len(gama_morph_slim)} rows, columns: {list(gama_morph_slim.columns)}")
    HAS_GAMA_MORPH = True
except FileNotFoundError:
    print(f"  WARNING: GAMA Morphology FITS not found at {GAMA_MORPH_FITS} — skipping")
    gama_morph_slim = None
    HAS_GAMA_MORPH = False

# --- GAMA Pairs ---
try:
    gama_pairs = load_fits_as_df(GAMA_PAIRS_FITS, 'GAMA Pairs')

    # Identify key columns — GAMA Pairs uses CATAID1, CATAID2, SepProjAng, SepVel
    for col in ['CATAID1', 'CATAID2', 'SEPPROJANG', 'SEPVEL']:
        if col not in gama_pairs.columns:
            raise KeyError(f"Expected column '{col}' not in GAMA Pairs. "
                           f"Columns: {list(gama_pairs.columns)}")

    gama_pairs['CATAID1'] = gama_pairs['CATAID1'].astype(str).str.strip().astype(int)
    gama_pairs['CATAID2'] = gama_pairs['CATAID2'].astype(str).str.strip().astype(int)
    print(f"  GAMA Pairs: {len(gama_pairs)} rows")
    HAS_GAMA_PAIRS = True
except FileNotFoundError:
    print(f"  WARNING: GAMA Pairs FITS not found at {GAMA_PAIRS_FITS} — skipping")
    gama_pairs = None
    HAS_GAMA_PAIRS = False

# =============================================================================
# STEP 4 — BUILD WANG CONCENTRATION ANCHOR + MSTAR
# =============================================================================

print('\n' + '='*60)
print('STEP 4 — Building Wang Concentration anchor + Mstar')
print('='*60)

# Use SF Mstar, fall back to composite
conc['Mstar_wang'] = conc['Mstar']
if 'Mstar_com' in conc.columns:
    fallback_mask = conc['Mstar_wang'].isna() & conc['Mstar_com'].notna()
    conc.loc[fallback_mask, 'Mstar_wang'] = conc.loc[fallback_mask, 'Mstar_com']
    print(f"  {fallback_mask.sum()} galaxies used Mstar_com as fallback in Wang")

shared_present = [c for c in SHARED_COLS if c in conc.columns and c != 'z']
missing_shared = [c for c in SHARED_COLS if c not in conc.columns]
if missing_shared:
    print(f"  WARNING: Shared columns missing from concentration: {missing_shared}")

conc_anchor = conc[['CATID', 'CIndex', 'Mstar_wang', 'z'] + shared_present].copy()
conc_anchor = conc_anchor.rename(columns={'z': 'z_wang'})  # keep z for kpc conversion

# =============================================================================
# STEP 5 — SLIM DOWN OTHER MORPHOLOGY TABLES
# =============================================================================

gini_slim   = gini[['CATID', 'gini_all_halpha']].copy()
asym_slim   = asym[['CATID', 'A_min']].copy()
smooth_slim = smooth[['CATID', 'S']].copy()

# =============================================================================
# STEP 6 — BUILD UNIVERSE OF ALL CATIDS (Wang ∪ AR2022 ∪ all morphology)
# =============================================================================

print('\n' + '='*60)
print('STEP 6 — Building full CATID universe')
print('='*60)

all_catids = pd.DataFrame({
    'CATID': pd.concat([
        conc_anchor['CATID'],
        ar2022_slim['CATID'],
        gini_slim['CATID'],
        asym_slim['CATID'],
        smooth_slim['CATID'],
    ]).drop_duplicates().reset_index(drop=True)
})
print(f"  Total unique CATIDs across all catalogues: {len(all_catids)}")

# =============================================================================
# STEP 7 — MERGE ALL SOURCES (left from universe)
# =============================================================================

print('\n' + '='*60)
print('STEP 7 — Merging all catalogues')
print('='*60)

master = all_catids.copy()

# Wang Concentration (priority 1)
master = master.merge(conc_anchor, on='CATID', how='left')
master['in_wang_conc'] = master['CIndex'].notna() | master['Mstar_wang'].notna()
# Fix: in_wang_conc should reflect actual membership in Wang catalogue
wang_catids = set(conc_anchor['CATID'])
master['in_wang_conc'] = master['CATID'].isin(wang_catids)
print(f"  Wang Concentration matched : {master['in_wang_conc'].sum()} / {len(master)}")

# AR2022
master = master.merge(ar2022_slim, on='CATID', how='left')
ar2022_catids = set(ar2022_slim['CATID'])
master['in_ar2022'] = master['CATID'].isin(ar2022_catids)
print(f"  AR2022 matched             : {master['in_ar2022'].sum()} / {len(master)}")

# Gini
master = master.merge(gini_slim, on='CATID', how='left')
gini_catids = set(gini_slim['CATID'])
master['in_gini'] = master['CATID'].isin(gini_catids)
print(f"  Gini matched               : {master['in_gini'].sum()} / {len(master)}")

# Asymmetry
master = master.merge(asym_slim, on='CATID', how='left')
asym_catids = set(asym_slim['CATID'])
master['in_asym'] = master['CATID'].isin(asym_catids)
print(f"  Asymmetry matched          : {master['in_asym'].sum()} / {len(master)}")

# Smoothness
master = master.merge(smooth_slim, on='CATID', how='left')
smooth_catids = set(smooth_slim['CATID'])
master['in_smooth'] = master['CATID'].isin(smooth_catids)
print(f"  Smoothness matched         : {master['in_smooth'].sum()} / {len(master)}")

# n_metrics count
master['n_metrics'] = (
    master['gini_all_halpha'].notna().astype(int) +
    master['CIndex'].notna().astype(int) +
    master['A_min'].notna().astype(int) +
    master['S'].notna().astype(int)
)

# =============================================================================
# STEP 8 — RESOLVE MSTAR AND SFR WITH PRIORITY CHAIN
# =============================================================================

print('\n' + '='*60)
print('STEP 8 — Resolving Mstar and SFR (Wang priority → AR2022 fallback)')
print('='*60)

# Mstar: Wang SF→composite first, then AR2022
master['Mstar'] = master['Mstar_wang']
ar2022_mstar_fallback = master['Mstar'].isna() & master['Mstar_ar2022'].notna()
master.loc[ar2022_mstar_fallback, 'Mstar'] = master.loc[ar2022_mstar_fallback, 'Mstar_ar2022']
print(f"  Mstar: Wang source  = {master['in_wang_conc'].sum() - ar2022_mstar_fallback.sum()} galaxies")
print(f"  Mstar: AR2022 fallback = {ar2022_mstar_fallback.sum()} galaxies")

# SFR: AR2022 is best estimate for all galaxies
master['SFR_raw'] = master['SFR_ar2022']
n_sfr = master['SFR_raw'].notna().sum()
print(f"  SFR from AR2022    : {n_sfr} / {len(master)}")

# Drop intermediate columns
master = master.drop(columns=['Mstar_wang', 'Mstar_ar2022', 'SFR_ar2022'])

# =============================================================================
# STEP 9 — COMPUTE SFR / sSFR (RAW AND LOG)
# =============================================================================

print('\n' + '='*60)
print('STEP 9 — Computing SFR, log_SFR, sSFR, log_sSFR')
print('='*60)

log_sfr_series, sfr_already_log = to_log_sfr(master['SFR_raw'])
if sfr_already_log:
    print('  SFR_raw already in log form — converting to linear for SFR column')
    master['SFR']     = 10 ** master['SFR_raw']
    master['log_SFR'] = master['SFR_raw']
else:
    print('  SFR_raw in linear form — applying log10')
    master['SFR']     = master['SFR_raw']
    master['log_SFR'] = log_sfr_series

# Mstar to linear for sSFR denominator
log_mstar_series, mstar_already_log = to_log_sfr(master['Mstar'])
if mstar_already_log:
    mstar_linear = 10 ** master['Mstar']
else:
    mstar_linear = master['Mstar']

master['sSFR']     = master['SFR'] / mstar_linear
master['log_sSFR'] = np.log10(master['sSFR'].replace(0, np.nan))

master = master.drop(columns=['SFR_raw'])

if master['Mstar'].median() > 100:
    master['log_Mstar'] = np.log10(master['Mstar'])
else:
    master['log_Mstar'] = master['Mstar']

# Restore z from z_wang (shared col has it; rename for clarity)
if 'z_wang' in master.columns and 'z' not in master.columns:
    master = master.rename(columns={'z_wang': 'z'})
elif 'z_wang' in master.columns:
    # z already exists from shared_cols merge — z_wang is duplicate
    master = master.drop(columns=['z_wang'])

# =============================================================================
# STEP 10 — GAMA MORPHOLOGY JOIN
# =============================================================================

print('\n' + '='*60)
print('STEP 10 — Joining GAMA Morphology')
print('='*60)

if HAS_GAMA_MORPH:
    master = master.merge(
        gama_morph_slim.rename(columns={'CATAID': 'CATID'}),
        on='CATID', how='left'
    )
    n_morph_matched = master['gama_hubble_type'].notna().sum() \
        if 'gama_hubble_type' in master.columns else \
        master['gama_final_class'].notna().sum()
    print(f"  GAMA Morphology matched : {n_morph_matched} / {len(master)}")
else:
    master['gama_hubble_type'] = np.nan
    master['gama_final_class'] = np.nan
    print('  Skipped (file not found) — columns set to NaN')

# =============================================================================
# STEP 11 — GAMA PAIRS ENVIRONMENTAL ENRICHMENT
# =============================================================================

print('\n' + '='*60)
print('STEP 11 — GAMA Pairs boolean flags')
print('='*60)

if HAS_GAMA_PAIRS:
    # Stack pairs so each galaxy appears as primary regardless of which
    # side of the pair it was on in the raw table
    pairs_as_1 = gama_pairs[['CATAID1', 'CATAID2', 'SEPPROJANG', 'SEPVEL']].rename(
        columns={'CATAID1': 'CATID', 'CATAID2': 'PARTNER'}
    )
    pairs_as_2 = gama_pairs[['CATAID1', 'CATAID2', 'SEPPROJANG', 'SEPVEL']].rename(
        columns={'CATAID2': 'CATID', 'CATAID1': 'PARTNER'}
    )
    all_pairs = pd.concat([pairs_as_1, pairs_as_2], ignore_index=True)
    all_pairs['CATID']   = all_pairs['CATID'].astype(str).str.strip().astype(int)
    all_pairs['ABS_VEL'] = all_pairs['SEPVEL'].abs()

    # Velocity cut — retain physically associated pairs only
    confirmed = all_pairs[all_pairs['ABS_VEL'] <= PAIR_VEL_CUT_KMS].copy()

    # Rename z → z_gal to avoid collision with any z already on confirmed
    z_lookup = master.set_index('CATID')['z'].dropna().to_dict()
    confirmed['z_gal'] = confirmed['CATID'].map(z_lookup)
    confirmed['SEP_KPC'] = arcsec_to_kpc(confirmed['SEPPROJANG'], confirmed['z_gal'])

    # For each galaxy: take the single closest confirmed pair
    nearest = (
        confirmed.sort_values('SEP_KPC')
        .groupby('CATID', as_index=False)
        .first()[['CATID', 'SEP_KPC']]
    )

    # Merge onto master and derive the three boolean flags
    master = master.merge(nearest, on='CATID', how='left')

    master['gama_in_pairs']          = master['SEP_KPC'].notna()
    master['gama_close_pair_100kpc'] = master['SEP_KPC'] <= 100.0
    master['gama_close_pair_50kpc']  = master['SEP_KPC'] <= 50.0

    # Drop the working separation column — not needed in output
    master = master.drop(columns=['SEP_KPC'])

    print(f"  Galaxies in GAMA pairs              : {master['gama_in_pairs'].sum()}")
    print(f"  Galaxies with close pair < 100 kpc  : {master['gama_close_pair_100kpc'].sum()}")
    print(f"  Galaxies with close pair < 50 kpc   : {master['gama_close_pair_50kpc'].sum()}")

else:
    master['gama_in_pairs']          = False
    master['gama_close_pair_100kpc'] = False
    master['gama_close_pair_50kpc']  = False

# =============================================================================
# STEP 12 — ASSIGN DATASET TAGS
# =============================================================================

print('\n' + '='*60)
print('STEP 12 — Assigning dataset tags')
print('='*60)

master['dataset_tag'] = master.apply(make_dataset_tag, axis=1)

print('\n  dataset_tag distribution:')
for tag, count in master['dataset_tag'].value_counts().items():
    print(f"    {tag:<20} : {count}")

# =============================================================================
# STEP 13 — FINAL COLUMN ORDER AND SAVE
# =============================================================================

print('\n' + '='*60)
print('STEP 13 — Ordering columns and saving')
print('='*60)

FINAL_COLS = [
    # Identity
    'CATID',
    # Provenance
    'dataset_tag',
    'in_wang_conc', 'in_ar2022', 'in_gini', 'in_asym', 'in_smooth',
    'n_metrics',
    # Morphology metrics
    'gini_all_halpha', 'CIndex', 'A_min', 'S',
    # Shared galaxy properties (from Wang, priority)
    'spectral_class', 'environment', 'galclass',
    're', 'ellip', 'z', 'g_i',
    # Physical quantities
    'Mstar', 'log_Mstar', 'SFR', 'log_SFR', 'sSFR', 'log_sSFR',
    # Quality flags
    'seeing', 'seeing_re', 'source_file', 'conflict_flag',
    # GAMA Morphology
    'gama_hubble_type', 'gama_final_class',
    # GAMA Pairs
    'gama_in_pairs', 'gama_close_pair_100kpc', 'gama_close_pair_50kpc',
]

# Only keep columns that actually exist (graceful GAMA skip)
final_cols_present = [c for c in FINAL_COLS if c in master.columns]
missing_final      = [c for c in FINAL_COLS if c not in master.columns]
if missing_final:
    print(f"  NOTE: The following expected columns are absent and will be omitted: {missing_final}")

master_out = master[final_cols_present]
master_out.to_csv(OUT_PATH, index=False)

# =============================================================================
# SUMMARY REPORT
# =============================================================================

print(f'\n{"="*60}')
print('  MASTER CSV v2 — BUILD COMPLETE')
print(f'{"="*60}')
print(f'  Total galaxies          : {len(master_out)}')
print(f'  In Wang Concentration   : {master_out["in_wang_conc"].sum()}')
print(f'  In AR2022               : {master_out["in_ar2022"].sum()}')
print(f'  Gini present            : {master_out["gini_all_halpha"].notna().sum()}')
print(f'  CIndex present          : {master_out["CIndex"].notna().sum()}')
print(f'  A_min present           : {master_out["A_min"].notna().sum()}')
print(f'  S present               : {master_out["S"].notna().sum()}')
print(f'  All 4 metrics present   : '
      f'{master_out[["gini_all_halpha","CIndex","A_min","S"]].notna().all(axis=1).sum()}')
print(f'  SFR present             : {master_out["SFR"].notna().sum()}')
print(f'  GAMA morphology present : '
      f'{master_out["gama_hubble_type"].notna().sum() if "gama_hubble_type" in master_out.columns else "N/A"}')
print(f'  Saved to                : {OUT_PATH}')
print(f'{"="*60}\n')

# Quick per-tag completeness audit
print('  Per-tag morphology completeness:')
for tag in master_out['dataset_tag'].unique():
    sub = master_out[master_out['dataset_tag'] == tag]
    n   = len(sub)
    n4  = sub[['gini_all_halpha','CIndex','A_min','S']].notna().all(axis=1).sum()
    print(f"    {tag:<20}: {n:>4} galaxies, {n4:>4} with all 4 metrics ({100*n4/n:.0f}%)")

print()


if __name__ == '__main__':
    pass