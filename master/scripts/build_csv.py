"""
Builds the master summary CSV collating results from Gini, Concentration, Asymmetry,
Smoothness for SAMI, enriched with GAMA environmental data and per-galaxy dataset tags.

Dataset Priority Chain
----------------------
1. Wang TXT files (SF + Composite)  → primary anchor for SFR, Mstar, and shared metadata
2. Wang Concentration CSV            → CIndex metric + fallback shared metadata
3. AR2022 SAMI DR3 catalogue         → fallback Mstar only (SFR no longer sourced here)
4. Gini / Asymmetry / Smoothness     → morphology metrics only (no metadata priority)

SFR Resolution Logic
--------------------
Galaxies are first classified as pure_SF or composite_SF from Wang TXT membership:
    pure_SF      : CATID appears in an SF TXT file but NOT in any composite TXT file
    composite_SF : CATID appears in BOTH an SF TXT file AND its matching composite TXT file

SFR is then assigned as:
    pure_SF      → SFR column from the matching SF TXT file  (total Hα, BPT-pure)
    composite_SF → SFR_n2ha_s from the matching composite TXT file
                   (SF-only component isolated via [NII]/Hα BPT demarcation)

For the 301 composite_SF galaxies where SFR_n2ha_s = 0 (i.e. no valid SF detection):
    Default: SFR set to NaN, galaxy retained in master with sfr_fallback_flag = 'nan'
    To switch to SF-file fallback instead, set COMPOSITE_SFR_FALLBACK = 'sf_file' below.
    Valid options: 'nan' | 'sf_file'

Mstar Resolution Logic
----------------------
    Primary   : Mstar column from Wang TXT files (SF file preferred; composite identical)
    Fallback 1: AR2022 Mstar (for CATIDs not in any Wang TXT file)

spectral_class Tag
------------------
Replaces the old (broken) spectral_class column. Derived entirely from Wang TXT membership:
    'pure_SF'      : in SF TXT file only
    'composite_SF' : in both SF and composite TXT files
    NaN            : CATID not in any Wang TXT file (morphology-only galaxies)

sfr_fallback_flag Column
------------------------
Tracks what SFR source was used per galaxy:
    'sf_file'     : SFR from Wang SF TXT (pure_SF galaxies, or composite fallback if enabled)
    'n2ha_s'      : SFR_n2ha_s from Wang composite TXT
    'nan_no_n2ha' : composite_SF galaxy where SFR_n2ha_s unavailable, set to NaN
    NaN           : galaxy not in any Wang TXT file (no Wang SFR available)

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
dataset_tag, in_ar2022, in_wang_sf, in_wang_composite, in_G, in_C, in_A, in_S, n_metrics,
G, C, A, S,
spectral_class, environment, galclass,
re, ellip, z,
Mstar, log_Mstar, SFR, log_SFR, sSFR, log_sSFR,
sfr_flag,
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

BASE_DIR   = '/Users/sophiesinclair/Desktop/sami'

# Wang TXT files (primary SFR + Mstar source)
WANG_SF_CLUSTER_TXT  = os.path.join(BASE_DIR, 'data/catalogs/SAMI_Cindex_data/clusterout_sf_hbsn3_v0.12.txt')
WANG_SF_GAMA_TXT     = os.path.join(BASE_DIR, 'data/catalogs/SAMI_Cindex_data/gamaout_sf_hbsn3_v0.12.txt')
WANG_COM_CLUSTER_TXT = os.path.join(BASE_DIR, 'data/catalogs/SAMI_Cindex_data/cluout_com_v0.12.txt')
WANG_COM_GAMA_TXT    = os.path.join(BASE_DIR, 'data/catalogs/SAMI_Cindex_data/gamaout_com_v0.12.txt')

# Morphology metric CSVs
GINI_CSV   = os.path.join(BASE_DIR, 'gini/gini_outputs/tables/compare_gini.csv')
CONC_CSV   = os.path.join(BASE_DIR, 'concentration/outputs/tables/concentration_results.csv')
ASYM_CSV   = os.path.join(BASE_DIR, 'asymmetry/outputs/tables/asymmetry_results.csv')
SMOOTH_CSV = os.path.join(BASE_DIR, 'smoothness/outputs/Tables/smoothness_results.csv')

# AR2022: now used only as Mstar fallback (not SFR)
AR2022_FITS = os.path.join(BASE_DIR, 'data/catalogs/AR2022_SAMI_DR3_Mstar_SFR_GSWL_rescaled.fits')

# GAMA catalogues
GAMA_PAIRS_FITS = os.path.join(BASE_DIR, 'data/fits/gama_pairs.fits')
GAMA_MORPH_FITS = os.path.join(BASE_DIR, 'data/fits/gama_morphology.fits')

OUT_DIR  = os.path.join(BASE_DIR, 'master')
OUT_PATH = os.path.join(OUT_DIR, 'outputs/tables/master_results.csv')

os.makedirs(os.path.join(OUT_DIR, 'outputs/tables'), exist_ok=True)

# ---------------------------------------------------------------------------
# SFR FALLBACK POLICY for composite galaxies with no valid SFR_n2ha_s
# ---------------------------------------------------------------------------
# 'nan'     → set SFR to NaN, flag as 'nan_no_n2ha'  (default, methodologically safe)
# 'sf_file' → substitute SF-file SFR, flag as 'sf_file' (includes AGN contamination)
COMPOSITE_SFR_FALLBACK = 'nan'   # <-- change here to switch policy

# Cosmology for arcsec → kpc conversion
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)

# Pair velocity cut (km/s)
PAIR_VEL_CUT_KMS = 500.0

# Pair separation thresholds (kpc)
GAMA_SEP_IN_KPC  = False
PAIR_SEP_100_KPC = 100.0
PAIR_SEP_50_KPC  =  50.0

# Shared metadata columns sourced from Wang concentration CSV (fallback only)
SHARED_COLS_CONC = [
    'environment', 'galclass', 'galclass_exclude',
    're', 'ellip', 'z', 'g_i',
    'seeing', 'seeing_re', 'source_file', 'conflict_flag',
]

# =============================================================================
# HELPERS
# =============================================================================

def load_wang_txt(path, label=''):
    """Load a Wang TXT file (whitespace-separated, header on first line prefixed #)."""
    with open(path) as f:
        header_line = f.readline().lstrip('#').split()
    df = pd.read_csv(path, sep=r'\s+', comment='#', header=0, names=header_line)
    df['CATID'] = df['CATID'].astype(str).str.strip().astype(int)
    # Replace sentinel zeros in SFR columns with NaN (0 = no valid detection in Wang)
    sfr_cols = [c for c in df.columns if c.upper().startswith('SFR')]
    for c in sfr_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df.loc[df[c] == 0, c] = np.nan
        df.loc[df[c] <= -999, c] = np.nan
    # Mstar sentinel
    if 'Mstar' in df.columns:
        df['Mstar'] = pd.to_numeric(df['Mstar'], errors='coerce')
        df.loc[df['Mstar'] <= -999, 'Mstar'] = np.nan
    print(f"  {len(df):>5} rows  ←  {os.path.basename(path)}  [{label}]")
    return df


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
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = df[col].str.decode('utf-8').str.strip()
        except AttributeError:
            pass
    return df


def normalise_catid(df, col):
    df[col] = df[col].astype(str).str.strip().astype(int)
    return df


def to_log_sfr(series):
    """
    Detect whether SFR values are raw linear or already log10.
    Returns (log_series, is_already_log).
    Heuristic: if median > 100 it is linear Msun/yr; otherwise already log10.
    """
    valid = series.dropna()
    if valid.empty:
        return series, False
    if valid.median() > 100:
        return np.log10(series.replace(0, np.nan)), False
    else:
        return series, True


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
    wang  = row['in_C']
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
# STEP 1 — LOAD WANG TXT FILES AND BUILD SPECTRAL CLASSIFICATION
# =============================================================================

print('\n' + '='*60)
print('STEP 1 — Loading Wang TXT files')
print('='*60)

wang_sf_cl  = load_wang_txt(WANG_SF_CLUSTER_TXT,  'Wang SF Cluster')
wang_sf_ga  = load_wang_txt(WANG_SF_GAMA_TXT,     'Wang SF GAMA')
wang_com_cl = load_wang_txt(WANG_COM_CLUSTER_TXT, 'Wang Composite Cluster')
wang_com_ga = load_wang_txt(WANG_COM_GAMA_TXT,    'Wang Composite GAMA')

# Build membership sets
sf_cluster_ids  = set(wang_sf_cl['CATID'])
sf_gama_ids     = set(wang_sf_ga['CATID'])
com_cluster_ids = set(wang_com_cl['CATID'])
com_gama_ids    = set(wang_com_ga['CATID'])

sf_all  = sf_cluster_ids  | sf_gama_ids
com_all = com_cluster_ids | com_gama_ids

pure_sf_ids     = sf_all - com_all      # SF files only
composite_sf_ids = sf_all & com_all     # in both SF and composite files

print(f"\n  pure_SF galaxies      : {len(pure_sf_ids)}")
print(f"  composite_SF galaxies : {len(composite_sf_ids)}")
print(f"  Total in Wang TXTs    : {len(sf_all)}")
print(f"  Composite-only (should be 0): {len(com_all - sf_all)}")

# =============================================================================
# STEP 2 — BUILD WANG SFR + MSTAR LOOKUP TABLES
# =============================================================================

print('\n' + '='*60)
print('STEP 2 — Building Wang SFR and Mstar lookup tables')
print('='*60)

# --- Mstar: identical across SF and composite files for shared galaxies ---
# Stack all four Wang files, take first non-null Mstar per CATID
wang_mstar_all = pd.concat([
    wang_sf_cl[['CATID', 'Mstar']],
    wang_sf_ga[['CATID', 'Mstar']],
    wang_com_cl[['CATID', 'Mstar']],
    wang_com_ga[['CATID', 'Mstar']],
], ignore_index=True)
wang_mstar = (
    wang_mstar_all
    .dropna(subset=['Mstar'])
    .drop_duplicates(subset='CATID', keep='first')
    [['CATID', 'Mstar']]
    .rename(columns={'Mstar': 'Mstar_wang'})
)
print(f"  Wang Mstar entries : {len(wang_mstar)}")

# --- SFR lookup tables ---
# Pure SF: SFR from SF files
sf_sfr_all = pd.concat([
    wang_sf_cl[['CATID', 'SFR']],
    wang_sf_ga[['CATID', 'SFR']],
], ignore_index=True).drop_duplicates(subset='CATID', keep='first')
sf_sfr_lookup = sf_sfr_all.set_index('CATID')['SFR'].to_dict()

# Composite: SFR_n2ha_s from composite files
com_sfr_all = pd.concat([
    wang_com_cl[['CATID', 'SFR_n2ha_s']],
    wang_com_ga[['CATID', 'SFR_n2ha_s']],
], ignore_index=True).drop_duplicates(subset='CATID', keep='first')
com_sfr_lookup = com_sfr_all.set_index('CATID')['SFR_n2ha_s'].to_dict()

print(f"  SF-file SFR entries valid   : {sum(1 for v in sf_sfr_lookup.values() if pd.notna(v))}")
print(f"  n2ha_s SFR entries valid    : {sum(1 for v in com_sfr_lookup.values() if pd.notna(v))}")

# --- galclass: from SF files (matches paper convention) ---
sf_galclass_all = pd.concat([
    wang_sf_cl[['CATID', 'galclass', 'galclass_exclude']],
    wang_sf_ga[['CATID', 'galclass', 'galclass_exclude']],
], ignore_index=True).drop_duplicates(subset='CATID', keep='first')

# --- Shared metadata: re, ellip, z, g_i, seeing, seeing_re ---
# These are present in SF files (and composite, but SF takes priority)
# g_i is NOT in composite files, so must come from SF files
WANG_META_COLS = ['CATID', 're', 'ellip', 'z', 'g_i', 'seeing', 'seeing_re']
wang_meta_sf = pd.concat([
    wang_sf_cl[[c for c in WANG_META_COLS if c in wang_sf_cl.columns]],
    wang_sf_ga[[c for c in WANG_META_COLS if c in wang_sf_ga.columns]],
], ignore_index=True).drop_duplicates(subset='CATID', keep='first')

# environment: only GAMA files have environment context (cluster vs field implicit in filename)
# Wang encodes environment via which file a galaxy is in — build this as a column
env_lookup = {}
for catid in sf_cluster_ids:
    env_lookup[catid] = 'cluster'
for catid in sf_gama_ids:
    env_lookup[catid] = 'field'   # GAMA = field/group environment

# =============================================================================
# STEP 3 — LOAD MORPHOLOGY CATALOGUES
# =============================================================================

print('\n' + '='*60)
print('STEP 3 — Loading morphology catalogues')
print('='*60)

gini   = load_csv(GINI_CSV,   'CATID', 'Gini')
conc   = load_csv(CONC_CSV,   'CATID', 'Wang Concentration')
asym   = load_csv(ASYM_CSV,   'catid', 'Asymmetry')
smooth = load_csv(SMOOTH_CSV, 'CATID', 'Smoothness')

# =============================================================================
# STEP 4 — LOAD AR2022 (Mstar fallback only)
# =============================================================================

print('\n' + '='*60)
print('STEP 4 — Loading AR2022 FITS (Mstar fallback only)')
print('='*60)

ar2022 = load_fits_as_df(AR2022_FITS, 'AR2022')
ar2022 = normalise_catid(ar2022, 'CATID')

mstar_col = 'MSTAR_BEST' if 'MSTAR_BEST' in ar2022.columns else \
            next((c for c in ar2022.columns if 'MSTAR' in c), None)
if mstar_col is None:
    raise KeyError(f'No Mstar column found in AR2022. Columns: {list(ar2022.columns)}')
print(f"  AR2022 Mstar fallback column : {mstar_col}")

ar2022_slim = ar2022[['CATID', mstar_col]].rename(columns={mstar_col: 'Mstar_ar2022'})

# =============================================================================
# STEP 5 — LOAD GAMA CATALOGUES
# =============================================================================

print('\n' + '='*60)
print('STEP 5 — Loading GAMA catalogues')
print('='*60)

# --- GAMA Morphology ---
try:
    gama_morph = load_fits_as_df(GAMA_MORPH_FITS, 'GAMA Morphology')
    cataid_col = next((c for c in gama_morph.columns if 'CATAID' in c or 'CATID' in c), None)
    if cataid_col is None:
        raise KeyError(f'No CATAID column. Columns: {list(gama_morph.columns)}')
    gama_morph = gama_morph.rename(columns={cataid_col: 'CATAID'})
    gama_morph['CATAID'] = gama_morph['CATAID'].astype(str).str.strip().astype(int)
    morph_cols_wanted  = ['CATAID', 'HUBBLE_TYPE', 'FINAL_CLASS']
    morph_cols_present = [c for c in morph_cols_wanted if c in gama_morph.columns]
    gama_morph_slim = gama_morph[morph_cols_present].rename(
        columns={'HUBBLE_TYPE': 'gama_hubble_type', 'FINAL_CLASS': 'gama_final_class'}
    )
    print(f"  GAMA Morphology: {len(gama_morph_slim)} rows")
    HAS_GAMA_MORPH = True
except FileNotFoundError:
    print(f"  WARNING: GAMA Morphology FITS not found — skipping")
    gama_morph_slim = None
    HAS_GAMA_MORPH = False

# --- GAMA Pairs ---
try:
    gama_pairs = load_fits_as_df(GAMA_PAIRS_FITS, 'GAMA Pairs')
    for col in ['CATAID1', 'CATAID2', 'SEPPROJANG', 'SEPVEL']:
        if col not in gama_pairs.columns:
            raise KeyError(f"Expected column '{col}' not in GAMA Pairs. "
                           f"Columns: {list(gama_pairs.columns)}")
    gama_pairs['CATAID1'] = gama_pairs['CATAID1'].astype(str).str.strip().astype(int)
    gama_pairs['CATAID2'] = gama_pairs['CATAID2'].astype(str).str.strip().astype(int)
    print(f"  GAMA Pairs: {len(gama_pairs)} rows")
    HAS_GAMA_PAIRS = True
except FileNotFoundError:
    print(f"  WARNING: GAMA Pairs FITS not found — skipping")
    gama_pairs = None
    HAS_GAMA_PAIRS = False

# =============================================================================
# STEP 6 — BUILD WANG CONCENTRATION ANCHOR (CIndex + fallback metadata)
# =============================================================================

print('\n' + '='*60)
print('STEP 6 — Building Wang Concentration anchor (CIndex)')
print('='*60)

# Only pulling CIndex and conc-specific columns from this CSV now.
# SFR/Mstar/spectral_class come from Wang TXT files instead.
conc_shared_present = [c for c in SHARED_COLS_CONC if c in conc.columns]
conc_anchor = conc[['CATID', 'CIndex'] + conc_shared_present].copy()
wang_conc_catids = set(conc_anchor['CATID'])
print(f"  Wang Concentration CATIDs : {len(wang_conc_catids)}")

# =============================================================================
# STEP 7 — BUILD CATID UNIVERSE
# =============================================================================

print('\n' + '='*60)
print('STEP 7 — Building full CATID universe')
print('='*60)

all_catids = pd.DataFrame({
    'CATID': pd.concat([
        conc_anchor['CATID'],
        ar2022_slim['CATID'],
        pd.Series(list(sf_all)),        # all Wang TXT galaxies
        gini['CATID'],
        asym['CATID'],
        smooth['CATID'],
    ]).drop_duplicates().reset_index(drop=True)
})
print(f"  Total unique CATIDs : {len(all_catids)}")

# =============================================================================
# STEP 8 — MERGE ALL SOURCES
# =============================================================================

print('\n' + '='*60)
print('STEP 8 — Merging all catalogues')
print('='*60)

master = all_catids.copy()

# Wang Concentration
master = master.merge(conc_anchor, on='CATID', how='left')
master['in_C'] = master['CATID'].isin(wang_conc_catids)
print(f"  Wang Concentration matched : {master['in_C'].sum()} / {len(master)}")

# AR2022 (Mstar only)
master = master.merge(ar2022_slim, on='CATID', how='left')
master['in_ar2022'] = master['CATID'].isin(set(ar2022_slim['CATID']))
print(f"  AR2022 matched             : {master['in_ar2022'].sum()} / {len(master)}")

# Morphology metrics
gini_slim   = gini[['CATID', 'gini_all_halpha']].copy()
asym_slim   = asym[['CATID', 'A_min']].copy()
smooth_slim = smooth[['CATID', 'S']].copy()

master = master.merge(gini_slim,   on='CATID', how='left')
master = master.merge(asym_slim,   on='CATID', how='left')
master = master.merge(smooth_slim, on='CATID', how='left')

master['in_G'] = master['CATID'].isin(set(gini['CATID']))
master['in_A'] = master['CATID'].isin(set(asym['CATID']))
master['in_S'] = master['CATID'].isin(set(smooth['CATID']))

# Rename metric columns to G, C, A, S
master = master.rename(columns={
    'gini_all_halpha': 'G',
    'CIndex':          'C',
    'A_min':           'A',
})
# S column already named S in smoothness CSV — no rename needed

master['n_metrics'] = (
    master['G'].notna().astype(int) +
    master['C'].notna().astype(int) +
    master['A'].notna().astype(int) +
    master['S'].notna().astype(int)
)

print(f"  Gini matched               : {master['in_G'].sum()} / {len(master)}")
print(f"  Asymmetry matched          : {master['in_A'].sum()} / {len(master)}")
print(f"  Smoothness matched         : {master['in_S'].sum()} / {len(master)}")

# Wang TXT membership flags
master['in_wang_sf']        = master['CATID'].isin(sf_all)
master['in_wang_composite'] = master['CATID'].isin(com_all)

# =============================================================================
# STEP 9 — ASSIGN SPECTRAL CLASS (replaces old broken spectral_class)
# =============================================================================

print('\n' + '='*60)
print('STEP 9 — Assigning spectral_class from Wang TXT membership')
print('='*60)

def assign_spectral_class(catid):
    if catid in composite_sf_ids:
        return 'composite_SF'
    elif catid in pure_sf_ids:
        return 'pure_SF'
    else:
        return np.nan

master['spectral_class'] = master['CATID'].apply(assign_spectral_class)

print(f"  pure_SF      : {(master['spectral_class'] == 'pure_SF').sum()}")
print(f"  composite_SF : {(master['spectral_class'] == 'composite_SF').sum()}")
print(f"  NaN (not in Wang TXTs) : {master['spectral_class'].isna().sum()}")

# =============================================================================
# STEP 10 — RESOLVE MSTAR (Wang TXT primary → AR2022 fallback)
# =============================================================================

print('\n' + '='*60)
print('STEP 10 — Resolving Mstar (Wang TXT → AR2022 fallback)')
print('='*60)

master = master.merge(wang_mstar, on='CATID', how='left')
master['Mstar'] = master['Mstar_wang']

ar2022_fallback_mask = master['Mstar'].isna() & master['Mstar_ar2022'].notna()
master.loc[ar2022_fallback_mask, 'Mstar'] = master.loc[ar2022_fallback_mask, 'Mstar_ar2022']

print(f"  Mstar from Wang TXTs   : {master['Mstar_wang'].notna().sum()}")
print(f"  Mstar from AR2022      : {ar2022_fallback_mask.sum()}")
print(f"  Mstar still NaN        : {master['Mstar'].isna().sum()}")

master = master.drop(columns=['Mstar_wang', 'Mstar_ar2022'])

# =============================================================================
# STEP 11 — RESOLVE SFR (Wang TXT, spectral_class-aware)
# =============================================================================

print('\n' + '='*60)
print('STEP 11 — Resolving SFR from Wang TXT files')
print(f'  Composite fallback policy : COMPOSITE_SFR_FALLBACK = "{COMPOSITE_SFR_FALLBACK}"')
print('='*60)

sfr_values = []
sfr_flags  = []

for catid, sc in zip(master['CATID'], master['spectral_class']):
    if sc == 'pure_SF':
        sfr  = sf_sfr_lookup.get(catid, np.nan)
        flag = 'sf_file' if pd.notna(sfr) else np.nan
    elif sc == 'composite_SF':
        sfr  = com_sfr_lookup.get(catid, np.nan)
        if pd.notna(sfr):
            flag = 'n2ha_s'
        else:
            # SFR_n2ha_s unavailable — apply fallback policy
            if COMPOSITE_SFR_FALLBACK == 'sf_file':
                sfr  = sf_sfr_lookup.get(catid, np.nan)
                flag = 'sf_file' if pd.notna(sfr) else np.nan
            else:
                sfr  = np.nan
                flag = 'nan_no_n2ha'
    else:
        sfr  = np.nan
        flag = np.nan
    sfr_values.append(sfr)
    sfr_flags.append(flag)

master['SFR_raw']  = sfr_values
master['sfr_flag'] = sfr_flags

flag_counts = pd.Series(sfr_flags).value_counts(dropna=False)
print(f"  SFR flag summary:")
for flag, n in flag_counts.items():
    print(f"    {str(flag):<15} : {n}")

# =============================================================================
# STEP 12 — COMPUTE SFR, log_SFR, sSFR, log_sSFR
# =============================================================================

print('\n' + '='*60)
print('STEP 12 — Computing SFR, log_SFR, sSFR, log_sSFR')
print('='*60)

log_sfr_series, sfr_already_log = to_log_sfr(master['SFR_raw'])

if sfr_already_log:
    print('  SFR_raw already in log form — storing as log_SFR, converting to linear SFR')
    master['log_SFR'] = master['SFR_raw']
    master['SFR']     = 10 ** master['SFR_raw']
else:
    print('  SFR_raw in linear form — computing log10')
    master['SFR']     = master['SFR_raw']
    master['log_SFR'] = log_sfr_series

master = master.drop(columns=['SFR_raw'])

# Mstar: detect linear vs log for sSFR denominator
_, mstar_already_log = to_log_sfr(master['Mstar'])
if mstar_already_log:
    master['log_Mstar'] = master['Mstar']
    mstar_linear = 10 ** master['Mstar']
else:
    master['log_Mstar'] = np.log10(master['Mstar'].replace(0, np.nan))
    mstar_linear = master['Mstar']

master['sSFR']     = master['SFR'] / mstar_linear
master['log_sSFR'] = np.log10(master['sSFR'].replace(0, np.nan))

# =============================================================================
# STEP 13 — ATTACH WANG TXT METADATA (re, ellip, z, g_i, seeing, seeing_re)
# =============================================================================

print('\n' + '='*60)
print('STEP 13 — Attaching Wang TXT metadata columns')
print('='*60)

master = master.merge(wang_meta_sf, on='CATID', how='left', suffixes=('', '_wang_txt'))

# Wang TXT metadata takes priority over conc CSV for overlapping columns
# (conc CSV already merged in step 8; fill gaps from Wang TXT)
for col in ['re', 'ellip', 'z', 'g_i', 'seeing', 'seeing_re']:
    txt_col = f'{col}_wang_txt'
    if txt_col in master.columns:
        if col in master.columns:
            # Fill NaNs in conc-sourced col with Wang TXT values
            mask = master[col].isna() & master[txt_col].notna()
            master.loc[mask, col] = master.loc[mask, txt_col]
        else:
            master[col] = master[txt_col]
        master = master.drop(columns=[txt_col])

# Attach environment from env_lookup (derived from which Wang TXT file galaxy is in)
master['environment_wang'] = master['CATID'].map(env_lookup)
# If conc CSV has environment, use it; fill gaps from Wang TXT derivation
if 'environment' in master.columns:
    env_fill_mask = master['environment'].isna() & master['environment_wang'].notna()
    master.loc[env_fill_mask, 'environment'] = master.loc[env_fill_mask, 'environment_wang']
else:
    master['environment'] = master['environment_wang']
master = master.drop(columns=['environment_wang'])

# Attach galclass from SF files (overrides conc CSV galclass if present)
master = master.merge(sf_galclass_all, on='CATID', how='left', suffixes=('_conc', '_wang'))
for col in ['galclass', 'galclass_exclude']:
    wang_col = f'{col}_wang'
    conc_col = f'{col}_conc'
    if wang_col in master.columns and conc_col in master.columns:
        # Wang TXT takes priority; fill gaps with conc value
        master[col] = master[wang_col].fillna(master[conc_col])
        master = master.drop(columns=[wang_col, conc_col])
    elif wang_col in master.columns:
        master = master.rename(columns={wang_col: col})
    elif conc_col in master.columns:
        master = master.rename(columns={conc_col: col})

print(f"  re present      : {master['re'].notna().sum()}")
print(f"  z present       : {master['z'].notna().sum()}")
print(f"  galclass present: {master['galclass'].notna().sum()}")

# =============================================================================
# STEP 14 — GAMA MORPHOLOGY JOIN
# =============================================================================

print('\n' + '='*60)
print('STEP 14 — Joining GAMA Morphology')
print('='*60)

if HAS_GAMA_MORPH:
    master = master.merge(
        gama_morph_slim.rename(columns={'CATAID': 'CATID'}),
        on='CATID', how='left'
    )
    n_matched = master['gama_hubble_type'].notna().sum() \
        if 'gama_hubble_type' in master.columns \
        else master['gama_final_class'].notna().sum()
    print(f"  GAMA Morphology matched : {n_matched} / {len(master)}")
else:
    master['gama_hubble_type'] = np.nan
    master['gama_final_class'] = np.nan
    print('  Skipped — columns set to NaN')

# =============================================================================
# STEP 15 — GAMA PAIRS ENVIRONMENTAL FLAGS
# =============================================================================

print('\n' + '='*60)
print('STEP 15 — GAMA Pairs boolean flags')
print('='*60)

if HAS_GAMA_PAIRS:
    pairs_as_1 = gama_pairs[['CATAID1', 'CATAID2', 'SEPPROJANG', 'SEPVEL']].rename(
        columns={'CATAID1': 'CATID', 'CATAID2': 'PARTNER'}
    )
    pairs_as_2 = gama_pairs[['CATAID1', 'CATAID2', 'SEPPROJANG', 'SEPVEL']].rename(
        columns={'CATAID2': 'CATID', 'CATAID1': 'PARTNER'}
    )
    all_pairs = pd.concat([pairs_as_1, pairs_as_2], ignore_index=True)
    all_pairs['CATID']   = all_pairs['CATID'].astype(str).str.strip().astype(int)
    all_pairs['ABS_VEL'] = all_pairs['SEPVEL'].abs()

    confirmed = all_pairs[all_pairs['ABS_VEL'] <= PAIR_VEL_CUT_KMS].copy()
    z_lookup  = master.set_index('CATID')['z'].dropna().to_dict()
    confirmed['z_gal'] = confirmed['CATID'].map(z_lookup)
    confirmed['SEP_KPC'] = arcsec_to_kpc(confirmed['SEPPROJANG'], confirmed['z_gal'])

    nearest = (
        confirmed.sort_values('SEP_KPC')
        .groupby('CATID', as_index=False)
        .first()[['CATID', 'SEP_KPC']]
    )

    master = master.merge(nearest, on='CATID', how='left')
    master['gama_in_pairs']          = master['SEP_KPC'].notna()
    master['gama_close_pair_100kpc'] = master['SEP_KPC'] <= PAIR_SEP_100_KPC
    master['gama_close_pair_50kpc']  = master['SEP_KPC'] <= PAIR_SEP_50_KPC
    master = master.drop(columns=['SEP_KPC'])

    print(f"  In GAMA pairs              : {master['gama_in_pairs'].sum()}")
    print(f"  Close pair < 100 kpc       : {master['gama_close_pair_100kpc'].sum()}")
    print(f"  Close pair <  50 kpc       : {master['gama_close_pair_50kpc'].sum()}")
else:
    master['gama_in_pairs']          = False
    master['gama_close_pair_100kpc'] = False
    master['gama_close_pair_50kpc']  = False

# =============================================================================
# STEP 16 — DATASET TAGS
# =============================================================================

print('\n' + '='*60)
print('STEP 16 — Assigning dataset tags')
print('='*60)

master['dataset_tag'] = master.apply(make_dataset_tag, axis=1)
print('\n  dataset_tag distribution:')
for tag, count in master['dataset_tag'].value_counts().items():
    print(f"    {tag:<20} : {count}")

# =============================================================================
# STEP 17 — FINAL COLUMN ORDER AND SAVE
# =============================================================================

print('\n' + '='*60)
print('STEP 17 — Ordering columns and saving')
print('='*60)

FINAL_COLS = [
    # Identity
    'CATID',
    # Provenance
    'dataset_tag',
    'in_ar2022',
    'in_wang_sf', 'in_wang_composite',
    'in_G', 'in_C', 'in_A', 'in_S',
    'n_metrics',
    # Morphology metrics
    'G', 'C', 'A', 'S',
    # Spectral / galaxy classification
    'spectral_class', 'environment', 'galclass',
    # Structural parameters
    're', 'ellip', 'z',
    # Physical quantities
    'Mstar', 'log_Mstar', 'SFR', 'log_SFR', 'sSFR', 'log_sSFR',
    # SFR provenance
    'sfr_flag',
    # Quality / observation flags
    'seeing', 'seeing_re', 'source_file', 'conflict_flag',
    # GAMA morphology
    'gama_hubble_type', 'gama_final_class',
    # GAMA pairs
    'gama_in_pairs', 'gama_close_pair_100kpc', 'gama_close_pair_50kpc',
]

final_cols_present = [c for c in FINAL_COLS if c in master.columns]
missing_final      = [c for c in FINAL_COLS if c not in master.columns]
if missing_final:
    print(f"  NOTE: Expected columns absent (will be omitted): {missing_final}")

master_out = master[final_cols_present]
master_out.to_csv(OUT_PATH, index=False)

# =============================================================================
# SUMMARY REPORT
# =============================================================================

print(f'\n{"="*60}')
print('  MASTER CSV — BUILD COMPLETE')
print(f'{"="*60}')
print(f'  Total galaxies           : {len(master_out)}')
print(f'  In Wang Concentration    : {master_out["in_C"].sum()}')
print(f'  In AR2022                : {master_out["in_ar2022"].sum()}')
print(f'  In Wang SF files         : {master_out["in_wang_sf"].sum()}')
print(f'  In Wang Composite files  : {master_out["in_wang_composite"].sum()}')
print(f'  pure_SF                  : {(master_out["spectral_class"]=="pure_SF").sum()}')
print(f'  composite_SF             : {(master_out["spectral_class"]=="composite_SF").sum()}')
print(f'  G present                : {master_out["G"].notna().sum()}')
print(f'  C present                : {master_out["C"].notna().sum()}')
print(f'  A present                : {master_out["A"].notna().sum()}')
print(f'  S present                : {master_out["S"].notna().sum()}')
print(f'  All 4 metrics present    : '
      f'{master_out[["G","C","A","S"]].notna().all(axis=1).sum()}')
print(f'  SFR present              : {master_out["SFR"].notna().sum()}')
print(f'  SFR flag: sf_file        : {(master_out["sfr_flag"]=="sf_file").sum()}')
print(f'  SFR flag: n2ha_s         : {(master_out["sfr_flag"]=="n2ha_s").sum()}')
print(f'  SFR flag: nan_no_n2ha    : {(master_out["sfr_flag"]=="nan_no_n2ha").sum()}')
print(f'  GAMA morphology present  : '
      f'{master_out["gama_hubble_type"].notna().sum() if "gama_hubble_type" in master_out.columns else "N/A"}')
print(f'  Saved to                 : {OUT_PATH}')
print(f'{"="*60}')

print('\n  Per-tag morphology completeness:')
for tag in master_out['dataset_tag'].unique():
    sub = master_out[master_out['dataset_tag'] == tag]
    n   = len(sub)
    n4  = sub[['G', 'C', 'A', 'S']].notna().all(axis=1).sum()
    print(f"    {tag:<20}: {n:>4} galaxies, {n4:>4} with all 4 metrics ({100*n4/n:.0f}%)")

print()

if __name__ == '__main__':
    pass