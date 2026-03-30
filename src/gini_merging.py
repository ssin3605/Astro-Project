import io
import os
import pandas as pd
from astropy.io import fits
from astropy.table import Table

### Used to merge Di's classifications made through her concentration work into my .csv's made in my gini analysisC

# ============================================================
# CONFIGURATION
# ============================================================

HALPHA_CSV    = 'gini/gini_outputs/tables/Halpha_gini.csv'
CONTINUUM_CSV = 'gini/gini_outputs/tables/continuum_gini.csv'
COMPARE_CSV   = 'gini/gini_outputs/tables/compare_gini.csv'

FITS_CATALOG  = 'data/catalogs/AR2022_SAMI_DR3_Mstar_SFR_GSWL_rescaled.fits'

TXT_FILES = {
    'SF_Cluster'        : 'data/catalogs/SAMI_Cindex_data/clusterout_sf_hbsn3_v0.12.txt',
    'SF_GAMA'           : 'data/catalogs/SAMI_Cindex_data/gamaout_sf_hbsn3_v0.12.txt',
    'Composite_Cluster' : 'data/catalogs/SAMI_Cindex_data/cluout_com_v0.12.txt',
    'Composite_GAMA'    : 'data/catalogs/SAMI_Cindex_data/gamaout_com_v0.12.txt',
}

OUTPUT_HALPHA    = 'gini/gini_outputs/tables/Halpha_gini.csv'
OUTPUT_CONTINUUM = 'gini/gini_outputs/tables/continuum_gini.csv'
OUTPUT_COMPARE   = 'gini/gini_outputs/tables/compare_gini.csv'

# ============================================================
# HELPER: load .txt classification file
# ============================================================

def load_txt(path, source_label):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines[0] = lines[0].lstrip('#').lstrip()
    df = pd.read_csv(io.StringIO(''.join(lines)), sep=r'\s+', engine='python')
    catid_col = next((c for c in df.columns if c.upper() == 'CATID'), None)
    df = df.rename(columns={catid_col: 'CATID'})
    df['CATID'] = df['CATID'].astype(int)
    df['source_file'] = source_label
    return df

# ============================================================
# STEP 1: Build classification lookup table
# ============================================================

print('Building classification lookup...')

# Load all four .txt files
sf_gama     = load_txt(TXT_FILES['SF_GAMA'],           'SF_GAMA')
sf_cluster  = load_txt(TXT_FILES['SF_Cluster'],        'SF_Cluster')
com_gama    = load_txt(TXT_FILES['Composite_GAMA'],    'Composite_GAMA')
com_cluster = load_txt(TXT_FILES['Composite_Cluster'], 'Composite_Cluster')

# Assign spectral class and environment from file membership
for df, spec, env in [
    (sf_gama,     'SF',        'GAMA'),
    (sf_cluster,  'SF',        'Cluster'),
    (com_gama,    'Composite', 'GAMA'),
    (com_cluster, 'Composite', 'Cluster'),
]:
    df['spectral_class'] = spec
    df['environment']    = env

# Build a unified lookup with one row per CATID per spectral class
# (a galaxy can appear as both SF and Composite — we keep both rows
# and flag the conflict, then let the user decide per-analysis)
all_classified = pd.concat(
    [sf_gama, sf_cluster, com_gama, com_cluster],
    ignore_index=True
)

# Core columns to carry forward into the lookup
lookup_cols = ['CATID', 'spectral_class', 'environment', 'galclass',
               're', 'ellip', 'z', 'g_i', 'Mstar',
               'seeing', 'seeing_re', 'source_file']

# Only keep columns that exist across all files
lookup_cols = [c for c in lookup_cols if c in all_classified.columns]
lookup = all_classified[lookup_cols].copy()

# Flag galaxies that appear under both SF and Composite
conflict_catids = (
    lookup.groupby('CATID')['spectral_class']
    .nunique()
    .pipe(lambda s: set(s[s > 1].index))
)
lookup['conflict_flag'] = lookup['CATID'].isin(conflict_catids)

print(f'  Total classified entries     : {len(lookup)}')
print(f'  Unique classified CATIDs     : {lookup["CATID"].nunique()}')
print(f'  CATIDs with SF+Composite     : {len(conflict_catids)}')

# ============================================================
# STEP 2: Build a single-row-per-galaxy summary lookup
#
# For galaxies appearing in both SF and Composite files,
# we create two resolved versions:
#   - lookup_sf_only : one row per galaxy, SF classification
#     preferred (use for Halpha Gini analysis)
#   - lookup_any     : one row per galaxy, first occurrence kept
#     (use for compare_gini where you want full coverage)
# ============================================================

# SF-preferred lookup: keep SF row if available, else Composite
lookup_sf_preferred = (
    lookup.sort_values('spectral_class',
                       key=lambda x: x.map({'SF': 0, 'Composite': 1}))
    .drop_duplicates(subset='CATID', keep='first')
    .copy()
)

print(f'\n  SF-preferred lookup (one row per galaxy):')
print(f'    SF         : {(lookup_sf_preferred["spectral_class"] == "SF").sum()}')
print(f'    Composite  : {(lookup_sf_preferred["spectral_class"] == "Composite").sum()}')
print(f'  Environment split:')
print(f'    GAMA       : {(lookup_sf_preferred["environment"] == "GAMA").sum()}')
print(f'    Cluster    : {(lookup_sf_preferred["environment"] == "Cluster").sum()}')
print(f'  galclass distribution:')
print(lookup_sf_preferred['galclass'].value_counts().sort_index().to_string())

# ============================================================
# STEP 3: Merge classifications into each CSV
# ============================================================

def merge_classifications(csv_path, lookup_df, label):
    """
    Left-join classification columns onto an existing Gini CSV.
    Preserves all existing rows and columns exactly.
    Adds: spectral_class, environment, galclass, re, ellip,
          z, g_i, Mstar_txt, seeing, seeing_re,
          source_file, conflict_flag
    """
    df = pd.read_csv(csv_path)
    original_len = len(df)

    # Rename Mstar from .txt to avoid collision with existing logMstar
    lookup_renamed = lookup_df.copy()
    if 'Mstar' in lookup_renamed.columns:
        lookup_renamed = lookup_renamed.rename(columns={'Mstar': 'Mstar_txt'})

    merged = df.merge(
        lookup_renamed,
        left_on='catid',
        right_on='CATID',
        how='left'
    ).drop(columns='CATID')

    # Fill unclassified galaxies
    merged['spectral_class'] = merged['spectral_class'].fillna('Unclassified')
    merged['environment']    = merged['environment'].fillna('Unknown')
    merged['conflict_flag']  = merged['conflict_flag'].fillna(False)

    # Verify row count unchanged
    assert len(merged) == original_len, \
        f'Row count changed in {label}: {original_len} → {len(merged)}'

    # Report
    print(f'\n  {label}:')
    print(f'    Rows          : {len(merged)} (unchanged)')
    print(f'    SF            : {(merged["spectral_class"] == "SF").sum()}')
    print(f'    Composite     : {(merged["spectral_class"] == "Composite").sum()}')
    print(f'    Unclassified  : {(merged["spectral_class"] == "Unclassified").sum()}')
    print(f'    Conflict flag : {merged["conflict_flag"].sum()}')
    print(f'    New columns   : {[c for c in merged.columns if c not in df.columns]}')

    return merged

print('\nMerging classifications into CSVs...')

halpha_merged    = merge_classifications(HALPHA_CSV,    lookup_sf_preferred, 'Halpha_gini')
continuum_merged = merge_classifications(CONTINUUM_CSV, lookup_sf_preferred, 'continuum_gini')
compare_merged   = merge_classifications(COMPARE_CSV,   lookup_sf_preferred, 'compare_gini')

# ============================================================
# STEP 4: Write outputs (overwrites existing CSVs in place)
# ============================================================

print('\nWriting updated CSVs...')

halpha_merged.to_csv(OUTPUT_HALPHA,    index=False)
continuum_merged.to_csv(OUTPUT_CONTINUUM, index=False)
compare_merged.to_csv(OUTPUT_COMPARE,  index=False)

print(f'  Written: {OUTPUT_HALPHA}')
print(f'  Written: {OUTPUT_CONTINUUM}')
print(f'  Written: {OUTPUT_COMPARE}')

# ============================================================
# STEP 5: Final schema summary
# ============================================================

print('\nFinal column schemas:')
for label, df in [
    ('Halpha_gini',    halpha_merged),
    ('continuum_gini', continuum_merged),
    ('compare_gini',   compare_merged),
]:
    print(f'\n  {label}:')
    print(f'    {list(df.columns)}')