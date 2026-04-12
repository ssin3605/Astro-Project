import io
import os
import pandas as pd

### Used to merge Di's classifications made through her concentration work into my .csv's made in my gini analysis

## There is a conflict flag that is added per galaxy at the end of the .csv
## This flag identifies galaxies that appear in both the SF and Composite txt files simultaneously
## The SF-preferred deduplication step resolves the conflict for practical purposes, keeping the SF row and drops the Composite row 
## for those galaxies. But the flag preserves the information that the classification was not clean-cut. 
## This means I can later filter them out entirely for a stricter analysis, or keep them and note the caveat in the thesis write-up.

# ============================================================
# CONFIGURATION
# ============================================================

HALPHA_CSV    = 'gini/gini_outputs/tables/Halpha_gini.csv'
CONTINUUM_CSV = 'gini/gini_outputs/tables/continuum_gini.csv'
COMPARE_CSV   = 'gini/gini_outputs/tables/compare_gini.csv'

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
# STEP 1: Load all four files
# ============================================================

print('=' * 60)
print('STEP 1: Loading classification files')
print('=' * 60)

sf_gama     = load_txt(TXT_FILES['SF_GAMA'],           'SF_GAMA',           'SF',        'GAMA')
sf_cluster  = load_txt(TXT_FILES['SF_Cluster'],        'SF_Cluster',        'SF',        'Cluster')
com_gama    = load_txt(TXT_FILES['Composite_GAMA'],    'Composite_GAMA',    'Composite', 'GAMA')
com_cluster = load_txt(TXT_FILES['Composite_Cluster'], 'Composite_Cluster', 'Composite', 'Cluster')

# ============================================================
# STEP 2: Concatenate — missing columns across files become NaN
# ============================================================

print('\n' + '=' * 60)
print('STEP 2: Concatenating all files')
print('=' * 60)

all_classified = pd.concat(
    [sf_gama, sf_cluster, com_gama, com_cluster],
    ignore_index=True
)

print(f"  Total rows after concat : {len(all_classified)}")
print(f"  Total columns           : {len(all_classified.columns)}")

# ============================================================
# STEP 3: Select and order columns
# ============================================================

print('\n' + '=' * 60)
print('STEP 3: Selecting and ordering columns')
print('=' * 60)

# Build ordered column list, keeping only those that actually loaded
desired_cols = (
    SHARED_COLS
    + SF_EXTRA_COLS
    + COMPOSITE_CINDEX_COLS
    + META_COLS
)

available_cols = [c for c in desired_cols if c in all_classified.columns]
missing_cols   = [c for c in desired_cols if c not in all_classified.columns]

if missing_cols:
    print(f"  Columns expected but not found in any file (will be absent from output):")
    for c in missing_cols:
        print(f"    - {c}")

all_classified = all_classified[available_cols].copy()

# ============================================================
# STEP 4: Flag CATIDs that appear in both SF and Composite files
# ============================================================

print('\n' + '=' * 60)
print('STEP 4: Flagging SF + Composite conflicts')
print('=' * 60)

conflict_catids = (
    all_classified.groupby('CATID')['spectral_class']
    .nunique()
    .pipe(lambda s: set(s[s > 1].index))
)
all_classified['conflict_flag'] = all_classified['CATID'].isin(conflict_catids)

print(f"  Total classified entries     : {len(all_classified)}")
print(f"  Unique CATIDs                : {all_classified['CATID'].nunique()}")
print(f"  CATIDs with SF + Composite   : {len(conflict_catids)}")

# ============================================================
# STEP 5: Merge SF and Composite rows per CATID into one row
# ============================================================

print('\n' + '=' * 60)
print('STEP 5: Merging SF and Composite data per CATID')
print('=' * 60)

sf_cols        = SHARED_COLS + SF_EXTRA_COLS
composite_cols = SHARED_COLS + COMPOSITE_CINDEX_COLS + CLUSTER_EXTRA_COLS

# Split into SF and Composite subsets, keeping only relevant columns
sf_rows  = all_classified[all_classified['spectral_class'] == 'SF'].copy()
com_rows = all_classified[all_classified['spectral_class'] == 'Composite'].copy()

# Keep only columns that exist in each subset
sf_keep  = [c for c in sf_cols  + META_COLS if c in sf_rows.columns]
com_keep = [c for c in composite_cols       if c in com_rows.columns]

sf_rows  = sf_rows[sf_keep].drop_duplicates(subset='CATID', keep='first')
com_rows = com_rows[com_keep].drop_duplicates(subset='CATID', keep='first')

# Left-join composite CIndex columns onto SF rows
# SF rows are the base — composite data is added where available
merged = sf_rows.merge(com_rows, on='CATID', how='left', suffixes=('', '_com'))

# For galaxies only in composite files (no SF counterpart), append them separately
composite_only_catids = set(com_rows['CATID']) - set(sf_rows['CATID'])
if composite_only_catids:
    composite_only = all_classified[
        all_classified['CATID'].isin(composite_only_catids)
    ].drop_duplicates(subset='CATID', keep='first')
    merged = pd.concat([merged, composite_only], ignore_index=True)
    print(f"  Composite-only CATIDs appended : {len(composite_only_catids)}")

merged = merged.reset_index(drop=True)

print(f"  Rows after merge         : {len(merged)}")
print(f"  SF CATIDs (base)         : {len(sf_rows)}")
print(f"  Composite CATIDs joined  : {com_rows['CATID'].isin(sf_rows['CATID']).sum()}")
print(f"  Conflict-flagged rows    : {merged['conflict_flag'].sum()}")

# Re-assign to lookup_sf_preferred for Step 6 compatibility
lookup_sf_preferred = merged

# ============================================================
# STEP 6: Write output CSV
# ============================================================

print('\n' + '=' * 60)
print('STEP 6: Writing output')
print('=' * 60)

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
lookup_sf_preferred.to_csv(OUTPUT_CSV, index=False)

print(f"  Written : {OUTPUT_CSV}")
print(f"\n  Final column schema ({len(lookup_sf_preferred.columns)} columns):")
for col in lookup_sf_preferred.columns:
    non_null = lookup_sf_preferred[col].notna().sum()
    print(f"    {col:<30} — {non_null:>4} non-null values")

print('\nDone.')