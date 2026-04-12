import io
import os
import pandas as pd

### Use to take all relevant Wang data and put it into a concentration_results.csv

## There is a conflict flag that is added per galaxy at the end of the .csv
## This flag identifies galaxies that appear in both the SF and Composite txt files simultaneously
## The SF-preferred deduplication step resolves the conflict for practical purposes, keeping the SF row and drops the Composite row 
## for those galaxies. But the flag preserves the information that the classification was not clean-cut. 
## This means I can later filter them out entirely for a stricter analysis, or keep them and note the caveat in the thesis write-up.

# ============================================================
# CONFIGURATION
# ============================================================

TXT_FILES = {
    'SF_Cluster'        : 'data/catalogs/SAMI_Cindex_data/clusterout_sf_hbsn3_v0.12.txt',
    'SF_GAMA'           : 'data/catalogs/SAMI_Cindex_data/gamaout_sf_hbsn3_v0.12.txt',
    'Composite_Cluster' : 'data/catalogs/SAMI_Cindex_data/cluout_com_v0.12.txt',
    'Composite_GAMA'    : 'data/catalogs/SAMI_Cindex_data/gamaout_com_v0.12.txt',
}

OUTPUT_CSV = 'concentration/outputs/tables/concentration_results.csv'

# ============================================================
# COLUMN DEFINITIONS
# ============================================================

# Columns present in all four files
SHARED_COLS = [
    'CATID', 're', 'PA', 'ellip', 'z', 'g_i', 'Mstar',
    'galclass', 'galclass_exclude', 'seeing', 'seeing_re',
]

# SF files only (single CIndex)
SF_EXTRA_COLS = [
    'CIndex', 'totalHagas', 'ha_r50', 'cont_r50', 'ha_EW', 'SFR',
]

# Composite files only — multiple CIndex variants per diagnostic and aperture
COMPOSITE_CINDEX_COLS = [
    'CIndex_n2ha_s', 'ha_EW_n_s', 'SFR_n2ha_s',
    'CIndex_n2ha_m', 'ha_EW_n_m', 'SFR_n2ha_m',
    'CIndex_n2ha_l', 'ha_EW_n_l', 'SFR_n2ha_l',
    'CIndex_o3hb_s', 'ha_EW_o_s', 'SFR_o3hb_s',
    'CIndex_o3hb_m', 'ha_EW_o_m', 'SFR_o3hb_m',
    'CIndex_o3hb_l', 'ha_EW_o_l', 'SFR_o3hb_l',
    'CIndex_s2ha_s', 'ha_EW_s_s', 'SFR_s2ha_s',
    'CIndex_s2ha_m', 'ha_EW_s_m', 'SFR_s2ha_m',
    'CIndex_s2ha_l', 'ha_EW_s_l', 'SFR_s2ha_l',
]

# Composite cluster file only
CLUSTER_EXTRA_COLS = ['BCGCATAID', 'MassA_1']

# Metadata columns added by this script
META_COLS = ['spectral_class', 'environment', 'conflict_flag', 'source_file']

# ============================================================
# HELPER: load .txt classification file
# ============================================================

def load_txt(path, source_label, spectral_class, environment):
    """
    Load a whitespace-delimited .txt catalogue file.
    Strips the leading '#' from the header line, assigns metadata columns,
    and standardises the CATID column name.
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    # Strip leading '#' and whitespace from the header line
    lines[0] = lines[0].lstrip('#').lstrip()

    df = pd.read_csv(io.StringIO(''.join(lines)), sep=r'\s+', engine='python')

    # Standardise CATID column name (case-insensitive match)
    catid_col = next((c for c in df.columns if c.upper() == 'CATID'), None)
    if catid_col is None:
        raise ValueError(f"No CATID column found in {path}. Columns: {list(df.columns)}")
    df = df.rename(columns={catid_col: 'CATID'})
    df['CATID'] = df['CATID'].astype(int)

    # Attach metadata
    df['source_file']    = source_label
    df['spectral_class'] = spectral_class
    df['environment']    = environment

    print(f"  Loaded {source_label:<25} — {len(df):>4} rows, {len(df.columns)} columns")
    return df

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