import io
import os
import pandas as pd

### Use after asymmetry_03.py
### This script will merges Wang's classifications into the asymmetry results CSV created.
### Have structured it to do so by writing over and editing the existing .csv created rather than produce a different .csv
### Mirrors the structure of src/gini_merging.py used for the Gini analysis.
### Important to note that any time asymmetry_03.py is run, this needs to be run after to update it will all needed data
### Also if any of the input or output paths change these must also be corrected in this script

## There is a conflict flag that is added per galaxy at the end of the .csv
## This flag identifies galaxies that appear in both the SF and Composite txt files simultaneously
## The SF-preferred deduplication step resolves the conflict for practical purposes, keeping the SF row and drops the Composite row 
## for those galaxies. But the flag preserves the information that the classification was not clean-cut. 
## This means you can later filter them out entirely for a stricter analysis, or keep them and note the caveat in your thesis write-up.


# ============================================================
# CONFIGURATION
# ============================================================

ASYMMETRY_CSV = 'asymmetry/outputs/tables/asymmetry_results.csv'

TXT_FILES = {
    'SF_Cluster'        : 'data/catalogs/SAMI_Cindex_data/clusterout_sf_hbsn3_v0.12.txt',
    'SF_GAMA'           : 'data/catalogs/SAMI_Cindex_data/gamaout_sf_hbsn3_v0.12.txt',
    'Composite_Cluster' : 'data/catalogs/SAMI_Cindex_data/cluout_com_v0.12.txt',
    'Composite_GAMA'    : 'data/catalogs/SAMI_Cindex_data/gamaout_com_v0.12.txt',
}

OUTPUT_CSV = 'asymmetry/outputs/tables/asymmetry_results.csv'

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

sf_gama     = load_txt(TXT_FILES['SF_GAMA'],           'SF_GAMA')
sf_cluster  = load_txt(TXT_FILES['SF_Cluster'],        'SF_Cluster')
com_gama    = load_txt(TXT_FILES['Composite_GAMA'],    'Composite_GAMA')
com_cluster = load_txt(TXT_FILES['Composite_Cluster'], 'Composite_Cluster')

for df, spec, env in [
    (sf_gama,     'SF',        'GAMA'),
    (sf_cluster,  'SF',        'Cluster'),
    (com_gama,    'Composite', 'GAMA'),
    (com_cluster, 'Composite', 'Cluster'),
]:
    df['spectral_class'] = spec
    df['environment']    = env

all_classified = pd.concat(
    [sf_gama, sf_cluster, com_gama, com_cluster],
    ignore_index=True
)

lookup_cols = ['CATID', 'spectral_class', 'environment', 'galclass',
               're', 'ellip', 'z', 'g_i', 'Mstar',
               'seeing', 'seeing_re', 'source_file']

lookup_cols = [c for c in lookup_cols if c in all_classified.columns]
lookup = all_classified[lookup_cols].copy()

conflict_catids = (
    lookup.groupby('CATID')['spectral_class']
    .nunique()
    .pipe(lambda s: set(s[s > 1].index))
)
lookup['conflict_flag'] = lookup['CATID'].isin(conflict_catids)

print(f'  Total classified entries : {len(lookup)}')
print(f'  Unique classified CATIDs : {lookup["CATID"].nunique()}')
print(f'  CATIDs with SF+Composite : {len(conflict_catids)}')

# ============================================================
# STEP 2: SF-preferred deduplication (one row per galaxy)
# ============================================================

lookup_sf_preferred = (
    lookup.sort_values('spectral_class',
                       key=lambda x: x.map({'SF': 0, 'Composite': 1}))
    .drop_duplicates(subset='CATID', keep='first')
    .copy()
)

print(f'\n  SF-preferred lookup (one row per galaxy):')
print(f'    SF        : {(lookup_sf_preferred["spectral_class"] == "SF").sum()}')
print(f'    Composite : {(lookup_sf_preferred["spectral_class"] == "Composite").sum()}')
print(f'  Environment split:')
print(f'    GAMA      : {(lookup_sf_preferred["environment"] == "GAMA").sum()}')
print(f'    Cluster   : {(lookup_sf_preferred["environment"] == "Cluster").sum()}')

# ============================================================
# STEP 3: Merge into asymmetry CSV
# ============================================================

print('\nMerging classifications into asymmetry CSV...')

df           = pd.read_csv(ASYMMETRY_CSV)
original_len = len(df)

# Rename Mstar to avoid collision with any existing mass column
lookup_renamed = lookup_sf_preferred.copy()
if 'Mstar' in lookup_renamed.columns:
    lookup_renamed = lookup_renamed.rename(columns={'Mstar': 'Mstar_txt'})

# catid in the asymmetry CSV is already an integer CATID
df['catid'] = df['catid'].astype(int)

merged = df.merge(
    lookup_renamed,
    left_on='catid',
    right_on='CATID',
    how='left'
).drop(columns='CATID')

# Fill unmatched galaxies
merged['spectral_class'] = merged['spectral_class'].fillna('Unclassified')
merged['environment']    = merged['environment'].fillna('Unknown')
merged['conflict_flag']  = merged['conflict_flag'].fillna(False)

assert len(merged) == original_len, \
    f'Row count changed: {original_len} → {len(merged)}'

print(f'  Rows          : {len(merged)} (unchanged)')
print(f'  SF            : {(merged["spectral_class"] == "SF").sum()}')
print(f'  Composite     : {(merged["spectral_class"] == "Composite").sum()}')
print(f'  Unclassified  : {(merged["spectral_class"] == "Unclassified").sum()}')
print(f'  Conflict flag : {merged["conflict_flag"].sum()}')
print(f'  New columns   : {[c for c in merged.columns if c not in df.columns]}')

# ============================================================
# STEP 4: Write output
# ============================================================

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
merged.to_csv(OUTPUT_CSV, index=False)

print(f'\nWritten: {OUTPUT_CSV}')
print(f'\nFinal column schema:')
print(f'  {list(merged.columns)}')