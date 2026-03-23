import io
import numpy as np
import pandas as pd
from astropy.io import fits

# ============================================================
# CONFIGURATION
# ============================================================

FITS_CATALOG = 'data/catalogs/AR2022_SAMI_DR3_Mstar_SFR_GSWL_rescaled.fits'

TXT_FILES = {
    'Composite_Cluster' : 'data/catalogs/SAMI_Cindex_data/cluout_com_v0.12.txt',
    'SF_Cluster'        : 'data/catalogs/SAMI_Cindex_data/clusterout_sf_hbsn3_v0.12.txt',
    'Composite_GAMA'    : 'data/catalogs/SAMI_Cindex_data/gamaout_com_v0.12.txt',
    'SF_GAMA'           : 'data/catalogs/SAMI_Cindex_data/gamaout_sf_hbsn3_v0.12.txt',
}

HALPHA_DIR = 'gini/gini_outputs/tables/masked_halpha_maps'

# galclass integer mapping from Wang et al. (2022)
# Adjust these if the paper specifies differently
GALCLASS_LABELS = {
    1: 'p_SF',      # Pure star forming
    2: 'c_SF',      # Central SF (AGN/LINER outskirts)
    3: 'c_L/A',     # Central LINER/AGN (SF outskirts)
    4: 'Pure AGN',  # Pure AGN / excluded
}

# ============================================================
# HELPER: load and parse a .txt classification file
# ============================================================

def load_txt(path, label):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines[0] = lines[0].lstrip('#').lstrip()
    cleaned = io.StringIO(''.join(lines))
    df = pd.read_csv(cleaned, sep=r'\s+', engine='python')
    catid_col = next((c for c in df.columns if c.upper() == 'CATID'), None)
    df = df.rename(columns={catid_col: 'CATID'})
    df['CATID'] = df['CATID'].astype(int)
    df['source_file'] = label
    return df

# ============================================================
# STEP 1: Load and merge all .txt files
# Since SF and Composite files overlap in CATID, we need to
# build one unified row per galaxy using the SF files as base
# (they contain galclass which is the key classifier)
# ============================================================

sf_cluster  = load_txt(TXT_FILES['SF_Cluster'],  'SF_Cluster')
sf_gama     = load_txt(TXT_FILES['SF_GAMA'],     'SF_GAMA')

# Combine SF files — these together cover all classified galaxies
# with one row per galaxy and the galclass column present
sf_all = pd.concat([sf_gama, sf_cluster], ignore_index=True)

# Confirm no cross-environment duplicates
assert sf_all['CATID'].nunique() == len(sf_all), \
    "Unexpected duplicate CATIDs between SF_GAMA and SF_Cluster"

print(f"Total unique classified galaxies (SF files): {len(sf_all)}")
print(f"  From SF_GAMA    : {len(sf_gama)}")
print(f"  From SF_Cluster : {len(sf_cluster)}")

# ============================================================
# STEP 2: Load FITS catalogue for sSFR
# (needed for the log(sSFR) > -11.25 cut)
# ============================================================

from astropy.table import Table
with fits.open(FITS_CATALOG) as hdul:
    cat = Table(hdul[1].data).to_pandas()
cat = cat.rename(columns={'catid': 'CATID'})
cat['CATID'] = cat['CATID'].astype(int)

# Compute log(sSFR) = SFR_best - Mstar_best (both already in log)
cat['logsSFR'] = cat['SFR_best'] - cat['Mstar_best']
cat_ssfr = cat[['CATID', 'logsSFR', 'Mstar_best', 'SFR_best']].copy()

# Merge sSFR into sf_all
sf_all = sf_all.merge(cat_ssfr, on='CATID', how='left')

# ============================================================
# STEP 3: Load your map CATIDs
# ============================================================

import os, re

def extract_catid(fname):
    match = re.search(r'(\d{4,})', fname)
    return int(match.group(1)) if match else None

map_files = [f for f in os.listdir(HALPHA_DIR) if f.endswith('.fits')]
map_catids = set(filter(None, [extract_catid(f) for f in map_files]))
print(f"\nGalaxies with Halpha maps: {len(map_catids)}")

# ============================================================
# STEP 4: Build the sample funnel — one cut at a time
# ============================================================

rows = []

def add_row(name, abbrev, description, df, note=''):
    n = df['CATID'].nunique() if 'CATID' in df.columns else len(df)
    rows.append({
        'Sample'      : name,
        'Abbreviation': abbrev,
        'Description' : description,
        'N'           : n,
    })
    if note:
        print(f"  [{abbrev}] {note}")
    return df

# --- Row 1: Full catalogue (FITS catalogue) ---
add_row(
    'Full Catalogue', 'Full_cat',
    'Full SAMI DR3 catalogue (Mstar/SFR catalogue)',
    cat
)

# --- Row 2: Galaxies with usable Halpha maps ---
has_map = sf_all[sf_all['CATID'].isin(map_catids)].copy()
add_row(
    'Has Halpha map', 'Map_sam',
    'Classified galaxies with a masked Halpha map',
    has_map
)

# --- Row 3: Wang Full_sam cuts applied sequentially ---
# Cut 1: EW_Hα ≥ 1 Å
ew_col = next((c for c in has_map.columns if 'EW' in c.upper() and 'ha' in c.lower()), None)
print(f"\n  EW column identified as: '{ew_col}'")
cut_ew = has_map[has_map[ew_col] >= 1.0].copy() if ew_col else has_map.copy()
add_row(
    'EW cut', 'EW_cut',
    f'EW_Hα ≥ 1 Å  ({ew_col})',
    cut_ew,
    note=f"Removed {len(has_map) - len(cut_ew)} galaxies"
)

# Cut 2: Re < 15 arcsec
cut_re = cut_ew[cut_ew['re'] < 15.0].copy()
add_row(
    'Re cut', 'Re_cut',
    'Re < 15 arcsec',
    cut_re,
    note=f"Removed {len(cut_ew) - len(cut_re)} galaxies"
)

# Cut 3: Ellipticity < 0.7
cut_ellip = cut_re[cut_re['ellip'] < 0.7].copy()
add_row(
    'Ellipticity cut', 'Ellip_cut',
    'Ellipticity < 0.7',
    cut_ellip,
    note=f"Removed {len(cut_re) - len(cut_ellip)} galaxies"
)

# Cut 4: Seeing quality cuts
cut_seeing = cut_ellip[
    (cut_ellip['seeing_re'] < 0.75) &
    (cut_ellip['seeing']    < 4.0)
].copy()
add_row(
    'Seeing cut', 'See_cut',
    'seeing/Re < 0.75 AND seeing < 4 arcsec',
    cut_seeing,
    note=f"Removed {len(cut_ellip) - len(cut_seeing)} galaxies"
)

# Cut 5: log(sSFR) > -11.25
cut_ssfr = cut_seeing[
    cut_seeing['logsSFR'].notna() &
    (cut_seeing['logsSFR'] > -11.25)
].copy()
add_row(
    'sSFR cut', 'sSFR_cut',
    'log(sSFR/yr⁻¹) > -11.25',
    cut_ssfr,
    note=f"Removed {len(cut_seeing) - len(cut_ssfr)} galaxies"
)

# Cut 6: Not pure AGN (galclass != 4, assuming 4 = pure AGN)
cut_agn = cut_ssfr[cut_ssfr['galclass'] != 4].copy()
add_row(
    'Full sample', 'Full_sam',
    'All cuts applied, not pure AGN (galclass ≠ 4)',
    cut_agn,
    note=f"Removed {len(cut_ssfr) - len(cut_agn)} pure AGN galaxies"
)

# ============================================================
# STEP 5: Classify Full_sam into Wang's sub-samples
# ============================================================

# galclass breakdown within Full_sam
print(f"\n  galclass value counts in Full_sam:")
print(cut_agn['galclass'].value_counts().sort_index().to_string())

full_sf = cut_agn[cut_agn['galclass'].isin([1, 2])].copy()
add_row(
    'Full Star Forming', 'Full_SF',
    'SF galaxies in Full_sam (galclass 1 or 2)',
    full_sf
)

p_sf = cut_agn[cut_agn['galclass'] == 1].copy()
add_row(
    'Pure Star Forming', 'p_SF',
    'Pure SF galaxies (galclass = 1)',
    p_sf
)

c_sf = cut_agn[cut_agn['galclass'] == 2].copy()
add_row(
    'Central SF', 'c_SF',
    'Central SF with LINER/AGN outskirts (galclass = 2)',
    c_sf
)

c_la = cut_agn[cut_agn['galclass'] == 3].copy()
add_row(
    'Central LINER/AGN', 'c_L/A',
    'Central LINER/AGN with extended SF (galclass = 3)',
    c_la
)

# ============================================================
# STEP 6: Also report environment breakdown
# ============================================================

print(f"\n  Environment breakdown in Full_sam:")
print(cut_agn['source_file'].value_counts().to_string())

# ============================================================
# STEP 7: Print the final table
# ============================================================

print('\n')
print('=' * 85)
print(f"  {'Sample':<22} {'Abbrev':<12} {'Description':<40} {'N':>5}")
print('=' * 85)
for row in rows:
    print(f"  {row['Sample']:<22} {row['Abbreviation']:<12} {row['Description']:<40} {row['N']:>5}")
print('=' * 85)

# Wang's numbers for direct comparison
print('\n  Wang et al. (2022) reference counts:')
print(f"  {'Full_cat':<12}: 2988")
print(f"  {'Full_sam':<12}: 778")
print(f"  {'Full_SF':<12}: 719")
print(f"  {'p_SF':<12}: 653")
print(f"  {'c_SF':<12}: 66")
print(f"  {'c_L/A':<12}: 59")