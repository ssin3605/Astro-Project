import os
import glob
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm

from src.gini import gini_from_mask
from src.signal_noise import sn_mask

## Use to create .csv file of Continuum based dataset with Gini, SFR, sSFR, Mass
### Have merged Di's classifications with this --> should not need to re-run or use anymore as .csv's are created


# Paths to datasets and output
catalog_path = "data/catalogs/AR2022_SAMI_DR3_Mstar_SFR_GSWL_rescaled.fits"
cont_dir = "gini/gini_outputs/tables/masked_continuum_maps"
output_path = "gini/gini_outputs/tables/continuum_gini.csv"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Bringing in SFR + Mstar catalogue
cat = Table.read(catalog_path)
df = cat.to_pandas()

# Keep only usable galaxies
df = df[np.isfinite(df["SFR_best"]) & np.isfinite(df["Mstar_best"])].copy()

# Loop 
results = []

found = 0
missing = 0
bad_open = 0
nan_gini_all = 0
nan_gini_sn3 = 0
too_few_pixels_sn3 = 0

MIN_PIXELS = 10

for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing Continuum Gini"):

    try:
        catid = int(row["catid"])
    except Exception:
        missing += 1
        continue

    # Glob matching is case-sensitive here; your real files use HALPHA
    pattern = os.path.join(cont_dir, f"{catid}_*_HALPHA_cont_map_masked.fits")
    matches = glob.glob(pattern)

    if not matches:
        missing += 1
        continue
    
    filepath = matches[0]
    found += 1
    
    # Reading in .fits data layers
    try:
        with fits.open(filepath, memmap=True) as hdul:

            # Given continuum structure is:
            # 0 PRIMARY                (50, 50)  flux map
            # 1 HALPHA_CONT_STD_DEV    (50, 50)  std dev map (error)
            flux_map = np.asarray(hdul["PRIMARY"].data, dtype=float)
            err_map  = np.asarray(hdul["HALPHA_CONT_STD_DEV"].data, dtype=float)

            if flux_map.ndim != 2 or err_map.ndim != 2:
                bad_open += 1
                continue

        # Building S/N mask (no S/N cut): finite & positive flux values taken
        mask_all = np.isfinite(flux_map) & (flux_map > 0)
        g_all = gini_from_mask(flux_map, mask_all, min_pixels=MIN_PIXELS)

        # S/N mask 
        mask_sn3 = sn_mask(
            flux_map,
            err_map,
            sn_cut=3.0,
            mode="error",   # change error type if dataset changes
        )
        n_sn3 = int(np.sum(mask_sn3))
        if n_sn3 < MIN_PIXELS:
            too_few_pixels_sn3 += 1

        g_sn3 = gini_from_mask(flux_map, mask_sn3, min_pixels=MIN_PIXELS)

        if not np.isfinite(g_all):
            nan_gini_all += 1
        if not np.isfinite(g_sn3):
            nan_gini_sn3 += 1

    except Exception:
        bad_open += 1
        continue

    logSFR = float(row["SFR_best"])
    logM = float(row["Mstar_best"])
    logsSFR = logSFR - logM  # since both are log10

    results.append({
        "catid": catid,
        "continuum_file": os.path.basename(filepath),
        "logSFR": logSFR,
        "logMstar": logM,
        "logsSFR": logsSFR,
        "gini_all": g_all,
        "gini_sn3": g_sn3,
        "n_sn3_pixels": n_sn3
    })

if len(results) == 0:
    raise RuntimeError("No continuum FITS files matched. Check cont_dir and filename pattern.")

out = pd.DataFrame(results)

# Saving calculations to .csv
out.to_csv(output_path, index=False)

print(f"Matched continuum files: {found}")
print(f"Missing continuum files: {missing}")
print(f"Bad/failed opens:     {bad_open}")
print(f"NaN Gini (no cut):    {nan_gini_all}")
print(f"NaN Gini (S/N>3):     {nan_gini_sn3}")
print(f"Too few SN pixels:       {too_few_pixels_sn3}")
print("Wrote:", output_path)
print("Rows written:", len(out))