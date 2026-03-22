import os
import glob
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm

from src.gini import gini_from_mask
from src.signal_noise import sn_mask

## Use to create .csv file of Halpha based dataset with Gini, SFR, sSFR, Mass

# Paths to datasets and output
catalog_path = "data/catalogs/AR2022_SAMI_DR3_Mstar_SFR_GSWL_rescaled.fits"
halpha_dir = "gini/gini_outputs/tables/masked_halpha_maps"
output_path = "gini/gini_outputs/tables/Halpha_gini_sfr_masked.csv"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Bringing in SFR + Mstar catalogue
cat = Table.read(catalog_path)
df = cat.to_pandas()

# Keep only usable galaxies
df = df[np.isfinite(df["SFR_best"]) & np.isfinite(df["Mstar_best"])].copy()

print("Galaxies with SFR + Mstar:", len(df))

# Loop 
results = []

found = 0
missing = 0
bad_open = 0
nan_gini_all = 0
nan_gini_sn3 = 0

# Selection of flux layer:
# 0 = Gaussian-fit flux
# 1 = summed/total flux
LAYER = 1

for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing Gini"):

    try:
        catid = int(row["catid"])
    except Exception:
        missing += 1
        continue

    # find correct 1-comp file
    pattern = os.path.join(halpha_dir, f"{catid}_*_Halpha_masked.fits") #change depending on dataset
    matches = glob.glob(pattern)

    if not matches:
        missing += 1
        continue
    
    filepath = matches[0]
    found += 1
    
    # Reading in .fits data layers
    try:
        with fits.open(filepath, memmap=True) as hdul:

            # Using HDU names
            flux_cube = hdul["PRIMARY"].data       # (2, 50, 50)
            err_cube  = hdul["HALPHA_ERR"].data    # (2, 50, 50)

            if (flux_cube is None or err_cube is None or
                np.ndim(flux_cube) != 3 or np.ndim(err_cube) != 3 or
                flux_cube.shape[0] < 2 or err_cube.shape[0] < 2):
                bad_open += 1
                continue

            flux_map = np.asarray(flux_cube[LAYER], dtype=float)
            err_map  = np.asarray(err_cube[LAYER], dtype=float)
        
        ### NEED TO CHECK FOR USE IN MAKED DATA SET CASE

        # Building S/N mask (no S/N cut): finite & positive flux values taken
        mask_all = np.isfinite(flux_map) & (flux_map > 0)
        g_all = gini_from_mask(flux_map, mask_all, min_pixels=10)

        # S/N mask 
        mask_sn3 = sn_mask(
            flux_map,
            err_map,
            sn_cut=3.0,
            mode="error",   # change error type if dataset changes
        )

        g_sn3 = gini_from_mask(flux_map, mask_sn3, min_pixels=10)

        # sanity check: number of pixels passing the mask used for gini_sn3
        n_sn3 = int(np.sum(mask_sn3))

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
        "halpha_file": os.path.basename(filepath),
        "logSFR": logSFR,
        "logMstar": logM,
        "logsSFR": logsSFR,
        "gini_all": g_all,
        "gini_sn3": g_sn3,
        "n_sn3_pixels": n_sn3
    })

if len(results) == 0:
    raise RuntimeError("No Halpha FITS files matched. Check halpha_dir and filename pattern.")

out = pd.DataFrame(results)

# Saving calculations to .csv
out.to_csv(output_path, index=False)

print(f"Matched Halpha files: {found}")
print(f"Missing Halpha files: {missing}")
print(f"Bad/failed opens:     {bad_open}")
print(f"NaN Gini (no cut):    {nan_gini_all}")
print(f"NaN Gini (S/N>3):     {nan_gini_sn3}")
print("Wrote:", output_path)
print("Rows written:", len(out))