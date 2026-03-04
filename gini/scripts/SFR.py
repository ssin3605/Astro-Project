from astropy.table import Table

t = Table.read("data/catalogs/AR2022_SAMI_DR3_Mstar_SFR_GSWL_rescaled.fits")  # path to your FITS
print(len(t))
print(t.colnames)
t.pprint(max_lines=5, max_width=120)

# column unit (if present)
print("SFR_best unit:", t["SFR_best"].unit)

# show min/max to sanity check
import numpy as np
s = np.array(t["SFR_best"])
print(np.nanmin(s), np.nanmax(s))

# quick search for likely ID columns
candidates = [c for c in t.colnames if "ID" in c.upper() or "CAT" in c.upper() or "GAMA" in c.upper()]
print(candidates)

