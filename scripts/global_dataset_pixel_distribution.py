import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.io import halpha_flux_map_from_dr3

# Execution script to run through all .fits data, extract the Halpha maps, flatten pixels and present the data

# Defines the path for the script to run. This file's location has .parent[0] of /sami/scripts, so .parents[1] allows the script to access other files in /sami including /data and /outputs
PROJECT_ROOT = Path(__file__).resolve().parents[1] # Project path is rooted to the base repo /sami
DATA_DIR = PROJECT_ROOT / "data" / "fits" / "sami_dr3_halpha_maps" # Use data from /sami/data/fits/sami_dr3_halpha_maps
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" # Save figures produced to /output

files = sorted(DATA_DIR.rglob("*_1-comp.fits")) # Only selecting the 1-comp.fits not recom-comp.fits

print(f"Found {len(files)} FITS files.")

# Containers
all_pix = []
n_ok = 0
n_empty = 0
n_failed = 0

for i, fn in enumerate(files):

    try:
        img = halpha_flux_map_from_dr3(fn)
    except Exception:
        n_failed += 1
        continue

    if i == 0:
        from astropy.io import fits
        with fits.open(fn) as hdul:
            print("RAW HDU0 shape:", np.asarray(hdul[0].data).shape)

        print("EXTRACTED flux shape:", img.shape)
        print("flux min/med/max:",
              np.nanmin(img),
              np.nanmedian(img),
              np.nanmax(img))

    pix = img.ravel()
    pix = pix[np.isfinite(pix)]
    pix = pix[pix > 0]

    if pix.size == 0:
        n_empty += 1
        continue

    all_pix.append(pix)
    n_ok += 1

print("\nSummary:")
print("  Successfully processed:", n_ok)
print("  Empty after filtering:", n_empty)
print("  Failed to load:", n_failed)

# If nothing worked 
if len(all_pix) == 0:
    raise RuntimeError("No usable pixels found.")

# Combine all galaxies in one pooled pixel array
all_pix = np.concatenate(all_pix)

# Printing of global stats
print(f"Total pixels used: {all_pix.size:,}")
print(f"Global min/median/max: "
      f"{all_pix.min():.4g}, "
      f"{np.median(all_pix):.4g}, "
      f"{all_pix.max():.4g}")


# Histogram (linear) - after inspection does not contribute much value 

plt.figure()
plt.hist(all_pix, bins=300)
plt.xlabel("Hα flux (native units)")
plt.ylabel("Number of pixels")
plt.title("All galaxies: pixel-value histogram (linear)")
plt.savefig(OUTPUT_DIR / "global_pixel_hist_linear.png", dpi=300)
plt.show()


# Histogram (log-x) - good for vast spans of magnitude

bins = np.logspace(
    np.log10(all_pix.min()),
    np.log10(all_pix.max()),
    300
)

plt.figure()
plt.hist(all_pix, bins=bins)
plt.xscale("log")
plt.xlabel("Hα flux (log scale)")
plt.ylabel("Number of pixels")
plt.title("All galaxies: pixel-value histogram (log x)")
plt.savefig(OUTPUT_DIR / "global_pixel_hist_log.png", dpi=300)
plt.show()


# Cumulative distribution function (useful for visualising quality cuts)

x = np.sort(all_pix)
cdf = np.arange(1, x.size + 1) / x.size

plt.figure()
plt.plot(x, cdf)
plt.xscale("log")
plt.xlabel("Hα flux (log scale)")
plt.ylabel("CDF")
plt.title("All galaxies: pixel-value CDF")
plt.savefig(OUTPUT_DIR / "global_pixel_cdf.png", dpi=300)
plt.show()
