from astropy.io import fits
import numpy as np

fitsfile = "data/fits/sami_dr3_Halpha_maps/6821_A_Halpha_default_1-comp.fits"

with fits.open(fitsfile) as hdul:
    print("\n=== HDU STRUCTURE ===")
    hdul.info()

    # Primary HDU (flux cube)
    primary_data = hdul[0].data
    print("\nPrimary HDU shape:", primary_data.shape)

    # Print both flux layers explicitly
    if primary_data.ndim == 3:
        # detect which axis has length 2
        axis2 = [i for i, n in enumerate(primary_data.shape) if n == 2][0]
        flux_cube = np.moveaxis(primary_data, axis2, 0)

        print("Flux layer 0 (Gaussian fit) min/max:",
              np.nanmin(flux_cube[0]), np.nanmax(flux_cube[0]))

        print("Flux layer 1 (Summed flux) min/max:",
              np.nanmin(flux_cube[1]), np.nanmax(flux_cube[1]))

    # Extension HDU (HALPHA_ERR)
    err_data = hdul[1].data
    print("\nHALPHA_ERR shape:", err_data.shape)

    print("Error map min/max:",
          np.nanmin(err_data), np.nanmax(err_data))