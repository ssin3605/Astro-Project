# File to: 
# - open SAMI .fits files
# - Extract data arrays
# - Produce clean NumPy objects

import numpy as np
from astropy.io import fits

# Function to read a SAMI Halpha flux map and corresponding error map.
def read_sami_halpha_map(fitsfile, errext="HALPHA_ERR"):
    """

    Parameters
    fitsfile : str
        Path to SAMI Halpha FITS file.
    errext : str
        Name of the FITS extension containing the error map.

    Returns
    fluxmap : 2D numpy array
        Halpha flux map.
    fluxerr : 2D numpy array
        Corresponding flux error map.
    header : astropy.io.fits.Header
        Primary header.
    """

    with fits.open(fitsfile) as hdul:
        # SAMI convention: data stored as (component, y, x)
        fluxmap = hdul[0].data[0, :, :]
        fluxerr = hdul[errext].data[0, :, :]
        header = hdul[0].header

    return fluxmap, fluxerr, header


# Function to take the Halpha flux map from DR3 Halpha map file

def halpha_flux_map_from_dr3(path):
    """
    Extract the Hα flux map from SAMI DR3 Halpha map FITS.
    Accomodated for both axis orderings as am unsure which it is:
      - (2, ny, nx)  -> flux is [0, :, :]
      - (ny, nx, 2)  -> flux is [:, :, 0]
    """
    with fits.open(path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data)

    if data.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {data.shape}")

    # Case 1: layer axis first (common in astropy for FITS with NAXIS3)
    if data.shape[0] == 2 and data.shape[1] == 50 and data.shape[2] == 50:
        flux = data[0, :, :]

    # Case 2: layer axis last
    elif data.shape[2] == 2 and data.shape[0] == 50 and data.shape[1] == 50:
        flux = data[:, :, 0]

    else:
        # More general fallback: identify the axis of length 2 and move it
        axes_len2 = [ax for ax, n in enumerate(data.shape) if n == 2]
        if len(axes_len2) != 1:
            raise ValueError(f"Can't identify 2-layer axis in shape {data.shape}")
        ax2 = axes_len2[0]
        data2 = np.moveaxis(data, ax2, 0)   # -> (2, ny, nx)
        flux = data2[0, :, :]

    return flux