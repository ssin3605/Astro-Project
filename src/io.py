# File to: 
# - open SAMI .fits files
# - Extract data arrays
# - Produce clean NumPy objects

import numpy as np
from astropy.io import fits


def read_sami_halpha_map(fitsfile, errext="HALPHA_ERR"):
    """
    Reads a SAMI Halpha flux map and corresponding error map.

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
