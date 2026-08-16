# File to: 
# - open SAMI .fits files
# - return flux and error layer values

import numpy as np
from astropy.io import fits

## Use for Halpha Data
def read_sami_Halpha_layers(fitsfile: str, flux_hdu="PRIMARY", err_hdu="HALPHA_ERR"):
    with fits.open(fitsfile, memmap=True) as hdul:
        flux = np.asarray(hdul[flux_hdu].data)
        err  = np.asarray(hdul[err_hdu].data)
        header = hdul[flux_hdu].header

    if flux.ndim != 3 or err.ndim != 3:
        raise ValueError(f"Expected 3D cubes. Got flux {flux.shape}, err {err.shape}")

    # move the axis of length 2 to the front -> (2, ny, nx)
    axf = [ax for ax, n in enumerate(flux.shape) if n == 2][0]
    axe = [ax for ax, n in enumerate(err.shape) if n == 2][0]
    flux2 = np.moveaxis(flux, axf, 0)
    err2  = np.moveaxis(err,  axe, 0)

    flux_layers = {"fit": flux2[0].astype(float), "sum": flux2[1].astype(float)}
    err_layers  = {"fit": err2[0].astype(float),  "sum": err2[1].astype(float)}

    meta = {
        "fitsfile": fitsfile,
        "flux_hdu": flux_hdu,
        "err_hdu": err_hdu,
        "flux_shape_raw": tuple(flux.shape),
        "err_shape_raw": tuple(err.shape),
        "flux_shape_std": tuple(flux2.shape),
        "err_shape_std": tuple(err2.shape),
    }

    return flux_layers, err_layers, header, meta

## Use for Continuum Data
def read_sami_continuum_layers(
    fitsfile,
    flux_hdu="PRIMARY",
    err_hdu="HALPHA_CONT_STD_DEV",
):
    """
    Read a SAMI continuum map and its std dev map.

    Returns
    -------
    fluxmap : 2D numpy array
    fluxerr : 2D numpy array
    header  : FITS header
    meta    : dict
    """

    with fits.open(fitsfile, memmap=True) as hdul:
        flux = np.asarray(hdul[flux_hdu].data, dtype=float)
        err  = np.asarray(hdul[err_hdu].data, dtype=float)
        header = hdul[flux_hdu].header

    if flux.ndim != 2 or err.ndim != 2:
        raise ValueError(
            f"Expected 2D maps. Got flux {flux.shape}, err {err.shape}"
        )

    meta = {
        "fitsfile": fitsfile,
        "flux_hdu": flux_hdu,
        "err_hdu": err_hdu,
        "flux_shape": tuple(flux.shape),
        "err_shape": tuple(err.shape),
    }

    return flux, err, header, meta