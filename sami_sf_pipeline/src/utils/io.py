"""
FITS readers for raw SAMI Halpha cubes and continuum maps.

Carried over from the audited repo's src/io.py with no logic changes -
only docstring/typing cleanup.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits


def read_sami_halpha_layers(fitsfile: str, flux_hdu: str = "PRIMARY", err_hdu: str = "HALPHA_ERR"):
    """
    Read a raw SAMI Halpha cube and its error cube.

    The two SAMI line-fit slices ([Gaussian fit, summed flux]) can be stored
    on either axis depending on file version, so the axis of length 2 is
    located and moved to the front, giving a consistent (2, ny, nx) layout.

    Returns
    -------
    flux_layers, err_layers : dict with keys "fit" (slice 0) and "sum" (slice 1)
    header : FITS header of the flux HDU
    meta : dict of shape/path metadata, useful for logging
    """
    with fits.open(fitsfile, memmap=True) as hdul:
        flux = np.asarray(hdul[flux_hdu].data)
        err = np.asarray(hdul[err_hdu].data)
        header = hdul[flux_hdu].header

    if flux.ndim != 3 or err.ndim != 3:
        raise ValueError(f"Expected 3D cubes. Got flux {flux.shape}, err {err.shape}")

    axf = [ax for ax, n in enumerate(flux.shape) if n == 2][0]
    axe = [ax for ax, n in enumerate(err.shape) if n == 2][0]
    flux2 = np.moveaxis(flux, axf, 0)
    err2 = np.moveaxis(err, axe, 0)

    flux_layers = {"fit": flux2[0].astype(float), "sum": flux2[1].astype(float)}
    err_layers = {"fit": err2[0].astype(float), "sum": err2[1].astype(float)}

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


def read_sami_continuum_layers(fitsfile: str, flux_hdu: str = "PRIMARY", err_hdu: str = "HALPHA_CONT_STD_DEV"):
    """
    Read a raw SAMI continuum flux map and its std-dev map.

    Returns
    -------
    fluxmap, fluxerr : 2D numpy arrays
    header : FITS header of the flux HDU
    meta : dict of shape/path metadata
    """
    with fits.open(fitsfile, memmap=True) as hdul:
        flux = np.asarray(hdul[flux_hdu].data, dtype=float)
        err = np.asarray(hdul[err_hdu].data, dtype=float)
        header = hdul[flux_hdu].header

    if flux.ndim != 2 or err.ndim != 2:
        raise ValueError(f"Expected 2D maps. Got flux {flux.shape}, err {err.shape}")

    meta = {
        "fitsfile": fitsfile,
        "flux_hdu": flux_hdu,
        "err_hdu": err_hdu,
        "flux_shape": tuple(flux.shape),
        "err_shape": tuple(err.shape),
    }

    return flux, err, header, meta


def read_masked_halpha(fitsfile: str):
    """
    Read a masked Halpha FITS file (as written by 02_continuum_mask.py).

    HDU 0: masked flux cube, shape (2, ny, nx) -> [fit, sum]
    HDU 1: masked error cube, shape (2, ny, nx)
    """
    with fits.open(fitsfile, memmap=True) as hdul:
        flux_cube = np.asarray(hdul[0].data, dtype=float)
        err_cube = np.asarray(hdul[1].data, dtype=float)
    return flux_cube, err_cube


def read_masked_continuum(fitsfile: str):
    """
    Read a masked continuum FITS file (as written by 02_continuum_mask.py).

    HDU 0: masked flux map, shape (ny, nx)
    HDU 1: masked error map, shape (ny, nx)
    """
    with fits.open(fitsfile, memmap=True) as hdul:
        flux = np.asarray(hdul[0].data, dtype=float)
        err = np.asarray(hdul[1].data, dtype=float)
    return flux, err
