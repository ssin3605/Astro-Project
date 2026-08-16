import numpy as np
from astropy.io import fits

from src.signal_noise import compute_sn, sn_mask


def load_continuum_maps(fits_path):
    """
    Load continuum flux and error maps.

    Assumes:
    - HDU 0: continuum flux map (2D)
    - HDU 1: continuum error/std-dev map (2D)
    """
    with fits.open(fits_path) as hdul:
        cont_flux = np.array(hdul[0].data, dtype=float)
        cont_err = np.array(hdul[1].data, dtype=float)
        flux_header = hdul[0].header.copy()
        err_header = hdul[1].header.copy()

    return cont_flux, cont_err, flux_header, err_header


def load_halpha_cubes(fits_path):
    """
    Load Halpha flux and error cubes.

    Assumes:
    - HDU 0: Halpha data cube, shape (2, ny, nx)
    - HDU 1: Halpha error cube, shape (2, ny, nx)

    Slice convention:
    - [0, :, :] = Gaussian fit
    - [1, :, :] = SUM
    """
    with fits.open(fits_path) as hdul:
        halpha_flux_cube = np.array(hdul[0].data, dtype=float)
        halpha_err_cube = np.array(hdul[1].data, dtype=float)
        flux_header = hdul[0].header.copy()
        err_header = hdul[1].header.copy()

    if halpha_flux_cube.ndim != 3:
        raise ValueError(f"Halpha flux cube is not 3D: shape={halpha_flux_cube.shape}")
    if halpha_err_cube.ndim != 3:
        raise ValueError(f"Halpha error cube is not 3D: shape={halpha_err_cube.shape}")

    return halpha_flux_cube, halpha_err_cube, flux_header, err_header


def make_continuum_mask(cont_flux, cont_err, sn_cut=3.0):
    """
    Build mask from continuum flux and error maps.

    Keeps pixels where:
    - flux is finite
    - error is finite and > 0
    - flux > 0
    - S/N >= sn_cut
    """
    sn_map = compute_sn(cont_flux, cont_err, mode="error")
    mask = sn_mask(
        cont_flux,
        cont_err,
        sn_cut=sn_cut,
        mode="error",
        positive_flux=True
    )
    return mask, sn_map


def apply_mask_2d(data_2d, mask, fill_value=0.0):
    """
    Apply 2D mask to 2D array.
    """
    data_2d = np.array(data_2d, dtype=float)
    data_2d[~np.isfinite(data_2d)] = fill_value
    return np.where(mask, data_2d, fill_value)


def apply_mask_3d(data_3d, mask, fill_value=0.0):
    """
    Apply a 2D mask to every slice of a 3D cube.

    Assumes cube shape:
    - (n_slice, ny, nx)

    For SAMI Halpha maps:
    - slice 0 = Gaussian fit
    - slice 1 = SUM
    """
    data_3d = np.array(data_3d, dtype=float)
    data_3d[~np.isfinite(data_3d)] = fill_value

    if data_3d.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {data_3d.shape}")

    if data_3d.shape[1:] != mask.shape:
        raise ValueError(
            f"Mask shape {mask.shape} is incompatible with data cube shape {data_3d.shape}"
        )

    mask_3d = mask[np.newaxis, :, :]
    return np.where(mask_3d, data_3d, fill_value)


def save_masked_continuum(output_path, masked_flux, masked_err,
                          flux_header=None, err_header=None):
    """
    Save masked continuum FITS:

    - HDU 0: masked continuum flux map, shape (ny, nx)
    - HDU 1: masked continuum error map, shape (ny, nx)
    """
    if flux_header is None:
        flux_header = fits.Header()
    if err_header is None:
        err_header = fits.Header()

    primary_hdu = fits.PrimaryHDU(
        data=masked_flux.astype(np.float32),
        header=flux_header
    )
    err_hdu = fits.ImageHDU(
        data=masked_err.astype(np.float32),
        header=err_header
    )

    fits.HDUList([primary_hdu, err_hdu]).writeto(output_path, overwrite=True)


def save_masked_halpha(output_path, masked_flux_cube, masked_err_cube,
                       flux_header=None, err_header=None):
    """
    Save masked Halpha FITS preserving original structure:

    - HDU 0: masked Halpha flux cube, shape (2, ny, nx)
    - HDU 1: masked Halpha error cube, shape (2, ny, nx)

    Slice convention:
    - [0, :, :] = Gaussian fit
    - [1, :, :] = SUM
    """
    if flux_header is None:
        flux_header = fits.Header()
    if err_header is None:
        err_header = fits.Header()

    primary_hdu = fits.PrimaryHDU(
        data=masked_flux_cube.astype(np.float32),
        header=flux_header
    )
    err_hdu = fits.ImageHDU(
        data=masked_err_cube.astype(np.float32),
        header=err_header
    )

    fits.HDUList([primary_hdu, err_hdu]).writeto(output_path, overwrite=True)