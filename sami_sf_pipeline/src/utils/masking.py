"""
Continuum-mask construction and application.

This is the single source of truth for masking logic. 02_continuum_mask.py
uses it to build data/masked/; every metric in 03_metrics/ reads the
already-masked FITS it produces rather than re-deriving its own mask, so the
same footprint is guaranteed across Gini, Concentration, Asymmetry, and
Smoothness (AUDIT_REPORT.md section 3.3).
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits

from .signal_noise import compute_sn, sn_mask


def make_continuum_mask(cont_flux: np.ndarray, cont_err: np.ndarray, sn_cut: float = 3.0):
    """
    Build the shared mask from the continuum flux/error maps.

    Keeps pixels where flux and error are finite, error > 0, flux > 0, and
    S/N >= sn_cut.
    """
    sn_map = compute_sn(cont_flux, cont_err, mode="error")
    mask = sn_mask(cont_flux, cont_err, sn_cut=sn_cut, mode="error", positive_flux=True)
    return mask, sn_map


def apply_mask_2d(data_2d: np.ndarray, mask: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Apply a 2D mask to a 2D array, zero-filling everything outside it."""
    data_2d = np.array(data_2d, dtype=float)
    data_2d[~np.isfinite(data_2d)] = fill_value
    return np.where(mask, data_2d, fill_value)


def apply_mask_3d(data_3d: np.ndarray, mask: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Apply a 2D mask to every slice of a (n_slice, ny, nx) cube."""
    data_3d = np.array(data_3d, dtype=float)
    data_3d[~np.isfinite(data_3d)] = fill_value

    if data_3d.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {data_3d.shape}")
    if data_3d.shape[1:] != mask.shape:
        raise ValueError(f"Mask shape {mask.shape} incompatible with cube shape {data_3d.shape}")

    return np.where(mask[np.newaxis, :, :], data_3d, fill_value)


def save_masked_continuum(output_path, masked_flux, masked_err, flux_header=None, err_header=None):
    """Write masked continuum FITS: HDU0 = flux, HDU1 = error, shape (ny, nx)."""
    flux_header = flux_header or fits.Header()
    err_header = err_header or fits.Header()

    primary_hdu = fits.PrimaryHDU(data=masked_flux.astype(np.float32), header=flux_header)
    err_hdu = fits.ImageHDU(data=masked_err.astype(np.float32), header=err_header)
    fits.HDUList([primary_hdu, err_hdu]).writeto(output_path, overwrite=True)


def save_masked_halpha(output_path, masked_flux_cube, masked_err_cube, flux_header=None, err_header=None):
    """Write masked Halpha FITS: HDU0 = flux cube, HDU1 = error cube, shape (2, ny, nx)."""
    flux_header = flux_header or fits.Header()
    err_header = err_header or fits.Header()

    primary_hdu = fits.PrimaryHDU(data=masked_flux_cube.astype(np.float32), header=flux_header)
    err_hdu = fits.ImageHDU(data=masked_err_cube.astype(np.float32), header=err_header)
    fits.HDUList([primary_hdu, err_hdu]).writeto(output_path, overwrite=True)
