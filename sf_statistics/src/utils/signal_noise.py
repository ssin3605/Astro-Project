"""
Signal-to-noise helpers used to build the continuum S/N mask.
"""

from __future__ import annotations

import numpy as np


def compute_sn(flux: np.ndarray, uncertainty: np.ndarray, mode: str = "error") -> np.ndarray:
    """
    Compute a S/N map.

    mode: "error" (uncertainty layer is 1-sigma error), "variance" (sigma^2),
    or "ivar" (1/sigma^2).
    """
    flux = np.asarray(flux)
    u = np.asarray(uncertainty)

    sn = np.zeros_like(flux, dtype=float)
    valid = np.isfinite(flux) & np.isfinite(u) & (u > 0)

    if mode == "error":
        sn[valid] = flux[valid] / u[valid]
    elif mode == "variance":
        sn[valid] = flux[valid] / np.sqrt(u[valid])
    elif mode == "ivar":
        sn[valid] = flux[valid] * np.sqrt(u[valid])
    else:
        raise ValueError("mode must be 'error', 'variance', or 'ivar'")

    return sn


def sn_mask(
    flux: np.ndarray,
    uncertainty: np.ndarray,
    sn_cut: float = 3.0,
    mode: str = "error",
    positive_flux: bool = True,
) -> np.ndarray:
    """Boolean mask of pixels passing the S/N cut."""
    flux = np.asarray(flux, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)

    sn = compute_sn(flux, uncertainty, mode=mode)

    mask = (
        np.isfinite(flux)
        & np.isfinite(uncertainty)
        & (uncertainty > 0)
        & np.isfinite(sn)
        & (sn >= sn_cut)
    )

    if positive_flux:
        mask &= flux > 0

    return mask
