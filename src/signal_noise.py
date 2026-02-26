import numpy as np


def compute_sn(flux: np.ndarray,
               uncertainty: np.ndarray,
               mode: str = "error") -> np.ndarray:
    """
    Compute S/N map.

    Parameters
    flux : 2D array, Flux map.
    uncertainty : 2D array, Uncertainty layer.
    mode : str, How uncertainty is stored:
        - "error"      : layer = 1-sigma error
        - "variance"   : layer = sigma^2
        - "ivar"       : layer = 1/sigma^2

    Returns
    sn : 2D array, Signal-to-noise map.
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


def sn_mask(flux: np.ndarray,
            uncertainty: np.ndarray,
            sn_cut: float = 3.0,
            mode: str = "error",
            positive_flux: bool = True) -> np.ndarray:
    """
    Return boolean mask of pixels passing S/N cut.
    """

    sn = compute_sn(flux, uncertainty, mode=mode)

    mask = np.isfinite(sn) & (sn >= sn_cut)

    if positive_flux:
        mask &= (flux > 0)

    return mask
