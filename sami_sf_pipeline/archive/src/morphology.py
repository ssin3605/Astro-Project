# This file is responsible for turning spatial maps of flux values into calculated numbers
# And non-parametric morphology

import numpy as np

## GINI CALCULATIONS
def gini_from_map(fluxmap, fluxerr=None, sn_cut=3.0):
    """
    Computes the Gini coefficient from a 2D flux map.
    """
    fluxmap = np.asarray(fluxmap)

    if fluxmap.ndim != 2:
        raise ValueError(f"fluxmap must be 2D (ny, nx). Got shape {fluxmap.shape}")

    flux = fluxmap.ravel()

    # baseline valid data must have finite and positive flux
    valid = np.isfinite(flux) & (flux > 0)

    if fluxerr is not None:
        fluxerr = np.asarray(fluxerr)
        if fluxerr.shape != fluxmap.shape:
            raise ValueError(f"fluxerr shape {fluxerr.shape} must match fluxmap shape {fluxmap.shape}")

        err = fluxerr.ravel()

        # require finite, positive errors
        valid &= np.isfinite(err) & (err > 0)

        # apply S/N cut safely
        valid &= (flux / err) >= sn_cut

    flux = flux[valid]

    if flux.size == 0:
        return np.nan

    flux_sorted = np.sort(flux)
    n = flux_sorted.size

    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * flux_sorted) / (n * np.sum(flux_sorted))) - (n + 1) / n
    return gini