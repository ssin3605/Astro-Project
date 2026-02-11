# This file is responsible for turning spatial maps into numbers
# And non-parametric morphology

import numpy as np


def gini_from_map(fluxmap, fluxerr=None, sn_cut=3.0):
    # importing 2D Halpha fluxmap, with room for signal-noise cutting/threshold (pixel selection)
    """
    Computes the Gini coefficient from a 2D flux map.

    Parameters
    fluxmap : 2D numpy array
        Flux map (e.g. Halpha).
    fluxerr : 2D numpy array, optional
        Flux error map.
    sn_cut : float
        Signal-to-noise threshold for including pixels.

    Returns
    gini : float
        Gini coefficient (0 = uniform, 1 = all flux in one pixel).
    """

    flux = fluxmap.flatten()
    #flattens data into just a distribution of values to treat each pixel as an individual agent

    # Masking invalid values as Gini undefined for negative values and negative flux has no physical meaning for SF
    valid = np.isfinite(flux) & (flux > 0)

    # signal to noise cut (need to evolve)
    if fluxerr is not None:
        err = fluxerr.flatten()
        valid &= (flux / err) >= sn_cut

    flux = flux[valid]

    if len(flux) == 0:
        return np.nan
        # what to return if no pixels make it through the data cuts

    # Sort fluxes, ix_(i)​
    flux_sorted = np.sort(flux)
    n = len(flux_sorted)

    # Gini coefficient formula
    index = np.arange(1, n + 1) # weighting brighter pixels more heavily 
    gini = (2.0 * np.sum(index * flux_sorted) / (n * np.sum(flux_sorted))) - (n + 1) / n

    return gini
