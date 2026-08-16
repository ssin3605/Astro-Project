import numpy as np

def gini_from_mask(fluxmap: np.ndarray,
                   mask: np.ndarray,
                   min_pixels: int = 10) -> float:
    """
    Computing Gini coefficient from flux values where mask == True.

    Parameters
    fluxmap : 2D array, flux map
    mask : 2D boolean array, pixels to include
    min_pixels : int, minimum number of valid pixels

    Returns
    float, Gini or NaN if insufficient pixels
    """

    flux = np.asarray(fluxmap)
    mask = np.asarray(mask)

    # Apply mask
    x = flux[mask]

    # Clean values
    x = x[np.isfinite(x) & (x > 0)]

    if x.size < min_pixels:
        return np.nan

    x_sorted = np.sort(x)
    n = x_sorted.size
    i = np.arange(1, n + 1)

    denom = n * np.sum(x_sorted)
    if denom <= 0:
        return np.nan

    g = (2.0 * np.sum(i * x_sorted) / denom) - ((n + 1.0) / n)

    return float(g)
