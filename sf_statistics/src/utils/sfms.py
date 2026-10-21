"""
Star-forming main sequence (SFMS) fitting with the Lee et al. (2015) turnover
form, fit to the ridge line of the data (the peak of the log(SFR) histogram
in each mass bin; Tuntipong et al. 2024).

Used by src/05_sfms.py, which fits the master catalogue and computes each
galaxy's offset from the fitted line (delta_ms).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def lee2015_sfms(log_mstar, S0, a1, Mt, t=13.5):
    """
    Lee et al. (2015) turnover SFMS functional form.

        log(SFR) = S0 - a1*t - log10(1 + 10^(Mt - M))

    Parameters
    ----------
    log_mstar : array-like
    S0, a1, Mt : float
        Free parameters to be fit.
    t : float
        Age of the universe in Gyr. Fixed at 13.5 (Tuntipong+2024).
    """
    return S0 - a1 * t - np.log10(1 + 10 ** (Mt - log_mstar))


def fit_ridge_line(
    log_mstar, log_sfr,
    mass_range=(9.0, 11.0), bin_width=0.2,
    min_galaxies=10, n_sfr_bins=20,
    sfr_floor=-1.0,
    bounds=((0.0, 0.0, 8.0), (20.0, 2.0, 12.0)),
):
    """
    Fit the Lee+2015 form to the ridge line of the data: the peak of the
    log(SFR) histogram per mass bin (Tuntipong+2024 approach).

    Parameters
    ----------
    log_mstar, log_sfr : array-like
    mass_range : tuple
        (min, max) log stellar mass range to bin over. Upper limit kept at
        11.0 to avoid quenched-dominated high-mass bins pulling the fit down.
    bin_width : float
        Width of each mass bin in dex.
    min_galaxies : int
        Minimum galaxies required in a bin to use it in the fit.
    n_sfr_bins : int
        Number of histogram bins used to find the SFR peak per mass bin.
    sfr_floor : float
        Minimum log(SFR) to include in each bin before finding the peak -
        excludes quenched galaxies from pulling the histogram peak down.
        Default -1.0 is appropriate for Hα-based SFRs.
    bounds : (lower, upper) tuples for (S0, a1, Mt)
        lee2015_sfms fixes t=13.5 for every fit point (this repo only ever
        fits a single cosmic epoch, never varies t), which makes S0 and a1
        individually non-identifiable - curve_fit can trade them off freely
        along the line S0 - a1*t = const without changing the fitted curve
        at all. Unbounded, this let S0/a1 drift to values in the hundreds
        or thousands (verified against the original unmodified code and old
        master_results.csv - this is a pre-existing issue, not introduced by
        this port). The default bounds keep the search within physically
        plausible values (matching the old p0=[2.9, 0.2, 9.9] initial guess)
        without changing the fitted curve or residuals at all - only which
        of the infinitely many equivalent (S0, a1) pairs gets reported.

    Returns
    -------
    sfms_params : array [S0, a1, Mt]
    ridge_mass, ridge_sfr : arrays - the per-bin points the fit was anchored on
    n_used : int - number of galaxies in the mass bins used to find the ridge
    """
    log_mstar = np.asarray(log_mstar, dtype=float)
    log_sfr = np.asarray(log_sfr, dtype=float)

    mass_bins = np.arange(mass_range[0], mass_range[1], bin_width)
    ridge_mass, ridge_sfr = [], []
    n_used = 0

    for m_low in mass_bins:
        m_high = m_low + bin_width
        mask = (
            (log_mstar >= m_low) & (log_mstar < m_high)
            & np.isfinite(log_sfr)
            & (log_sfr >= sfr_floor)  # exclude quenched tail
        )
        if mask.sum() < min_galaxies:
            continue
        n_used += int(mask.sum())

        hist, edges = np.histogram(log_sfr[mask], bins=n_sfr_bins)
        peak_idx = np.argmax(hist)
        peak_sfr = (edges[peak_idx] + edges[peak_idx + 1]) / 2

        ridge_mass.append(m_low + bin_width / 2)
        ridge_sfr.append(peak_sfr)

    ridge_mass = np.array(ridge_mass)
    ridge_sfr = np.array(ridge_sfr)

    sfms_params, _ = curve_fit(
        lee2015_sfms, ridge_mass, ridge_sfr,
        p0=[2.9, 0.2, 9.9], maxfev=10000,
        bounds=bounds,
    )

    return sfms_params, ridge_mass, ridge_sfr, n_used


def compute_delta_ms(log_mstar, log_sfr, sfms_params):
    """
    Vertical offset of each galaxy from the fitted SFMS ridge line.

    delta_ms > 0 -> above the main sequence (elevated star formation)
    delta_ms < 0 -> below the main sequence (suppressed star formation)

    NaN is preserved wherever log_sfr is not finite, rather than producing
    a spurious offset.
    """
    log_mstar = np.asarray(log_mstar, dtype=float)
    log_sfr = np.asarray(log_sfr, dtype=float)

    S0, a1, Mt = sfms_params
    log_sfr_ms = lee2015_sfms(log_mstar, S0, a1, Mt)
    delta_ms = log_sfr - log_sfr_ms
    delta_ms[~np.isfinite(log_sfr)] = np.nan

    return delta_ms
