import numpy as np
from scipy.optimize import curve_fit

def lee2015_sfms(log_mstar, S0, a1, Mt, t=13.5):
    """
    Lee et al. (2015) turnover SFMS functional form.
    Used by Fraser-McKelvie+2021 and Tuntipong+2024.

    log(SFR) = S0 - a1*t - log10(1 + 10^(Mt - M))

    Parameters
    ----------
    log_mstar : array-like
        Log stellar mass values.
    S0, a1, Mt : float
        Free parameters to be fit.
    t : float
        Age of the universe in Gyr. Fixed at 13.5 (Tuntipong+2024).

    Returns
    -------
    log_sfr_ms : array
        Predicted log(SFR) on the main sequence at each log_mstar.
    """
    return S0 - a1 * t - np.log10(1 + 10**(Mt - log_mstar))


def fit_ridge_line(log_mstar, log_sfr,
                   mass_range=(9.0, 11.0), bin_width=0.2,
                   min_galaxies=10, n_sfr_bins=20,
                   sfr_floor=-1.0):
    """
    Fit the Lee+2015 form to the ridge line of the data.
    Follows Tuntipong+2024: peak of log(SFR) histogram per mass bin.

    Parameters
    ----------
    log_mstar, log_sfr : array-like
    mass_range : tuple
        (min, max) log stellar mass range to bin over.
        Upper limit reduced to 11.0 to avoid quenched-dominated bins.
    bin_width : float
        Width of each mass bin in dex.
    min_galaxies : int
        Minimum galaxies required in a bin to use it in the fit.
    n_sfr_bins : int
        Number of histogram bins used to find the SFR peak per mass bin.
    sfr_floor : float
        Minimum log(SFR) to include in each bin before finding the peak.
        Excludes quenched galaxies from pulling the histogram peak down.
        Default -1.0 is appropriate for Hα-based SFRs.

    Returns
    -------
    sfms_params : array [S0, a1, Mt]
    ridge_mass  : array
    ridge_sfr   : array
    """
    mass_bins = np.arange(mass_range[0], mass_range[1], bin_width)
    ridge_mass, ridge_sfr = [], []

    for m_low in mass_bins:
        m_high = m_low + bin_width
        mask = (
            (log_mstar >= m_low) & (log_mstar < m_high) &
            np.isfinite(log_sfr) &
            (log_sfr >= sfr_floor)          # ← exclude quenched tail
        )
        if mask.sum() < min_galaxies:
            continue

        hist, edges = np.histogram(log_sfr[mask], bins=n_sfr_bins)
        peak_idx = np.argmax(hist)
        peak_sfr = (edges[peak_idx] + edges[peak_idx + 1]) / 2

        ridge_mass.append(m_low + bin_width / 2)
        ridge_sfr.append(peak_sfr)

    ridge_mass = np.array(ridge_mass)
    ridge_sfr  = np.array(ridge_sfr)

    sfms_params, _ = curve_fit(
        lee2015_sfms,
        ridge_mass, ridge_sfr,
        p0=[2.9, 0.2, 9.9],
        maxfev=10000                        # ← allow more iterations
    )

    return sfms_params, ridge_mass, ridge_sfr


def compute_delta_ms(log_mstar, log_sfr, sfms_params):
    """
    Compute vertical offset of each galaxy from the SFMS ridge line.

    ∆MS > 0  →  above main sequence (elevated star formation)
    ∆MS < 0  →  below main sequence (suppressed star formation)
    ∆MS = 0  →  on the main sequence

    Parameters
    ----------
    log_mstar : array-like
        Log stellar mass for all galaxies.
    log_sfr : array-like
        Log SFR (Hα-based) for all galaxies.
    sfms_params : array [S0, a1, Mt]
        Best-fit parameters returned by fit_ridge_line().

    Returns
    -------
    delta_ms : array
        ∆MS value for each galaxy. NaN preserved for missing SFR entries.
    """
    S0, a1, Mt = sfms_params
    log_sfr_ms  = lee2015_sfms(log_mstar, S0, a1, Mt)
    delta_ms    = np.array(log_sfr) - log_sfr_ms

    # Preserve NaNs from the input rather than producing a spurious offset
    delta_ms[~np.isfinite(log_sfr)] = np.nan

    return delta_ms