"""
Centralised SFR/Mstar log-vs-linear detection.

AUDIT_REPORT.md flagged this heuristic as independently duplicated across
six call sites in the old repo (gini/scripts/galaxy_stats.py,
gini/scripts/representative_galaxies.py,
smoothness/scripts/representative_galaxies.py,
smoothness/scripts/smoothness_plotting.py,
asymmetry/scripts/asymmetry_plotting.py, master/scripts/build_csv.py). This
is now the single implementation; every script in this pipeline that needs
to detect whether a physical-quantity column is already log10-scaled should
import it from here rather than re-implementing it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Heuristic threshold: linear SFR/Mstar values are >> 100 in typical SAMI
# samples (Mstar in Msun, SFR ratios etc.), log10 values are single/double
# digit. This is unchanged from the original heuristic - only centralised.
LINEAR_MEDIAN_THRESHOLD = 100.0


def to_log10(series: pd.Series, name: str = "", verbose: bool = True) -> pd.Series:
    """
    Convert a Series to log10 if it looks linear (median > threshold),
    otherwise return it unchanged (assumed already log10).

    Zeros are replaced with NaN before taking log10 to avoid -inf.
    """
    valid = series.dropna()
    if valid.empty:
        if verbose:
            print(f"  {name}: no valid values - skipping log detection")
        return series

    if valid.median() > LINEAR_MEDIAN_THRESHOLD:
        if verbose:
            print(f"  {name}: raw/linear values detected (median={valid.median():.2f}) -> applying log10")
        return np.log10(series.replace(0, np.nan))

    if verbose:
        print(f"  {name}: already log-scaled (median={valid.median():.2f}) -> using as-is")
    return series


def to_log_sfr(series: pd.Series):
    """
    Same detection as to_log10, but also returns whether the input was
    already log-scaled - used by 04_build_master_csv.py to decide whether
    it needs to *also* back out the linear SFR value.
    """
    valid = series.dropna()
    if valid.empty:
        return series, False
    if valid.median() > LINEAR_MEDIAN_THRESHOLD:
        return np.log10(series.replace(0, np.nan)), False
    return series, True
