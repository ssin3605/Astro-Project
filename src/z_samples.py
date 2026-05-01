# Every sample used in my analysis's definition lives here as a function 
# that takes the full dataframe and returns a boolean mask. Nothing else.
import numpy as np
import pandas as pd


def all_galaxies(df):
    """No filtering — full master_sfms sample."""
    return pd.Series(np.ones(len(df), dtype=bool), index=df.index)


def wang_only(df):
    """Galaxies present in the Wang+2022 concentration SF catalogue."""
    return df["in_wang_conc"] == True


def wang_sf_only(df):
    """Wang+2022 galaxies that are also spectroscopically star-forming."""
    return wang_only(df) & (df["spectral_class"] == "SF")


def sf_only(df):
    """All spectroscopically classified SF galaxies across any source."""
    return df["spectral_class"] == "SF"


def close_pairs_100kpc(df):
    """Galaxies with a close pair within 100 kpc."""
    return df["gama_close_pair_100kpc"] == True


def close_pairs_50kpc(df):
    """Galaxies with a close pair within 50 kpc."""
    return df["gama_close_pair_50kpc"] == True


def gama_region(df):
    """Galaxies in the GAMA environment field."""
    return df["environment"] == "GAMA"


def cluster_region(df):
    """Galaxies in the Cluster environment field."""
    return df["environment"] == "Cluster"


def sfr_cut(df, min_log_sfr=-2.0):
    """
    Galaxies above a minimum log(SFR) threshold.
    Combine with other filters as needed.
    """
    return df["log_SFR"] > min_log_sfr


# ── Sample registry ───────────────────────────────────────────────────────────
# Maps string names used in plot configs to filter functions.
# Filters can be combined with & in the notebook or here as named combos.

SAMPLE_REGISTRY = {
    "all"              : all_galaxies,
    "wang"             : wang_only,
    "wang_sf"          : wang_sf_only,
    "sf_only"          : sf_only,
    "close_pairs_100"  : close_pairs_100kpc,
    "close_pairs_50"   : close_pairs_50kpc,
    "gama"             : gama_region,
    "cluster"          : cluster_region,
}


def get_sample(df, sample_key, min_log_sfr=None):
    """
    Apply a named sample filter to df.

    Parameters
    ----------
    df : DataFrame
    sample_key : str
        Key from SAMPLE_REGISTRY, or "all".
    min_log_sfr : float or None
        If provided, additionally applies an SFR floor cut.

    Returns
    -------
    df_filtered : DataFrame
        Subset of df passing the filter.
    """
    if sample_key not in SAMPLE_REGISTRY:
        raise ValueError(
            f"Unknown sample '{sample_key}'. "
            f"Valid options: {list(SAMPLE_REGISTRY.keys())}"
        )

    mask = SAMPLE_REGISTRY[sample_key](df)

    if min_log_sfr is not None:
        mask = mask & sfr_cut(df, min_log_sfr)

    return df[mask].copy()