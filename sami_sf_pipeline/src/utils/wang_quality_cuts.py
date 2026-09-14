"""
Wang et al. (2022)'s six quality/purity cuts that reduce their BPT-classified
candidate catalogue down to Full_sam, the sample they actually ran CAS/
environment analysis on. concentration.csv (this pipeline's copy of Wang's
SAMI_CIndex_data catalogue, 2959 galaxies) is *not* already purity-cut - it's
the pre-selection candidate list, confirmed while investigating why this
pipeline's "galaxies classified by Wang" count (2959) didn't match the
paper's reported Full_sam (778).

Cuts, in the order Wang applies them (order doesn't change any individual
galaxy's pass/fail on any one cut - each criterion only reads that galaxy's
own columns - but does change the "N removed at this step" bookkeeping if
you report cuts sequentially, e.g. in notebooks/sample_summary.ipynb):

    1. Size       : re <= 15 arcsec
    2. Inclination : ellip <= 0.7
    3. Seeing      : seeing_re < 0.75 AND seeing < 4 arcsec
    4. Star-forming: dust-corrected integrated EW(Halpha) >= 1 Angstrom
    5. Not pure AGN: galclass == 4
    6. sSFR        : log10(sSFR/yr^-1) > -11.25, using SFR resolved by each
                      galaxy's true pure_SF/composite_SF membership (not
                      concentration.csv's own "spectral_class" column, which
                      is mislabeled - it's tagged per source file at load
                      time and is "SF" for every row merged from the SF
                      files, regardless of a galaxy's actual Wang
                      classification)

Verified against Wang's published per-cut removal counts, even though this
pipeline's base population (2959) differs from the paper's (2988): applied
sequentially to concentration.csv, inclination removes exactly 307 (paper:
307), seeing exactly 714 (paper: 714), EW exactly 783 (paper: 783), and
pure AGN exactly 121 (paper: 121) - only size differs (70 here vs. the
paper's 99, a gap that matches the 2988-2959=29 galaxies this pipeline's
base doesn't include). Confirmed 2026-08-17; counts will drift if
concentration.csv is regenerated from updated source data.

Pure-AGN exclusion (cut 5) - resolved 2026-08-17: `galclass==4` is the
right flag after all. It was initially ruled out because a flat filter on
the whole 2959-galaxy catalogue gives 545 galclass==4 galaxies, nowhere
near Wang's reported 121 - but 121 is the count of galclass==4 galaxies
*remaining after* the size/inclination/seeing/EW cuts have already run
(1085 galaxies enter this step, 121 of them are galclass==4), not a filter
against the full catalogue. Since cut order doesn't affect any individual
cut's own pass/fail value (see above), evaluating `cut_not_pure_agn`
against the full 2959 here still gives the correct per-galaxy result -
only a plain unconditional galclass==4 count against the *full* catalogue
would have been misleading, and was.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.wang import load_all_wang_files, wang_membership_sets

RE_MAX_ARCSEC = 15.0
ELLIP_MAX = 0.7
SEEING_RE_MAX = 0.75
SEEING_MAX_ARCSEC = 4.0
EW_HALPHA_MIN = 1.0
LOG_SSFR_MIN = -11.25

CUT_COLUMNS = [
    "cut_size",
    "cut_inclination",
    "cut_seeing",
    "cut_star_forming",
    "cut_not_pure_agn",
    "cut_ssfr",
]


def resolve_spectral_class(catid: pd.Series, membership: dict[str, set]) -> pd.Series:
    """pure_SF = in an SF file but not any Composite file; composite_SF = in both -
    the same definition wang_membership_sets() uses, applied here directly rather
    than trusting concentration.csv's own (mislabeled) spectral_class column."""
    return pd.Series(
        np.where(
            catid.isin(membership["pure_sf_ids"]),
            "pure_SF",
            np.where(catid.isin(membership["composite_sf_ids"]), "composite_SF", np.nan),
        ),
        index=catid.index,
    )


def resolve_log_ssfr(conc: pd.DataFrame, spectral_class: pd.Series) -> pd.Series:
    """Same SFR resolution policy as 04_build_master_csv.py: pure_SF galaxies use
    their own SFR; composite_SF galaxies use SFR_n2ha_s (the BPT-isolated
    star-forming component, mitigating AGN/LINER contamination) - "correcting
    Halpha for non star-formation contamination in non-pure_sf", per Wang (2022)."""
    sfr = np.where(
        spectral_class == "pure_SF",
        conc["SFR"],
        np.where(spectral_class == "composite_SF", conc["SFR_n2ha_s"], np.nan),
    )
    sfr = pd.Series(sfr, index=conc.index)
    return sfr - conc["Mstar"]  # both already log10 in Wang's raw catalogue


def apply_quality_cuts(conc: pd.DataFrame, membership: dict[str, set] | None = None) -> pd.DataFrame:
    """Adds spectral_class_true, one boolean cut_* column per criterion (CUT_COLUMNS),
    and in_full_sam (AND of all six) to a copy of conc. Missing/NaN inputs fail their
    cut (a galaxy Wang has no measurement for cannot be confirmed as passing it)."""
    df = conc.copy()

    if membership is None:
        membership = wang_membership_sets(load_all_wang_files())

    df["spectral_class_true"] = resolve_spectral_class(df["CATID"], membership)

    df["cut_size"] = df["re"] <= RE_MAX_ARCSEC
    df["cut_inclination"] = df["ellip"] <= ELLIP_MAX
    df["cut_seeing"] = (df["seeing_re"] < SEEING_RE_MAX) & (df["seeing"] < SEEING_MAX_ARCSEC)
    df["cut_star_forming"] = df["ha_EW"] >= EW_HALPHA_MIN

    df["cut_not_pure_agn"] = df["galclass"] != 4

    log_ssfr = resolve_log_ssfr(df, df["spectral_class_true"])
    df["cut_ssfr"] = log_ssfr.notna() & (log_ssfr > LOG_SSFR_MIN)

    df["in_full_sam"] = df[CUT_COLUMNS].all(axis=1)

    return df
