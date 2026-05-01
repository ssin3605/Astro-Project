"""
================================================================================
SAMI GALAXY SURVEY — ENVIRONMENTAL ANALYSIS PLOTTING SCAFFOLD
================================================================================
Structure:
  1. IMPORTS & SETTINGS
  2. DATA LOADING
  3. ══════════════════════════════════════════
     PLOT CONFIGURATION  ← change things here
  ══════════════════════════════════════════
  4. DERIVED QUANTITIES & CUTS
  5. PLOTTING FUNCTIONS
  6. MAIN EXECUTION

Quick-start: edit Section 3, then run the script.
================================================================================
"""

# ── 1. IMPORTS & SETTINGS ─────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import warnings
warnings.filterwarnings("ignore")

# Consistent figure aesthetics
plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "serif",
    "axes.labelsize":    13,
    "axes.titlesize":    14,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   10,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})


# ── 2. DATA LOADING ────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the SAMI catalogue.
    Adjust the reader (read_csv / read_fits / etc.) to match your file format.
    Expected minimum columns (rename inside this function if yours differ):
        catid, logMstar, logSFR, logsSFR,
        environment, n_neighbours,
        gini_ha, gini_cont, concentration, asymmetry, clumpiness
    """
    # ── swap this block for your actual loader ─────────────────────────────
    df = pd.read_csv(filepath)          # CSV example
    # from astropy.table import Table
    # df = Table.read(filepath).to_pandas()   # FITS example
    # ───────────────────────────────────────────────────────────────────────

    # ── column rename map — update keys to match your catalogue ───────────
    rename = {
        # "catalogue_name": "scaffold_name",
        # e.g. "log_mstar":  "logMstar",
    }
    df = df.rename(columns=rename)

    return df


# ── 3. PLOT CONFIGURATION ══════════════════════════════════════════════════════
#
#   Change the values below; the rest of the script adapts automatically.
#
# ══════════════════════════════════════════════════════════════════════════════

# ── 3a. DATA FILE ─────────────────────────────────────────────────────────────

DATA_PATH = "master/outputs/tables/master_results.csv"          # ← path to your catalogue


# ── 3b. ENVIRONMENT FILTER ────────────────────────────────────────────────────
#
#   Set ENVIRONMENT_COL to the column that labels environment.
#   Set ENVIRONMENT_SELECT to the value(s) you want to keep.
#   Set to None to use the full sample.
#
#   Examples:
#       ENVIRONMENT_COL    = "environment"
#       ENVIRONMENT_SELECT = "field"           # single group
#       ENVIRONMENT_SELECT = ["group", "cluster"]  # multiple groups

ENVIRONMENT_COL    = "spectral_class"        # ← column name in your catalogue
ENVIRONMENT_SELECT = 'SF'                 # ← None | str | list[str]

#   Secondary filter: number of nearby neighbours
#   Set to None to ignore, or (min, max) tuple to apply a range cut.
#   Example: N_NEIGHBOURS_RANGE = (1, 5)  → keep galaxies with 1–5 neighbours

N_NEIGHBOURS_COL   = "n_neighbours"       # ← column name (or None to skip)
N_NEIGHBOURS_RANGE = None                 # ← None | (min, max)


# ── 3c. AXES ──────────────────────────────────────────────────────────────────

X_COL   = "log_Mstar"      # ← x-axis column  (log stellar mass is default)
Y_COL   = "log_sSFR"       # ← y-axis column  (log sSFR is default)

X_LABEL = r"$\log\,(M_\star\,/\,M_\odot)$"
Y_LABEL = r"$\log\,(\mathrm{sSFR}\,/\,\mathrm{yr}^{-1})$"


# ── 3d. COLOUR METRIC ─────────────────────────────────────────────────────────
#
#   The column whose value colours each data point.
#
#   Built-in options (match your catalogue column names):
#       "gini_all_halpha"        Gini coefficient — Hα emission
#       "CIndex"  CAS concentration
#       "A_min"      CAS asymmetry
#       "S"     CAS clumpiness/smoothness
#       "log_SFR"         log SFR
#       "log_Mstar"       log M★  (useful when Y_COL is something else)
#
#   You can also type any other numeric column from your catalogue.

COLOUR_COL   = "gini_all_halpha"            # ← column to colour by
COLOUR_LABEL = "Gini (Hα)"         # ← colourbar label
CMAP         = "plasma"             # ← matplotlib colourmap name

# Colourbar limits — set to None for automatic (min/max of data)
CBAR_VMIN = None                    # ← float | None
CBAR_VMAX = None                    # ← float | None


# ── 3e. sSFR CUT ──────────────────────────────────────────────────────────────
#
#   Horizontal reference line on the plot AND an optional hard data cut.
#
#   SSFR_LINE_VALUE : draws a dashed line at this y-value if Y_COL is logsSFR.
#                     Set to None to suppress the line.
#   SSFR_CUT_APPLY  : if True, *removes* galaxies below the cut from the plot.
#   SSFR_CUT_COL    : column used for the cut (default matches Y_COL).
#
#   Example — main-sequence cut:
#       SSFR_LINE_VALUE = -10.8
#       SSFR_CUT_APPLY  = True

SSFR_LINE_VALUE = -12             # ← float | None
SSFR_CUT_APPLY  = True             # ← True = hard cut | False = line only
SSFR_CUT_COL    = "log_sSFR"        # ← column used for the cut


# ── 3f. POINT STYLE ───────────────────────────────────────────────────────────

MARKER_SIZE  = 20          # scatter point size (pt²)
MARKER_ALPHA = 0.75        # transparency (0–1)
MARKER_STYLE = "o"         # matplotlib marker string


# ── 3g. OUTPUT ────────────────────────────────────────────────────────────────

SAVE_FIGURE  = True
OUTPUT_PATH  = "master/outputs/figures/sami_plot.png"      # ← .pdf | .png | .svg
SHOW_FIGURE  = True

# ══ END OF CONFIGURATION ══════════════════════════════════════════════════════


# ── 4. DERIVED QUANTITIES & CUTS ───────────────────────────────────────────────

def apply_cuts(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all filters defined in Section 3 and return the surviving subset."""

    n_start = len(df)

    # Environment filter
    if ENVIRONMENT_SELECT is not None and ENVIRONMENT_COL in df.columns:
        targets = (
            [ENVIRONMENT_SELECT]
            if isinstance(ENVIRONMENT_SELECT, str)
            else list(ENVIRONMENT_SELECT)
        )
        df = df[df[ENVIRONMENT_COL].isin(targets)]
        print(f"  Environment cut  ({ENVIRONMENT_COL} ∈ {targets}): "
              f"{n_start} → {len(df)}")

    # Neighbour range filter
    if (N_NEIGHBOURS_RANGE is not None
            and N_NEIGHBOURS_COL is not None
            and N_NEIGHBOURS_COL in df.columns):
        lo, hi = N_NEIGHBOURS_RANGE
        df = df[df[N_NEIGHBOURS_COL].between(lo, hi)]
        print(f"  Neighbour cut    ({N_NEIGHBOURS_COL} ∈ [{lo}, {hi}]): "
              f"→ {len(df)}")

    # sSFR hard cut
    if SSFR_CUT_APPLY and SSFR_LINE_VALUE is not None and SSFR_CUT_COL in df.columns:
        df = df[df[SSFR_CUT_COL] >= SSFR_LINE_VALUE]
        print(f"  sSFR cut         ({SSFR_CUT_COL} ≥ {SSFR_LINE_VALUE}): "
              f"→ {len(df)}")

    # Drop rows with NaN in the columns we actually need
    required = [c for c in [X_COL, Y_COL, COLOUR_COL] if c in df.columns]
    df = df.dropna(subset=required)
    print(f"  After NaN drop: {len(df)} galaxies remain\n")

    return df


# ── 5. PLOTTING FUNCTIONS ──────────────────────────────────────────────────────

def build_title(df: pd.DataFrame) -> str:
    """Construct a descriptive plot title from the active configuration."""
    parts = []

    if ENVIRONMENT_SELECT is not None:
        env = (ENVIRONMENT_SELECT
               if isinstance(ENVIRONMENT_SELECT, str)
               else " + ".join(ENVIRONMENT_SELECT))
        parts.append(env.capitalize())
    else:
        parts.append("All environments")

    if N_NEIGHBOURS_RANGE is not None and N_NEIGHBOURS_COL is not None:
        parts.append(f"{N_NEIGHBOURS_COL} ∈ [{N_NEIGHBOURS_RANGE[0]}, "
                     f"{N_NEIGHBOURS_RANGE[1]}]")

    if SSFR_LINE_VALUE is not None:
        tag = "cut" if SSFR_CUT_APPLY else "line"
        parts.append(f"sSFR {tag} = {SSFR_LINE_VALUE}")

    parts.append(f"colour: {COLOUR_LABEL}")
    parts.append(f"N = {len(df):,}")

    return "  |  ".join(parts)


def make_scatter_plot(df: pd.DataFrame) -> plt.Figure:
    """Produce the main scatter plot."""

    # Colour normalisation
    vmin = CBAR_VMIN if CBAR_VMIN is not None else df[COLOUR_COL].quantile(0.02)
    vmax = CBAR_VMAX if CBAR_VMAX is not None else df[COLOUR_COL].quantile(0.98)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(CMAP)

    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(
        df[X_COL],
        df[Y_COL],
        c=df[COLOUR_COL],
        cmap=cmap,
        norm=norm,
        s=MARKER_SIZE,
        alpha=MARKER_ALPHA,
        marker=MARKER_STYLE,
        linewidths=0,
        zorder=3,
    )

    # Colourbar
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        pad=0.02,
        fraction=0.04,
    )
    cbar.set_label(COLOUR_LABEL, fontsize=12)

    # sSFR reference line
    if SSFR_LINE_VALUE is not None and Y_COL == SSFR_CUT_COL:
        ax.axhline(
            SSFR_LINE_VALUE,
            color="crimson",
            linestyle="--",
            linewidth=1.4,
            label=f"sSFR = {SSFR_LINE_VALUE}",
            zorder=4,
        )
        ax.legend(loc="upper left")

    # Labels & title
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_title(build_title(df), fontsize=11, pad=10)

    fig.tight_layout()
    return fig


# ── 6. MAIN EXECUTION ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SAMI Environmental Analysis — Plotting Scaffold")
    print("=" * 60)

    # Load
    print(f"\nLoading data from: {DATA_PATH}")
    df_raw = load_data(DATA_PATH)
    print(f"  Catalogue loaded: {len(df_raw):,} galaxies, "
          f"{df_raw.shape[1]} columns\n")

    # Cut
    print("Applying cuts:")
    df = apply_cuts(df_raw)

    # Plot
    print("Generating plot …")
    fig = make_scatter_plot(df)

    if SAVE_FIGURE:
        fig.savefig(OUTPUT_PATH, bbox_inches="tight")
        print(f"  Saved → {OUTPUT_PATH}")

    if SHOW_FIGURE:
        plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()