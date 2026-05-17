# src/plotting.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd       
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.sfms       import lee2015_sfms
from src.z_samples    import get_sample
from src.z_plot_config import (AXIS_LABELS, CMAPS, VALID_X, VALID_Y, VALID_C,
                              GALCLASS_ORDER, GALCLASS_COLOURS, GALCLASS_LABELS)

def draw_highlights(ax, df_plot, x_col, y_col, catids):
    """
    Circle up to 4 galaxies by CATID on an existing axes.
    Each galaxy gets a distinct hollow marker and a CATID label.

    Parameters
    ----------
    ax      : matplotlib Axes
    df_plot : DataFrame   already filtered plotting sample
    x_col   : str         x-axis column
    y_col   : str         y-axis column
    catids  : list        up to 4 CATID values (int or str)
    """
    from src.z_plot_config import HIGHLIGHT_STYLES

    if not catids:
        return

    legend_elements = []

    for i, catid in enumerate(catids[:4]):
        style = HIGHLIGHT_STYLES[i]
        row   = df_plot[df_plot["CATID"].astype(str) == str(catid)]

        if row.empty:
            print(f"  [highlight] CATID {catid} not found in current sample")
            continue

        xval = row[x_col].values[0]
        yval = row[y_col].values[0]

        ax.scatter(xval, yval,
                   marker=style["marker"],
                   s=120,
                   facecolors="none",
                   edgecolors="black",
                   linewidths=1.8,
                   zorder=10)

        ax.annotate(str(catid),
                    xy=(xval, yval),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=7,
                    fontweight="bold",
                    color="black",
                    zorder=11)

        legend_elements.append(
            plt.scatter([], [],
                        marker=style["marker"],
                        s=60,
                        facecolors="none",
                        edgecolors="black",
                        linewidths=1.5,
                        label=str(catid))
        )

    if legend_elements:
        ax.legend(handles=legend_elements,
                  title="Highlighted CATID",
                  fontsize=7,
                  title_fontsize=7,
                  loc="upper left")


def _validate_config(config):
    """Raise informative errors for invalid axis/colour choices."""
    x = config.get("x")
    y = config.get("y")
    c = config.get("colour_by")

    if x not in VALID_X:
        raise ValueError(f"x='{x}' not valid. Choose from {VALID_X}")
    if y not in VALID_Y:
        raise ValueError(f"y='{y}' not valid. Choose from {VALID_Y}")
    if c not in VALID_C:
        raise ValueError(f"colour_by='{c}' not valid. Choose from {VALID_C}")


def plot_scatter(df, sfms_params, config, ax=None):
    """
    Core scatter plot function driven entirely by config dict.

    Config keys
    -----------
    x                : str    x-axis column
    y                : str    y-axis column
    colour_by        : str    colour scale column
    sample           : str    key from SAMPLE_REGISTRY (default "all")
    sfr_cut          : float  minimum log(SFR) filter (default None)
    show_sfms        : bool   overlay SFMS line (default True)
    show_sfms_bands  : bool   overlay ±0.3 dex bands (default False)
    show_median_bins : bool   overlay median per mass bin (default False)
    median_sfr_floor : float  SFR floor for median bins (default -1.0)
    title            : str    optional plot title
    vmin / vmax      : float  optional colour scale limits
    """
    _validate_config(config)

    x_col      = config["x"]
    y_col      = config["y"]
    colour_col = config["colour_by"]
    sample_key = config.get("sample",    "all")
    min_sfr    = config.get("sfr_cut",   None)
    show_sfms  = config.get("show_sfms", True)
    show_bands = config.get("show_sfms_bands",  False)
    show_meds  = config.get("show_median_bins", False)
    title      = config.get("title", "")

    # ── Apply sample filter ───────────────────────────────────────
    df_plot = get_sample(df, sample_key, min_log_sfr=min_sfr)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    # ── Scatter: continuous colour scale ─────────────────────────
    if colour_col != "galclass":
        vmin = config.get("vmin", np.nanpercentile(df_plot[colour_col], 5))
        vmax = config.get("vmax", np.nanpercentile(df_plot[colour_col], 95))
        cmap = CMAPS.get(colour_col, "viridis")

        sc = ax.scatter(
            df_plot[x_col], df_plot[y_col],
            c=df_plot[colour_col],
            cmap=cmap, vmin=vmin, vmax=vmax,
            s=12, alpha=0.75, linewidths=0
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(AXIS_LABELS.get(colour_col, colour_col), fontsize=10)

    # ── Scatter: discrete galclass colour ────────────────────────
    else:
        df_plot = df_plot.copy()
        df_plot["galclass"] = pd.to_numeric(
            df_plot["galclass"], errors="coerce"
        ).astype("Int64")

        for gc in GALCLASS_ORDER:
            mask = df_plot["galclass"] == gc
            if mask.sum() == 0:
                continue
            ax.scatter(
                df_plot.loc[mask, x_col],
                df_plot.loc[mask, y_col],
                color=GALCLASS_COLOURS[gc],
                label=GALCLASS_LABELS[gc],
                s=12, alpha=0.75, linewidths=0
            )
        ax.legend(fontsize=8, loc="upper left",
                  title="Environment", title_fontsize=8)

    # ── SFMS line: log_SFR vs log_Mstar ───────────────────────────
    if show_sfms and x_col == "log_Mstar" and y_col == "log_SFR":
        mass_range = np.linspace(
            df_plot[x_col].min() - 0.1,
            df_plot[x_col].max() + 0.1,
            300
        )
        sfms_line = lee2015_sfms(mass_range, *sfms_params)

        ax.plot(mass_range, sfms_line,
                color="gold", lw=2, ls="-",
                label="SFMS (Lee+2015 fit)", zorder=5)

        if show_bands:
            ax.fill_between(mass_range,
                            sfms_line - 0.3, sfms_line + 0.3,
                            color="gold", alpha=0.15, zorder=4)

    # ── SFMS line: log_sSFR vs log_Mstar ──────────────────────────
    if show_sfms and x_col == "log_Mstar" and y_col == "log_sSFR":
        mass_range = np.linspace(
            df_plot[x_col].min() - 0.1,
            df_plot[x_col].max() + 0.1,
            300
        )
        sfms_ssfr = lee2015_sfms(mass_range, *sfms_params) - mass_range
        ax.plot(mass_range, sfms_ssfr,
                color="gold", lw=2, ls="-",
                label="SFMS", zorder=5)

    # ── Median bin overlay ────────────────────────────────────────
    if show_meds and x_col == "log_Mstar" and y_col == "log_SFR":
        sfr_floor = config.get("median_sfr_floor", -1.0)
        bins      = np.arange(8.0, 11.8, 0.2)
        bin_mids  = bins[:-1] + 0.1
        medians, valid_mid = [], []

        for i in range(len(bins) - 1):
            mask = (
                (df_plot[x_col] >= bins[i]) &
                (df_plot[x_col] <  bins[i + 1]) &
                np.isfinite(df_plot[y_col]) &
                (df_plot[y_col] >= sfr_floor)
            )
            if mask.sum() >= 5:
                medians.append(np.median(df_plot.loc[mask, y_col]))
                valid_mid.append(bin_mids[i])

        ax.plot(valid_mid, medians,
                color="#2d8a4e", lw=1.5, ls="--",
                marker="o", markersize=4,
                label="Median per bin", zorder=6)

    # ── Legend for SFMS/median overlays ───────────────────────────
    if x_col == "log_Mstar" and y_col in ("log_SFR", "log_sSFR"):
        ax.legend(fontsize=8, loc="upper left")

    # ── Highlights ───────────────────────────────────────────────
    highlight_ids = config.get("highlight", [])
    if highlight_ids and x_col in ("log_Mstar",) and \
       y_col in ("log_SFR", "log_sSFR"):
        draw_highlights(ax, df_plot, x_col, y_col, highlight_ids)


    # ── Axis labels and title ─────────────────────────────────────
    ax.set_xlabel(AXIS_LABELS.get(x_col, x_col), fontsize=11)
    ax.set_ylabel(AXIS_LABELS.get(y_col, y_col), fontsize=11)
    if title:
        ax.set_title(title, fontsize=11)

    fig.tight_layout()
    return fig, ax