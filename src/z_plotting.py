# src/plotting.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd       
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.sfms       import lee2015_sfms
from src.z_samples    import get_sample
from src.z_plot_config import (AXIS_LABELS, CMAPS, VALID_X, VALID_Y, VALID_C,
                              GALCLASS_ORDER, GALCLASS_COLOURS, GALCLASS_LABELS)


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
    Core scatter plot function. Driven entirely by config dict.

    Parameters
    ----------
    df : DataFrame
        Full master_sfms dataframe (unfiltered).
    sfms_params : array [S0, a1, Mt]
        SFMS best-fit parameters from sfms_prep.py.
    config : dict with keys:
        x          : str   column for x-axis
        y          : str   column for y-axis
        colour_by  : str   column for colour scale
        sample     : str   key from SAMPLE_REGISTRY (default "all")
        sfr_cut    : float minimum log(SFR) filter (default None)
        show_sfms  : bool  overlay SFMS line (default True)
        show_sfms_bands : bool  overlay ±0.3 dex bands (default False)
        title      : str   optional plot title
        vmin/vmax  : float optional colour scale limits
    ax : matplotlib Axes or None
        If None, a new figure is created.

    Returns
    -------
    fig, ax
    """
    _validate_config(config)

    x_col      = config["x"]
    y_col      = config["y"]
    colour_col = config["colour_by"]
    sample_key = config.get("sample",    "all")
    min_sfr    = config.get("sfr_cut",   None)
    show_sfms  = config.get("show_sfms", True)
    show_bands = config.get("show_sfms_bands", False)
    title      = config.get("title",     "")

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

    # ── SFMS overlay (only meaningful when y=log_SFR, x=log_Mstar) ──
    if show_sfms and x_col == "log_Mstar" and y_col == "log_SFR":
        mass_range = np.linspace(
            df_plot[x_col].min() - 0.1,
            df_plot[x_col].max() + 0.1,
            300
        )
        sfms_line = lee2015_sfms(mass_range, *sfms_params)

        ax.plot(mass_range, sfms_line,
                color="gold", lw=2, ls="-",
                label="SFMS", zorder=5)

        if show_bands:
            ax.fill_between(mass_range,
                            sfms_line - 0.3, sfms_line + 0.3,
                            color="gold", alpha=0.15, zorder=4)

        ax.legend(fontsize=8, loc="upper left")

    # ── Labels ───────────────────────────────────────────────────
    ax.set_xlabel(AXIS_LABELS.get(x_col, x_col), fontsize=11)
    ax.set_ylabel(AXIS_LABELS.get(y_col, y_col), fontsize=11)

    if title:
        ax.set_title(title, fontsize=11)

    fig.tight_layout()
    return fig, ax