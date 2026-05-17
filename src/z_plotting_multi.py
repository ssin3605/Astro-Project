# src/z_plot_multi.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.sfms          import lee2015_sfms
from src.z_samples     import get_sample
from src.z_plot_config import (AXIS_LABELS, CMAPS, GALCLASS_ORDER,
                               GALCLASS_COLOURS, GALCLASS_LABELS)

# ── Shared helpers ────────────────────────────────────────────────────────────

def _sfms_ssfr_line(sfms_params, x_min, x_max, n=300):
    """Return (mass_range, log_sSFR_line) for SFMS overlay on sSFR panels."""
    mass_range = np.linspace(x_min, x_max, n)
    sfms_ssfr  = lee2015_sfms(mass_range, *sfms_params) - mass_range
    return mass_range, sfms_ssfr


def _scatter_panel(ax, x, y, c_vals, cmap, vmin, vmax,
                   sfms_params, show_sfms=True):
    """
    Draw a single scatter panel with optional SFMS overlay.
    Returns the scatter PathCollection for colorbar use.
    """
    sc = ax.scatter(x, y,
                    c=c_vals, cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    s=10, alpha=0.75, linewidths=0)

    if show_sfms and len(x) > 0:
        mr, sl = _sfms_ssfr_line(sfms_params,
                                  np.nanmin(x) - 0.1,
                                  np.nanmax(x) + 0.1)
        ax.plot(mr, sl, color="gold", lw=1.5, ls="-",
                label="SFMS", zorder=5)

    return sc


def _apply_panel_labels(ax, title=None, xlabel=True, ylabel=True):
    if xlabel:
        ax.set_xlabel(AXIS_LABELS["log_Mstar"], fontsize=9)
    if ylabel:
        ax.set_ylabel(AXIS_LABELS["log_sSFR"], fontsize=9)
    if title:
        ax.set_title(title, fontsize=9)


def _global_vmin_vmax(df_plot, metric, percentiles=(5, 95)):
    """Compute colour scale limits from the full plotting sample."""
    vals = df_plot[metric].dropna()
    return (np.nanpercentile(vals, percentiles[0]),
            np.nanpercentile(vals, percentiles[1]))


# ── Option 1: Corner plot ─────────────────────────────────────────────────────

def plot_corner(df, sfms_params, metric, sample="wang",
                sfr_cut=None, figsize=(10, 9)):
    """
    True lower-triangular corner plot for three driving properties:
    log_Mstar, log_sSFR, galclass (as integer 1-4).

    Layout (3x3 lower triangle)
    ---------------------------
    [0,0] histogram of log_Mstar
    [1,0] log_sSFR vs log_Mstar        ← SFMS overlay
    [1,1] histogram of log_sSFR
    [2,0] galclass vs log_Mstar
    [2,1] galclass vs log_sSFR
    [2,2] histogram of galclass (bar chart)

    All scatter panels colour-coded by metric with fixed shared scale.
    Environment plotted as integer 1-4 with custom y-tick labels.
    """
    df_plot = get_sample(df, sample, min_log_sfr=sfr_cut).copy()
    df_plot["galclass"] = pd.to_numeric(
        df_plot["galclass"], errors="coerce"
    ).astype("Int64")
    df_plot = df_plot.dropna(subset=["log_Mstar", "log_sSFR",
                                     "galclass", metric])

    cmap         = CMAPS.get(metric, "plasma")
    vmin, vmax   = _global_vmin_vmax(df_plot, metric)
    metric_label = AXIS_LABELS.get(metric, metric)

    x  = df_plot["log_Mstar"].values.astype(float)
    y  = df_plot["log_sSFR"].values.astype(float)
    gc = df_plot["galclass"].values.astype(float)
    c  = df_plot[metric].values.astype(float)

    env_labels = {1: "Ungrouped", 2: "Low-mass\ngroup",
                  3: "High-mass\ngroup", 4: "Cluster"}

    fig, axes = plt.subplots(3, 3, figsize=figsize)

    # Hide upper triangle
    for row in range(3):
        for col in range(3):
            if col > row:
                axes[row, col].set_visible(False)

    # ── Diagonal: distributions ───────────────────────────────────
    axes[0, 0].hist(x,  bins=30, color="steelblue", alpha=0.8)
    axes[0, 0].set_ylabel("Count", fontsize=8)
    axes[0, 0].set_xlabel(AXIS_LABELS["log_Mstar"], fontsize=8)

    axes[1, 1].hist(y,  bins=30, color="steelblue", alpha=0.8)
    axes[1, 1].set_ylabel("Count", fontsize=8)
    axes[1, 1].set_xlabel(AXIS_LABELS["log_sSFR"], fontsize=8)

    axes[2, 2].bar([1, 2, 3, 4],
                   [df_plot["galclass"].value_counts().get(i, 0)
                    for i in [1, 2, 3, 4]],
                   color=[GALCLASS_COLOURS[i] for i in [1, 2, 3, 4]],
                   alpha=0.85)
    axes[2, 2].set_xticks([1, 2, 3, 4])
    axes[2, 2].set_xticklabels(["Ungrouped", "Low-mass\ngroup",
                                 "High-mass\ngroup", "Cluster"],
                               fontsize=6)
    axes[2, 2].set_ylabel("Count", fontsize=8)
    axes[2, 2].set_xlabel("Environment", fontsize=8)

    # ── [1,0]: log_sSFR vs log_Mstar ─────────────────────────────
    sc = axes[1, 0].scatter(x, y, c=c, cmap=cmap,
                            vmin=vmin, vmax=vmax,
                            s=8, alpha=0.7, linewidths=0)
    mr, sl = _sfms_ssfr_line(sfms_params,
                              np.nanmin(x) - 0.1,
                              np.nanmax(x) + 0.1)
    axes[1, 0].plot(mr, sl, color="gold", lw=1.5, zorder=5,
                    label="SFMS")
    axes[1, 0].set_xlabel(AXIS_LABELS["log_Mstar"], fontsize=8)
    axes[1, 0].set_ylabel(AXIS_LABELS["log_sSFR"], fontsize=8)
    axes[1, 0].legend(fontsize=7, loc="upper right")

    # ── [2,0]: galclass vs log_Mstar ─────────────────────────────
    rng    = np.random.default_rng(42)
    jitter = rng.uniform(-0.15, 0.15, size=len(gc))

    axes[2, 0].scatter(x, gc + jitter, c=c, cmap=cmap,
                       vmin=vmin, vmax=vmax,
                       s=8, alpha=0.7, linewidths=0)
    axes[2, 0].set_xlabel(AXIS_LABELS["log_Mstar"], fontsize=8)
    axes[2, 0].set_ylabel("Environment", fontsize=8)
    axes[2, 0].set_yticks([1, 2, 3, 4])
    axes[2, 0].set_yticklabels([env_labels[i] for i in [1, 2, 3, 4]],
                               fontsize=7)

    # ── [2,1]: galclass vs log_sSFR ──────────────────────────────
    axes[2, 1].scatter(y, gc + jitter, c=c, cmap=cmap,
                       vmin=vmin, vmax=vmax,
                       s=8, alpha=0.7, linewidths=0)
    axes[2, 1].set_xlabel(AXIS_LABELS["log_sSFR"], fontsize=8)
    axes[2, 1].set_ylabel("Environment", fontsize=8)
    axes[2, 1].set_yticks([1, 2, 3, 4])
    axes[2, 1].set_yticklabels([env_labels[i] for i in [1, 2, 3, 4]],
                               fontsize=7)

    # ── Shared colourbar ──────────────────────────────────────────
    fig.subplots_adjust(right=0.85, hspace=0.45, wspace=0.45)
    cax  = fig.add_axes([0.88, 0.35, 0.02, 0.4])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label(metric_label, fontsize=10)

    fig.suptitle(f"Corner plot — {metric_label}  |  sample: {sample}",
                 fontsize=11, y=1.01)

    return fig

# ── Option 2: Faceted grid (scatter) ─────────────────────────────────────────

def plot_faceted_grid(df, sfms_params, sample="wang",
                      sfr_cut=None, figsize=(14, 4)):
    """
    Produces 4 figures, one per galclass environment bin.
    Each figure has 4 panels, one per metric (G, C, A, S).
    x = log_Mstar, y = log_sSFR, colour = metric.
    SFMS overlaid in each panel. Fixed colour scale per metric.

    Parameters
    ----------
    df          : DataFrame
    sfms_params : array [S0, a1, Mt]
    sample      : str   SAMPLE_REGISTRY key
    sfr_cut     : float|None
    figsize     : tuple
    """
    METRICS = ["G", "C", "A", "S"]

    df_plot = get_sample(df, sample, min_log_sfr=sfr_cut).copy()
    df_plot["galclass"] = pd.to_numeric(
        df_plot["galclass"], errors="coerce"
    ).astype("Int64")

    # Compute fixed vmin/vmax per metric across full sample
    vranges = {
        m: _global_vmin_vmax(df_plot, m) for m in METRICS
    }

    figs = []

    for gc_val in GALCLASS_ORDER:
        mask  = df_plot["galclass"] == gc_val
        df_gc = df_plot[mask]

        fig, axes = plt.subplots(1, 4, figsize=figsize,
                                 sharey=True, sharex=True)

        fig.suptitle(
            f"{GALCLASS_LABELS[gc_val]}  |  sample: {sample}",
            fontsize=11, y=1.02
        )

        for ax, metric in zip(axes, METRICS):
            x      = df_gc["log_Mstar"].values
            y      = df_gc["log_sSFR"].values
            c_vals = df_gc[metric].values
            cmap   = CMAPS.get(metric, "plasma")
            vmin, vmax = vranges[metric]

            sc = _scatter_panel(ax, x, y, c_vals, cmap, vmin, vmax,
                                sfms_params, show_sfms=True)

            n_plotted = int(np.isfinite(c_vals).sum())

            # Colourbar attached to each panel individually
            cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
            cbar.set_label(AXIS_LABELS.get(metric, metric), fontsize=8)
            cbar.ax.tick_params(labelsize=7)

            ax.set_xlabel(AXIS_LABELS["log_Mstar"], fontsize=9)
            ax.set_title(f"{AXIS_LABELS.get(metric, metric)}\n"
                         f"n={n_plotted}",
                         fontsize=9, pad=4)

            # y-label on leftmost panel only
            if metric == "G":
                ax.set_ylabel(AXIS_LABELS["log_sSFR"], fontsize=9)

            ax.legend(fontsize=7, loc="upper right")

        fig.tight_layout()
        figs.append(fig)

    return figs


# ── Option 3: 3D plot ─────────────────────────────────────────────────────

def plot_3d_scatter(df, sfms_params, metric, sample="wang",
                    sfr_cut=None, figsize=(10, 8), elev=25, azim=-60):
    """
    3D scatter plot of the three driving properties with metric as colour.

    Axes
    ----
    x : log_Mstar
    y : galclass (1-4, jittered slightly for visibility)
    z : log_sSFR

    Points are coloured by metric value with fixed shared scale.
    SFMS projected onto the x-z plane as a reference curve.

    Parameters
    ----------
    df          : DataFrame
    sfms_params : array [S0, a1, Mt]
    metric      : str   one of "G", "C", "A", "S"
    sample      : str   SAMPLE_REGISTRY key
    sfr_cut     : float|None
    figsize     : tuple
    elev        : float initial elevation angle (degrees)
    azim        : float initial azimuth angle (degrees)
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    df_plot = get_sample(df, sample, min_log_sfr=sfr_cut).copy()
    df_plot["galclass"] = pd.to_numeric(
        df_plot["galclass"], errors="coerce"
    ).astype("Int64")
    df_plot = df_plot.dropna(subset=["log_Mstar", "log_sSFR",
                                     "galclass", metric])

    cmap         = CMAPS.get(metric, "plasma")
    vmin, vmax   = _global_vmin_vmax(df_plot, metric)
    metric_label = AXIS_LABELS.get(metric, metric)

    x  = df_plot["log_Mstar"].values.astype(float)
    z  = df_plot["log_sSFR"].values.astype(float)
    c  = df_plot[metric].values.astype(float)

    # Jitter galclass slightly so points don't stack on discrete planes
    rng = np.random.default_rng(42)
    y   = df_plot["galclass"].values.astype(float)
    y  += rng.uniform(-0.15, 0.15, size=len(y))

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    # ── 3D scatter ────────────────────────────────────────────────
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sc   = ax.scatter(x, y, z,
                      c=c, cmap=cmap, norm=norm,
                      s=12, alpha=0.7, linewidths=0,
                      depthshade=True)

    # ── SFMS projected onto x-z (mass-sSFR) plane ────────────────
    # Drawn at y=0.5 so it sits just in front of the ungrouped plane
    mr, sl = _sfms_ssfr_line(sfms_params,
                              np.nanmin(x) - 0.1,
                              np.nanmax(x) + 0.1)
    ax.plot(mr,
            np.full_like(mr, 0.5),
            sl,
            color="gold", lw=2, ls="-",
            label="SFMS", zorder=10)

    # ── Axes labels and ticks ─────────────────────────────────────
    ax.set_xlabel(AXIS_LABELS["log_Mstar"], fontsize=9, labelpad=8)
    ax.set_ylabel("Environment", fontsize=9, labelpad=8)
    ax.set_zlabel(AXIS_LABELS["log_sSFR"], fontsize=9, labelpad=8)

    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["Ungrouped", "Low-mass\ngroup",
                         "High-mass\ngroup", "Cluster"],
                       fontsize=7)

    ax.legend(fontsize=8, loc="upper left")

    # ── Colourbar ─────────────────────────────────────────────────
    cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.6, aspect=15)
    cbar.set_label(metric_label, fontsize=10)

    ax.set_title(f"3D scatter — {metric_label}  |  sample: {sample}",
                 fontsize=11, pad=12)

    fig.tight_layout()
    return fig

# ── Option 4: Heatmap grid ────────────────────────────────────────────────────

def plot_heatmap_grid(df, sfms_params, metric, sample="wang",
                      sfr_cut=None, bin_width=0.3,
                      min_per_cell=5, figsize=(14, 4)):
    """
    1×4 row of binned median heatmap panels, one per galclass bin.
    x = log_Mstar, y = log_sSFR. Cell colour = median metric value.
    Fixed colour scale across all panels.
    Cells with fewer than min_per_cell galaxies are left blank (NaN).
    Galaxy counts per cell and minimum count printed to terminal.

    Parameters
    ----------
    df           : DataFrame
    sfms_params  : array [S0, a1, Mt]  (kept for API consistency)
    metric       : str
    sample       : str
    sfr_cut      : float|None
    bin_width    : float  dex width of each cell (default 0.3)
    min_per_cell : int    minimum galaxies to colour a cell
    figsize      : tuple
    """
    df_plot = get_sample(df, sample, min_log_sfr=sfr_cut).copy()
    df_plot["galclass"] = pd.to_numeric(
        df_plot["galclass"], errors="coerce"
    ).astype("Int64")

    cmap         = CMAPS.get(metric, "plasma")
    vmin, vmax   = _global_vmin_vmax(df_plot, metric)
    metric_label = AXIS_LABELS.get(metric, metric)

    # ── Bin edges from full sample extent ────────────────────────
    x_min = np.floor(df_plot["log_Mstar"].min() / bin_width) * bin_width
    x_max = np.ceil( df_plot["log_Mstar"].max() / bin_width) * bin_width
    y_min = np.floor(df_plot["log_sSFR"].min()  / bin_width) * bin_width
    y_max = np.ceil( df_plot["log_sSFR"].max()  / bin_width) * bin_width

    x_edges = np.arange(x_min, x_max + bin_width, bin_width)
    y_edges = np.arange(y_min, y_max + bin_width, bin_width)
    nx, ny  = len(x_edges) - 1, len(y_edges) - 1

    fig, axes = plt.subplots(1, 4, figsize=figsize,
                             sharey=True, sharex=True)

    im = None
    print(f"\n=== Heatmap cell counts — {metric_label} | sample: {sample} ===")

    for ax, gc_val in zip(axes, GALCLASS_ORDER):
        mask  = df_plot["galclass"] == gc_val
        df_gc = df_plot[mask].dropna(subset=["log_Mstar", "log_sSFR", metric])

        grid_med   = np.full((ny, nx), np.nan)
        grid_count = np.zeros((ny, nx), dtype=int)

        for i in range(nx):
            for j in range(ny):
                cell_mask = (
                    (df_gc["log_Mstar"] >= x_edges[i]) &
                    (df_gc["log_Mstar"] <  x_edges[i + 1]) &
                    (df_gc["log_sSFR"]  >= y_edges[j]) &
                    (df_gc["log_sSFR"]  <  y_edges[j + 1])
                )
                n = cell_mask.sum()
                grid_count[j, i] = n
                if n >= min_per_cell:
                    grid_med[j, i] = df_gc.loc[cell_mask, metric].median()

        # ── Terminal output ───────────────────────────────────────
        occupied     = grid_count[grid_count > 0]
        filled_cells = (grid_count >= min_per_cell).sum()
        total_cells  = nx * ny
        min_count    = int(occupied.min()) if len(occupied) > 0 else 0

        print(f"\n  {GALCLASS_LABELS[gc_val]}")
        print(f"    total grid cells    : {total_cells}")
        print(f"    occupied cells (>0) : {len(occupied)}")
        print(f"    filled cells (>={min_per_cell})  : {filled_cells}")
        print(f"    min galaxies in any occupied cell: {min_count}")

        # ── Draw heatmap ──────────────────────────────────────────
        im = ax.imshow(grid_med,
                       origin="lower",
                       extent=[x_min, x_max, y_min, y_max],
                       aspect="auto",
                       cmap=cmap,
                       vmin=vmin, vmax=vmax,
                       interpolation="nearest")

        n_plotted = int((grid_count >= min_per_cell).sum())
        ax.set_title(f"{GALCLASS_LABELS[gc_val]}\n"
                     f"({n_plotted} filled cells)",
                     fontsize=9)
        ax.set_xlabel(AXIS_LABELS["log_Mstar"], fontsize=9)
        if gc_val == 1:
            ax.set_ylabel(AXIS_LABELS["log_sSFR"], fontsize=9)

    # ── Shared colourbar ──────────────────────────────────────────
    fig.subplots_adjust(right=0.88, wspace=0.08)
    cax  = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(f"Median {metric_label}", fontsize=10)

    fig.suptitle(f"Heatmap grid — {metric_label}  |  sample: {sample}",
                 fontsize=11)
    return fig