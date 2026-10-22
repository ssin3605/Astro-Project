"""
Thesis figures: Random Forest actual vs. out-of-fold predicted morphology,
as a function of the most important input features.

Each panel plots the metric against one feature for every galaxy: the actual
value (small grey circles) and the out-of-fold predicted value (small
triangles, in the metric's Chapter 3 colour). The predictions scatter less than the actual values, because the
model can only reproduce the part of the metric that the features explain.
On top, galaxies are binned by the feature (10 equal-number bins) and the mean
actual (black circles) and mean predicted (triangles) values are shown
with standard-error bars. If the model has captured how the metric depends on
that feature, the two binned trends lie on top of each other. Each bin is
placed at the median feature value of its galaxies.

Two figures, drawn at the printed size (6.3 in = full A4 text width):
    RF_oof_trends_G_S.pdf/.png   2x4: Gini (top row) and Clumpiness (bottom row)
                                 against log M*, r_e, z and Delta MS, with
                                 shared x-axes per column and y-axes per row
    RF_oof_trends_C.pdf/.png     1x4: Concentration against Sigma_5, r_e,
                                 ellipticity and Delta MS

The R^2 box in each row is the cross-validated R^2 (mean +/- std over the 5
folds), the same number as in the importance figures.

Reads from outputs/ch4/predictions/RF_{G,S,C}_oof.csv (written by
ml/notebooks/forward/random_forest/RF_{G,S,C}.ipynb), the matching
outputs/ch4/importances/RF_{metric}_performance.csv, and the feature values
from outputs/ch4/csv/ml_data.csv. Saves to outputs/ch4/figures/.

Usage
    python ml/figures/make_oof_figures.py
"""
import os
import string
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "sf_statistics"))
from src.utils.paths import (  # noqa: E402
    CH4_FIGURES_DIR, ML_DATA_CSV, ML_IMPORTANCES_DIR, ML_PREDICTIONS_DIR,
)

OUTPUT_DIR = str(CH4_FIGURES_DIR)

METRIC_NAMES = {"G": "Gini (G)", "C": "Concentration (C)", "S": "Clumpiness (S)"}

# Axis names match the Chapter 3 figures and the importance figures
FEATURE_LABELS = {
    "log_Mstar": r"$\log(M_\star / M_\odot)$",
    "re":        r"$r_e$ (arcsec)",
    "z":         r"$z$",
    "delta_ms":  r"$\Delta\mathrm{MS}$ (dex)",
    "5NN":       r"$\Sigma_5$ (Mpc$^{-2}$)",
    "ellip":     "ellipticity",
}
LOG_X_FEATURES = {"5NN"}   # spans ~4 orders of magnitude

GS_FEATURES = ["log_Mstar", "re", "z", "delta_ms"]
C_FEATURES = ["5NN", "re", "ellip", "delta_ms"]

N_BINS = 10
MIN_PER_BIN = 3

# point = individual galaxies, colour = binned trend on top
ACTUAL = dict(marker="o", color="black", point="0.45", label="Actual")
PREDICTED = dict(marker="^", label="Predicted (out-of-fold)")   # coloured by metric, below

# Predicted values use each metric's colour from the Chapter 3 GCAS figures
METRIC_COLOURS = {"G": "#7B3294", "C": "#0868AC", "A": "#D7301F", "S": "#238B45"}
POINT_SIZE = 3
POINT_ALPHA = 0.35

STYLE = {
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def load_metric(metric):
    """OOF predictions joined to the feature values, plus the CV R^2 (mean, std)."""
    oof = pd.read_csv(os.path.join(ML_PREDICTIONS_DIR, f"RF_{metric}_oof.csv"))
    features = pd.read_csv(ML_DATA_CSV)
    data = oof.merge(features, on="CATID", how="left", validate="one_to_one")
    perf = pd.read_csv(os.path.join(ML_IMPORTANCES_DIR, f"RF_{metric}_performance.csv"))
    r2 = perf.loc[perf["Metric"] == "R2"].iloc[0]
    return data, r2["Mean"], r2["Std"]


def binned_trend(x, y, n_bins=N_BINS):
    """Equal-number bins in x: median x, mean y and standard error of y per bin."""
    edges = np.unique(np.nanquantile(x, np.linspace(0, 1, n_bins + 1)))
    idx = np.digitize(x, edges[1:-1])
    bx, by, berr = [], [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() < MIN_PER_BIN:
            continue
        bx.append(np.median(x[sel]))
        by.append(np.mean(y[sel]))
        berr.append(np.std(y[sel]) / np.sqrt(sel.sum()))
    return np.array(bx), np.array(by), np.array(berr)


def plot_trend_panel(ax, data, feature, metric):
    x = data[feature].to_numpy(float)
    ok = np.isfinite(x)
    # (column, marker, colour of individual points, colour of binned means)
    series = [("actual", ACTUAL["marker"], ACTUAL["point"], ACTUAL["color"]),
              ("predicted", PREDICTED["marker"], METRIC_COLOURS[metric], METRIC_COLOURS[metric])]
    # Every galaxy: actual and out-of-fold predicted values (rasterised to keep
    # the PDF small)
    for col, marker, point_colour, _ in series:
        ax.scatter(x[ok], data[col].to_numpy(float)[ok], s=POINT_SIZE, marker=marker,
                   color=point_colour, alpha=POINT_ALPHA, linewidths=0, rasterized=True, zorder=2)
    # Binned trends on top
    for col, marker, _, trend_colour in series:
        bx, by, berr = binned_trend(x[ok], data[col].to_numpy(float)[ok])
        ax.errorbar(bx, by, yerr=berr, fmt=marker, color=trend_colour,
                    ms=4, elinewidth=0.9, capsize=1.5,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=4)
    if feature in LOG_X_FEATURES:
        ax.set_xscale("log")
    ax.grid(alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)


def r2_box(ax, r2_mean, r2_std):
    # Cross-validated R^2 (mean +/- std over the 5 folds); "CV" is left to the caption.
    # Sits just below the panel letter so the two don't collide.
    ax.text(0.97, 0.84, f"$R^2$ = {r2_mean:.2f} ± {r2_std:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.9))


def panel_letter(ax, letter):
    ax.text(0.04, 0.95, f"({letter})", transform=ax.transAxes, ha="left", va="top",
            fontsize=9, fontweight="bold")


def add_legend(fig, y_anchor, metrics):
    """Actual (black) plus one predicted entry per metric colour in the figure."""
    entries = [(ACTUAL["marker"], ACTUAL["color"], ACTUAL["label"])]
    if len(metrics) == 1:
        entries.append((PREDICTED["marker"], METRIC_COLOURS[metrics[0]], PREDICTED["label"]))
    else:
        entries += [(PREDICTED["marker"], METRIC_COLOURS[m], f"Predicted ({m})") for m in metrics]
    handles = [Line2D([], [], color=colour, marker=marker, ms=4.5, ls="none",
                      markeredgecolor="white", markeredgewidth=0.4, label=label)
               for marker, colour, label in entries]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, y_anchor), handletextpad=0.3, columnspacing=1.8)


def headroom(axes_row, frac=0.30):
    """Extend a row's shared y-range upwards so the trends sit below the R^2 box."""
    lo, hi = axes_row[0].get_ylim()
    axes_row[0].set_ylim(lo, hi + frac * (hi - lo))


def save(fig, stem):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, stem)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}.pdf / .png")


def make_gs_figure():
    """2x4: Gini (top) and Clumpiness (bottom) against the same four features."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 4, figsize=(6.3, 3.7), sharex="col", sharey="row")
        letters = iter(string.ascii_lowercase)
        for row, metric in enumerate(["G", "S"]):
            data, r2_mean, r2_std = load_metric(metric)
            for col, feature in enumerate(GS_FEATURES):
                ax = axes[row, col]
                plot_trend_panel(ax, data, feature, metric)
                panel_letter(ax, next(letters))
                if row == 1:
                    ax.set_xlabel(FEATURE_LABELS[feature])
            axes[row, 0].set_ylabel(METRIC_NAMES[metric])
            headroom(axes[row])
            r2_box(axes[row, -1], r2_mean, r2_std)
        add_legend(fig, 1.0, ["G", "S"])
        fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=0.6, w_pad=0.4)
        save(fig, "RF_oof_trends_G_S")


def make_c_figure():
    """1x4: Concentration against its four most important features."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(6.3, 2.1), sharey=True)
        data, r2_mean, r2_std = load_metric("C")
        for ax, feature, letter in zip(axes, C_FEATURES, string.ascii_lowercase):
            plot_trend_panel(ax, data, feature, "C")
            panel_letter(ax, letter)
            ax.set_xlabel(FEATURE_LABELS[feature])
        axes[0].set_ylabel(METRIC_NAMES["C"])
        headroom(axes)
        r2_box(axes[-1], r2_mean, r2_std)
        add_legend(fig, 1.02, ["C"])
        fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=0.4)
        save(fig, "RF_oof_trends_C")


if __name__ == "__main__":
    make_gs_figure()
    make_c_figure()
