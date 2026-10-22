"""
Thesis figure: per-algorithm 2x2 permutation-importance summary.

One figure per algorithm (RF, XGBoost, CatBoost, NN), each a 2x2 grid with
one panel per morphology metric (A, C, G, S). Each panel shows the top 6
features by permutation importance (RANDOM excluded from ranking, kept as a
dashed reference line), styled after the reference figure
"ML Model Thesis Figure Example.png" (bar chart + top-right performance
inset). The category pie chart from that example is intentionally left out
until local/environmental/intrinsic feature groupings are defined -- see the
PIE_CHART_HERE marker below.

Also makes the thesis-body figure RF_XGBoost_importance_paired.pdf/.png: a 2x2
dot plot comparing Random Forest and XGBoost permutation importances for each
metric (see make_paired_figure).

Reads from outputs/ch4/importances/{Algorithm}_{Metric}.csv and
outputs/ch4/importances/{Algorithm}_{Metric}_performance.csv, written by the
16 forward-model notebooks in ml/notebooks/forward/. Does not touch those notebooks.
Saves to outputs/ch4/figures/.

Usage
    python ml/figures/make_importance_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "sf_statistics"))
from src.utils.paths import CH4_FIGURES_DIR, ML_IMPORTANCES_DIR  # noqa: E402

IMPORTANCES_DIR = str(ML_IMPORTANCES_DIR)
OUTPUT_DIR = str(CH4_FIGURES_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALGORITHMS = {
    "RF": "Random Forest",
    "XGBoost": "XGBoost",
    "CatBoost": "CatBoost",
    "NN": "Neural Network",
}

# Panel order: top-left, top-right, bottom-left, bottom-right (same as Chapter 3)
METRICS = ["G", "C", "A", "S"]
METRIC_NAMES = {
    "G": "Gini (G)",
    "C": "Concentration (C)",
    "A": "Asymmetry (A)",
    "S": "Clumpiness (S)",
}

TOP_N = 6

# One colour per feature, shared by the per-algorithm figures and the paired
# RF vs XGBoost figure, so a feature is always the same colour across every
# panel and every figure. The seven features in the thesis figures use the
# Okabe-Ito colour-blind-safe palette; the rest keep their earlier colours.
FEATURE_COLOURS = {
    "log_Mstar":             "#0072B2",   # blue
    "log_sSFR":              "#009E73",   # bluish green
    "delta_ms":              "#CC79A7",   # reddish purple
    "re":                    "#E69F00",   # orange
    "ellip":                 "#56B4E9",   # sky blue
    "z":                     "#D55E00",   # vermillion
    "5NN":                   "#000000",   # black
    "environment":           "#F4A582",
    "is_isolated":           "#FEE090",
    "closest_companion_kpc": "#92C5DE",
    "log_MassA_1":           "#762A83",
    "is_central":            "#B2ABD2",
    "is_satellite":          "#8C96C6",
    "is_cluster":            "#D9D9D9",
    "spectral_class":        "#E08214",
    "Dfil":                  "#A6761D",
    "Dnode":                 "#E7298A",
    "Dvoid":                 "#666666",
    "RANDOM":                "#BBBBBB",
}


def load_panel_data(algo, metric):
    imp = pd.read_csv(os.path.join(IMPORTANCES_DIR, f"{algo}_{metric}.csv"))
    perf = pd.read_csv(os.path.join(IMPORTANCES_DIR, f"{algo}_{metric}_performance.csv"))

    random_val = imp.loc[imp["Feature"] == "RANDOM", "Perm_Mean"].values[0]
    ranked = imp[imp["Feature"] != "RANDOM"].sort_values("Perm_Mean", ascending=False)
    top = ranked.head(TOP_N).iloc[::-1]  # reversed so the largest bar ends up on top

    r2_row = perf.loc[perf["Metric"] == "R2"].iloc[0]
    return top, random_val, r2_row["Mean"], r2_row["Std"]


def plot_panel(ax, algo, metric):
    top, random_val, r2_mean, r2_std = load_panel_data(algo, metric)

    features = top["Feature"].tolist()
    means = top["Perm_Mean"].values
    stds = top["Perm_Std"].values
    colours = [FEATURE_COLOURS.get(f, "#888888") for f in features]
    y_pos = np.arange(len(features))

    ax.barh(y_pos, means, xerr=stds, capsize=3, color=colours,
            edgecolor="white", linewidth=0.6,
            error_kw={"elinewidth": 1.0, "ecolor": "#333333", "capthick": 1.0})
    ax.axvline(random_val, color="#555555", linestyle="--", linewidth=1.2, zorder=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10)
    ax.set_title(METRIC_NAMES[metric], fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlabel("Permutation Importance\n(mean decrease in R²)", fontsize=9)

    # Performance inset, top-right -- this session's analogue of the example
    # figure's Train/Test AUC box.
    ax.text(0.97, 0.05, f"CV R² = {r2_mean:.2f} ± {r2_std:.2f}",
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#4c4b4b", alpha=0.9))

    # PIE_CHART_HERE: once features are tagged Local / Intrinsic / Environmental,
    # add a small inset pie (e.g. via ax.inset_axes([0.62, 0.62, 0.35, 0.35]))
    # showing each category's share of total permutation importance, mirroring
    # the reference figure. Left out for now per instruction.


def make_figure(algo):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, metric in zip(axes.flat, METRICS):
        plot_panel(ax, algo, metric)

    fig.suptitle(f"{ALGORITHMS[algo]} — Permutation Feature Importance by Morphology Metric",
                 fontsize=15, y=0.99)
    fig.text(0.5, 0.005,
             f"Top {TOP_N} features by permutation importance (mean ± std over 5-fold CV). "
             "Dashed line: RANDOM noise-feature baseline.",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    out_path = os.path.join(OUTPUT_DIR, f"{algo}_importance_summary.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


# --- Paired thesis-body figure (RF vs XGBoost) -------------------------------
#
# 2x2 dot plot, one panel per metric. Each row is a feature, coloured by
# feature; the marker shape gives the model (RF filled circle, XGBoost open
# diamond), drawn on the same row so agreement between the models is easy to
# see. Features are sorted within each panel by the mean of the two models'
# permutation importance, so each panel reads as a ranking. The grey band is
# the noise level: up to the larger of the two models' RANDOM mean + 1 std.
# The four binary environment flags (is_central, is_satellite, is_isolated,
# is_cluster) are left out: |Perm_Mean| < 0.008 in every panel.

PAIRED_FEATURES = ["log_Mstar", "log_sSFR", "delta_ms", "re", "ellip", "z", "5NN"]

# Names match the axis labels used in the Chapter 3 figures
PAIRED_LABELS = {
    "log_Mstar": r"$\log M_\star$",
    "log_sSFR":  r"$\log \mathrm{sSFR}$",
    "delta_ms":  r"$\Delta\mathrm{MS}$",
    "re":        r"$r_e$",
    "ellip":     "ellipticity",
    "z":         r"$z$",
    "5NN":       r"$\Sigma_5$",
}

# Same feature colours as the per-algorithm figures
PAIRED_FEATURE_COLOURS = {f: FEATURE_COLOURS[f] for f in PAIRED_FEATURES}

PAIRED_ALGOS = {
    "RF":      dict(marker="o", filled=True,  label="Random Forest", short="RF"),
    "XGBoost": dict(marker="D", filled=False, label="XGBoost",       short="XGB"),
}

# Drawn at the printed size (A4 text width with ~2.5 cm margins is ~6.3 in),
# so font sizes are the true sizes in the thesis
PAIRED_FIGSIZE = (6.3, 4.9)
PAIRED_STYLE = {
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "xtick.major.size": 2.5,
    "ytick.major.size": 0,
}
MODEL_OFFSET = 0.18   # vertical offset of each model's marker within a row


def load_paired(algo, metric):
    imp = pd.read_csv(os.path.join(IMPORTANCES_DIR, f"{algo}_{metric}.csv")).set_index("Feature")
    perf = pd.read_csv(os.path.join(IMPORTANCES_DIR, f"{algo}_{metric}_performance.csv"))
    r2_mean = perf.loc[perf["Metric"] == "R2", "Mean"].iloc[0]
    return imp["Perm_Mean"], imp["Perm_Std"], r2_mean


def plot_paired_panel(ax, metric):
    data = {algo: load_paired(algo, metric) for algo in PAIRED_ALGOS}

    # Rank features by the mean of the two models' importance (most important on top)
    order = sorted(PAIRED_FEATURES,
                   key=lambda f: -np.mean([data[a][0][f] for a in PAIRED_ALGOS]))
    rows = np.arange(len(order))

    # Noise level from the RANDOM feature
    noise = max(data[a][0]["RANDOM"] + data[a][1]["RANDOM"] for a in PAIRED_ALGOS)
    ax.axvspan(-1, noise, color="0.88", lw=0, zorder=0)
    ax.axvline(0, color="0.4", lw=0.6, zorder=1)

    for row, feat in zip(rows, order):
        colour = PAIRED_FEATURE_COLOURS[feat]
        xs = [data[a][0][feat] for a in PAIRED_ALGOS]
        ys = [row + (k - 0.5) * MODEL_OFFSET for k in range(len(PAIRED_ALGOS))]
        for (algo, st), x, y in zip(PAIRED_ALGOS.items(), xs, ys):
            ax.errorbar(x, y, xerr=data[algo][1][feat], fmt="none", ecolor=colour,
                        elinewidth=0.8, alpha=0.6, zorder=3)
            ax.plot(x, y, ls="none", marker=st["marker"], ms=4.5, mec=colour, mew=1.1,
                    mfc=colour if st["filled"] else "white", zorder=4)

    ax.set_yticks(rows)
    ax.set_yticklabels([PAIRED_LABELS[f] for f in order])
    ax.set_ylim(len(order) - 0.5, -0.6)

    lo = min((data[a][0][order] - data[a][1][order]).min() for a in PAIRED_ALGOS)
    hi = max((data[a][0][order] + data[a][1][order]).max() for a in PAIRED_ALGOS)
    ax.set_xlim(min(lo, 0) - 0.02 * hi, hi * 1.05)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.grid(axis="x", alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    r2_text = " · ".join(f"{st['short']} {data[a][2]:.2f}" for a, st in PAIRED_ALGOS.items())
    ax.set_title(METRIC_NAMES[metric], fontsize=10, pad=12)
    ax.text(0.5, 1.01, f"CV $R^2$: {r2_text}", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8, color="0.1")


def make_paired_figure(out_dir=OUTPUT_DIR):
    with plt.rc_context(PAIRED_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=PAIRED_FIGSIZE)
        for ax, metric in zip(axes.flat, METRICS):
            plot_paired_panel(ax, metric)
        for ax in axes[1]:
            ax.set_xlabel(r"Permutation Importance ($\Delta R^2$)")

        # Legend: marker shape = model (in neutral grey), grey band = noise level
        handles = [Line2D([], [], ls="none", marker=st["marker"], ms=5, mew=1.1, mec="0.25",
                          mfc="0.25" if st["filled"] else "white", label=st["label"])
                   for st in PAIRED_ALGOS.values()]
        handles.append(Patch(color="0.88", label="RANDOM noise level"))
        fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9, frameon=False,
                   bbox_to_anchor=(0.5, 1.0), handletextpad=0.4, columnspacing=1.5)
        fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.0, w_pad=1.2)

        stem = os.path.join(out_dir, "RF_XGBoost_importance_paired")
        fig.savefig(stem + ".pdf", bbox_inches="tight")
        fig.savefig(stem + ".png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {stem}.pdf / .png")


if __name__ == "__main__":
    for algo in ALGORITHMS:
        make_figure(algo)
    make_paired_figure()
