"""
Thesis figure: per-algorithm 2x2 permutation-importance summary for the
inverse problem (predicting global galaxy properties FROM G, C, A, S).

One figure per algorithm, each a 2x2 grid with one panel per continuous target
(delta_ms, log_sSFR, log_SFR, log_Mstar). Each panel shows the top 6 features
by permutation importance of the GCAS + Baseline model (RANDOM excluded from
ranking, kept as a dashed reference line). Control / conditioning features are
hatched, as in the notebooks, because their importance is not interpreted
physically. The inset gives the full-model CV R² and the headline number: the
gain in R² from adding G, C, A, S to the observational baseline.

Styled to match make_importance_figures.py (the forward-model figures).

Reads from outputs/ch4/importances/{Algorithm}_morph_{target}.csv and
outputs/ch4/importances/{Algorithm}_morph_{target}_performance.csv, written by the
"Save to shared importances folder" cell of
ml/notebooks/inverse/morphology_*.ipynb, and saves to outputs/ch4/figures/.
Targets not yet run are shown as empty "not yet run" panels.

Every panel must come from the same controls / condition_on config, otherwise
the figure is refused. Set ALLOW_MIXED_CONFIG = True to plot anyway (each
panel is then captioned with its own config).

Usage
    python ml/figures/make_morphology_importance_figures.py
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

ALGORITHMS = {
    "RF": "Random Forest",
    "XGBoost": "XGBoost",
    "CatBoost": "CatBoost",
    "NN": "Neural Network",
}

# Panel order: top-left, top-right, bottom-left, bottom-right
TARGETS = ["delta_ms", "log_sSFR", "log_SFR", "log_Mstar"]
TARGET_NAMES = {
    "delta_ms":  "SFMS offset (Δ$_{MS}$)",
    "log_sSFR":  "Specific SFR (log sSFR)",
    "log_SFR":   "Star formation rate (log SFR)",
    "log_Mstar": "Stellar mass (log M$_*$)",
}

TOP_N = 6
ALLOW_MIXED_CONFIG = False

# Same palette as the morphology notebooks (and, for shared features, the
# forward-model figures) so a feature is always the same colour everywhere.
FEATURE_COLOURS = {
    "G":              "#1B9E77",
    "C":              "#7570B3",
    "A":              "#D95F02",
    "S":              "#E6AB02",
    "z":              "#C7EAE5",
    "seeing":         "#8C510A",
    "n_res":          "#01665E",
    "ellip":          "#4D9221",
    "log_Mstar":      "#2166AC",
    "log_sSFR":       "#4DAC26",
    "log_SFR":        "#B8E186",
    "delta_ms":       "#9B30D9",
    "5NN":            "#D6604D",
    "environment":    "#F4A582",
    "spectral_class": "#E08214",
    "re":             "#35978F",
    "re_kpc":         "#80CDC1",
    "RANDOM":         "#BBBBBB",
}


def _paths(algo, target):
    stem = os.path.join(IMPORTANCES_DIR, f"{algo}_morph_{target}")
    return f"{stem}.csv", f"{stem}_performance.csv"


def load_panel_data(algo, target):
    imp_path, perf_path = _paths(algo, target)
    if not (os.path.exists(imp_path) and os.path.exists(perf_path)):
        return None
    imp = pd.read_csv(imp_path, keep_default_na=False, na_values=[""])
    perf = pd.read_csv(perf_path, keep_default_na=False, na_values=[""])

    random_val = imp.loc[imp["Feature"] == "RANDOM", "Perm_Mean"].values[0]
    ranked = imp[imp["Feature"] != "RANDOM"].sort_values("Perm_Mean", ascending=False)
    top = ranked.head(TOP_N).iloc[::-1]  # reversed so the largest bar ends up on top

    def row(metric):
        r = perf.loc[perf["Metric"] == metric].iloc[0]
        return r["Mean"], r["Std"]

    meta = perf.iloc[0]
    config = (str(meta["controls"]) if pd.notna(meta["controls"]) else "",
              str(meta["condition_on"]) if pd.notna(meta["condition_on"]) else "")
    return {
        "top": top,
        "random_val": random_val,
        "r2": row("R2"),
        "gain": row("Gain_GCAS"),
        "config": config,
        "n": int(meta["n_galaxies"]),
    }


def config_label(config):
    controls, condition_on = config
    label = f"controls: {controls.replace('|', ', ') or 'none'}"
    if condition_on:
        label += f"  |  conditioned on: {condition_on.replace('|', ', ')}"
    return label


def plot_panel(ax, target, data, show_config):
    # Extra title padding leaves room for the per-panel config caption
    ax.set_title(TARGET_NAMES[target], fontsize=12, pad=16 if show_config else 6)

    if data is None:
        ax.text(0.5, 0.5, "not yet run", transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="#888888", style="italic")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#dddddd")
        return

    top = data["top"]
    features = top["Feature"].tolist()
    means = top["Perm_Mean"].values
    stds = top["Perm_Std"].values
    colours = [FEATURE_COLOURS.get(f, "#888888") for f in features]
    y_pos = np.arange(len(features))

    bars = ax.barh(y_pos, means, xerr=stds, capsize=3, color=colours,
                   edgecolor="white", linewidth=0.6,
                   error_kw={"elinewidth": 1.0, "ecolor": "#333333", "capthick": 1.0})
    for bar, role in zip(bars, top["Role"]):
        if role in ("control", "conditioning"):
            bar.set_hatch("//")
            bar.set_edgecolor("#555555")
    ax.axvline(data["random_val"], color="#555555", linestyle="--", linewidth=1.2, zorder=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlabel("Permutation importance\n(mean decrease in R²)", fontsize=9)

    r2_mean, r2_std = data["r2"]
    gain_mean, gain_std = data["gain"]
    ax.text(0.97, 0.05,
            f"CV R² = {r2_mean:.2f} ± {r2_std:.2f}\n"
            f"ΔR² from G,C,A,S = {gain_mean:+.2f} ± {gain_std:.2f}",
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.9))

    if show_config:
        ax.text(0.5, 1.012, config_label(data["config"]), transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7.5, color="#666666")


def make_figure(algo, show=False):
    panel_data = {t: load_panel_data(algo, t) for t in TARGETS}
    present = {t: d for t, d in panel_data.items() if d is not None}
    if not present:
        print(f"{algo}: no saved morphology importances found in {IMPORTANCES_DIR} — skipping")
        return None

    configs = {t: d["config"] for t, d in present.items()}
    mixed = len(set(configs.values())) > 1
    if mixed and not ALLOW_MIXED_CONFIG:
        detail = "\n".join(f"  {t}: {config_label(c)}" for t, c in configs.items())
        raise ValueError(
            f"{algo}: panels were run with different configs:\n{detail}\n"
            "Rerun the odd one(s) out, or set ALLOW_MIXED_CONFIG = True to plot anyway.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, target in zip(axes.flat, TARGETS):
        plot_panel(ax, target, panel_data[target], show_config=mixed)

    fig.suptitle(f"{ALGORITHMS.get(algo, algo)} — Predicting Galaxy Properties from "
                 "Hα Morphology (G, C, A, S)", fontsize=15, y=0.99)

    handles = [Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.2),
               Patch(facecolor="white", edgecolor="#555555", hatch="//")]
    labels = ["RANDOM noise-feature baseline", "control feature"]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.025),
               ncol=2, fontsize=9, frameon=False)

    n_values = sorted({d["n"] for d in present.values()})
    footnote = (f"Top {TOP_N} features by permutation importance of the G,C,A,S + baseline model "
                "(mean ± std over 5-fold CV). ΔR²: gain over the baseline-only model "
                "(repeated 5-fold CV). ")
    if not mixed:
        footnote += config_label(next(iter(configs.values()))) + ". "
    footnote += f"N = {', '.join(map(str, n_values))}."
    fig.text(0.5, 0.005, footnote, ha="center", fontsize=8.5, color="#444444", wrap=True)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{algo}_morph_importance_summary.png")
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


if __name__ == "__main__":
    for algo in ALGORITHMS:
        make_figure(algo)
