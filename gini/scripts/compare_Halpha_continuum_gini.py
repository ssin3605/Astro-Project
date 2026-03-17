import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Locating relevant directories --> change between using masked and unmasked dataset
halpha_csv = "gini/gini_outputs/tables/Halpha_gini_sfr_masked.csv"
cont_csv   = "gini/gini_outputs/tables/continuum_gini_sfr_masked.csv"

out_csv    = "gini/gini_outputs/tables/halpha_vs_continuum_gini_masked_comparison.csv"
plot_dir   = "gini/gini_outputs/figures"

# Using S/N-cut gini values
GINI_COL = "gini_all"   # or can switch for "gini_all"/"gini_sn3" as needed

# Quality cuts for ratio stability
EPS = 1e-6                 # avoid division by zero
MIN_CONT_GINI = 0.02       # skip extreme small denominators
MIN_PIXELS_COL = "n_sn3_pixels"  # will be used if present
MIN_PIXELS = 10

MAKE_PLOTS = True

def require_cols(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}. Found: {list(df.columns)}")
    
def save_scatter(x, y, xlabel, ylabel, title, filename, diag_11=False, hline0=False):
    m = np.isfinite(x) & np.isfinite(y)
    plt.figure()
    plt.scatter(x[m], y[m], s=10)
    if diag_11:
        lo = np.nanmin(np.concatenate([x[m], y[m]]))
        hi = np.nanmax(np.concatenate([x[m], y[m]]))
        plt.plot([lo, hi], [lo, hi], linestyle="--")
    if hline0:
        plt.axhline(0, linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename), dpi=200)
    plt.close()

def main():
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if MAKE_PLOTS:
        os.makedirs(plot_dir, exist_ok=True)

    h = pd.read_csv(halpha_csv)
    c = pd.read_csv(cont_csv)

    require_cols(h, ["catid", GINI_COL], "Halpha CSV")
    require_cols(c, ["catid", GINI_COL], "Continuum CSV")

    # Keep relevant columns
    keep_h = ["catid", GINI_COL]
    keep_c = ["catid", GINI_COL]

    # Extra columns like SFT, Mstar and sSFR will also be used for plotting
    extra_cols = ["logSFR", "logMstar", "logsSFR"]
    for col in extra_cols:
        if col in h.columns and col not in keep_h:
            keep_h.append(col)
        if col in c.columns and col not in keep_c:
            keep_c.append(col)

    # If n_sn3_pixels exists (which I believe it does only for Halpha gini), bring it in too
    if MIN_PIXELS_COL in h.columns:
        keep_h.append(MIN_PIXELS_COL)
    if MIN_PIXELS_COL in c.columns:
        keep_c.append(MIN_PIXELS_COL)

    # Collating Halpha and Continuum data
    h2 = h[keep_h].copy()
    c2 = c[keep_c].copy()

    # Merge data sets
    merged = pd.merge(
        h2,
        c2,
        on="catid",
        how="inner",
        suffixes=("_halpha", "_cont")
    )

    # Compute metrics
    gh = merged[f"{GINI_COL}_halpha"].astype(float)
    gc = merged[f"{GINI_COL}_cont"].astype(float)

    merged["gini_ratio"] = gh / (gc + EPS) # don't want anything /0
    merged["gini_delta"] = gh - gc
    merged["log_gini_ratio"] = np.log10(merged["gini_ratio"])

    # Quality cut masks
    mask = np.isfinite(gh) & np.isfinite(gc)
    mask &= (gc > MIN_CONT_GINI)  # stabilise ratio when continuum gini ~ 0

    # Optional pixel-count cut (if both exist, require both; if one exists, require it)
    if f"{MIN_PIXELS_COL}_halpha" in merged.columns:
        mask &= merged[f"{MIN_PIXELS_COL}_halpha"].fillna(0).astype(int) >= MIN_PIXELS
    if f"{MIN_PIXELS_COL}_cont" in merged.columns:
        mask &= merged[f"{MIN_PIXELS_COL}_cont"].fillna(0).astype(int) >= MIN_PIXELS

    merged_qc = merged[mask].copy()

    # Save merged table
    merged.to_csv(out_csv, index=False)
    print("Wrote merged table:", out_csv)
    print("Rows merged (inner join):", len(merged))
    print("Rows after QC mask:", len(merged_qc))

    if not MAKE_PLOTS:
        return

# 1) Hα vs Continuum (1:1 line)
    save_scatter(
        merged_qc[f"{GINI_COL}_cont"].to_numpy(),
        merged_qc[f"{GINI_COL}_halpha"].to_numpy(),
        xlabel=f"Continuum {GINI_COL}",
        ylabel=f"Hα {GINI_COL}",
        title=f"Masked: Hα vs Continuum Gini ({GINI_COL})",
        filename=f"masked_halpha_vs_continuum_{GINI_COL}.png",
        diag_11=True
    )

    # 2) ΔG = G(Hα) - G(cont) vs logM*
    if "logMstar_halpha" in merged_qc.columns or "logMstar_cont" in merged_qc.columns:
        # Prefer halpha side if present, else continuum
        xcol = "logMstar_halpha" if "logMstar_halpha" in merged_qc.columns else "logMstar_cont"
        save_scatter(
            merged_qc[xcol].to_numpy(),
            merged_qc["gini_delta"].to_numpy(),
            xlabel="log M*",
            ylabel="ΔG = G(Hα) − G(cont)",
            title="Masked: ΔG vs log M*",
            filename="masked_deltaG_vs_logMstar.png",
            hline0=True
        )

    # 3) ΔG vs logsSFR
    if "logsSFR_halpha" in merged_qc.columns or "logsSFR_cont" in merged_qc.columns:
        xcol = "logsSFR_halpha" if "logsSFR_halpha" in merged_qc.columns else "logsSFR_cont"
        save_scatter(
            merged_qc[xcol].to_numpy(),
            merged_qc["gini_delta"].to_numpy(),
            xlabel="log sSFR = logSFR − logM*",
            ylabel="ΔG = G(Hα) − G(cont)",
            title="Masked ΔG vs log sSFR",
            filename="masked_deltaG_vs_logsSFR.png",
            hline0=True
        )

    # 4) Ratio vs logM* (optional but useful)
    if "logMstar_halpha" in merged_qc.columns or "logMstar_cont" in merged_qc.columns:
        xcol = "logMstar_halpha" if "logMstar_halpha" in merged_qc.columns else "logMstar_cont"
        save_scatter(
            merged_qc[xcol].to_numpy(),
            merged_qc["gini_ratio"].to_numpy(),
            xlabel="log M*",
            ylabel="G(Hα) / G(cont)",
            title="Masked Gini ratio vs log M*",
            filename="masked_ratio_vs_logMstar.png"
        )

    # 5) Histogram of ΔG
    plt.figure()
    vals = merged_qc["gini_delta"].to_numpy()
    vals = vals[np.isfinite(vals)]
    plt.hist(vals, bins=40)
    plt.xlabel("ΔG = G(Hα) − G(cont)")
    plt.ylabel("Count")
    plt.title("Masked Distribution of ΔG")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "masked_deltaG_hist.png"), dpi=200)
    plt.close()

    print("Wrote plots to:", plot_dir)


if __name__ == "__main__":
    main()