import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Use to plot data calulated in Halpha .csv file

infile = "gini/gini_outputs/tables/Halpha_gini.csv"
outdir = "gini/gini_outputs/figures"
os.makedirs(outdir, exist_ok=True)

df = pd.read_csv(infile)

highlight_ids = [617665, 107259, 105420, 9008500868]  # 4 selected catids
highlight_markers = ["o", "s", "^", "D"]         # circle, square, triangle, diamond

def make_plot(x, y, c, xlabel, ylabel, clabel, title, outname, highlight_ids=None, highlight_markers=None, annotate=True):
    d = df[np.isfinite(df[x]) & np.isfinite(df[y]) & np.isfinite(df[c])].copy()
    
    plt.figure()
    sc = plt.scatter(d[x], d[y], c=d[c], s=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title} (N={len(d)})") 
    cb = plt.colorbar(sc)
    cb.set_label(clabel)

    # Highlighting markers
    if highlight_ids is not None and len(highlight_ids) > 0:
        # default marker set (4 distinct shapes)
        if highlight_markers is None:
            highlight_markers = ["o", "s", "^", "D"]

        for i, CATID in enumerate(highlight_ids):
            sel = d[d["CATID"] == CATID]
            if sel.empty:
                print(f"WARNING: CATID={CATID} not found in plotted subset for {outname}")
                continue

            m = highlight_markers[i % len(highlight_markers)]

            plt.scatter(sel[x], sel[y],
                        s=180, marker=m,
                        facecolors="none", edgecolors="k",
                        linewidths=2.0, zorder=5,
                        label=f"{CATID}")

            if annotate:
                # if duplicate rows exist, annotate the first
                r = sel.iloc[0]
                plt.annotate(str(int(CATID)),
                             (r[x], r[y]),
                             textcoords="offset points",
                             xytext=(6, 6),
                             fontsize=9, weight="bold",
                             color="k")

        plt.legend(title="Highlighted catid", frameon=True, loc="best")
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, outname), dpi=200)
    plt.close()

# 1) logSFR vs logMstar
make_plot(
    "logMstar", "logSFR", "gini_all",
    r"$\log(M_\star)$",
    r"$\log(\mathrm{SFR})$",
    r"Gini(H$\alpha$)",
    r"Gini(H$\alpha$)",
    "Halpha_masked__logSFR_vs_logMstar.png",
    highlight_ids=highlight_ids,
    highlight_markers=highlight_markers,
    annotate=True
)

# 2) logsSFR vs logMstar
make_plot(
    "logMstar", "logsSFR", "gini_all",
    r"$\log(M_\star)$",
    r"$\log(\mathrm{sSFR})$",
    r"Gini(H$\alpha$)",
    r"Gini(H$\alpha$ Masked)",
    "Halpha_masked__logsSFR_vs_logMstar.png",
    highlight_ids=highlight_ids,
    highlight_markers=highlight_markers,
    annotate=True
)

print("Wrote figures to:", outdir)