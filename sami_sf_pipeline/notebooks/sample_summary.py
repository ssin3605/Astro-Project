"""
Sample summary table for the thesis.

Follows the sample from the raw SAMI DR3 Halpha maps, through Wang et al.
(2022)'s six sequential quality cuts, to the clean sample. Then breaks the
clean sample down by survey and spectral class, metric validity, and
environment coverage.

Every row in the selection section is a strict subset of the row above it.
A metric counts as valid only when it has an actual (finite) value - an
in_<metric> flag on its own isn't enough.

Read-only: reads master_final.csv and the raw map filenames, and writes
outputs/csv/sample_summary_<date>.csv.

Run from sami_sf_pipeline/:
    python notebooks/sample_summary.py
"""

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
RAW_HALPHA_DIR = PIPELINE_ROOT / "data" / "raw" / "sami_dr3_Halpha_maps"
CSV_DIR = PIPELINE_ROOT / "outputs" / "csv"
MASTER_CSV = CSV_DIR / "master_final.csv"

# Wang (2022) cuts, in the order the paper applies them.
CUTS = [
    ("cut_size", "Size (r_e <= 15 arcsec)"),
    ("cut_inclination", "Inclination (ellip <= 0.7)"),
    ("cut_seeing", "Seeing (seeing/r_e < 0.75, seeing < 4 arcsec)"),
    ("cut_star_forming", "Star-forming (EW(Halpha) >= 1 A)"),
    ("cut_not_pure_agn", "Not pure AGN (galclass != 4)"),
    ("cut_ssfr", "sSFR (log sSFR > -11.25)"),
]
METRICS = [("G", "Gini (G)"), ("C", "Concentration (C)"),
           ("A", "Asymmetry (A)"), ("S", "Smoothness (S)")]


def raw_catids(directory: Path) -> set[int]:
    """Unique CATIDs from filenames like '100162_A_Halpha_default_1-comp.fits'.
    Several files can share a CATID (map variants, OS duplicate copies)."""
    pattern = re.compile(r"^(\d+)_")
    return {int(m.group(1)) for f in directory.iterdir() if (m := pattern.match(f.name))}


def has_value(series: pd.Series) -> pd.Series:
    """Finite and not the -999 sentinel."""
    values = pd.to_numeric(series, errors="coerce")
    return np.isfinite(values) & (values != -999)


def main() -> None:
    master = pd.read_csv(MASTER_CSV, low_memory=False)
    raw_ids = raw_catids(RAW_HALPHA_DIR)

    rows = []

    def add(section, stage, n, removed=None):
        rows.append({"section": section, "stage": stage,
                     "n_removed": removed, "n": int(n)})

    # Selection: raw maps -> Wang classified -> six sequential cuts.
    sel = "Selection"
    add(sel, "SAMI DR3 Halpha maps (unique CATID)", len(raw_ids))

    sample = master[master["in_C"] & master["CATID"].isin(raw_ids)]
    add(sel, "Classified SF or composite by Wang (2022)", len(sample),
        removed=len(raw_ids) - len(sample))

    for col, label in CUTS:
        kept = sample[sample[col] == True]  # NaN counts as fail
        add(sel, label, len(kept), removed=len(sample) - len(kept))
        sample = kept

    # Sanity check: sequential funnel should equal the pipeline's own flag.
    n_flag = int((master["in_full_sam"] & master["CATID"].isin(raw_ids)).sum())
    assert len(sample) == n_flag, f"funnel gives {len(sample)}, in_full_sam gives {n_flag}"
    add(sel, "Clean sample", len(sample))

    clean = sample

    # Breakdown of the clean sample.
    brk = "Clean sample breakdown"
    for env in ["GAMA", "Cluster"]:
        add(brk, f"Survey: {env}", (clean["environment"] == env).sum())
    for cls, label in [("pure_SF", "pure SF"), ("composite_SF", "composite")]:
        add(brk, f"Spectral class: {label}", (clean["spectral_class"] == cls).sum())

    # Metric validity within the clean sample.
    met = "Valid metrics (clean sample)"
    valid = pd.DataFrame({m: has_value(clean[m]) for m, _ in METRICS})
    for m, label in METRICS:
        add(met, label, valid[m].sum())
    add(met, "All four (G, C, A, S)", valid.all(axis=1).sum())

    # Environment coverage within the clean sample.
    env = "Environment coverage (clean sample)"
    status = clean["group_status"]
    add(env, "Group status available (GAMA only)", status.notna().sum())
    for s in ["central", "satellite", "isolated"]:
        add(env, f"  {s}", (status == s).sum())
    add(env, "5th-nearest-neighbour density (5NN)", has_value(clean["5NN"]).sum())

    summary = pd.DataFrame(rows)
    summary["n_removed"] = summary["n_removed"].astype("Int64")

    with pd.option_context("display.max_rows", None, "display.width", 120):
        for section, block in summary.groupby("section", sort=False):
            print(f"\n{section}")
            print(block.drop(columns="section").to_string(index=False))

    out_path = CSV_DIR / f"sample_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nWritten: {out_path.relative_to(PIPELINE_ROOT)}")


if __name__ == "__main__":
    main()
