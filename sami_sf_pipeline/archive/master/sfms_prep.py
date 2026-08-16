import pandas as pd
import numpy as np
import sys
import os

## Run once to get the SFMS line and calculate the \delta MS values for each galaxy and add these to master_sfms.csv

# Make src/ importable regardless of where the script is run from
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.sfms import fit_ridge_line, compute_delta_ms

# ── Paths ────────────────────────────────────────────────────────

INPUT_PATH  = "master/outputs/tables/master_results.csv"
OUTPUT_PATH = "master/outputs/tables/master_sfms.csv"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ── Load data ────────────────────────────────────────────────────

df = pd.read_csv(INPUT_PATH)

log_mstar = df['log_Mstar'].values
log_sfr   = df['log_SFR'].values     # adjust to your actual column name

# ── Fit SFMS on star-forming galaxies only ───────────────────────
# Rough SF selection: finite SFR and above a plausible lower limit.
# Refine this threshold once you have looked at your data distribution.

sf_mask = np.isfinite(log_sfr) & (log_sfr > -2.0)

sfms_params, ridge_mass, ridge_sfr = fit_ridge_line(
    log_mstar[sf_mask],
    log_sfr[sf_mask]
)

S0, a1, Mt = sfms_params
print("SFMS fit complete:")
print(f"  S0 = {S0:.4f}")
print(f"  a1 = {a1:.4f}")
print(f"  Mt = {Mt:.4f}")
print(f"  Fit anchored on {len(ridge_mass)} mass bins")

# ── Compute ∆MS for all galaxies ─────────────────────────────────
# Galaxies without a valid SFR will receive NaN, not a spurious value.

df['delta_ms'] = compute_delta_ms(log_mstar, log_sfr, sfms_params)

print(f"\ndelta_ms computed for {df['delta_ms'].notna().sum()} "
      f"of {len(df)} galaxies")

# ── Save ─────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved to {OUTPUT_PATH}")