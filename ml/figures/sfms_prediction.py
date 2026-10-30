"""Precision of predicting the SFMS offset (delta_ms) from Halpha morphology (G, C, A, S).

ONE random forest predicts delta_ms directly from G, C, A, S + the z/seeing controls.
Out-of-fold: 5-fold CV, so every galaxy is predicted by a model that never saw it.

Why predict delta_ms directly, instead of predicting log_SFR (or log_sSFR) and subtracting a
second model's prediction back to delta_ms?
    delta_ms = log_SFR - log_SFR_MS(M*) is already the mass-independent quantity -- the SFMS's
    mass dependence is subtracted out *before* training, as a fixed formula, not learned by the
    model. Predicting it directly needs no second model and no subtraction step.

    An earlier version of this script instead fit a second model (log_SFR or log_sSFR) and
    subtracted it from the delta_ms model's prediction. That is unreliable once log_Mstar is an
    input feature to both models: each one partially "looks up" the same low-noise M*-SFMS curve
    from that single feature, and subtracting cancels almost all of that shared, non-morphological
    signal -- confirmed by a test where removing G/C/A/S entirely *shrank* the apparent error
    (0.052 dex) rather than growing it. That result was an artifact of the construction, not a
    measurement of morphology's predictive power.

Since log_SFR = log_SFR_MS(M*_true) + delta_ms exactly, the error on an "implied log_SFR" built
from (the true SFMS curve at the galaxy's TRUE mass) + (the predicted delta_ms) is identical to
the delta_ms error -- no cancellation risk, because the curve term is a deterministic formula
evaluated at the true mass, not a second fitted model. This is confirmed below rather than assumed.

Two runs:
    'Morphology alone'    features = G, C, A, S, z, seeing
    'Morphology + mass'   features = G, C, A, S, z, seeing, log_Mstar
The second checks whether morphology is doing real work beyond standing in for stellar mass: if
the error barely changes when mass is already known to the model, morphology carries information
about delta_ms beyond what it reveals about mass. 'Morphology alone' is the headline run (no mass
leakage risk at all) and is what the figure below uses.

Errors are in dex:
    RMSE = sqrt(mean(error^2))   -- typical error, weights big misses more; same kind of number
                                     as a scatter (e.g. the literature's 0.3 dex)
    MAE  = mean(|error|)         -- plain average miss
The "+/-" is the spread of that number across the 5 test folds, i.e. how much it changes
depending on which galaxies happen to be tested.

Figure: (a) measured galaxies around the fitted SFMS, with the literature and this-sample scatter;
        (b) each galaxy's implied-SFR error (= its delta_ms error) against its true mass. A
            perfect model puts every point on zero.

Run:  python sfms_prediction.py   (from any folder)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "sf_statistics"))
from src.utils.paths import CH4_FIGURES_DIR, MASTER_CSV  # noqa: E402

# outputs/pipeline/master.csv in, outputs/ch4/figures/ out
DATA_PATH = str(MASTER_CSV)
OUTPUT_DIR = str(CH4_FIGURES_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
MORPH_FEATURES = ['G', 'C', 'A', 'S']
CONTROL_FEATURES = ['z', 'seeing']
LITERATURE_SCATTER = 0.3   # dex, literature SFMS scatter
N_FOLDS = 5
SEED = 42

RUNS = {
    'Morphology alone':  MORPH_FEATURES + CONTROL_FEATURES,
    'Morphology + mass': MORPH_FEATURES + CONTROL_FEATURES + ['log_Mstar'],
}
HEADLINE_RUN = 'Morphology alone'

# ── Load the same sample as the morphology notebooks ──
df = pd.read_csv(DATA_PATH)
needed_cols = list(dict.fromkeys(MORPH_FEATURES + CONTROL_FEATURES + ['log_Mstar', 'delta_ms', 'log_SFR']))
df = df[df['in_full_sam'] == True].dropna(subset=needed_cols)
df = df[~(df[needed_cols] == -999).any(axis=1)].reset_index(drop=True)

mstar = df['log_Mstar'].values
true_dms = df['delta_ms'].values
true_sfr = df['log_SFR'].values
true_ms = true_sfr - true_dms   # SFMS curve evaluated at each galaxy's true mass (exact, pointwise)

folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(df)), df['environment']))


def rmse(err):
    return np.sqrt(np.mean(err ** 2))


def mae(err):
    return np.mean(np.abs(err))


def fold_uncertainty(err, stat):
    return np.std([stat(err[test]) for _, test in folds])


def run_model(features):
    # Out-of-fold delta_ms predictions for one feature set
    X = df[features].values
    pred = np.zeros(len(df))
    for k, (train, test) in enumerate(folds):
        rf = RandomForestRegressor(n_estimators=500, max_features='sqrt', min_samples_leaf=5,
                                   random_state=SEED + k, n_jobs=-1)
        rf.fit(X[train], true_dms[train])
        pred[test] = rf.predict(X[test])
    return pred


print(f"N galaxies: {len(df)}\n")
print("Scatter of delta_ms about the SFMS (the reference every model below is compared against):")
print(f"  literature                         {LITERATURE_SCATTER:.2f} dex")
print(f"  this sample (RMS of true delta_ms) {rmse(true_dms):.3f} dex\n")

print("Out-of-fold delta_ms prediction (5-fold CV):")
run_preds = {}
for name, features in RUNS.items():
    pred = run_model(features)
    run_preds[name] = pred
    err = pred - true_dms
    print(f"  {name} (features = {features})")
    print(f"    RMSE = {rmse(err):.3f} ± {fold_uncertainty(err, rmse):.3f} dex   "
          f"MAE = {mae(err):.3f} ± {fold_uncertainty(err, mae):.3f} dex")

pred_dms = run_preds[HEADLINE_RUN]

# ── Sanity check: implied log_SFR error (true SFMS curve at the TRUE mass + predicted delta_ms)
# is identical to the delta_ms error -- confirming no reconstruction step is needed to go from
# delta_ms precision to SFR precision. ──
implied_sfr = true_ms + pred_dms
max_diff = np.max(np.abs((implied_sfr - true_sfr) - (pred_dms - true_dms)))
print(f"\nImplied log_SFR error == delta_ms error exactly (max diff {max_diff:.2e} dex):")
print("  predicting delta_ms directly already gives the SFR-recovery precision --")
print("  no second model or subtraction step needed.")

# Error in the implied SFR (= delta_ms error), binned in true stellar mass (8 equal-count bins)
ms_err = pred_dms - true_dms
N_MASS_BINS = 8
bin_edges = np.quantile(mstar, np.linspace(0, 1, N_MASS_BINS + 1))
bin_idx = np.digitize(mstar, bin_edges[1:-1])
bin_mass = np.array([mstar[bin_idx == i].mean() for i in range(N_MASS_BINS)])
bin_err = np.array([np.median(ms_err[bin_idx == i]) for i in range(N_MASS_BINS)])
print(f"\nMedian implied-SFR error by stellar mass ({HEADLINE_RUN}):")
print('  ' + '   '.join(f"logM*={m:.1f}: {e:+.2f}" for m, e in zip(bin_mass, bin_err)) + ' dex')

# ── Figure: error in the SFR implied by the predicted delta_ms, vs. stellar mass ──
fig, ax_b = plt.subplots(figsize=(6.5, 5.2))

ax_b.scatter(mstar, ms_err, s=10, color='#1F77E8', alpha=0.5, label='Each galaxy')
ax_b.plot(bin_mass, bin_err, 'o-', color='black', linewidth=2, label='Binned median')
ax_b.axhline(0, color='black', linewidth=1)
ax_b.axhspan(-rmse(ms_err), rmse(ms_err), color='#1F77E8', alpha=0.12,
             label=f'± RMSE = {rmse(ms_err):.2f} dex')
ax_b.set_xlabel('log M* [M☉]  (true)')
ax_b.set_ylabel('Predicted SFMS offset [dex]')
ax_b.legend(fontsize=9)

plt.tight_layout()
for ext in ('pdf', 'png'):
    out_path = os.path.join(OUTPUT_DIR, f'sfms_prediction.{ext}')
    fig.savefig(out_path, dpi=200)
    print(f'Saved {out_path}')
plt.show()