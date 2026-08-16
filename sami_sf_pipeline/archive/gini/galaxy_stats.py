"""
Prints key parameters for a given CATID from Halpha_gini.csv

Outputs: CATID, log(M*), log(sSFR), log(SFR), Gini, Environment
"""

import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

CATID   = 107259  # <-- change this
GINI_CSV = 'gini/gini_outputs/tables/Halpha_gini.csv'

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(GINI_CSV)

# Ensure consistent column name
if 'catid' in df.columns:
    df = df.rename(columns={'catid': 'CATID'})

df['CATID'] = df['CATID'].astype(int)

# ============================================================
# LOOKUP AND PRINT
# ============================================================

row = df[df['CATID'] == CATID]

divider = '=' * 40

if row.empty:
    print(f'\n{divider}')
    print(f'  CATID {CATID} not found in Halpha_gini.csv.')
    print(f'{divider}\n')
else:
    r = row.iloc[0]

    print(f'\n{divider}')
    print(f'  CATID       : {int(r["CATID"])}')

    # Use existing log columns directly
    print(f'  log(M*)     : {r["logMstar"]:.3f}')
    print(f'  log(sSFR)   : {r["logsSFR"]:.3f}')
    print(f'  log(SFR)    : {r["logSFR"]:.3f}')

    # Gini coefficient
    print(f'  Gini        : {r["gini_all"]:.4f}')

    # Environment (if present)
    if 'environment' in r:
        print(f'  Environment : {r["environment"]}')

    print(f'{divider}\n')