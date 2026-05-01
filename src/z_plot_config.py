# This is the single place where you define what every column is called 
# in a plot, what colour scales to use, and what the environment bins mean. 
# Changing a label here changes it everywhere.

# ── Axis labels ───────────────────────────────────────────────────────────────

AXIS_LABELS = {
    "log_Mstar"  : r"$\log(M_\star / M_\odot)$",
    "log_SFR"    : r"$\log(\mathrm{SFR} / M_\odot\,\mathrm{yr}^{-1})$",
    "log_sSFR"   : r"$\log(\mathrm{sSFR} / \mathrm{yr}^{-1})$",
    "delta_ms"   : r"$\Delta\mathrm{MS}$",
    "galclass"   : "Environment (halo mass bin)",
    "gini_all_halpha" : "Gini",
    "CIndex"     : "Concentration",
    "A_min"      : "Asymmetry",
    "S"          : "Smoothness",
}

# ── Colour scales per metric ──────────────────────────────────────────────────

CMAPS = {
    "gini_all_halpha" : "plasma",
    "CIndex"          : "plasma",
    "A_min"           : "plasma",
    "S"               : "plasma",
    "delta_ms"        : "coolwarm",
    "log_SFR"         : "viridis",
    "log_sSFR"        : "viridis",
    "log_Mstar"       : "viridis",
}

# ── Environment bin ordering and display labels ───────────────────────────────
# Matches the 4 galclass categories from Wang+2022

GALCLASS_ORDER  = [1, 2, 3, 4]
GALCLASS_LABELS = {
    1      : r"Ungrouped",
    2      : r"Low-mass group ($M_{200} \leq 10^{12.5}$)",
    3      : r"High-mass group ($10^{12.5} < M_{200} \leq 10^{14}$)",
    4      : r"Cluster ($M_{200} > 10^{14}$)",
}

# Discrete colour for each environment bin (used when galclass is
# the colour axis rather than a continuous metric)
GALCLASS_COLOURS = {
    1      : "#4477AA",
    2      : "#66CCEE",
    3      : "#CCBB44",
    4      : "#EE6677",
}

# ── Valid axis columns ────────────────────────────────────────────────────────

VALID_X = {"log_Mstar", "galclass"}
VALID_Y = {"log_SFR", "log_sSFR", "log_Mstar", "delta_ms", "galclass"}
VALID_C = {"gini_all_halpha", "CIndex", "A_min", "S",
           "delta_ms", "log_SFR", "log_sSFR", "log_Mstar", "galclass"}