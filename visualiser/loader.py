"""
Handles FITS file discovery and catalogue lookup for SAMI DR3 data.

SAMI DR3 file structure assumed:
  <data_root>/<CATID>/<CATID>_<component>_<type>.fits

"""

import os
import glob
import pandas as pd
from pathlib import Path


DATA_ROOT = os.environ.get("SAMI_DATA_ROOT", "./data/fits")
CATALOGUE_PATH = os.environ.get("SAMI_CATALOGUE", "./master/outputs/tables/master_results.csv")
# ─────────────────────────────────────────────────────────────────────────────

# Map keys → (filename fragment patterns to try, FITS extension)
# Extend this dict as you add more map types.
MAP_PATTERNS = {
    # Key               Filename fragments (tried in order)          HDU ext
    "continuum":       (["*_continuum*", "*_stellar_cont*",
                         "*white*", "*cont*"],                        0),
    "halpha_flux":     (["*Halpha*flux*", "*Ha_flux*",
                         "*HALPHA*FLUX*", "*halpha*"],                0),
    "halpha_snr":      (["*Halpha*SNR*", "*Ha_SNR*",
                         "*HALPHA*SNR*"],                             0),
    "halpha_ew":       (["*Halpha*EW*", "*Ha_EW*"],                  0),
    "gas_velocity":    (["*gas*velocity*", "*Ha*velocity*",
                         "*Halpha*vel*", "*gas_vel*"],                0),
    "gas_dispersion":  (["*gas*disp*", "*Ha*disp*",
                         "*Halpha*disp*"],                            0),
    "stellar_vel":     (["*stellar*velocity*", "*stars*vel*",
                         "*stel*vel*"],                               0),
    "stellar_disp":    (["*stellar*disp*", "*stel*disp*"],           0),
    "hbeta_flux":      (["*Hbeta*flux*", "*HB*flux*"],               0),
    "nii_flux":        (["*NII*flux*", "*N2*flux*"],                  0),
    "oiii_flux":       (["*OIII*flux*", "*O3*flux*"],                 0),
}

# Human-readable titles for each map key
MAP_TITLES = {
    "continuum":      "Continuum",
    "halpha_flux":    "Hα Flux",
    "halpha_snr":     "Hα SNR",
    "halpha_ew":      "Hα EW",
    "gas_velocity":   "Gas Velocity",
    "gas_dispersion": "Gas Dispersion",
    "stellar_vel":    "Stellar Velocity",
    "stellar_disp":   "Stellar Dispersion",
    "hbeta_flux":     "Hβ Flux",
    "nii_flux":       "[NII] Flux",
    "oiii_flux":      "[OIII] Flux",
}

# Units label per map type (for colourbar)
MAP_UNITS = {
    "continuum":      "counts",
    "halpha_flux":    r"$10^{-16}$ erg/s/cm²/Å",
    "halpha_snr":     "SNR",
    "halpha_ew":      "Å",
    "gas_velocity":   "km/s",
    "gas_dispersion": "km/s",
    "stellar_vel":    "km/s",
    "stellar_disp":   "km/s",
    "hbeta_flux":     r"$10^{-16}$ erg/s/cm²/Å",
    "nii_flux":       r"$10^{-16}$ erg/s/cm²/Å",
    "oiii_flux":      r"$10^{-16}$ erg/s/cm²/Å",
}


def find_map_file(catid: str, map_key: str, data_root: str = DATA_ROOT):
    """
    Search for a SAMI map FITS file for a given galaxy and map type.

    Tries multiple filename patterns to handle DR3 naming variations.
    Returns the file path string, or None if not found.
    """
    catid = str(catid)
    patterns, _ = MAP_PATTERNS.get(map_key, ([], 0))

    # Search locations: galaxy-specific subfolder first, then flat root
    search_dirs = [
        os.path.join(data_root, catid),
        data_root,
    ]

    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        for pattern in patterns:
            # Case-insensitive glob on Linux requires manual check
            matches = (
                glob.glob(os.path.join(search_dir, pattern))
                + glob.glob(os.path.join(search_dir, pattern.lower()))
                + glob.glob(os.path.join(search_dir, pattern.upper()))
            )
            # Filter: file must contain the CATID
            matches = [m for m in matches if catid in os.path.basename(m)]
            if matches:
                return matches[0]
    return None


def list_available_maps(catid: str, data_root: str = DATA_ROOT) -> dict:
    """
    Returns a dict of {map_key: filepath} for all maps found for a galaxy.
    Missing maps have value None.
    """
    return {
        key: find_map_file(catid, key, data_root)
        for key in MAP_PATTERNS
    }


# ── Catalogue ─────────────────────────────────────────────────────────────────

_catalogue_cache = None


def load_catalogue(path: str = CATALOGUE_PATH) -> pd.DataFrame:
    """Load master_results.csv with caching."""
    global _catalogue_cache
    if _catalogue_cache is None:
        _catalogue_cache = pd.read_csv(path)
        # Normalise CATID column: try common column names
        for col in ["CATID", "catid", "CatID", "galaxy_id", "ID"]:
            if col in _catalogue_cache.columns:
                _catalogue_cache = _catalogue_cache.rename(columns={col: "CATID"})
                break
        _catalogue_cache["CATID"] = _catalogue_cache["CATID"].astype(str)
    return _catalogue_cache


def get_galaxy_params(catid: str, catalogue_path: str = CATALOGUE_PATH) -> dict:
    """
    Retrieve catalogue parameters for a galaxy as a dict.
    Returns empty dict if CATID not found.
    """
    df = load_catalogue(catalogue_path)
    row = df[df["CATID"] == str(catid)]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()