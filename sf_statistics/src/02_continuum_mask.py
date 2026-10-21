#!/usr/bin/env python3
"""
Computes the continuum-derived S/N mask for every galaxy and applies it
to both the continuum map and the Halpha cube.

Single masking step in the pipeline. Every downstream metric reads its input
directly from the product of this step, so Gini, Asymmetry, and Clumpiness 
are guaranteed to share the same pixel mask (Concentration is the one metric 
that does not use this mask, because it is sourced directly from Wang's 
SAMI_CIndex_data catalogue).

Mask definition:
    keep pixel if continuum flux and error are finite, error > 0,
    flux > 0, and S/N >= SN_CUT (default 3.0)

Input : data/fits/continuum_maps/, data/fits/sami_dr3_Halpha_maps/
Output: data/masked/continuum/<CATID>_<IFU>_continuum_masked.fits
        data/masked/halpha/<CATID>_<IFU>_Halpha_masked.fits    

Usage
    python src/02_continuum_mask.py                # full sample
    python src/02_continuum_mask.py --catid 7139    # single galaxy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from tqdm import tqdm

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from src.utils.masking import (  
    apply_mask_2d,
    apply_mask_3d,
    make_continuum_mask,
    save_masked_continuum,
    save_masked_halpha,
)
from src.utils.paths import MASKED_CONT_DIR, MASKED_HALPHA_DIR, RAW_CONT_DIR, RAW_HALPHA_DIR

OUT_CONT_DIR = MASKED_CONT_DIR
OUT_HALPHA_DIR = MASKED_HALPHA_DIR

SN_CUT = 3.0

def galaxy_id_from_continuum_name(filename: str) -> str:
    parts = filename.replace(".fits", "").split("_")
    return f"{parts[0]}_{parts[1]}"

def halpha_filename(galaxy_id: str) -> str:
    return f"{galaxy_id}_Halpha_default_1-comp.fits"

def out_continuum_filename(galaxy_id: str) -> str:
    return f"{galaxy_id}_continuum_masked.fits"

def out_halpha_filename(galaxy_id: str) -> str:
    return f"{galaxy_id}_Halpha_masked.fits"

def process_one(cont_path: Path, sn_cut: float = SN_CUT) -> dict:
    # Mask one galaxy's continuum + Halpha data
    galaxy_id = galaxy_id_from_continuum_name(cont_path.name)
    halpha_path = RAW_HALPHA_DIR / halpha_filename(galaxy_id)

    if not halpha_path.exists():
        return {"galaxy_id": galaxy_id, "status": "missing_halpha"}

    with fits.open(cont_path) as hdul:
        cont_flux = np.array(hdul[0].data, dtype=float)
        cont_err = np.array(hdul[1].data, dtype=float)
        cont_flux_header = hdul[0].header.copy()
        cont_err_header = hdul[1].header.copy()

    mask, _sn_map = make_continuum_mask(cont_flux, cont_err, sn_cut=sn_cut)

    masked_cont_flux = apply_mask_2d(cont_flux, mask)
    masked_cont_err = apply_mask_2d(cont_err, mask)

    with fits.open(halpha_path) as hdul:
        halpha_flux_raw = np.array(hdul[0].data, dtype=float)
        halpha_err_raw = np.array(hdul[1].data, dtype=float)
        halpha_flux_header = hdul[0].header.copy()
        halpha_err_header = hdul[1].header.copy()

    # Standardise to (2, ny, nx) - the axis of length 2 may be first or last
    # depending on file version (mirrors src/utils/io.py's convention).
    axf = [ax for ax, n in enumerate(halpha_flux_raw.shape) if n == 2][0]
    axe = [ax for ax, n in enumerate(halpha_err_raw.shape) if n == 2][0]
    halpha_flux_cube = np.moveaxis(halpha_flux_raw, axf, 0)
    halpha_err_cube = np.moveaxis(halpha_err_raw, axe, 0)

    if halpha_flux_cube.shape[1:] != mask.shape:
        return {"galaxy_id": galaxy_id, "status": "shape_mismatch"}

    masked_halpha_flux = apply_mask_3d(halpha_flux_cube, mask)
    masked_halpha_err = apply_mask_3d(halpha_err_cube, mask)

    OUT_CONT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_HALPHA_DIR.mkdir(parents=True, exist_ok=True)

    save_masked_continuum(
        OUT_CONT_DIR / out_continuum_filename(galaxy_id),
        masked_cont_flux, masked_cont_err,
        flux_header=cont_flux_header, err_header=cont_err_header,
    )
    save_masked_halpha(
        OUT_HALPHA_DIR / out_halpha_filename(galaxy_id),
        masked_halpha_flux, masked_halpha_err,
        flux_header=halpha_flux_header, err_header=halpha_err_header,
    )

    n_valid = int(mask.sum())
    n_pix = int(mask.size)
    return {
        "galaxy_id": galaxy_id,
        "status": "ok",
        "n_valid_pixels": n_valid,
        "n_pixels": n_pix,
        "valid_frac": n_valid / n_pix if n_pix else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catid", type=int, default=None, help="Process a single CATID instead of the full sample.")
    parser.add_argument("--sn-cut", type=float, default=SN_CUT, help=f"Continuum S/N threshold (default {SN_CUT}).")
    args = parser.parse_args()

    if args.catid is not None:
        cont_files = sorted(RAW_CONT_DIR.glob(f"{args.catid}_*_HALPHA_cont_map_default.fits"))
        if not cont_files:
            raise FileNotFoundError(f"No raw continuum FITS found for CATID {args.catid}")
    else:
        cont_files = sorted(RAW_CONT_DIR.glob("*_HALPHA_cont_map_default.fits"))
        if not cont_files:
            raise FileNotFoundError(f"No continuum files found in {RAW_CONT_DIR}")

    print(f"Found {len(cont_files)} continuum file(s). S/N cut = {args.sn_cut}")

    counts = {"ok": 0, "missing_halpha": 0, "shape_mismatch": 0}
    for cont_path in tqdm(cont_files, desc="Masking"):
        result = process_one(cont_path, sn_cut=args.sn_cut)
        counts[result["status"]] = counts.get(result["status"], 0) + 1
        if result["status"] != "ok":
            tqdm.write(f"  [{result['status']}] {result['galaxy_id']}")

    print("\n--- Summary ---")
    print(f"  Total continuum files : {len(cont_files)}")
    print(f"  Masked OK             : {counts.get('ok', 0)}")
    print(f"  Missing Halpha        : {counts.get('missing_halpha', 0)}")
    print(f"  Shape mismatch        : {counts.get('shape_mismatch', 0)}")
    print(f"  Output (continuum)    : {OUT_CONT_DIR}")
    print(f"  Output (Halpha)       : {OUT_HALPHA_DIR}")


if __name__ == "__main__":
    main()
