from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from src.masking import (
    load_continuum_maps,
    load_halpha_cubes,
    make_continuum_mask,
    apply_mask_2d,
    apply_mask_3d,
    save_masked_continuum,
    save_masked_halpha,
)

CONT_DIR = PROJECT_ROOT / "data" / "fits" / "continuum_maps"
HALPHA_DIR = PROJECT_ROOT / "data" / "fits" / "sami_dr3_Halpha_maps"

OUT_CONT_DIR = PROJECT_ROOT / "gini" / "gini_outputs" / "tables" / "masked_continuum_maps"
OUT_HALPHA_DIR = PROJECT_ROOT / "gini" / "gini_outputs" / "tables" / "masked_halpha_maps"

SN_CUT = 3.0


def get_galaxy_id_from_continuum_name(filename):
    """
    6837_A_HALPHA_cont_map_default.fits -> 6837_A
    """
    parts = filename.replace(".fits", "").split("_")
    return f"{parts[0]}_{parts[1]}"


def build_halpha_filename(galaxy_id):
    """
    6837_A -> 6837_A_Halpha_default_1-comp.fits
    """
    return f"{galaxy_id}_Halpha_default_1-comp.fits"


def build_output_cont_filename(galaxy_id):
    return f"{galaxy_id}_HALPHA_cont_map_masked.fits"


def build_output_halpha_filename(galaxy_id):
    return f"{galaxy_id}_Halpha_masked.fits"

def main():
    OUT_CONT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_HALPHA_DIR.mkdir(parents=True, exist_ok=True)

    cont_files = sorted(CONT_DIR.glob("*_HALPHA_cont_map_default.fits"))

    if not cont_files:
        print(f"No continuum files found in: {CONT_DIR}")
        return

    n_total = 0
    n_done = 0
    n_missing_halpha = 0
    n_failed = 0

    for cont_path in cont_files:
        n_total += 1
        galaxy_id = get_galaxy_id_from_continuum_name(cont_path.name)
        halpha_path = HALPHA_DIR / build_halpha_filename(galaxy_id)

        if not halpha_path.exists():
            print(f"[MISSING Halpha] {galaxy_id} -> {halpha_path.name}")
            n_missing_halpha += 1
            continue

        try:
            # Load continuum 2D maps
            cont_flux, cont_err, cont_flux_header, cont_err_header = load_continuum_maps(cont_path)

            # Build continuum-derived mask
            mask, sn_map = make_continuum_mask(cont_flux, cont_err, sn_cut=SN_CUT)

            # Apply to continuum flux and error
            masked_cont_flux = apply_mask_2d(cont_flux, mask, fill_value=0.0)
            masked_cont_err = apply_mask_2d(cont_err, mask, fill_value=0.0)

            # Load Halpha 3D cubes
            halpha_flux_cube, halpha_err_cube, halpha_flux_header, halpha_err_header = load_halpha_cubes(halpha_path)

            # Apply SAME 2D continuum mask across all slices
            masked_halpha_flux_cube = apply_mask_3d(halpha_flux_cube, mask, fill_value=0.0)
            masked_halpha_err_cube = apply_mask_3d(halpha_err_cube, mask, fill_value=0.0)

            # Save outputs
            cont_out = OUT_CONT_DIR / build_output_cont_filename(galaxy_id)
            halpha_out = OUT_HALPHA_DIR / build_output_halpha_filename(galaxy_id)

            print("Saving continuum to:", cont_out)
            print("Saving Halpha to:", halpha_out)

            save_masked_continuum(
                cont_out,
                masked_cont_flux,
                masked_cont_err,
                flux_header=cont_flux_header,
                err_header=cont_err_header
            )

            save_masked_halpha(
                halpha_out,
                masked_halpha_flux_cube,
                masked_halpha_err_cube,
                flux_header=halpha_flux_header,
                err_header=halpha_err_header
            )

            n_valid = int(mask.sum())
            n_pix = int(mask.size)
            frac = n_valid / n_pix if n_pix > 0 else np.nan

            print(f"[DONE] {galaxy_id} | valid pixels = {n_valid}/{n_pix} ({frac:.3f})")
            n_done += 1

        except Exception as e:
            print(f"[FAILED] {galaxy_id}: {type(e).__name__}: {e}")
            raise

    print("\n--- Summary ---")
    print(f"Total continuum files: {n_total}")
    print(f"Completed:             {n_done}")
    print(f"Missing Halpha files:  {n_missing_halpha}")
    print(f"Failed:                {n_failed}")


if __name__ == "__main__":
    main()