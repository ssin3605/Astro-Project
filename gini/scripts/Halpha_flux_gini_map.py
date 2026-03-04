import os
import os.path

from src.io import read_sami_Halpha_layers
from src.morphology import gini_from_map
from src.plotting import plot_map

def main():
    fitsfile = "data/fits/sami_dr3_Halpha_maps/6837_A_Halpha_default_1-comp.fits"

    outdir = "gini/gini_outputs/figures"
    os.makedirs(outdir, exist_ok=True)

    flux_layers, err_layers, header, meta = read_sami_Halpha_layers(fitsfile)
    base = os.path.splitext(os.path.basename(fitsfile))[0]

    # Extract galaxy ID
    catid = base.split("_")[0]

    # Layer 1 (sum)
    flux_sum = flux_layers["sum"]
    err_sum = err_layers["sum"]
    g_sum = gini_from_map(flux_sum, err_sum, sn_cut=3.0)

    print(f"File: {fitsfile}")
    #print(f"Gini (fit layer 0): {g_fit:.3f}")
    print(f"Gini (sum layer 1): {g_sum:.3f}")
    print("Meta:", meta)

    # Plot flux sum layer
    plot_map(
        flux_sum,
        title=f"{catid} Hα | FLUX | Gini={g_sum:.2f}",
        outfile=os.path.join(outdir, f"{catid}_Halpah_flux.png"),
    )

    # Plot error sum layer 
    plot_map(
        err_sum,
        title=f"{catid} Hα | ERR",
        outfile=os.path.join(outdir, f"{catid}_Halpha_error.png"),
    )


if __name__ == "__main__":
    main()