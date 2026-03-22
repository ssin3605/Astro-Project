import os
import os.path

from src.io import read_sami_continuum_layers
from src.morphology import gini_from_map
from src.plotting import plot_map

## Use to plot a continuum flux map

def main():
    fitsfile = "gini/gini_outputs/tables/masked_continuum_maps/6837_A_HALPHA_cont_map_masked.fits"

    outdir = "gini/gini_outputs/figures"
    os.makedirs(outdir, exist_ok=True)

    flux, err, header, meta = read_sami_continuum_layers(fitsfile)
    base = os.path.splitext(os.path.basename(fitsfile))[0]

    # Extract galaxy ID
    catid = base.split("_")[0]

    g = gini_from_map(flux, err, sn_cut=3.0)

    print(f"File: {fitsfile}")
    print(f"Gini (continuum): {g:.3f}")
    print("Meta:", meta)

    # Plot flux sum layer
    plot_map(
        flux,
        title=f"{catid} Continuum | FLUX | Gini={g:.2f}",
        outfile=os.path.join(outdir, f"{catid}_Continuum_masked_flux.png"),
    )

    # Plot error sum layer 
    plot_map(
        err,
        title=f"{catid} Continuum | ERR",
        outfile=os.path.join(outdir, f"{catid}_Continuum_masked_error.png"),
    )


if __name__ == "__main__":
    main()