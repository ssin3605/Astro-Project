# analysis/run_gini.py

from src.io import read_sami_halpha_map
from src.morphology import gini_from_map
from src.plotting import plot_halpha_map

fitsfile = "data/fits/sami_dr3_Halpha_maps/9008500890_A_Halpha_default_1-comp.fits"

fluxmap, fluxerr, header = read_sami_halpha_map(fitsfile)

gini = gini_from_map(fluxmap, fluxerr)
print(f"Gini coefficient: {gini:.3f}")

plot_halpha_map(
    fluxmap,
    title=f"Gini = {gini:.2f}",
    outfile="outputs/figures/9008500890_A_Halpha_default_1-comp.png"
)
# run as module with: python -m scripts.gini_sample_run, otherwise won't find "src"
