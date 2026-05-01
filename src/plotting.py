# Using this to turn generated arrays into figures
# Also saving figures created

import matplotlib.pyplot as plt
import numpy as np

def plot_map(fluxmap, title=None, outfile=None): #, logscale=True
    """
    Plot an Halpha flux map.

    Parameters
    fluxmap : 2D numpy array
        Flux map.
    title : str, optional
        Plot title.
    outfile : str, optional
        If provided, save figure to this path.
    logscale : bool
        If True, plot log10(flux).
    """
    fluxmap = np.asarray(fluxmap)

    fig, ax = plt.subplots(figsize=(5, 5))

    image = fluxmap
    label = "Flux"
    #if logscale:
        # Avoid log of zero
        #safe = np.where(fluxmap > 0, fluxmap, np.nan)
        #image = np.log10(safe)
        #label = "log10(Hα Flux)"
    #else:
        #image = fluxmap
        #label = "Flux"
    
    im = ax.imshow(image, origin="lower")
    plt.colorbar(im, ax=ax, label=label)

    if title:
        ax.set_title(title)

    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")

    if outfile:
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
