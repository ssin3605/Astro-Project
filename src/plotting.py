# Using this to turn generated arrays into figures
# Also saving figures created

import matplotlib.pyplot as plt


def plot_halpha_map(fluxmap, title=None, outfile=None):
    """
    Plot an Halpha flux map.

    Parameters
    fluxmap : 2D numpy array
        Flux map.
    title : str, optional
        Plot title.
    outfile : str, optional
        If provided, save figure to this path.
    """

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(fluxmap, origin="lower")
    plt.colorbar(im, ax=ax, label="Halpha Flux")

    if title:
        ax.set_title(title)

    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")

    if outfile:
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
