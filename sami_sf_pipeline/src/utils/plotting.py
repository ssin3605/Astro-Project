"""Minimal shared plotting helper, carried over from src/plotting.py."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_map(
    fluxmap,
    title: str | None = None,
    outfile: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "YlOrRd",
    ax=None,
    background_color: str | None = None,
    ellipse: dict | None = None,
):
    """
    Plot a 2D flux map and optionally save it to outfile.

    vmin/vmax default to None (matplotlib auto-scales to each map's own
    min/max) - pass fixed values to use the same colour scale across
    multiple galaxies, e.g. for a side-by-side comparison figure where the
    absolute flux level (not just each map's own relative structure) needs
    to be visually comparable.

    background_color, if given, fills the axes background with this colour
    before drawing the image. NaN pixels are transparent by default (the
    colormap's "bad" colour has alpha=0), so this shows through wherever
    there is no data - making "no flux" (background) visually distinct from
    real but faint flux values near the bottom of the colour scale.

    ellipse, if given, is a dict of matplotlib.patches.Ellipse kwargs (must
    include xy, width, height, angle) drawn on top of the map - e.g. to mark
    a galaxy's r_e. Styling defaults (unfilled, black dashed outline) are
    used for any keys not supplied.

    If ax is given, draws into it instead of creating a new figure (used
    for multi-panel comparisons) and does not save/show - the caller owns
    the figure. Returns the imshow AxesImage either way (handy for a shared
    colorbar across subplots).
    """
    fluxmap = np.asarray(fluxmap)

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(5, 5))

    if background_color is not None:
        ax.set_facecolor(background_color)

    im = ax.imshow(fluxmap, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)

    if ellipse is not None:
        ellipse_kwargs = {"fill": False, "edgecolor": "black", "linewidth": 1.5, "linestyle": "--"}
        ellipse_kwargs.update(ellipse)
        ax.add_patch(Ellipse(**ellipse_kwargs))

    if title:
        ax.set_title(title)
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")

    if not owns_figure:
        return im

    plt.colorbar(im, ax=ax, label="Flux")

    if outfile:
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return im
