import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from math import ceil
from typing import Optional, List, Union


def _plot_image_grid(
    images: List[np.ndarray],
    labels: Optional[List[str]] = None,
    group_ids: Optional[List] = None,
    n_cols: int = 10,
    patch_size: float = 2.0,
    border_extend: float = 0.1,
    border_alpha: float = 1.0,
    fontsize: float = 6,
    cmap: str = 'tab10',
) -> tuple:
    """
    Primitive: render a list of image arrays as a grid with optional
    colored borders and text labels.

    Parameters
    ----------
    images : list of np.ndarray
        Each element is (H, W, 3) uint8 or float image.
    labels : list of str, optional
        Per-image text label shown in top-left corner.
    group_ids : list, optional
        Categorical group per image used to color the border.
        If None, no border is drawn.
    n_cols : int
        Number of columns in the grid.
    patch_size : float
        Size in inches of each subplot.
    border_extend : float
        How far (in axes-fraction units) the border extends outside the image.
    border_alpha : float
        Opacity of the border rectangle.
    cmap : str
        Matplotlib colormap name used to map group_ids to colors.
    fontsize : float
        Font size for labels, in points. Default is 6, but may need to be
        increased for very small patch_size.
    Returns
    -------
    fig, axs : tuple
    """
    n = len(images)
    n_rows = ceil(n / n_cols)

    fig, axs = plt.subplots(
        n_rows, n_cols,
        figsize=(patch_size * n_cols, patch_size * n_rows),
        layout='constrained'
    )
    axs = np.array(axs).flatten() if n > 1 else np.array([axs])

    # Build color lookup once
    if group_ids is not None:
        if isinstance(cmap, ListedColormap):
            # index directly by group_id — color[3] is always group 3's color
            group_to_color = {g: cmap(g) for g in set(group_ids)}
        else:
            # string cmap: normalize by total number of colors expected
            colormap = cm.get_cmap(cmap)
            n = colormap.N if hasattr(colormap, 'N') else 256
            group_to_color = {g: colormap(g / n) for g in set(group_ids)}

    for i, image in enumerate(images):
        ax = axs[i]
        ax.imshow(image)
        ax.set_zorder(10)

        if group_ids is not None:
            color = group_to_color[group_ids[i]]
            border = mpatches.Rectangle(
                (-border_extend * patch_size, -border_extend * patch_size),
                1 + 2 * border_extend * patch_size,
                1 + 2 * border_extend,
                transform=ax.transAxes,
                facecolor=color,
                alpha=border_alpha,
                zorder=-10,
                clip_on=False,
            )
            ax.add_patch(border)

        if labels is not None:
            ax.text(
                0, 0.95, str(labels[i]),
                ha='left', va='top',
                transform=ax.transAxes,
                fontsize=max(fontsize, patch_size * 4),
                color='black',
                zorder=20,
            )

        ax.axis('off')

    # Hide unused axes
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
        axs[j].set_visible(False)

    return fig, axs