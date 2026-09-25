import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from math import ceil
from typing import Optional, List, Union


def _group_color_lookup(group_ids: List, cmap: str) -> dict:
    """Map each distinct group id in `group_ids` to a color from `cmap`."""
    if isinstance(cmap, ListedColormap):
        # index directly by group_id — color[3] is always group 3's color.
        # Wrap ids >= N; cmap(N) would return the "over" color (the last entry).
        return {g: cmap(int(g) % cmap.N) for g in set(group_ids)}
    # string cmap: index directly (with wraparound) rather than normalizing
    # -- colormap(x) for a float x >= 1.0 clips to the colormap's last
    # entry instead of wrapping, so `g / n` collapses every group_id >= n
    # onto one identical color once there are more groups than the
    # colormap has entries.
    colormap = colormaps[cmap]
    n = colormap.N if hasattr(colormap, 'N') else 256
    return {g: colormap(g % n) for g in set(group_ids)}


def _draw_group_border(ax, color, patch_size: float, border_extend: float, alpha: float) -> None:
    """Draw a colored rectangle behind an axes' image, extending past its edges."""
    border = mpatches.Rectangle(
        (-border_extend * patch_size, -border_extend * patch_size),
        1 + 2 * border_extend * patch_size,
        1 + 2 * border_extend,
        transform=ax.transAxes,
        facecolor=color,
        alpha=alpha,
        zorder=-10,
        clip_on=False,
    )
    ax.add_patch(border)


def _draw_corner_label(ax, text: str, fontsize: float, patch_size: float) -> None:
    """Draw a text label in the top-left corner of an axes."""
    ax.text(
        0, 0.95, str(text),
        ha='left', va='top',
        transform=ax.transAxes,
        fontsize=max(fontsize, patch_size * 4),
        color='black',
        zorder=20,
    )


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
    axs = np.atleast_1d(axs).ravel()

    # Build color lookup once
    if group_ids is not None:
        group_to_color = _group_color_lookup(group_ids, cmap)

    for i, image in enumerate(images):
        ax = axs[i]
        ax.imshow(image)
        ax.set_zorder(10)

        if group_ids is not None:
            _draw_group_border(ax, group_to_color[group_ids[i]], patch_size, border_extend, border_alpha)

        if labels is not None:
            _draw_corner_label(ax, labels[i], fontsize, patch_size)

        ax.axis('off')

    # Hide unused axes
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
        axs[j].set_visible(False)

    return fig, axs