import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Union, Tuple


def get_transparent_colormap(
    color: Union[str, Tuple[float, float, float]],
    alpha: float = 0.5,
    name: str = None
) -> LinearSegmentedColormap:
    """
    Create a colormap that transitions from transparent white to a specified color.
    
    Parameters
    ----------
    color : str or tuple of float
        Color specification. Can be:
        - 'green', 'red', 'blue' for predefined colors
        - Tuple of (r, g, b) values in range [0, 1]
    alpha : float, default=0.5
        Alpha transparency of the final color (0=transparent, 1=opaque).
    name : str, optional
        Name for the colormap. If None, auto-generated from color.
        
    Returns
    -------
    LinearSegmentedColormap
        Colormap transitioning from transparent white to the specified color.
        
    Examples
    --------
    >>> # Create transparent-to-green colormap
    >>> cmap = get_transparent_colormap('green', alpha=0.3)
    >>> 
    >>> # Create custom color
    >>> cmap = get_transparent_colormap((0.8, 0.2, 0.5), alpha=0.5)
    """
    # Parse color specification
    if isinstance(color, str):
        color_map = {
            'green': (0, 1, 0),
            'red': (1, 0, 0),
            'blue': (0, 0, 1)
        }
        if color.lower() in color_map:
            rgb = color_map[color.lower()]
            if name is None:
                name = f'transparent_to_{color.lower()}'
        else:
            raise ValueError(
                f"Unknown color '{color}'. Use 'green', 'red', 'blue' "
                "or provide an (r, g, b) tuple."
            )
    elif isinstance(color, (tuple, list)) and len(color) == 3:
        rgb = tuple(color)
        if name is None:
            name = f'transparent_to_rgb_{rgb[0]:.2f}_{rgb[1]:.2f}_{rgb[2]:.2f}'
    else:
        raise ValueError(
            "Color must be a string ('green', 'red', 'blue') "
            "or an (r, g, b) tuple."
        )
    
    # Create colormap
    cmap = LinearSegmentedColormap.from_list(
        name,
        [(1, 1, 1, 0), (*rgb, alpha)],
        N=256
    )
    
    return cmap


def make_transparent_to_color_colormaps(N, base_cmap, show_plot=True, final_alpha=0.5):
    """
    Generate N colormaps, each going from transparent white to a color
    sampled from the base colormap at regular intervals.
    Optionally, visualize the endpoint colors as a discrete colorbar.
    
    Parameters:
        N (int): Number of colormaps to generate.
        base_cmap (Colormap): A matplotlib colormap instance.
        show_plot (bool): Whether to show the colorbar plot.
        
    Returns:
        List[LinearSegmentedColormap]: List of custom colormaps.
    """
    colormaps = []
    endpoint_colors = []
    for i, frac in enumerate(np.linspace(0, 1, N)):
        color = base_cmap(frac)
        rgb = color[:3]
        cmap = LinearSegmentedColormap.from_list(
            f'transparent_to_color_{i}',
            [(1, 1, 1, 0), (*rgb, final_alpha)],
            N=256
        )
        colormaps.append(cmap)
        endpoint_colors.append(rgb)
    
    if show_plot:
        # Plot the endpoint colors as a discrete colorbar
        fig, ax = plt.subplots(figsize=(N, 1))
        for i, color in enumerate(endpoint_colors):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
        ax.set_xlim(0, N)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(N) + 0.5)
        ax.set_xticklabels([f'{i+1}' for i in range(N)])
        ax.set_yticks([])
        ax.set_title('Endpoint Colors')
        plt.show()
        
        # Optionally, show the gradients for each colormap
        fig, axes = plt.subplots(1, N, figsize=(N*2, 1))
        for i, cmap in enumerate(colormaps):
            gradient = np.linspace(0, 1, 256).reshape(1, -1)
            axes[i].imshow(gradient, aspect='auto', cmap=cmap)
            axes[i].axis('off')
        plt.suptitle('Colormap Gradients')
        plt.show()
    
    return colormaps