import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from typing import Union, Tuple
from PIL import Image

# Multimodal imaging color definitions
FLUOROPHORE_COLORS = {
    "red": (1, 0, 0),
    "green": (0, 1, 0),
    "blue": (0.2, 0.2, 1),
    "yellow": (1, 1, 0),
    "cyan": (0, 1, 1),
    "magenta": (1, 0, 1),
    "white": (1, 1, 1),
    "orange": (1, 0.5, 0),
}

# Create colormaps (black to color)
FLUOROPHORE_CMAPS = {
    name: mcolors.LinearSegmentedColormap.from_list(f"black_{name}", [(0, 0, 0), rgb])
    for name, rgb in FLUOROPHORE_COLORS.items()
}

# Default marker color assignments for CyCIF/CODEX imaging
MARKER_COLOR_DEFAULTS = {
    # DNA
    'DNA1': 'blue', 'DNA2': 'blue', 'DNA3': 'blue', 'DNA4': 'blue',
    'DNA5': 'blue', 'DNA6': 'blue', 'DNA7': 'blue', 'DNA8': 'blue',
    'DNA9': 'blue', 'DNA10': 'blue', 'DNA11': 'blue',
    
    # Immune lineage
    'CD3E': 'magenta',
    'CD4': 'green',
    'CD8a': 'red',
    'CD20': 'white',
    'CD21': 'green',
    'CD23': 'red',
    'CD45': 'white',

    # Immune state / checkpoints
    'PD1': 'cyan',
    'PDL1': 'yellow',
    'CTLA4': 'cyan',
    'TIGIT': 'orange',
    'ICOS': 'cyan',
    'FOXP3': 'green',
    'TCF1_TCF7': 'cyan',
    'IRF-4': 'magenta',
    'CD25': 'yellow',
    'CD86': 'cyan',
    'CD36': 'orange',

    # Cytotoxic / effector
    'GZMB': 'yellow',
    'MPO': 'magenta',

    # Myeloid
    'CD11c': 'white',
    'CD11b': 'green',
    'CD68': 'yellow',
    'CD163': 'red',
    'CD206': 'green',
    'MCT': 'cyan',
    'HLADRB1': 'white',

    # Tissue architecture
    'panCK': 'cyan',
    'aSMA': 'magenta',
    'CD31': 'red',

    # TLS markers
    'CXCL13': 'cyan',
    'PNAd': 'yellow',

    # Proliferation
    'Ki-67': 'red',
    'PCNA': 'magenta',

    # Other
    'Ag85B': 'green',
    'LaminB1AC': 'cyan',
    'NGFR': 'red',
    'ABCG1': 'orange',
}


def get_marker_colormap(marker_name: str, custom_mapping: dict = None):
    """
    Get colormap for a specific marker.
    
    Parameters
    ----------
    marker_name : str
        Name of the marker (e.g., 'CD3E', 'DNA1').
    custom_mapping : dict, optional
        Custom marker-to-color mapping. If provided, overrides defaults.
        
    Returns
    -------
    cmap : LinearSegmentedColormap
        Colormap for the marker (black to color).
        
    Examples
    --------
    >>> cmap = get_marker_colormap('CD3E')
    >>> plt.imshow(image, cmap=cmap)
    """
    if custom_mapping and marker_name in custom_mapping:
        color = custom_mapping[marker_name]
    elif marker_name in MARKER_COLOR_DEFAULTS:
        color = MARKER_COLOR_DEFAULTS[marker_name]
    else:
        # Default to white if not found
        color = 'white'
    
    return FLUOROPHORE_CMAPS[color]


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


def resize_image_to_fit(image: Image.Image, max_width: int, max_height: int) -> Tuple[Image.Image, int, int]:
    """
    Resize image to fit within max dimensions while maintaining aspect ratio.
    
    Parameters
    ----------
    image : PIL.Image.Image
        Input image to resize.
    max_width : int
        Maximum width in pixels.
    max_height : int
        Maximum height in pixels.
        
    Returns
    -------
    resized_image : PIL.Image.Image
        Resized image.
    new_width : int
        Width of resized image.
    new_height : int
        Height of resized image.
        
    Examples
    --------
    >>> from PIL import Image
    >>> from meson.plotting import resize_image_to_fit
    >>> 
    >>> img = Image.open('large_image.png')
    >>> resized, w, h = resize_image_to_fit(img, 1920, 1080)
    >>> print(f"Resized to {w}x{h}")
    """
    img_width, img_height = image.size
    
    # Calculate scaling factor
    width_ratio = max_width / img_width
    height_ratio = max_height / img_height
    scale_ratio = min(width_ratio, height_ratio)
    
    # Calculate new dimensions
    new_width = int(img_width * scale_ratio)
    new_height = int(img_height * scale_ratio)
    
    # Resize image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image, new_width, new_height