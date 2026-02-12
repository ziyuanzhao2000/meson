import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, List, Union, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
import spatialdata
from tqdm import tqdm

from ._utils import get_transparent_colormap

if TYPE_CHECKING:
    from spatialdata import SpatialData


def plot_feature_spatial_distribution(
    sdata: "SpatialData",
    feature_prefix: str,
    feature_idx: int,
    image_names: Union[List[str], str] = 'all',
    output_path: Optional[str] = None,
    point_name: str = 'grid_point',
    bbox_name: str = 'bbox',
    cmap: Union[str, LinearSegmentedColormap] = 'transparent_to_green',
    fill_alpha: float = 0.3,
    nrows: Optional[int] = None,
    ncols: int = 5,
    figsize_per_image: tuple = (8, 6),
    dpi: int = 150,
    colorbar: bool = False,
    datashader_method: bool = True,
    show_titles: bool = False,
    return_fig: bool = False
) -> Optional[plt.Figure]:
    """
    Plot spatial distribution of a feature across multiple whole slide images.
    
    Creates a grid of WSI images with the feature overlaid as colored shapes,
    saving the result as a single composite figure.
    
    Parameters
    ----------
    sdata : SpatialData
        Spatial data object containing images and feature annotations.
    feature_prefix : str
        Prefix of the feature column in obs table. For example:
        - 'kmeans_label' for k-means clustering
        - 'kmeans_k25_label' for specific k-means model
        - 'UNI_SAE' for SAE features
    feature_idx : int
        Index of the specific feature to visualize.
        Will look for column '{feature_prefix}_{feature_idx}' in obs.
    image_names : list of str or 'all', default='all'
        List of image names to include in the visualization.
        If 'all', uses all images in sdata.
    output_path : str, optional
        Directory path to save the output figure.
        If None, figure is not saved to disk.
    point_name : str, default='grid_point'
        Name of the point element in spatial data.
    bbox_name : str, default='bbox'
        Name of the bounding box/shape element to render.
    cmap : str or LinearSegmentedColormap, default='transparent_to_green'
        Colormap for visualization. Can be:
        - 'transparent_to_green', 'transparent_to_red', 'transparent_to_blue'
        - matplotlib colormap name
        - LinearSegmentedColormap instance
    fill_alpha : float, default=0.3
        Alpha transparency for the feature overlay (0=transparent, 1=opaque).
    nrows : int, optional
        Number of rows in the grid. If None, calculated from ncols.
    ncols : int, default=5
        Number of columns in the grid layout.
    figsize_per_image : tuple, default=(8, 6)
        Size (width, height) in inches for each subplot.
    dpi : int, default=150
        Resolution for saving the figure.
    colorbar : bool, default=False
        Whether to show colorbar in individual plots.
    datashader_method : bool, default=True
        Whether to use datashader rendering method for shapes.
    show_titles : bool, default=False
        Whether to show image names as titles on subplots.
    return_fig : bool, default=False
        Whether to return the figure object instead of closing it.
        
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object if return_fig=True, otherwise None.
        
    Examples
    --------
    >>> from meson.plotting import plot_feature_spatial_distribution
    >>> 
    >>> # Plot k-means cluster 0 across all images
    >>> plot_feature_spatial_distribution(
    ...     sdata,
    ...     feature_prefix='kmeans_label',
    ...     feature_idx=0,
    ...     output_path='/path/to/output/feature_reports'
    ... )
    >>> 
    >>> # Plot SAE feature 42 on specific images
    >>> plot_feature_spatial_distribution(
    ...     sdata,
    ...     feature_prefix='UNI_SAE',
    ...     feature_idx=42,
    ...     image_names=['slide_001', 'slide_002', 'slide_003'],
    ...     cmap='transparent_to_red',
    ...     ncols=3
    ... )
    """
    if isinstance(image_names, str) and image_names == 'all':
        image_names = list(set([name.split('_')[0] for name in sdata.images]))
        image_names.sort()
    elif not isinstance(image_names, list):
        raise ValueError("image_names must be 'all' or a list of image names")
    
    if isinstance(cmap, str):
        if cmap in ['transparent_to_green', 'transparent_to_red', 'transparent_to_blue']:
            # Extract color name from string like 'transparent_to_green' -> 'green'
            color_name = cmap.split('_')[-1]
            cmap = get_transparent_colormap(color_name, alpha=fill_alpha)
        else:
            # Assume it's a matplotlib colormap name
            cmap = plt.get_cmap(cmap)
            
    n_images = len(image_names)
    if nrows is None:
        nrows = int(np.ceil(n_images / ncols))
    feature_col = f'{feature_prefix}_{feature_idx}'
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    
    try:
        for i, image_name in enumerate(tqdm(image_names, desc="Rendering images")):
            # Create minimal SpatialData with only necessary elements
            sdata_mini = spatialdata.SpatialData()
            sdata_mini[image_name] = sdata[image_name]
            
            bbox_full_name = f'{image_name}_{point_name}_{bbox_name}'
            if bbox_full_name in sdata.shapes:
                sdata_mini[bbox_full_name] = sdata[bbox_full_name]
            else:
                print(f"Warning: {bbox_full_name} not found in sdata.shapes")
                continue
            
            patch_full_name = f'{image_name}_{point_name}_patch'
            if patch_full_name in sdata.tables:
                sdata_mini[patch_full_name] = sdata[patch_full_name]
            else:
                print(f"Warning: {patch_full_name} not found in sdata.tables")
                continue
            
            # Check if feature column exists
            if feature_col not in sdata_mini[patch_full_name].obs.columns:
                print(f"Warning: {feature_col} not found in {patch_full_name}.obs")
                continue
            
            # Render with spatialdata-plot
            if datashader_method:
                ax = sdata_mini.pl.render_images(element=image_name)\
                    .pl.render_shapes(
                        element=bbox_full_name,
                        color=feature_col,
                        cmap=cmap,
                        fill_alpha=fill_alpha,
                        method='datashader',
                        datashader_reduction='max'
                    )\
                    .pl.show(
                        image_name,
                        colorbar=colorbar,
                        title=image_name if show_titles else '',
                        return_ax=True
                    )
            else:
                ax = sdata_mini.pl.render_images(element=image_name)\
                    .pl.render_shapes(
                        element=bbox_full_name,
                        color=feature_col,
                        cmap=cmap,
                        fill_alpha=fill_alpha
                    )\
                    .pl.show(
                        image_name,
                        colorbar=colorbar,
                        title=image_name if show_titles else '',
                        return_ax=True
                    )
            
            img_path = os.path.join(temp_dir, f'plot_{i:03d}.png')
            ax.figure.savefig(img_path, dpi=dpi, bbox_inches='tight')
            image_paths.append(img_path)
            plt.close(ax.figure)
        
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(figsize_per_image[0] * ncols, figsize_per_image[1] * nrows)
        )
        
        # Ensure axes is 2D array
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)
        
        axes_flat = axes.flatten()
        
        for i, img_path in enumerate(image_paths):
            img = mpimg.imread(img_path)
            axes_flat[i].imshow(img)
            axes_flat[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(image_paths), len(axes_flat)):
            axes_flat[i].axis('off')
            axes_flat[i].set_visible(False)
        
        plt.tight_layout()
        
        if output_path is not None:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            output_file = os.path.join(
                output_path,
                f'{feature_prefix}_{feature_idx:05d}_spatial_distribution.png'
            )
            fig.savefig(output_file, bbox_inches='tight', dpi=dpi)
            print(f"Saved: {output_file}")
        
        if return_fig:
            return fig
        else:
            plt.close(fig)
            return None
            
    finally:
        for path in image_paths:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)