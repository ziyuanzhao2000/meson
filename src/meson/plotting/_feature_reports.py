import os
import io
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, List, Union, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
import spatialdata
from tqdm import tqdm
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import inch

from ._utils import get_transparent_colormap, resize_image_to_fit

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


def create_feature_pdf(
    top_image_path: str,
    bottom_image_path: str,
    output_pdf_path: str,
    feature_name: Optional[str] = None,
    page_width_inches: float = 13.33,
    page_height_inches: float = 7.5,
    image_dpi: int = 300,
    margin_dots: int = 150
) -> None:
    """
    Create a PDF with two images arranged vertically on a landscape slide.
    
    Designed for feature reports with spatial distribution (top) and
    patch gallery (bottom) images. Uses 16:9 aspect ratio by default.
    
    Parameters
    ----------
    top_image_path : str
        Path to image for top of PDF (e.g., spatial distribution).
    bottom_image_path : str
        Path to image for bottom of PDF (e.g., patch gallery).
    output_pdf_path : str
        Path where PDF will be saved.
    feature_name : str, optional
        Feature name to display as title at top of page.
    page_width_inches : float, default=13.33
        Page width in inches (13.33 = 16:9 aspect ratio).
    page_height_inches : float, default=7.5
        Page height in inches (7.5 = 16:9 aspect ratio).
    image_dpi : int, default=300
        DPI of the images being embedded.
    margin_dots : int, default=150
        Margin size in dots/pixels.
        
    Examples
    --------
    >>> from meson.plotting import create_feature_pdf
    >>> 
    >>> create_feature_pdf(
    ...     'feature_00017_samples_1-20.png',
    ...     'feature_00017_spatial_distribution.png',
    ...     'feature_00017_report.pdf',
    ...     feature_name='Necrotic Core'
    ... )
    
    Notes
    -----
    Images are automatically resized to fit within the page while
    maintaining aspect ratio. The scaling accounts for conversion
    between points (PDF units) and dots (image units).
    """
    # Set up page dimensions
    page_width = page_width_inches * inch  # Convert to points
    page_height = page_height_inches * inch
    
    # Create PDF canvas
    c = canvas.Canvas(output_pdf_path, pagesize=(page_width, page_height))
    
    # Scale factor: 72 points per inch / image_dpi dots per inch
    scale_factor = 72 / image_dpi
    c.scale(scale_factor, scale_factor)
    
    # Convert page dimensions to dots
    page_width_dots = page_width / scale_factor
    page_height_dots = page_height / scale_factor
    
    # Define margins and available space
    available_width = page_width_dots - 2 * margin_dots
    available_height = page_height_dots - 3 * margin_dots  # Extra margin for spacing
    max_image_height = available_height
    
    try:
        # Load and process top image
        top_img = Image.open(top_image_path)
        top_img_resized, top_width, top_height = resize_image_to_fit(
            top_img, int(available_width), int(max_image_height)
        )
        
        # Load and process bottom image
        bottom_img = Image.open(bottom_image_path)
        bottom_img_resized, bottom_width, bottom_height = resize_image_to_fit(
            bottom_img, int(available_width), int(max_image_height)
        )
        
        # Calculate positions (center images horizontally)
        top_x = margin_dots + (available_width - top_width) / 2
        top_y = page_height_dots - margin_dots - top_height
        
        bottom_x = margin_dots + (available_width - bottom_width) / 2
        bottom_y = margin_dots
        
        # Convert PIL images to ImageReader objects
        top_img_buffer = io.BytesIO()
        top_img_resized.save(top_img_buffer, format='PNG')
        top_img_buffer.seek(0)
        top_img_reader = ImageReader(top_img_buffer)
        
        bottom_img_buffer = io.BytesIO()
        bottom_img_resized.save(bottom_img_buffer, format='PNG')
        bottom_img_buffer.seek(0)
        bottom_img_reader = ImageReader(bottom_img_buffer)
        
        # Draw images on PDF
        c.drawImage(top_img_reader, top_x, top_y, width=top_width, height=top_height)
        c.drawImage(bottom_img_reader, bottom_x, bottom_y, width=bottom_width, height=bottom_height)
        
        # Add feature name as title
        if feature_name is not None:
            c.setFont("Helvetica-Bold", 16)
            title_x = page_width_dots / 2
            title_y = page_height_dots - 100
            c.drawCentredString(title_x, title_y, f"Feature: {feature_name}")
        
        # Save PDF
        c.save()
        print(f"Created PDF: {output_pdf_path}")
        
    except Exception as e:
        print(f"Error creating PDF for {feature_name}: {str(e)}")
        raise