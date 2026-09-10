import os
import io
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, List, Union, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from PIL import Image

from mesoslide._slides import DEFAULT_TILE_KEY
from mesoslide._deprecated import (
    ELEMENT_NAME_HINT, SLIDES_HINT, deprecated_kwargs, drop, removed,
)
from ._utils import get_transparent_colormap, resize_image_to_fit
from ._feature_map import plot_feature_map

if TYPE_CHECKING:
    from wsidata import WSIData


def _iter_plot_slides(slides, tile_key):
    """Yield (slide_id, wsi) with images attached, from whatever `slides` is."""
    import pandas as pd
    from wsidata import WSIData
    from mesoslide._slides import iter_slides, slide_id_from

    if isinstance(slides, pd.DataFrame):
        yield from iter_slides(slides, attach_images=True)
        return
    if isinstance(slides, WSIData):
        slides = [slides]
    if isinstance(slides, dict):
        yield from slides.items()
        return
    for wsi in slides:
        yield slide_id_from(wsi), wsi


@deprecated_kwargs(
    image_names=removed(SLIDES_HINT),
    point_name=drop('grid_point', ELEMENT_NAME_HINT),
    bbox_name=drop('bbox', ELEMENT_NAME_HINT),
    feature_prefix=removed(
        "feature_prefix and feature_idx collapse to a single feature_name; "
        "pass f'{prefix}_{idx}'."
    ),
    feature_idx=removed(
        "feature_prefix and feature_idx collapse to a single feature_name; "
        "pass f'{prefix}_{idx}'."
    ),
)
def plot_feature_spatial_distribution(
    slides,
    feature_name: str,
    *,
    output_path: Optional[str] = None,
    tile_key: str = DEFAULT_TILE_KEY,
    image_size: int = 2000,
    cmap: Union[str, LinearSegmentedColormap] = 'transparent_to_green',
    fill_alpha: float = 0.3,
    nrows: Optional[int] = None,
    ncols: int = 5,
    figsize_per_image: tuple = (8, 6),
    dpi: int = 150,
    colorbar: bool = False,
    datashader_method: bool = True,
    show_titles: bool = False,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    Plot the spatial distribution of one feature across a cohort of slides.

    Renders each slide with :func:`mesoslide.plotting.plot_feature_map` and
    composites the results into a single grid figure.

    Parameters
    ----------
    slides : slides_table, WSIData, list of WSIData, or {slide_id: WSIData}
        The cohort. `plot_feature_map`'s background is read lazily via each
        slide's `wsi.reader`, so `attach_images=True` is not required either
        way this is passed.
    feature_name : str
        Feature to plot, e.g. 'UNI_SAE_42' or 'kmeans_label_3'. An .obs column
        of the tile table, or a .var name in its .X.
    output_path : str, optional
        Directory to save the composite figure into. Not saved if None.
    tile_key : str, default='tiles'
    image_size : int, default=2000
        Max dimension of each panel's background thumbnail; see
        :func:`mesoslide.plotting.plot_feature_map`.
    cmap : str or LinearSegmentedColormap, default='transparent_to_green'
        'transparent_to_green' / '_red' / '_blue', a matplotlib colormap name,
        or a colormap instance.
    fill_alpha : float, default=0.3
    nrows : int, optional
        Grid rows. Derived from ncols if None.
    ncols : int, default=5
    figsize_per_image : tuple, default=(8, 6)
    dpi : int, default=150
    colorbar : bool, default=False
    datashader_method : bool, default=True
    show_titles : bool, default=False
        Label each panel with its slide id.
    return_fig : bool, default=False

    Returns
    -------
    matplotlib Figure if return_fig=True, else None

    Examples
    --------
    >>> import mesoslide as ms
    >>> ms.plotting.plot_feature_spatial_distribution(
    ...     manifest, 'UNI_SAE_42', output_path='reports/', ncols=3
    ... )
    """
    if isinstance(cmap, str):
        if cmap in ('transparent_to_green', 'transparent_to_red', 'transparent_to_blue'):
            cmap = get_transparent_colormap(cmap.split('_')[-1], alpha=fill_alpha)
        else:
            cmap = plt.get_cmap(cmap)

    temp_dir = tempfile.mkdtemp()
    image_paths = []

    try:
        rendered = 0
        for slide_id, wsi in tqdm(
            _iter_plot_slides(slides, tile_key), desc="Rendering slides"
        ):
            try:
                fig = plot_feature_map(
                    wsi,
                    feature_name,
                    tile_key=tile_key,
                    image_size=image_size,
                    cmap=cmap,
                    fill_alpha=fill_alpha,
                    figsize=figsize_per_image,
                    colorbar=colorbar,
                    title=str(slide_id) if show_titles else '',
                    method='datashader' if datashader_method else 'rasterize',
                    datashader_reduction='max',
                    return_ax=False,
                )
            except (KeyError, ValueError) as e:
                print(f"Warning: skipping slide {slide_id!r}: {e}")
                continue

            img_path = os.path.join(temp_dir, f'plot_{rendered:03d}.png')
            fig.savefig(img_path, dpi=dpi, bbox_inches='tight')
            image_paths.append(img_path)
            plt.close(fig)
            rendered += 1

        if not image_paths:
            raise ValueError(
                f"No slide could be rendered for feature '{feature_name}'."
            )

        if nrows is None:
            nrows = int(np.ceil(len(image_paths) / ncols))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(figsize_per_image[0] * ncols, figsize_per_image[1] * nrows)
        )

        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)

        axes_flat = axes.flatten()

        for i, img_path in enumerate(image_paths):
            axes_flat[i].imshow(mpimg.imread(img_path))
            axes_flat[i].axis('off')

        for i in range(len(image_paths), len(axes_flat)):
            axes_flat[i].axis('off')
            axes_flat[i].set_visible(False)

        plt.tight_layout()

        if output_path is not None:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            output_file = os.path.join(
                output_path, f'{feature_name}_spatial_distribution.png'
            )
            fig.savefig(output_file, bbox_inches='tight', dpi=dpi)
            print(f"Saved: {output_file}")

        if return_fig:
            return fig
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
    >>> from mesoslide.plotting import create_feature_pdf
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
    # reportlab is only needed for the PDF path, so it is imported here rather
    # than at module scope -- otherwise plot_feature_spatial_distribution, which
    # does not use it, cannot be imported without it installed.
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.units import inch

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