import io
from pathlib import Path
from typing import TYPE_CHECKING, Union, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from PIL import Image

from mesoslide._slides import DEFAULT_TILE_KEY
from ._utils import (
    get_transparent_colormap, resize_image_to_fit, _finish_plot, _load_image_source,
)
from ._feature_map import plot_feature_map

if TYPE_CHECKING:
    from wsidata import WSIData


def _iter_plot_slides(slides, tile_key):
    """Yield (slide_id, wsi) from whatever `slides` is.

    Images are not attached: `plot_feature_map` reads the background via
    `wsi.reader`/`get_thumbnail`, which works regardless of `attach_images`.
    """
    import pandas as pd
    from wsidata import WSIData
    from mesoslide._slides import iter_slides, slide_id_from

    if isinstance(slides, pd.DataFrame):
        yield from iter_slides(slides, attach_images=False)
        return
    if isinstance(slides, WSIData):
        slides = [slides]
    if isinstance(slides, dict):
        yield from slides.items()
        return
    for wsi in slides:
        yield slide_id_from(wsi), wsi


def _render_feature_grid(
    slides,
    feature_name: str,
    *,
    tile_key: str = DEFAULT_TILE_KEY,
    image_size: int = 2000,
    cmap: Union[str, LinearSegmentedColormap] = 'transparent_to_green',
    fill_alpha: float = 0.3,
    nrows: Optional[int] = None,
    ncols: int = 5,
    figsize_per_image: tuple = (8, 6),
    colorbar: bool = False,
    show_titles: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    """Composite each slide's plot_feature_map render into one grid figure.

    Renders each slide with :func:`mesoslide.plotting.plot_feature_map` and
    composites the results into a single grid figure. Used internally by
    :func:`create_feature_pdf` for its top panel. Caller owns the returned
    (open) figure -- must close it.
    """
    if isinstance(cmap, str):
        if cmap in ('transparent_to_green', 'transparent_to_red', 'transparent_to_blue'):
            cmap = get_transparent_colormap(cmap.split('_')[-1], alpha=fill_alpha)
        else:
            cmap = plt.get_cmap(cmap)

    rendered_images = []

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
                return_ax=False,
            )
        except (KeyError, ValueError) as e:
            print(f"Warning: skipping slide {slide_id!r}: {e}")
            continue

        buf = io.BytesIO()
        fig.savefig(buf, dpi=dpi, format='png')
        plt.close(fig)
        buf.seek(0)
        rendered_images.append(Image.open(buf))

    if not rendered_images:
        raise ValueError(
            f"No slide could be rendered for feature '{feature_name}'."
        )

    if nrows is None:
        nrows = int(np.ceil(len(rendered_images) / ncols))

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

    for i, image in enumerate(rendered_images):
        axes_flat[i].imshow(image)
        axes_flat[i].axis('off')

    for i in range(len(rendered_images), len(axes_flat)):
        axes_flat[i].axis('off')
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    return fig


def create_feature_pdf(
    slides,
    feature_name: str,
    bottom_image,
    *,
    output_path: Optional[str] = None,
    return_buffer: bool = False,
    tile_key: str = DEFAULT_TILE_KEY,
    image_size: int = 2000,
    cmap: Union[str, LinearSegmentedColormap] = 'transparent_to_green',
    fill_alpha: float = 0.3,
    ncols: int = 5,
    nrows: Optional[int] = None,
    figsize_per_image: tuple = (8, 6),
    colorbar: bool = False,
    show_titles: bool = False,
    page_width_inches: float = 13.33,
    page_height_inches: float = 7.5,
    image_dpi: int = 300,
    margin_dots: int = 150,
) -> Optional[io.BytesIO]:
    """
    Create a one-page PDF report for one feature: a grid of each slide's
    feature map (top) and a supplied patch gallery image (bottom), arranged
    vertically on a landscape page.

    Parameters
    ----------
    slides : slides_table, WSIData, list of WSIData, or {slide_id: WSIData}
        The cohort for the top panel. See :func:`mesoslide.plotting.plot_feature_map`.
    feature_name : str
        Feature to plot in the top panel, and shown as the page title.
    bottom_image : str, Path, io.BytesIO, PIL.Image, or matplotlib Figure
        Image for the bottom of the page (e.g. from
        :func:`mesoslide.plotting.plot_patch_gallery`).
    output_path : str, optional
        Path to save the PDF to. Not saved if None.
    return_buffer : bool, default=False
        Return an in-memory PDF BytesIO.
    tile_key : str, default='tiles'
    image_size : int, default=2000
        Max dimension of each top-panel slide's background thumbnail; see
        :func:`mesoslide.plotting.plot_feature_map`.
    cmap : str or LinearSegmentedColormap, default='transparent_to_green'
    fill_alpha : float, default=0.3
    ncols : int, default=5
    nrows : int, optional
        Grid rows for the top panel. Derived from ncols if None.
    figsize_per_image : tuple, default=(8, 6)
    colorbar : bool, default=False
    show_titles : bool, default=False
        Label each top-panel slide with its slide id.
    page_width_inches : float, default=13.33
        Page width in inches (13.33 = 16:9 aspect ratio).
    page_height_inches : float, default=7.5
        Page height in inches (7.5 = 16:9 aspect ratio).
    image_dpi : int, default=300
        DPI used both to render the top panel and to size images on the
        page (assumes both images were rendered at this DPI).
    margin_dots : int, default=150
        Margin size in dots/pixels.

    Returns
    -------
    io.BytesIO if return_buffer=True, else None

    Examples
    --------
    >>> from mesoslide.plotting import create_feature_pdf, plot_patch_gallery
    >>>
    >>> bottom = plot_patch_gallery(patches, slides=slides, return_buffer=True)[0]
    >>> create_feature_pdf(
    ...     slides, 'UNI_SAE_42', bottom,
    ...     output_path='feature_00017_report.pdf',
    ... )

    Notes
    -----
    Images are automatically resized to fit within the page while
    maintaining aspect ratio. The scaling accounts for conversion
    between points (PDF units) and dots (image units).
    """
    if output_path is None and not return_buffer:
        raise ValueError("Provide output_path and/or return_buffer=True.")

    # reportlab is only needed for the PDF path, so it is imported here rather
    # than at module scope -- otherwise this module cannot be imported
    # without it installed unless create_feature_pdf is actually used.
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.units import inch

    top_fig = _render_feature_grid(
        slides, feature_name, tile_key=tile_key, image_size=image_size, cmap=cmap,
        fill_alpha=fill_alpha, figsize_per_image=figsize_per_image, colorbar=colorbar,
        show_titles=show_titles, ncols=ncols, nrows=nrows, dpi=image_dpi,
    )
    # Reuses the same _finish_plot helper as plot_patch_gallery, rather than
    # hand-rolling the same savefig-to-buffer-and-close steps again here.
    top_buf = _finish_plot(top_fig, None, show=False, return_buffer=True, dpi=image_dpi)

    # Set up page dimensions
    page_width = page_width_inches * inch  # Convert to points
    page_height = page_height_inches * inch

    # Create PDF canvas directly into an in-memory buffer
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=(page_width, page_height))

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

    # Load and process top image
    top_img = _load_image_source(top_buf)
    top_img_resized, top_width, top_height = resize_image_to_fit(
        top_img, int(available_width), int(max_image_height)
    )

    # Load and process bottom image
    bottom_img = _load_image_source(bottom_image)
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
    c.setFont("Helvetica-Bold", 16)
    title_x = page_width_dots / 2
    title_y = page_height_dots - 100
    c.drawCentredString(title_x, title_y, f"Feature: {feature_name}")

    # Save PDF
    c.save()
    pdf_buffer.seek(0)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(pdf_buffer.getvalue())
        print(f"Created PDF: {output_path}")

    if return_buffer:
        pdf_buffer.seek(0)
        return pdf_buffer
    return None