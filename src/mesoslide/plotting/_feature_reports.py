import io
from pathlib import Path
from typing import TYPE_CHECKING, Union, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from PIL import Image

from mesoslide._slides import DEFAULT_TILE_KEY
from mesoslide._patch_selector import select_top_patches
from ._utils import get_transparent_colormap, resize_image_to_fit, _finish_plot, _load_image_source
from ._feature_map import plot_feature_map
from ._patch_gallery import plot_patch_gallery

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


def _count_plot_slides(slides) -> int:
    """How many slides `_iter_plot_slides` will yield, without opening any of them."""
    import pandas as pd
    from wsidata import WSIData

    if isinstance(slides, (pd.DataFrame, dict)):
        return len(slides)
    if isinstance(slides, WSIData):
        return 1
    return len(list(slides))


def _render_feature_grid(
    slides,
    feature_name: str,
    *,
    tile_key: str = DEFAULT_TILE_KEY,
    image_size: int = 2000,
    cmap: Union[str, LinearSegmentedColormap] = 'transparent_to_green',
    fill_alpha: float = 1.0,
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
    :func:`create_feature_report` for its top panel. Caller owns the returned
    (open) figure -- must close it.
    """
    if isinstance(cmap, str):
        if cmap in ('transparent_to_green', 'transparent_to_red', 'transparent_to_blue'):
            cmap = get_transparent_colormap(cmap.split('_')[-1], alpha=0.5)
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

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0.02)
    return fig


def create_feature_report(
    slides,
    feature_name: str,
    *,
    output_path: Optional[str] = None,
    return_buffer: bool = False,
    tile_key: str = DEFAULT_TILE_KEY,
    image_size: int = 2000,
    cmap: Union[str, LinearSegmentedColormap] = 'transparent_to_green',
    fill_alpha: float = 1.0,
    ncols: int = 5,
    nrows: Optional[int] = 2,
    figsize_per_image: Optional[tuple] = None,
    colorbar: bool = False,
    show_titles: bool = False,
    gallery_cmap: str = 'tab10',
    gallery_nrows: int = 2,
    top_fraction: float = 0.1,
    patches_per_row: int = 10,
    patch_display_size: float = 2.0,
    show_slide_ids: bool = False,
    show_scores: bool = False,
    group_col: Optional[str] = None,
    border_alpha: float = 1.0,
    border_extend: float = 0.1,
    page_width_inches: float = 13.33,
    page_height_inches: float = 7.5,
    image_dpi: int = 300,
    margin_dots: int = 150,
    progress_bar: bool = True,
) -> Optional[io.BytesIO]:
    """
    Create a one-page PDF report for one feature: a patch gallery (top) and
    a grid of each slide's feature map (bottom), arranged vertically on a
    landscape page.

    Parameters
    ----------
    slides : slides_table, WSIData, list of WSIData, or {slide_id: WSIData}
        The cohort for the bottom panel and for selecting/extracting the
        top panel's patches.
    feature_name : str
        Feature to plot in the bottom panel, to select the top panel's
        patches by, and shown as the page title.
    output_path : str, optional
        Path to save the PDF to. Not saved if None.
    return_buffer : bool, default=False
        Return an in-memory PDF BytesIO.
    tile_key : str, default='tiles'
    image_size : int, default=2000
        Max dimension of each bottom-panel slide's background thumbnail; see
        :func:`mesoslide.plotting.plot_feature_map`.
    cmap : str or LinearSegmentedColormap, default='transparent_to_green'
        Feature-map colormap for the bottom panel.
    fill_alpha : float, default=1.0
    ncols : int, default=5
    nrows : int, default=2
        Grid columns and rows for the bottom panel. If nrows is None, it is
        computed from ncols and the number of slides.
    figsize_per_image : tuple, optional
        Per-slide figure size for the bottom panel. Computed automatically
        from the space left below the (already-sized) top panel if None.
    colorbar : bool, default=False
    show_titles : bool, default=False
        Label each bottom-panel slide with its slide id.
    gallery_cmap : str, default='tab10'
        Border colormap for the top panel; see :func:`plot_patch_gallery`'s
        ``cmap``.
    gallery_nrows : int, default=2
        Row count for the top panel's patch gallery grid. Together with
        ``patches_per_row`` this determines how many patches are selected
        (``gallery_nrows * patches_per_row``, or fewer if not enough
        qualifying patches exist).
    top_fraction : float, default=0.1
        Passed to :func:`mesoslide.select_top_patches` to restrict the top
        panel's patches to the top fraction of qualifying, score-sorted
        patches before evenly sampling ``gallery_nrows * patches_per_row``
        of them.
    patches_per_row : int, default=10
    patch_display_size : float, default=2.0
    show_slide_ids : bool, default=False
    show_scores : bool, default=False
    group_col : str, optional
        Column in patches.obs for the top panel's border colour-coding.
    border_alpha : float, default=1.0
    border_extend : float, default=0.1
    page_width_inches : float, default=13.33
        Page width in inches (13.33 = 16:9 aspect ratio).
    page_height_inches : float, default=7.5
        Page height in inches (7.5 = 16:9 aspect ratio).
    image_dpi : int, default=300
        DPI used to render both panels and to size images on the page.
    margin_dots : int, default=150
        Margin size in dots/pixels.
    progress_bar : bool, default=True

    Returns
    -------
    io.BytesIO if return_buffer=True, else None

    Examples
    --------
    >>> from mesoslide.plotting import create_feature_report
    >>>
    >>> create_feature_report(
    ...     slides, 'UNI_SAE_42',
    ...     output_path='feature_00017_report.pdf',
    ... )

    Notes
    -----
    Images are automatically resized to fit within the page while
    maintaining aspect ratio. The scaling accounts for conversion
    between points (PDF units) and dots (image units). The patch gallery
    is forced onto a single page regardless of how many patches are given.
    """
    if output_path is None and not return_buffer:
        raise ValueError("Provide output_path and/or return_buffer=True.")

    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.units import inch

    page_width = page_width_inches * inch 
    page_height = page_height_inches * inch

    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=(page_width, page_height))

    scale_factor = 72 / image_dpi
    c.scale(scale_factor, scale_factor)

    page_width_dots = page_width / scale_factor
    page_height_dots = page_height / scale_factor

    available_width = page_width_dots - 2 * margin_dots
    available_height = page_height_dots - 2 * margin_dots 
    middle_gap = 0
    patches = select_top_patches(
        slides, feature_name,
        n=gallery_nrows * patches_per_row, top_fraction=top_fraction, tile_key=tile_key,
    )
    n_patches = len(patches.obs)
    top_buf = plot_patch_gallery(
        patches, slides=slides,
        samples_per_figure=max(n_patches, 1), patches_per_row=patches_per_row,
        patch_display_size=patch_display_size, show_slide_ids=show_slide_ids,
        show_scores=show_scores, group_col=group_col, border_alpha=border_alpha,
        border_extend=border_extend, cmap=gallery_cmap, dpi=image_dpi,
        return_buffer=True, progress_bar=progress_bar,
    )[0]
    top_img = _load_image_source(top_buf)
    top_width = available_width
    top_height = top_img.height * (available_width / top_img.width)

    height_budget = available_height -  middle_gap
    min_bottom_height = 0.3 * height_budget
    if height_budget - top_height < min_bottom_height:
        top_height = height_budget - min_bottom_height
        top_width = top_img.width * (top_height / top_img.height)
    top_img_resized = top_img.resize((int(top_width), int(top_height)), Image.Resampling.LANCZOS)

    remaining_height = height_budget - top_height
    if figsize_per_image is None:
        n_rows_effective = nrows if nrows is not None else int(np.ceil(
            _count_plot_slides(slides) / ncols
        ))
        figsize_per_image = (
            (available_width / image_dpi) / ncols,
            (remaining_height / image_dpi) / max(n_rows_effective, 1),
        )

    num_slides = nrows * ncols if nrows is not None else len(slides)
    if num_slides == len(slides):
        slides_subset = slides
    else:
        import itertools
        # skip-N sampling to reduce the number of slides to fit in the grid
        skip = max(1, len(slides) // num_slides)
        slides_subset = dict(itertools.islice(slides.items(), None, None, skip))
    
    bottom_fig = _render_feature_grid(
        slides_subset, feature_name, tile_key=tile_key, image_size=image_size, cmap=cmap,
        fill_alpha=fill_alpha, figsize_per_image=figsize_per_image, colorbar=colorbar,
        show_titles=show_titles, ncols=ncols, nrows=nrows, dpi=image_dpi,
    )
    bottom_buf = _finish_plot(bottom_fig, None, show=False, return_buffer=True, dpi=image_dpi)
    bottom_img = _load_image_source(bottom_buf)
    bottom_img_resized, bottom_width, bottom_height = resize_image_to_fit(
        bottom_img, int(available_width), int(remaining_height)
    )

    top_x = margin_dots + (available_width - top_width) / 2
    top_y = page_height_dots - margin_dots - top_height

    bottom_x = margin_dots + (available_width - bottom_width) / 2
    bottom_y = margin_dots

    top_img_buffer = io.BytesIO()
    top_img_resized.save(top_img_buffer, format='PNG')
    top_img_buffer.seek(0)
    top_img_reader = ImageReader(top_img_buffer)

    bottom_img_buffer = io.BytesIO()
    bottom_img_resized.save(bottom_img_buffer, format='PNG')
    bottom_img_buffer.seek(0)
    bottom_img_reader = ImageReader(bottom_img_buffer)

    c.drawImage(top_img_reader, top_x, top_y, width=top_width, height=top_height)
    c.drawImage(bottom_img_reader, bottom_x, bottom_y, width=bottom_width, height=bottom_height)

    c.setFont("Helvetica-Bold", 60)
    title_x = page_width_dots / 2
    title_y = page_height_dots - 100
    c.drawCentredString(title_x, title_y, f"Feature: {feature_name}")

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