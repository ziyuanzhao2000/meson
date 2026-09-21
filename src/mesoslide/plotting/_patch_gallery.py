import io
from typing import TYPE_CHECKING, Optional, Tuple, Union, List
import numpy as np
import matplotlib.pyplot as plt

from mesoslide._slides import DEFAULT_TILE_KEY, HE_PATCH_IMG_KEY, SLIDE_ID, SLIDE_REF
from mesoslide._deprecated import SLIDES_HINT, deprecated_kwargs, removed, rename
from ._gallery_plan import GalleryPlan
from ._image_grid import _plot_image_grid

if TYPE_CHECKING:
    import anndata as ad
    from wsidata import WSIData
    from mesoslide.tools.segmenters import TokenClusterizer


@deprecated_kwargs(
    sdata=removed(SLIDES_HINT),
    show_image_names=rename('show_slide_ids'),
)
def plot_patch_gallery_with_saliency(
    patches: "ad.AnnData",
    clusterizers: List["TokenClusterizer"],
    model=None,
    slides=None,
    output_path: Optional[str] = None,
    tile_key: str = DEFAULT_TILE_KEY,
    samples_per_figure: int = 100,
    patches_per_row: int = 10,
    patch_display_size: float = 2.0,
    show_slide_ids: bool = False,
    show_scores: bool = False,
    title: Optional[str] = None,
    dpi: int = 300,
    return_fig: bool = False,
    return_buffer: bool = False,
    progress_bar: bool = True,
    saliency_alpha_power: float = 1.0,
    batch_size: int = 16,
    cache: bool = False,
    blend_with_previous: bool = True,
    cmap='viridis_r',
) -> Optional[Union[Tuple[plt.Figure, np.ndarray], List[io.BytesIO]]]:
    """
    Create a gallery of patches with H&E images and token cluster saliency maps.

    Displays patches in a grid where each patch occupies a column-group of rows:
    - Row 0: Original H&E image
    - Row 1..K: Cluster-map overlay for each clusterizer, colorized with viridis

    A thin wrapper around :class:`GalleryPlan`
    (``GalleryPlan(patches).add_he_row(...).add_cluster_map_rows(...)``); use
    `GalleryPlan` directly for other row combinations (CyCIF channels, custom
    row ordering, etc).

    Pixel data and cluster maps are always read through
    :func:`mesoslide.preprocessing.extract_patch_images`/:func:`mesoslide.preprocessing.extract_cluster_maps`,
    which check `patches.obsm` first and only fall back to `slides` for
    whatever isn't already cached there -- pass `cache=True` to persist
    freshly computed results back into `patches` for reuse across calls.

    Parameters
    ----------
    patches : AnnData
        Selected tiles, e.g. from :func:`mesoslide.select_top_patches`.
        Required .obs columns: 'x', 'y' (plus 'slide_id' across slides).
        Optional column: 'score' (used when show_scores=True).
    clusterizers : list of TokenClusterizer
        Each produces one cluster-map row. Must have distinct, non-empty
        `feature_name`s (see :func:`extract_cluster_maps`).
    model : str or lazyslide_models.ImageModel, optional
        The vision model to embed patches with -- forwarded to
        :func:`extract_cluster_maps` for every clusterizer. Optional: when
        omitted, each clusterizer resolves its own model from its stored
        `model_name` (see `TokenClusterizer`'s class docstring). Pass this
        explicitly when clusterizers share one model, to resolve it once
        here instead of once per clusterizer.
    slides : WSIData, list of WSIData, or {slide_id: WSIData}, optional
        Not required in the common case -- falls back to
        `patches.obs['_slide_ref']`, populated automatically by
        :func:`mesoslide.select_top_patches` and friends. Pass this
        explicitly to read from slides you already have open (skips
        re-opening) rather than relying on that column. Build with
        :func:`mesoslide.open_slides`; slides must have image data attached.
    output_path : str, optional
        File path to save to. Required when n_patches > samples_per_figure
        unless return_buffer=True. For multiple pages, the patch range is
        inserted before the extension (e.g. ``gallery_1-100.png``).
    samples_per_figure : int, default=100
    patches_per_row : int, default=10
    patch_display_size : float, default=2.0
        Subplot size in inches.
    show_slide_ids : bool, default=False
    show_scores : bool, default=False
    title : str, optional
        Figure suptitle (single-page only).
    dpi : int, default=300
    return_fig : bool, default=False
        Return (fig, axes) -- only when there is exactly one page.
    return_buffer : bool, default=False
        Return a list of in-memory PNG buffers, one per page (always a
        list, even for a single page).
    progress_bar : bool, default=True
    saliency_alpha_power : float, default=1.0
        Exponent applied to normalised cluster values for alpha contrast.
    batch_size : int, default=16
        Batch size passed to clusterizers during inference.
    cache : bool, default=False
        Forwarded to :func:`extract_patch_images`/:func:`extract_cluster_maps`:
        persist freshly extracted pixels/cluster maps into `patches.obsm` so
        later calls on the same `patches` skip re-reading from slides.

    Returns
    -------
    (fig, axes), a list of buffers, or None

    Raises
    ------
    ValueError
        If required inputs are missing or columns absent from patches.obs.

    Examples
    --------
    >>> # Fully automatic -- model resolved once here and shared across c1, c2
    >>> slides = ms.open_slides(manifest)
    >>> plot_patch_gallery_with_saliency(
    ...     patches, clusterizers=[c1, c2], model='uni2', slides=slides,
    ...     output_path='output/saliency.png'
    ... )
    >>>
    >>> # Cache once, plot many times without slides
    >>> plot_patch_gallery_with_saliency(
    ...     patches, clusterizers=[c1, c2], model='uni2', slides=slides,
    ...     cache=True, output_path='output/saliency_1.png'
    ... )
    >>> plot_patch_gallery_with_saliency(
    ...     patches, clusterizers=[c1, c2],   # no slides needed -- all cached
    ...     output_path='output/saliency_2.png'
    ... )
    """
    plan = (
        GalleryPlan(patches)
        .add_he_row(slides, tile_key=tile_key, cache=cache)
        .add_cluster_map_rows(
            clusterizers, model, slides,
            tile_key=tile_key, 
            saliency_alpha_power=saliency_alpha_power,
            batch_size=batch_size, cache=cache,
            blend_with_previous=blend_with_previous,
            cmap=cmap
        )
    )
    return plan.render(
        output_path=output_path,
        patches_per_row=patches_per_row,
        samples_per_figure=samples_per_figure,
        patch_display_size=patch_display_size,
        title=title,
        show_slide_ids=show_slide_ids,
        show_scores=show_scores,
        dpi=dpi,
        return_fig=return_fig,
        return_buffer=return_buffer,
        progress_bar=progress_bar,
    )


@deprecated_kwargs(
    sdata=removed(SLIDES_HINT),
    show_image_names=rename('show_slide_ids'),
)
def plot_patch_gallery(
    patches: "ad.AnnData",
    slides=None,
    output_path: Optional[str] = None,
    tile_key: str = DEFAULT_TILE_KEY,
    samples_per_figure: int = 100,
    patches_per_row: int = 10,
    patch_display_size: float = 2.0,
    show_slide_ids: bool = False,
    show_scores: bool = False,
    group_col: Optional[str] = None,
    border_alpha: float = 1.0,
    border_extend: float = 0.1,
    cmap: str = 'tab10',
    title: Optional[str] = None,
    dpi: int = 300,
    return_fig: bool = False,
    return_buffer: bool = False,
    progress_bar: bool = True,
    cache: bool = False,
) -> Optional[Union[Tuple[plt.Figure, np.ndarray], List[io.BytesIO]]]:
    """
    Create a grid gallery of tissue patches.

    A thin wrapper around :class:`GalleryPlan`
    (``GalleryPlan(patches).add_he_row(...)``); use `GalleryPlan` directly to
    combine H&E with other row types (cluster maps, CyCIF channels, ...).

    Parameters
    ----------
    patches : AnnData
        Selected tiles, e.g. from :func:`mesoslide.select_top_patches`.
        Required .obs columns: 'x', 'y' (plus 'slide_id' across slides).
    slides : WSIData, list of WSIData, or {slide_id: WSIData}, optional
        Not required in the common case -- falls back to
        `patches.obs['_slide_ref']`, populated automatically by
        :func:`mesoslide.select_top_patches` and friends. Build with
        :func:`mesoslide.open_slides`.
    output_path : str, optional
        File path to save to. For multiple pages, the patch range is
        inserted before the extension (e.g. ``gallery_1-100.png``).
    samples_per_figure : int
    patches_per_row : int
    patch_display_size : float
    show_slide_ids : bool
    show_scores : bool
    group_col : str, optional
        Column in patches.obs for border colour-coding.
    border_alpha : float
    cmap : str
    title : str, optional
    dpi : int
    return_fig : bool
        Return (fig, axes) -- only when there is exactly one page.
    return_buffer : bool
        Return a list of in-memory PNG buffers, one per page (always a
        list, even for a single page).
    progress_bar : bool
    cache : bool, default=False
        Forwarded to :func:`extract_patch_images`: cache the extracted array
        in ``patches.obsm['he_patch_img']`` so later calls on the same `patches`
        skip re-reading from slides.

    Returns
    -------
    (fig, axes), a list of buffers, or None

    Examples
    --------
    >>> # Automatic extraction
    >>> slides = ms.open_slides(manifest)
    >>> plot_patch_gallery(patches, slides=slides, output_path='output/gallery.png')
    >>>
    >>> # Cache once, plot many times without slides
    >>> plot_patch_gallery(patches, slides=slides, cache=True,
    ...                     output_path='output/gallery_1.png')
    >>> plot_patch_gallery(patches, output_path='output/gallery_2.png')  # no slides needed
    """
    has_slide_ref = SLIDE_REF in patches.obs.columns and patches.obs[SLIDE_REF].notna().any()
    if slides is None and HE_PATCH_IMG_KEY not in patches.obsm and not has_slide_ref:
        raise ValueError(
            "slides is required unless patches.obsm['he_patch_img'] is already "
            "cached (see extract_patch_images(..., cache=True)), or "
            "patches.obs['_slide_ref'] is populated (set automatically by "
            "mesoslide.select_top_patches and friends)."
        )

    plan = GalleryPlan(patches).add_he_row(
        slides, tile_key=tile_key, group_col=group_col, cmap=cmap,
        border_extend=border_extend, border_alpha=border_alpha, cache=cache,
    )

    return plan.render(
        output_path=output_path,
        patches_per_row=patches_per_row,
        samples_per_figure=samples_per_figure,
        patch_display_size=patch_display_size,
        title=title,
        show_slide_ids=show_slide_ids,
        show_scores=show_scores,
        dpi=dpi,
        return_fig=return_fig,
        return_buffer=return_buffer,
        progress_bar=progress_bar,
    )


def plot_feature_gallery(
    images: List[np.ndarray],
    group_ids: List,
    labels: Optional[List[str]] = None,
    n_cols: int = 10,
    patch_size: float = 2.0,
    border_extend: float = 0.1,
    border_alpha: float = 1.0,
    cmap: str = 'tab10',
    fontsize: float = 6,
) -> tuple:
    """
    Plot a grid of pre-extracted image arrays with cluster-coloured borders.

    Thin wrapper around _plot_image_grid for the SAE feature gallery use-case
    where images are already in memory.

    Parameters
    ----------
    images : list of np.ndarray, each (H, W, 3)
    group_ids : list
    labels : list of str, optional
    n_cols : int
    patch_size : float
    border_extend : float
    border_alpha : float
    cmap : str
    fontsize : float

    Returns
    -------
    fig, axs : tuple
    """
    return _plot_image_grid(
        images=images,
        labels=labels,
        group_ids=group_ids,
        n_cols=n_cols,
        patch_size=patch_size,
        border_extend=border_extend,
        border_alpha=border_alpha,
        cmap=cmap,
        fontsize=fontsize,
    )
