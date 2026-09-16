import io
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union, List
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from mesoslide._slides import DEFAULT_TILE_KEY, PATCH_IMG_KEY, SLIDE_ID
from mesoslide._deprecated import SLIDES_HINT, deprecated_kwargs, removed, rename
from mesoslide.preprocessing._extract_patches import extract_patch_images
from mesoslide.preprocessing._extract_cluster_maps import cluster_img_key, extract_cluster_maps
from ._image_grid import _plot_image_grid
from ._utils import _finish_plot

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
) -> Optional[Union[Tuple[plt.Figure, np.ndarray], List[io.BytesIO]]]:
    """
    Create a gallery of patches with H&E images and token cluster saliency maps.

    Displays patches in a grid where each patch occupies a column-group of rows:
    - Row 0: Original H&E image
    - Row 1..K: Cluster-map overlay for each clusterizer

    Pixel data and cluster maps are always read through
    :func:`extract_patch_images`/:func:`extract_cluster_maps`, which check
    `patches.obsm` first and only fall back to `slides` for whatever isn't
    already cached there -- pass `cache=True` to persist freshly computed
    results back into `patches` for reuse across calls.

    Parameters
    ----------
    patches : AnnData
        Selected tiles, e.g. from :func:`mesoslide.select_top_patches`.
        Required .obs columns: 'x', 'y' (plus 'slide_id' across slides).
        Optional column: 'score' (used when show_scores=True).
    clusterizers : list of TokenClusterizer
        Each produces one cluster-map row. Must have distinct, non-empty
        `feature_name`s (see :func:`extract_cluster_maps`).
    slides : WSIData, list of WSIData, or {slide_id: WSIData}, optional
        Required unless the pixel array and every clusterizer's cluster map
        are already cached in `patches.obsm` (see `cache`). Build with
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
    >>> # Fully automatic
    >>> slides = ms.open_slides(manifest)
    >>> plot_patch_gallery_with_saliency(
    ...     patches, clusterizers=[c1, c2], slides=slides,
    ...     output_path='output/saliency.png'
    ... )
    >>>
    >>> # Cache once, plot many times without slides
    >>> plot_patch_gallery_with_saliency(
    ...     patches, clusterizers=[c1, c2], slides=slides,
    ...     cache=True, output_path='output/saliency_1.png'
    ... )
    >>> plot_patch_gallery_with_saliency(
    ...     patches, clusterizers=[c1, c2],   # no slides needed -- all cached
    ...     output_path='output/saliency_2.png'
    ... )
    """

    if not clusterizers:
        raise ValueError("At least one clusterizer must be provided.")
    names = [c.feature_name for c in clusterizers]
    if any(not n for n in names):
        raise ValueError(
            "Every clusterizer must have a non-empty feature_name -- it's "
            "used as this clusterizer's patches.obsm cache key."
        )
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(
            f"clusterizers must have distinct feature_name values to avoid "
            f"colliding patches.obsm keys; duplicates: {dupes}"
        )
    if slides is None:
        missing_pixel_cache = PATCH_IMG_KEY not in patches.obsm
        missing_cluster_cache = any(
            cluster_img_key(c) not in patches.obsm for c in clusterizers
        )
        if missing_pixel_cache or missing_cluster_cache:
            raise ValueError(
                "slides is required unless patches.obsm already has the "
                "cached pixel array and every clusterizer's cached cluster "
                "map (see extract_patch_images/extract_cluster_maps "
                "cache=True)."
            )

    if show_slide_ids and SLIDE_ID not in patches.obs.columns:
        raise ValueError(f"show_slide_ids=True requires a '{SLIDE_ID}' column in patches.obs")
    if show_scores and 'score' not in patches.obs.columns:
        raise ValueError("show_scores=True requires 'score' column in patches.obs")

    patch_df = patches.obs
    n_patches = len(patch_df)
    n_pages = int(np.ceil(n_patches / samples_per_figure))

    if n_pages > 1 and output_path is None and not return_buffer:
        raise ValueError(
            f"Dataset has {n_patches} patches requiring {n_pages} pages. "
            "Please provide output_path and/or return_buffer=True for multi-page figures."
        )

    n_clusterizers = len(clusterizers)
    row_labels = ["H&E"] + [c.feature_name for c in clusterizers]

    import anndata as ad

    if progress_bar:
        print("Extracting patches from slides...")
    patches_array = extract_patch_images(
        patches, slides,
        tile_key=tile_key,
        channel_first=False,   # (N, H, W, C) for display
        progress_bar=progress_bar,
        skip_errors=True,
        cache=cache,
    )

    # extract_cluster_maps handles one clusterizer at a time -- the "shared
    # embedding computed once" property comes from run_model_stages' own
    # per-key caching in patches.obsm, not from anything special about
    # calling it with a list, so stacking the K results is done here.
    per_clusterizer_maps = [
        extract_cluster_maps(
            patches, slides, c,
            batch_size=batch_size,
            progress_bar=progress_bar,
            cache=cache,
        )
        for c in clusterizers
    ]

    is_list_result = isinstance(per_clusterizer_maps[0], list)
    if is_list_result:
        cluster_maps = [
            np.stack(
                [per_clusterizer_maps[k][i] for k in range(n_clusterizers)],
                axis=0,
            ).astype(np.uint8)  # (K, H, W)
            for i in range(n_patches)
        ]
    else:
        stacked = np.stack(per_clusterizer_maps, axis=0)  # (K, N, H, W)
        cluster_maps = stacked.transpose(1, 0, 2, 3).astype(np.uint8)  # (N, K, H, W)

    # Normalise to lists for uniform downstream indexing
    patches_list = (
        list(patches_array) if isinstance(patches_array, np.ndarray)
        else patches_array
    )
    cluster_maps_list = (
        list(cluster_maps) if isinstance(cluster_maps, np.ndarray)
        else cluster_maps
    )

    # Per-row title text, resolved once from patches.obs
    patch_titles = []
    for _, row in patch_df.iterrows():
        title_parts = []
        if show_slide_ids:
            title_parts.append(str(row[SLIDE_ID]))
        if show_scores:
            title_parts.append(f"Score: {row.get('score', 0):.3f}")
        patch_titles.append("\n".join(title_parts) if title_parts else None)

    buffers = [] if return_buffer else None

    for page_idx in range(n_pages):
        start_idx = page_idx * samples_per_figure
        end_idx = min(start_idx + samples_per_figure, n_patches)

        batch_images = patches_list[start_idx:end_idx]        # list of (H, W, C)
        batch_maps = cluster_maps_list[start_idx:end_idx]      # list of (K, H, W)
        batch_titles = patch_titles[start_idx:end_idx]

        if progress_bar and n_pages > 1:
            print(f"Rendering page {page_idx+1}/{n_pages} "
                  f"(patches {start_idx+1}–{end_idx})...")

        fp = None
        if output_path is not None:
            fp = output_path if n_pages == 1 else _paged_path(output_path, start_idx, end_idx)
            Path(fp).parent.mkdir(parents=True, exist_ok=True)

        result = _render_saliency_page(
            batch_images, batch_maps, batch_titles, row_labels, n_clusterizers,
            patches_per_row=patches_per_row,
            patch_display_size=patch_display_size,
            saliency_alpha_power=saliency_alpha_power,
            title=title if n_pages == 1 else None,
            dpi=dpi,
            output_path=fp,
            return_fig=return_fig and n_pages == 1,
            return_buffer=return_buffer,
        )
        if n_pages == 1 and return_fig and not return_buffer:
            return result
        if return_buffer:
            buffers.append(result)

    if return_buffer:
        return buffers
    return None


def _render_saliency_page(
    images: List[np.ndarray],
    cluster_maps: List[np.ndarray],
    patch_titles: List[Optional[str]],
    row_labels: List[str],
    n_clusterizers: int,
    *,
    patches_per_row: int,
    patch_display_size: float,
    saliency_alpha_power: float,
    title: Optional[str],
    dpi: int,
    output_path: Optional[str],
    return_fig: bool,
    return_buffer: bool,
) -> Union[Tuple[plt.Figure, np.ndarray], io.BytesIO, None]:
    """Render one page of the H&E + saliency-overlay grid from already-extracted arrays."""
    n_samples = len(images)
    n_cols = min(patches_per_row, n_samples)
    n_rows_per_patch = 1 + n_clusterizers
    n_rows_total = int(np.ceil(n_samples / patches_per_row)) * n_rows_per_patch

    fig, axes = plt.subplots(
        n_rows_total, n_cols,
        figsize=(patch_display_size * n_cols,
                 patch_display_size * n_rows_total)
    )

    # Normalise axes to 2-D array
    if n_rows_total == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows_total == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_samples):
        row_base = (i // patches_per_row) * n_rows_per_patch
        col = i % patches_per_row

        image_display = images[i]          # (H, W, C)
        cluster_maps_i = cluster_maps[i]   # (K, H, W) uint8

        # H&E row
        ax_he = axes[row_base, col]
        ax_he.imshow(image_display)
        if patch_titles[i]:
            ax_he.set_title(patch_titles[i], fontsize=8)
        ax_he.axis('off')

        # Saliency rows
        for k in range(n_clusterizers):
            cluster_map = cluster_maps_i[k].astype(np.float32)  # (H, W)
            n_clusters = 3
            alpha_values = (cluster_map / n_clusters) ** saliency_alpha_power

            overlay = np.zeros((*alpha_values.shape, 4), dtype=np.float32)
            overlay[..., 3] = alpha_values  # black with variable alpha

            ax_sal = axes[row_base + 1 + k, col]
            ax_sal.imshow(image_display)
            ax_sal.imshow(overlay)
            ax_sal.axis('off')

    # Hide unused axes
    for i in range(n_samples, n_rows_total // n_rows_per_patch * n_cols):
        for k in range(n_rows_per_patch):
            r = (i // n_cols) * n_rows_per_patch + k
            c = i % n_cols
            if r < n_rows_total:
                axes[r, c].axis('off')
                axes[r, c].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    # Row labels on left margin
    for row_idx, label in enumerate(row_labels):
        y = 1 - (row_idx + 0.5) / n_rows_per_patch
        fig.text(-0.01, y, label, fontsize=12,
                 rotation=90, va="center", ha="center")

    keep_alive = return_fig and not return_buffer
    result = _finish_plot(fig, axes, show=keep_alive, save=output_path,
                           return_fig=False, return_buffer=return_buffer, dpi=dpi)
    if output_path is not None:
        print(f"Saved: {output_path}")
    if return_fig and not return_buffer:
        return fig, axes
    return result


def _paged_path(output_path: str, start_idx: int, end_idx: int) -> str:
    """Derive a per-page file path by inserting the patch range before the extension."""
    p = Path(output_path)
    return str(p.with_name(f"{p.stem}_{start_idx + 1}-{end_idx}{p.suffix}"))


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

    Parameters
    ----------
    patches : AnnData
        Selected tiles, e.g. from :func:`mesoslide.select_top_patches`.
        Required .obs columns: 'x', 'y' (plus 'slide_id' across slides).
    slides : WSIData, list of WSIData, or {slide_id: WSIData}, optional
        Required unless the pixel array is already cached in
        `patches.obsm['patch_img']` (see `cache`). Build with
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
        in ``patches.obsm['patch_img']`` so later calls on the same `patches`
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
    if slides is None and PATCH_IMG_KEY not in patches.obsm:
        raise ValueError(
            "slides is required unless patches.obsm['patch_img'] is already "
            "cached (see extract_patch_images(..., cache=True))."
        )

    if show_slide_ids and SLIDE_ID not in patches.obs.columns:
        raise ValueError(f"show_slide_ids=True requires a '{SLIDE_ID}' column in patches.obs")
    if show_scores and 'score' not in patches.obs.columns:
        raise ValueError("show_scores=True requires 'score' column in patches.obs")
    if group_col is not None and group_col not in patches.obs.columns:
        raise ValueError(f"group_col '{group_col}' not found in patches.obs")

    import anndata as ad

    patch_df = patches.obs
    n_patches = len(patch_df)
    n_pages = int(np.ceil(n_patches / samples_per_figure))

    if n_pages > 1 and output_path is None and not return_buffer:
        raise ValueError(
            f"Dataset has {n_patches} patches requiring {n_pages} pages. "
            "Please provide output_path and/or return_buffer=True for multi-page figures."
        )

    # Full-dataset extraction (once, before paging)
    if progress_bar:
        print("Extracting patches from slides...")
    patches_array = extract_patch_images(
        patches, slides,
        tile_key=tile_key,
        channel_first=False,
        progress_bar=progress_bar,
        skip_errors=True,
        cache=cache,
    )

    patches_list = (
        list(patches_array) if isinstance(patches_array, np.ndarray)
        else patches_array
    )

    buffers = [] if return_buffer else None

    for page_idx in range(n_pages):
        start_idx = page_idx * samples_per_figure
        end_idx = min(start_idx + samples_per_figure, n_patches)
        batch_df = patch_df.iloc[start_idx:end_idx]
        batch_images = patches_list[start_idx:end_idx]

        if progress_bar and n_pages > 1:
            print(f"Rendering page {page_idx+1}/{n_pages} "
                  f"(patches {start_idx+1}–{end_idx})...")

        labels = []
        for _, row in batch_df.iterrows():
            parts = []
            if show_slide_ids:
                parts.append(str(row[SLIDE_ID]))
            if show_scores:
                parts.append(f"Score: {row.get('score', 0):.3f}")
            labels.append('\n'.join(parts) if parts else None)

        group_ids = (
            batch_df[group_col].tolist() if group_col is not None else None
        )

        fig, axs = _plot_image_grid(
            images=batch_images,
            border_extend=border_extend,
            labels=labels if any(l is not None for l in labels) else None,
            group_ids=group_ids,
            n_cols=patches_per_row,
            patch_size=patch_display_size,
            border_alpha=border_alpha,
            cmap=cmap,
        )

        if title and n_pages == 1:
            fig.suptitle(title, fontsize=16)

        fp = None
        if output_path is not None:
            fp = output_path if n_pages == 1 else _paged_path(output_path, start_idx, end_idx)
            Path(fp).parent.mkdir(parents=True, exist_ok=True)

        keep_alive = n_pages == 1 and return_fig and not return_buffer
        result = _finish_plot(fig, axs, show=keep_alive, save=fp,
                               return_fig=False, return_buffer=return_buffer, dpi=dpi)
        if fp is not None:
            print(f"Saved: {fp}")
        if return_buffer:
            buffers.append(result)
        if n_pages == 1 and return_fig and not return_buffer:
            return fig, axs

    if return_buffer:
        return buffers
    return None

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