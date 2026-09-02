import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union, List
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from meson.preprocessing._extract_patches import extract_patches
from meson.preprocessing._extract_saliency_maps import extract_saliency_maps
from ._image_grid import _plot_image_grid

if TYPE_CHECKING:
    from spatialdata import SpatialData
    import anndata as ad
    from meson.tools.segmenters import TokenClusterizer

def plot_patch_gallery_with_saliency(
    patches: "ad.AnnData",
    clusterizers: Optional[List["TokenClusterizer"]] = None,
    sdata: Optional["SpatialData"] = None,
    patches_array: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    saliency_maps: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    output_path: Optional[str] = None,
    filename_prefix: str = 'patch_gallery_saliency',
    samples_per_figure: int = 100,
    patches_per_row: int = 10,
    patch_display_size: float = 2.0,
    show_image_names: bool = False,
    show_scores: bool = False,
    title: Optional[str] = None,
    dpi: int = 300,
    return_fig: bool = False,
    progress_bar: bool = True,
    saliency_alpha_power: float = 1.0,
    batch_size: int = 16,
) -> Optional[Tuple[plt.Figure, np.ndarray]]:
    """
    Create a gallery of patches with H&E images and token cluster saliency maps.

    Displays patches in a grid where each patch occupies a column-group of rows:
    - Row 0: Original H&E image
    - Row 1..K: Saliency overlay for each clusterizer / pre-computed map

    Accepts either raw SpatialData + clusterizers (extraction done internally)
    or pre-computed arrays (useful when plotting multiple times or formats).

    Parameters
    ----------
    patches : AnnData
        Patch metadata in .obs.
        Required columns: 'image', 'xmin', 'xmax', 'ymin', 'ymax'.
        Optional column: 'score' (used when show_scores=True).
    clusterizers : list of TokenClusterizer, optional
        Required when saliency_maps is None. Each produces one saliency row.
    sdata : SpatialData, optional
        Required when patches_array is None.
    patches_array : np.ndarray or list of np.ndarray, optional
        Pre-extracted patches, channel-last (N, H, W, C) or list of (H, W, C).
        If provided, sdata is not used.
    saliency_maps : np.ndarray or list of np.ndarray, optional
        Pre-computed cluster label maps (uint8).
        np.ndarray shape: (N, K, H, W); list: N elements of (K, H, W).
        If provided, clusterizers are not called (but their feature_names are
        still used for row labels if clusterizers is also given).
    output_path : str, optional
        Directory to save figures. Required when n_patches > samples_per_figure.
    filename_prefix : str, default='patch_gallery_saliency'
    samples_per_figure : int, default=100
    patches_per_row : int, default=10
    patch_display_size : float, default=2.0
        Subplot size in inches.
    show_image_names : bool, default=False
    show_scores : bool, default=False
    title : str, optional
        Figure suptitle (single-page only).
    dpi : int, default=300
    return_fig : bool, default=False
        Return (fig, axes) for single-page figures.
    progress_bar : bool, default=True
    saliency_alpha_power : float, default=1.0
        Exponent applied to normalised cluster values for alpha contrast.
    batch_size : int, default=16
        Batch size passed to clusterizers during inference.

    Returns
    -------
    (fig, axes) or None
        Returned only when return_fig=True and a single page is produced.

    Raises
    ------
    ValueError
        If required inputs are missing or columns absent from patches.obs.

    Examples
    --------
    >>> # Fully automatic
    >>> plot_patch_gallery_with_saliency(
    ...     patches, clusterizers=[c1, c2], sdata=sdata,
    ...     output_path='output/saliency'
    ... )
    >>>
    >>> # Pre-computed (extract once, plot many times)
    >>> imgs = extract_patches(sdata, patches, channel_first=False)
    >>> maps = extract_saliency_maps(
    ...     extract_patches(sdata, patches, channel_first=True), [c1, c2]
    ... )
    >>> np.save("imgs.npy", imgs); np.save("maps.npy", maps)
    >>>
    >>> plot_patch_gallery_with_saliency(
    ...     patches,
    ...     clusterizers=[c1, c2],   # still used for row labels
    ...     patches_array=np.load("imgs.npy"),
    ...     saliency_maps=np.load("maps.npy"),
    ...     output_path='output/saliency'
    ... )
    """
    
    if patches_array is None and sdata is None:
        raise ValueError("Either sdata or patches_array must be provided.")
    if saliency_maps is None and (clusterizers is None or len(clusterizers) == 0):
        raise ValueError(
            "Either saliency_maps or at least one clusterizer must be provided."
        )

    required_cols = ['image', 'xmin', 'xmax', 'ymin', 'ymax']
    missing_cols = [c for c in required_cols if c not in patches.obs.columns]
    if missing_cols:
        raise ValueError(f"patches.obs missing required columns: {missing_cols}")
    if show_scores and 'score' not in patches.obs.columns:
        raise ValueError("show_scores=True requires 'score' column in patches.obs")

    patch_df = patches.obs
    n_patches = len(patch_df)
    n_pages = int(np.ceil(n_patches / samples_per_figure))

    if n_pages > 1 and output_path is None:
        raise ValueError(
            f"Dataset has {n_patches} patches requiring {n_pages} pages. "
            "Please provide output_path for multi-page figures."
        )

    
    if saliency_maps is not None:
        # Infer K from pre-computed maps
        if isinstance(saliency_maps, np.ndarray):
            n_clusterizers = saliency_maps.shape[1]
        else:
            n_clusterizers = saliency_maps[0].shape[0]
        row_labels = (
            ["H&E"] + [c.feature_name for c in clusterizers]
            if clusterizers
            else ["H&E"] + [f"Clusterizer {k+1}" for k in range(n_clusterizers)]
        )
    else:
        n_clusterizers = len(clusterizers)
        row_labels = ["H&E"] + [c.feature_name for c in clusterizers]

    import anndata as ad

    if patches_array is None:
        if progress_bar:
            print("Extracting patches from sdata...")
        patches_array = extract_patches(
            sdata, patches,
            channel_first=False,   # (N, H, W, C) for display
            progress_bar=progress_bar,
            skip_errors=True,
        )

    if saliency_maps is None:
        if progress_bar:
            print("Extracting patches (channel-first) for clusterizers...")
        patches_cf = extract_patches(
            sdata, patches,
            channel_first=True,
            progress_bar=progress_bar,
            skip_errors=True,
        )
        saliency_maps = extract_saliency_maps(
            patches_cf,
            clusterizers=clusterizers,
            batch_size=batch_size,
            progress_bar=progress_bar,
            shared_embedder=True
        )

    # Normalise to lists for uniform downstream indexing
    patches_list = (
        list(patches_array) if isinstance(patches_array, np.ndarray)
        else patches_array
    )
    saliency_list = (
        list(saliency_maps) if isinstance(saliency_maps, np.ndarray)
        else saliency_maps
    )
    
    for page_idx in range(n_pages):
        start_idx = page_idx * samples_per_figure
        end_idx = min(start_idx + samples_per_figure, n_patches)
        batch_df = patch_df.iloc[start_idx:end_idx]

        batch_images = patches_list[start_idx:end_idx]   # list of (H, W, C)
        batch_maps = saliency_list[start_idx:end_idx]    # list of (K, H, W)

        if progress_bar and n_pages > 1:
            print(f"Rendering page {page_idx+1}/{n_pages} "
                  f"(patches {start_idx+1}–{end_idx})...")

        n_samples = len(batch_images)
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

            image_display = batch_images[i]          # (H, W, C)
            patch_meta = batch_df.iloc[i]
            cluster_maps_i = batch_maps[i]           # (K, H, W) uint8

            # Title for the H&E row
            title_parts = []
            if show_image_names:
                title_parts.append(str(patch_meta['image']))
            if show_scores:
                title_parts.append(f"Score: {patch_meta.get('score', 0):.3f}")
            patch_title = "\n".join(title_parts)

            # H&E row
            ax_he = axes[row_base, col]
            ax_he.imshow(image_display)
            if patch_title:
                ax_he.set_title(patch_title, fontsize=8)
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

        if title and n_pages == 1:
            fig.suptitle(title, fontsize=16)

        plt.tight_layout()

        # Row labels on left margin
        for row_idx, label in enumerate(row_labels):
            y = 1 - (row_idx + 0.5) / n_rows_per_patch
            fig.text(-0.01, y, label, fontsize=12,
                     rotation=90, va="center", ha="center")

        if output_path is not None:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            fp = os.path.join(
                output_path,
                f'{filename_prefix}_samples_{start_idx+1}-{end_idx}.png'
            )
            fig.savefig(fp, bbox_inches='tight', dpi=dpi)
            print(f"Saved: {fp}")

        if n_pages == 1 and return_fig:
            return fig, axes

        plt.close(fig)

    return None

def plot_patch_gallery(
    patches: "ad.AnnData",
    sdata: Optional["SpatialData"] = None,
    patches_array: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    output_path: Optional[str] = None,
    filename_prefix: str = 'patch_gallery',
    samples_per_figure: int = 100,
    patches_per_row: int = 10,
    patch_display_size: float = 2.0,
    show_image_names: bool = False,
    show_scores: bool = False,
    group_col: Optional[str] = None,
    border_alpha: float = 1.0,
    border_extend: float = 0.1,
    cmap: str = 'tab10',
    title: Optional[str] = None,
    dpi: int = 300,
    return_fig: bool = False,
    progress_bar: bool = True,
) -> Optional[Tuple[plt.Figure, np.ndarray]]:
    """
    Create a grid gallery of tissue patches.

    Parameters
    ----------
    patches : AnnData
        Patch metadata in .obs.
        Required columns: 'image', 'xmin', 'xmax', 'ymin', 'ymax'.
    sdata : SpatialData, optional
        Required when patches_array is None.
    patches_array : np.ndarray or list of np.ndarray, optional
        Pre-extracted patches, channel-last (N, H, W, C) or list of (H, W, C).
        If provided, sdata is not used for extraction.
    output_path : str, optional
    filename_prefix : str
    samples_per_figure : int
    patches_per_row : int
    patch_display_size : float
    show_image_names : bool
    show_scores : bool
    group_col : str, optional
        Column in patches.obs for border colour-coding.
    border_alpha : float
    cmap : str
    title : str, optional
    dpi : int
    return_fig : bool
    progress_bar : bool

    Returns
    -------
    (fig, axes) or None

    Examples
    --------
    >>> # Automatic extraction
    >>> plot_patch_gallery(patches, sdata=sdata, output_path='output/gallery')
    >>>
    >>> # Pre-computed
    >>> imgs = extract_patches(sdata, patches, channel_first=False)
    >>> plot_patch_gallery(patches, patches_array=imgs, output_path='output/gallery')
    """
    if patches_array is None and sdata is None:
        raise ValueError("Either sdata or patches_array must be provided.")

    required_cols = ['image', 'xmin', 'xmax', 'ymin', 'ymax']
    missing_cols = [c for c in required_cols if c not in patches.obs.columns]
    if missing_cols:
        raise ValueError(f"patches.obs missing required columns: {missing_cols}")
    if show_scores and 'score' not in patches.obs.columns:
        raise ValueError("show_scores=True requires 'score' column in patches.obs")
    if group_col is not None and group_col not in patches.obs.columns:
        raise ValueError(f"group_col '{group_col}' not found in patches.obs")

    import anndata as ad

    patch_df = patches.obs
    n_patches = len(patch_df)
    n_pages = int(np.ceil(n_patches / samples_per_figure))

    if n_pages > 1 and output_path is None:
        raise ValueError(
            f"Dataset has {n_patches} patches requiring {n_pages} pages. "
            "Please provide output_path for multi-page figures."
        )

    # Full-dataset extraction (once, before paging)
    if patches_array is None:
        if progress_bar:
            print("Extracting patches from sdata...")
        patches_array = extract_patches(
            sdata, patches,
            channel_first=False,
            progress_bar=progress_bar,
            skip_errors=True,
        )

    patches_list = (
        list(patches_array) if isinstance(patches_array, np.ndarray)
        else patches_array
    )

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
            if show_image_names:
                parts.append(str(row['image']))
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

        if output_path is not None:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            fp = os.path.join(
                output_path,
                f'{filename_prefix}_samples_{start_idx+1}-{end_idx}.png'
            )
            fig.savefig(fp, bbox_inches='tight', dpi=dpi)
            print(f"Saved: {fp}")

        if n_pages == 1 and return_fig:
            return fig, axs

        plt.close(fig)

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