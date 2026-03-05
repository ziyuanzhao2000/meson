import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import torch
import cv2

from meson._readwrite import get_base_level
from meson.preprocessing._extract_patches import extract_patches
from ._image_grid import _plot_image_grid

if TYPE_CHECKING:
    from spatialdata import SpatialData
    import anndata as ad
    from meson.tools.segmenters import TokenClusterizer

import matplotlib.patches as mpatches
import matplotlib.cm as cm
from math import ceil

def extract_patch_images(
    sdata,
    patch_df,
    progress: bool = True,
) -> list:
    """
    Extract raw image arrays from sdata for each row in patch_df.

    Parameters
    ----------
    sdata : SpatialData
    patch_df : pd.DataFrame or AnnData
        Must have .obs columns: image, ymin, ymax, xmin, xmax.
        If AnnData is passed, .obs is used automatically.
    progress : bool
        Show tqdm progress bar.

    Returns
    -------
    images : list of np.ndarray, each (H, W, 3)
    """
    from meson._readwrite import get_base_level
    from tqdm import tqdm

    if hasattr(patch_df, "obs"):
        obs = patch_df.obs
    else:
        obs = patch_df

    iterator = tqdm(obs.iterrows(), total=len(obs), desc="Extracting patches") \
        if progress else obs.iterrows()

    images = []
    for _, row in iterator:
        img = (
            get_base_level(sdata[row.image])
            [:, row.ymin:row.ymax, row.xmin:row.xmax]
            .transpose("y", "x", "c")
            .compute()
            .data
        )
        images.append(img)
    return images

def plot_patch_gallery_with_saliency(
    sdata: "SpatialData",
    patches: "ad.AnnData",
    clusterizers: List["TokenClusterizer"],
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
    saliency_alpha_power: float = 1.0
) -> Optional[Tuple[plt.Figure, np.ndarray]]:
    """
    Create a gallery of patches with H&E images and token cluster saliency maps.
    
    Displays patches in a grid where each patch has:
    - First row: Original H&E image
    - Subsequent rows: Saliency maps from each TokenClusterizer
    
    The saliency maps use transparency (alpha) based on cluster ordering to highlight
    regions of interest. If cluster_order is set in a TokenClusterizer, it uses that
    ordering; otherwise uses raw cluster labels.
    
    Parameters
    ----------
    sdata : SpatialData
        Spatial data object containing the full WSI images.
    patches : AnnData
        AnnData object containing patch metadata in .obs.
        Must have columns: 'image', 'xmin', 'xmax', 'ymin', 'ymax'.
        Optional column: 'score' (for show_scores=True).
    clusterizers : list of TokenClusterizer
        List of K TokenClusterizer instances to generate saliency maps.
        Each produces one row of saliency maps below the H&E row.
    output_path : str, optional
        Path to save the figure(s). Required when len(patches) > samples_per_figure.
    filename_prefix : str, default='patch_gallery_saliency'
        Prefix for saved figure filenames.
    samples_per_figure : int, default=100
        Maximum number of patches to display per figure page.
    patches_per_row : int, default=10
        Number of patches per row in the grid.
    patch_display_size : float, default=2.0
        Size of each patch subplot in inches.
    show_image_names : bool, default=False
        Whether to show source image name above each patch.
    show_scores : bool, default=False
        Whether to show score value above each patch.
        Requires 'score' column in patches.obs.
    title : str, optional
        Overall title for the figure (only used for single-page figures).
    dpi : int, default=300
        Resolution for saving figures.
    return_fig : bool, default=False
        Whether to return figure and axes. Only applies when creating
        a single page. Multi-page figures are always saved and not returned.
    progress_bar : bool, default=True
        Whether to show tqdm progress bar while processing.
    saliency_alpha_power : float, default=1.0
        Power to apply to normalized cluster values for alpha channel.
        Higher values increase contrast. E.g., 2.0 squares the values,
        making low values more transparent.
        
    Returns
    -------
    fig, axes : tuple of (plt.Figure, np.ndarray) or None
        Returns figure and axes array only if:
        - return_fig=True AND
        - len(patches) <= samples_per_figure (single page)
        Otherwise returns None.
        
    Raises
    ------
    ValueError
        If output_path is None when multiple pages are needed.
        If required columns are missing from patches.obs.
        If no clusterizers provided.
        
    Examples
    --------
    >>> from meson.plotting import plot_patch_gallery_with_saliency
    >>> from meson.tools.segmenters import TokenClusterizer
    >>> import joblib
    >>> 
    >>> # Create clusterizers
    >>> kmeans1 = joblib.load('classifier1.joblib')
    >>> kmeans2 = joblib.load('classifier2.joblib')
    >>> 
    >>> clusterizer1 = TokenClusterizer(uni, kmeans1, token_grid_size=14)
    >>> clusterizer2 = TokenClusterizer(uni, kmeans2, token_grid_size=14)
    >>> 
    >>> # Optionally set cluster orders
    >>> clusterizer1.cluster_order = np.array([2, 0, 1])
    >>> 
    >>> # Plot gallery
    >>> plot_patch_gallery_with_saliency(
    ...     sdata,
    ...     selected_patches,
    ...     clusterizers=[clusterizer1, clusterizer2],
    ...     show_scores=True,
    ...     patches_per_row=10,
    ...     output_path='output/saliency_maps'
    ... )
    """
    # Validate inputs
    required_cols = ['image', 'xmin', 'xmax', 'ymin', 'ymax']
    missing_cols = [col for col in required_cols if col not in patches.obs.columns]
    if missing_cols:
        raise ValueError(f"patches.obs missing required columns: {missing_cols}")
    
    if show_scores and 'score' not in patches.obs.columns:
        raise ValueError("show_scores=True requires 'score' column in patches.obs")
    
    if not clusterizers or len(clusterizers) == 0:
        raise ValueError("At least one TokenClusterizer must be provided")
    
    patch_df = patches.obs
    n_patches = len(patch_df)
    n_pages = int(np.ceil(n_patches / samples_per_figure))
    n_clusterizers = len(clusterizers)
    
    # Check if we need multiple pages but no output path provided
    if n_pages > 1 and output_path is None:
        raise ValueError(
            f"Dataset has {n_patches} patches requiring {n_pages} pages. "
            "Please provide output_path for multi-page figures."
        )
    
    for page_idx in range(n_pages):
        start_idx = page_idx * samples_per_figure
        end_idx = min(start_idx + samples_per_figure, n_patches)
        batch_df = patch_df.iloc[start_idx:end_idx]
        
        # Extract all patches for this page at once
        import anndata as ad
        batch_patches = ad.AnnData(obs=batch_df.reset_index(drop=True))
        
        if progress_bar and n_pages > 1:
            print(f"Processing page {page_idx+1}/{n_pages}...")
        
        extracted_patches = extract_patches(
            sdata,
            batch_patches,
            channel_first=True,  # (N, C, H, W) for TokenClusterizer
            progress_bar=progress_bar,
            skip_errors=True
        )
        
        # Generate cluster maps for each clusterizer
        all_cluster_maps = []
        for clust_idx, clusterizer in enumerate(clusterizers):
            if progress_bar:
                print(f"  Generating saliency maps with clusterizer {clust_idx+1}/{n_clusterizers}...")
            
            # Use the __call__ method to get cluster maps
            cluster_maps = clusterizer(
                extracted_patches,
                sdata=None,  # Already extracted
                output_size=None,  # Use patch size
                batch_size=16,
                show_progress=progress_bar
            )
            all_cluster_maps.append(cluster_maps)
        
        # Calculate grid dimensions
        n_samples = len(extracted_patches)
        n_cols = min(patches_per_row, n_samples)
        # Each patch has 1 H&E row + K saliency rows
        n_rows_per_patch = 1 + n_clusterizers
        n_rows_total = int(np.ceil(n_samples / patches_per_row)) * n_rows_per_patch
        
        # Create figure
        fig, axes = plt.subplots(
            n_rows_total, n_cols,
            figsize=(patch_display_size * n_cols, patch_display_size * n_rows_total)
        )
        
        # Ensure axes is 2D array
        if n_rows_total == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows_total == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Display patches
        for i in range(n_samples):
            row_base = (i // patches_per_row) * n_rows_per_patch
            col = i % patches_per_row
            
            # Get patch data and metadata
            image_data = extracted_patches[i]  # (C, H, W)
            patch = batch_df.iloc[i]
            
            # Convert to (H, W, C) for display
            image_display = image_data.transpose(1, 2, 0)
            
            # Build title (only for H&E row)
            title_parts = []
            if show_image_names:
                title_parts.append(str(patch.image))
            if show_scores:
                score_val = patch.score if hasattr(patch, 'score') else 0
                title_parts.append(f"Score: {score_val:.3f}")
            patch_title = "\n".join(title_parts) if title_parts else ""
            
            # Display H&E image (first row)
            axes[row_base, col].imshow(image_display)
            if patch_title:
                axes[row_base, col].set_title(patch_title, fontsize=8)
            axes[row_base, col].axis('off')
            
            # Display saliency maps (subsequent rows)
            for clust_idx in range(n_clusterizers):
                saliency_row = row_base + 1 + clust_idx
                cluster_map = all_cluster_maps[clust_idx][i]  # (H, W)
                
                # Get cluster order if available
                clusterizer = clusterizers[clust_idx]
                if clusterizer.cluster_order is not None:
                    # Remap cluster labels using the order
                    cluster_order = clusterizer.cluster_order
                    n_clusters = len(cluster_order)
                    # Map each cluster to its position in the ordering
                    alpha_values = cluster_order[cluster_map.astype(np.uint8)] / n_clusters
                else:
                    # Use raw cluster labels
                    n_clusters = cluster_map.max() + 1
                    alpha_values = cluster_map.astype(float) / n_clusters
                
                # Apply power for contrast adjustment
                alpha_values = alpha_values ** saliency_alpha_power
                
                # Create black overlay with alpha
                black_overlay = np.zeros((*alpha_values.shape, 4), dtype=np.float32)
                black_overlay[..., 3] = alpha_values
                
                # Display H&E with overlay
                axes[saliency_row, col].imshow(image_display, vmin=0, vmax=1)
                axes[saliency_row, col].imshow(black_overlay, vmin=0, vmax=1)
                print(np.unique(black_overlay[..., 3]))
                axes[saliency_row, col].axis('off')
        
        # Hide unused subplots
        for i in range(n_samples * n_rows_per_patch, n_rows_total * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
            axes[row, col].set_visible(False)
        
        # Add overall title for single-page figures
        if title and n_pages == 1:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        row_labels = ["H&E"] + [clust.feature_name for clust in clusterizers]
        for row_idx, label in enumerate(row_labels):
            y = 1 - (row_idx + 0.5) / n_rows_per_patch  # row center in figure coords
            fig.text(
                -0.01, y,  # -0.01 positions labels close to left edge
                label,
                fontsize=12,
                rotation=90,
                va="center", ha="center"
            )
    
        # Save if output_path provided
        if output_path is not None:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            
            # Format: prefix_samples_start-end.png
            filename = f'{filename_prefix}_samples_{start_idx+1}-{end_idx}.png'
            filepath = os.path.join(output_path, filename)
            fig.savefig(filepath, bbox_inches='tight', dpi=dpi)
            print(f"Saved: {filepath}")
        
        # Handle return behavior
        if n_pages == 1 and return_fig:
            return fig, axes
        else:
            plt.close(fig)
    
    return None

def plot_patch_gallery(
    sdata: "SpatialData",
    patches: "ad.AnnData",
    output_path: Optional[str] = None,
    filename_prefix: str = 'patch_gallery',
    samples_per_figure: int = 100,
    patches_per_row: int = 10,
    patch_display_size: float = 2.0,
    show_image_names: bool = False,
    show_scores: bool = False,
    group_col: Optional[str] = None,
    border_alpha: float = 1.0,
    cmap: str = 'tab10',
    title: Optional[str] = None,
    dpi: int = 300,
    return_fig: bool = False,
    progress_bar: bool = True
) -> Optional[Tuple[plt.Figure, np.ndarray]]:
    """
    Create a grid gallery of tissue patches from selected observations.

    Parameters
    ----------
    sdata : SpatialData
    patches : AnnData
        Must have obs columns: 'image', 'xmin', 'xmax', 'ymin', 'ymax'.
    output_path : str, optional
    filename_prefix : str
    samples_per_figure : int
    patches_per_row : int
    patch_display_size : float
    show_image_names : bool
    show_scores : bool
    group_col : str, optional
        Column in patches.obs used to color-code patch borders.
        E.g. a cluster assignment column.
    border_alpha : float
    cmap : str
    title : str, optional
    dpi : int
    return_fig : bool
    progress_bar : bool

    Returns
    -------
    fig, axes or None
    """
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

    for page_idx in range(n_pages):
        start_idx = page_idx * samples_per_figure
        end_idx = min(start_idx + samples_per_figure, n_patches)
        batch_df = patch_df.iloc[start_idx:end_idx]

        batch_patches = ad.AnnData(obs=batch_df.reset_index(drop=True))
        if progress_bar and n_pages > 1:
            print(f"Processing page {page_idx+1}/{n_pages}...")

        extracted = extract_patches(
            sdata, batch_patches,
            channel_first=False,
            progress_bar=progress_bar,
            skip_errors=True
        )

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
            images=extracted,
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
            fp = os.path.join(output_path,
                              f'{filename_prefix}_samples_{start_idx+1}-{end_idx}.png')
            fig.savefig(fp, bbox_inches='tight', dpi=dpi)
            print(f"Saved: {fp}")

        if n_pages == 1 and return_fig:
            return fig, axs
        else:
            plt.close(fig)

    return None


def plot_feature_gallery(
    images: List[np.ndarray],
    group_ids: List,
    labels: Optional[List[str]] = None,
    n_cols: int = 10,
    patch_size: float = 2.0,
    border_extend: float = 0.05,
    border_alpha: float = 1.0,
    cmap: str = 'tab10',
    fontsize: float = 6,
) -> tuple:
    """
    Plot a grid of pre-extracted image arrays with cluster-colored borders.

    This is a thin wrapper around _plot_image_grid for the common SAE
    feature gallery use-case where images are already in memory.

    Parameters
    ----------
    images : list of np.ndarray
        Pre-extracted patch images, each (H, W, 3).
    group_ids : list
        Cluster/group assignment per image for border coloring.
    labels : list of str, optional
        Per-image text label (e.g. feature index). Pass None to suppress.
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
        fontsize=fontsize
    )