# /n/scratch/users/z/ziz531/meson/src/meson/plotting/_patch_gallery.py

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from meson._readwrite import get_base_level
from meson.preprocessing._extract_patches import extract_patches

if TYPE_CHECKING:
    from spatialdata import SpatialData
    import anndata as ad


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
    title: Optional[str] = None,
    dpi: int = 300,
    return_fig: bool = False,
    progress_bar: bool = True
) -> Optional[Tuple[plt.Figure, np.ndarray]]:
    """
    Create a grid gallery of tissue patches from selected observations.
    
    Displays patches in a grid layout with optional image names and scores.
    For large datasets (>samples_per_figure), creates multiple figure pages
    that are saved to disk.
    
    Parameters
    ----------
    sdata : SpatialData
        Spatial data object containing the full WSI images.
    patches : AnnData
        AnnData object containing patch metadata in .obs.
        Must have columns: 'image', 'xmin', 'xmax', 'ymin', 'ymax'.
        Optional column: 'score' (for show_scores=True).
    output_path : str, optional
        Path to save the figure(s). If None and multiple pages needed,
        raises ValueError. Required when len(patches) > samples_per_figure.
    filename_prefix : str, default='patch_gallery'
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
        Whether to show tqdm progress bar while rendering patches.
        
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
        
    Examples
    --------
    >>> from meson.plotting import plot_patch_gallery
    >>> import anndata as ad
    >>> 
    >>> # Simple single-page gallery
    >>> fig, axes = plot_patch_gallery(
    ...     sdata,
    ...     patches_subset,
    ...     patches_per_row=10,
    ...     return_fig=True
    ... )
    >>> 
    >>> # Multi-page gallery with scores
    >>> plot_patch_gallery(
    ...     sdata,
    ...     all_patches,
    ...     output_path='/path/to/output/gallery',
    ...     show_image_names=True,
    ...     show_scores=True,
    ...     samples_per_figure=100
    ... )
    """
    # Validate inputs
    required_cols = ['image', 'xmin', 'xmax', 'ymin', 'ymax']
    missing_cols = [col for col in required_cols if col not in patches.obs.columns]
    if missing_cols:
        raise ValueError(f"patches.obs missing required columns: {missing_cols}")
    
    if show_scores and 'score' not in patches.obs.columns:
        raise ValueError("show_scores=True requires 'score' column in patches.obs")
    
    patch_df = patches.obs
    n_patches = len(patch_df)
    n_pages = int(np.ceil(n_patches / samples_per_figure))
    
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
            channel_first=False,  # (N, Y, X, C) for matplotlib
            progress_bar=progress_bar,
            skip_errors=True
        )
        
        # Calculate grid dimensions
        n_samples = len(extracted_patches)
        n_rows = int(np.ceil(n_samples / patches_per_row))
        n_cols = min(patches_per_row, n_samples)
        
        # Create figure
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(patch_display_size * n_cols, patch_display_size * n_rows)
        )
        
        # Ensure axes is 2D array
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Display patches
        for i in range(n_samples):
            row = i // patches_per_row
            col = i % patches_per_row
            
            # Get patch data and metadata
            image_data = extracted_patches[i]
            patch = batch_df.iloc[i]
            
            # Build title
            title_parts = []
            if show_image_names:
                title_parts.append(str(patch.image))
            if show_scores:
                score_val = patch.score if hasattr(patch, 'score') else 0
                title_parts.append(f"Score: {score_val:.3f}")
            patch_title = "\n".join(title_parts) if title_parts else ""
            
            # Display patch
            axes[row, col].imshow(image_data)
            if patch_title:
                axes[row, col].set_title(patch_title, fontsize=8)
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(n_samples, n_rows * n_cols):
            row = i // patches_per_row
            col = i % patches_per_row
            axes[row, col].axis('off')
            axes[row, col].set_visible(False)
        
        # Add overall title for single-page figures
        if title and n_pages == 1:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        # Save if output_path provided
        if output_path is not None:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            
            # Format: prefix_samples_start-end.png (e.g., feature_00017_samples_1-100.png)
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


def sample_patches_for_feature(
    sdata: "SpatialData",
    feature_prefix: str,
    feature_idx: int,
    n_samples: int = 20,
    image_names: Optional[list] = None,
    point_name: str = 'grid_point',
    random_state: Optional[int] = None
) -> "ad.AnnData":
    """
    Sample patches where a specific feature is active across images.
    
    Collects patches where the binary feature column equals 1,
    then randomly samples n_samples patches from across all images.
    
    Parameters
    ----------
    sdata : SpatialData
        Spatial data object.
    feature_prefix : str
        Prefix of the feature column (e.g., 'kmeans_label', 'UNI_SAE').
    feature_idx : int
        Feature index to sample from.
    n_samples : int, default=20
        Number of patches to sample.
    image_names : list of str, optional
        List of image names to sample from. If None, uses all images.
    point_name : str, default='grid_point'
        Name of the point element.
    random_state : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    sampled_patches : AnnData
        Concatenated AnnData with sampled patches.
        Contains 'score' column set to 1.0 for all samples.
        
    Examples
    --------
    >>> from meson.plotting import sample_patches_for_feature
    >>> 
    >>> patches = sample_patches_for_feature(
    ...     sdata,
    ...     feature_prefix='kmeans_label',
    ...     feature_idx=0,
    ...     n_samples=20
    ... )
    """
    import anndata as ad
    
    if image_names is None:
        image_names = list(set([name.split('_')[0] for name in sdata.images]))
        image_names.sort()
    
    feature_col = f'{feature_prefix}_{feature_idx}'
    
    # Collect all active patch indices
    all_active_indices = []
    for image_name in image_names:
        patch_table_name = f'{image_name}_{point_name}_patch'
        if patch_table_name not in sdata.tables:
            continue
        
        patch_data = sdata[patch_table_name]
        if feature_col not in patch_data.obs.columns:
            continue
        
        active_mask = patch_data.obs[feature_col] == 1
        active_indices = np.where(active_mask)[0]
        all_active_indices.extend([(image_name, idx) for idx in active_indices])
    
    if len(all_active_indices) == 0:
        raise ValueError(f"No active patches found for {feature_col}")
    
    # Sample patches
    if random_state is not None:
        np.random.seed(random_state)
    
    n_available = len(all_active_indices)
    n_to_sample = min(n_samples, n_available)
    
    if n_available < n_samples:
        print(f"Warning: Only {n_available} patches available, sampling all.")
    
    sampled_idx = np.random.choice(n_available, size=n_to_sample, replace=False)
    sampled_pairs = [all_active_indices[i] for i in sampled_idx]
    
    # Extract patches
    sampled_patches_list = []
    for image_name, idx in sampled_pairs:
        patch_table_name = f'{image_name}_{point_name}_patch'
        patch_subset = sdata[patch_table_name][idx:idx+1, []]
        sampled_patches_list.append(patch_subset)
    
    # Concatenate
    result = ad.concat(sampled_patches_list)
    result.obs['score'] = 1.0
    
    return result