from typing import TYPE_CHECKING, Optional
import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from spatialdata import SpatialData
    import anndata as ad


def extract_patches(
    sdata: "SpatialData",
    patches: "ad.AnnData",
    channel_first: bool = True,
    progress_bar: bool = True,
    skip_errors: bool = True
) -> np.ndarray:
    """
    Extract image patches from WSI based on patch table metadata.
    
    Extracts patches from whole slide images using bounding box coordinates
    stored in the patch table. Returns a stacked numpy array suitable for
    batch processing, visualization, or model inference.
    
    Parameters
    ----------
    sdata : SpatialData
        Spatial data object containing the full WSI images.
    patches : AnnData
        AnnData object containing patch metadata in .obs.
        Required columns: 'image', 'xmin', 'xmax', 'ymin', 'ymax'.
    channel_first : bool, default=True
        If True, returns shape (N_patches, N_channels, Y, X) (PyTorch convention).
        If False, returns shape (N_patches, Y, X, N_channels) (matplotlib/numpy convention).
    progress_bar : bool, default=True
        Whether to show tqdm progress bar during extraction.
    skip_errors : bool, default=True
        If True, failed extractions are skipped and excluded from output.
        If False, raises exception on first failure.
        
    Returns
    -------
    patches_array : np.ndarray
        Array of extracted patches with shape depending on channel_first:
        - If channel_first=True: (N_patches, N_channels, Y, X)
        - If channel_first=False: (N_patches, Y, X, N_channels)
        
        Note: If skip_errors=True, N_patches may be less than len(patches.obs)
        due to failed extractions.
        
    Raises
    ------
    ValueError
        If required columns are missing from patches.obs.
    Exception
        If skip_errors=False and patch extraction fails.
        
    Examples
    --------
    >>> from meson.preprocessing import extract_patches
    >>> 
    >>> # Extract patches in PyTorch format
    >>> patches_array = extract_patches(sdata, patch_table, channel_first=True)
    >>> print(patches_array.shape)  # (100, 3, 224, 224)
    >>> 
    >>> # Extract for matplotlib visualization
    >>> patches_array = extract_patches(sdata, patch_table, channel_first=False)
    >>> plt.imshow(patches_array[0])
    
    Notes
    -----
    - Patches are computed eagerly into memory (via .compute())
    - For large batches, consider processing in chunks
    - Currently optimized for H&E (RGB) images
    """
    from meson._readwrite import get_base_level
    
    # Validate inputs
    required_cols = ['image', 'xmin', 'xmax', 'ymin', 'ymax']
    missing_cols = [col for col in required_cols if col not in patches.obs.columns]
    if missing_cols:
        raise ValueError(f"patches.obs missing required columns: {missing_cols}")
    
    patch_df = patches.obs
    extracted_patches = []
    
    # Setup iterator
    iterator = patch_df.iterrows()
    if progress_bar:
        iterator = tqdm(iterator, total=len(patch_df), desc="Extracting patches")
    
    for _, patch in iterator:
        try:
            # Extract patch from WSI at base level
            image_data = get_base_level(sdata[patch.image])[
                :,
                int(patch.ymin):int(patch.ymax),
                int(patch.xmin):int(patch.xmax)
            ]
            
            # Compute and convert to numpy
            if channel_first:
                # Keep as (C, Y, X)
                patch_array = image_data.compute().values
            else:
                # Transpose to (Y, X, C)
                patch_array = image_data.transpose('y', 'x', 'c').compute().values
            
            extracted_patches.append(patch_array)
            
        except Exception as e:
            if skip_errors:
                if progress_bar:
                    tqdm.write(f"Warning: Failed to load patch from {patch.image}: {e}")
                else:
                    print(f"Warning: Failed to load patch from {patch.image}: {e}")
                continue
            else:
                raise
    
    if len(extracted_patches) == 0:
        raise ValueError("No patches were successfully extracted")
    
    # Stack all patches
    patches_array = np.stack(extracted_patches, axis=0)
    
    return patches_array