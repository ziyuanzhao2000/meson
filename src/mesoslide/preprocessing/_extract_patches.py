from typing import TYPE_CHECKING, Optional, Union, List
import warnings
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
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Extract image patches from WSI based on patch table metadata.

    Extracts patches from whole slide images using bounding box coordinates
    stored in the patch table. Returns a stacked numpy array if all patches
    share the same shape, otherwise a list of arrays.

    Parameters
    ----------
    sdata : SpatialData
        Spatial data object containing the full WSI images.
    patches : AnnData
        AnnData object containing patch metadata in .obs.
        Required columns: 'image', 'xmin', 'xmax', 'ymin', 'ymax'.
    channel_first : bool, default=True
        If True, each patch has shape (C, H, W) → stacked: (N, C, H, W).
        If False, each patch has shape (H, W, C) → stacked: (N, H, W, C).
    progress_bar : bool, default=True
        Whether to show tqdm progress bar during extraction.
    skip_errors : bool, default=True
        If True, failed extractions are skipped and excluded from output.
        If False, raises exception on first failure.

    Returns
    -------
    patches_array : np.ndarray or list of np.ndarray
        If all patches share the same shape, returns a stacked np.ndarray:
        - channel_first=True:  (N, C, H, W)
        - channel_first=False: (N, H, W, C)
        If shapes differ, returns a list of arrays and emits a warning.

        Note: If skip_errors=True, N may be less than len(patches.obs).

    Raises
    ------
    ValueError
        If required columns are missing from patches.obs, or if no patches
        were successfully extracted.
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

    required_cols = ['image', 'xmin', 'xmax', 'ymin', 'ymax']
    missing_cols = [col for col in required_cols if col not in patches.obs.columns]
    if missing_cols:
        raise ValueError(f"patches.obs missing required columns: {missing_cols}")

    patch_df = patches.obs
    extracted_patches = []

    iterator = patch_df.iterrows()
    if progress_bar:
        iterator = tqdm(iterator, total=len(patch_df), desc="Extracting patches")

    for _, patch in iterator:
        try:
            image_data = get_base_level(sdata[patch.image])[
                :,
                int(patch.ymin):int(patch.ymax),
                int(patch.xmin):int(patch.xmax)
            ]

            if channel_first:
                patch_array = image_data.compute().values          # (C, H, W)
            else:
                patch_array = image_data.transpose('y', 'x', 'c').compute().values  # (H, W, C)

            extracted_patches.append(patch_array)

        except Exception as e:
            if skip_errors:
                msg = f"Warning: Failed to load patch from {patch.image}: {e}"
                tqdm.write(msg) if progress_bar else print(msg)
                continue
            else:
                raise

    if len(extracted_patches) == 0:
        raise ValueError("No patches were successfully extracted")

    # Stack if all shapes are identical, otherwise return list
    shapes = [p.shape for p in extracted_patches]
    if len(set(shapes)) == 1:
        return np.stack(extracted_patches, axis=0)
    else:
        warnings.warn(
            f"Patches have inconsistent shapes ({len(set(shapes))} distinct shapes). "
            "Returning a list instead of a stacked array.",
            UserWarning,
            stacklevel=2,
        )
        return extracted_patches