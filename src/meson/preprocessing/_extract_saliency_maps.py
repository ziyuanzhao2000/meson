from typing import TYPE_CHECKING, List, Optional, Union
import warnings
import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from meson.tools.segmenters import TokenClusterizer

def extract_saliency_maps(
    patches_array: Union[np.ndarray, List[np.ndarray]],
    clusterizers: List["TokenClusterizer"],
    batch_size: int = 16,
    progress_bar: bool = True,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Generate token cluster saliency maps for a set of pre-extracted patches.

    For each clusterizer, runs inference on the patches to produce a spatial
    map of cluster labels per patch. The raw cluster label maps are returned
    (as uint8), keeping visualization concerns (alpha, power, colourmaps)
    separate.

    Parameters
    ----------
    patches_array : np.ndarray or list of np.ndarray
        Pre-extracted patches in channel-first format.
        - np.ndarray: shape (N, C, H, W)
        - list: each element shape (C, H, W); used when patches differ in size.
    clusterizers : list of TokenClusterizer
        K clusterizer instances. Each must be callable and accept a batch of
        patches in (N, C, H, W) format.
    batch_size : int, default=16
        Number of patches to process at once per clusterizer call.
    progress_bar : bool, default=True
        Whether to show tqdm progress bars.

    Returns
    -------
    saliency_maps : np.ndarray or list of np.ndarray
        Cluster label maps as uint8.
        - If all maps share the same spatial shape AND patches_array was a
          stacked array: np.ndarray of shape (N, K, H, W), dtype uint8.
        - Otherwise: list of length N, each element shape (K, H, W), dtype uint8.

    Raises
    ------
    ValueError
        If clusterizers is empty.
        If patches_array is empty.

    Examples
    --------
    >>> from meson.preprocessing import extract_patches, extract_saliency_maps
    >>>
    >>> patches = extract_patches(sdata, patch_table, channel_first=True)
    >>> maps = extract_saliency_maps(patches, clusterizers=[c1, c2])
    >>> print(maps.shape)   # (N, 2, H, W)  dtype=uint8
    >>>
    >>> # Save for later reuse
    >>> np.save("patches.npy", patches)
    >>> np.save("saliency_maps.npy", maps)

    Notes
    -----
    - Alpha mapping, power transforms, and colourmap selection are purely
      visualization concerns and belong in the plotting layer, not here.
    - Each clusterizer's __call__ must return an array of shape (N, H, W)
      with integer cluster labels.
    """
    if not clusterizers:
        raise ValueError("At least one TokenClusterizer must be provided.")

    is_list_input = isinstance(patches_array, list)

    if is_list_input:
        if len(patches_array) == 0:
            raise ValueError("patches_array is empty.")
        n_patches = len(patches_array)
    else:
        if patches_array.ndim != 4:
            raise ValueError(
                f"patches_array must be 4-D (N, C, H, W), got shape {patches_array.shape}."
            )
        n_patches = patches_array.shape[0]

    n_clusterizers = len(clusterizers)

    # Run each clusterizer and collect maps: list of (N, H, W) arrays
    per_clusterizer_maps = []
    clust_iter = enumerate(clusterizers)
    if progress_bar:
        clust_iter = tqdm(
            clust_iter,
            total=n_clusterizers,
            desc="Running clusterizers",
        )

    for clust_idx, clusterizer in clust_iter:
        if progress_bar:
            tqdm.write(
                f"  Clusterizer {clust_idx + 1}/{n_clusterizers}: "
                f"{getattr(clusterizer, 'feature_name', str(clusterizer))}"
            )

        # clusterizer expects (N, C, H, W); handle list input by stacking
        # temporarily if all shapes agree, else process one-by-one
        if is_list_input:
            shapes = [p.shape for p in patches_array]
            if len(set(shapes)) == 1:
                batch = np.stack(patches_array, axis=0)
                cluster_maps = clusterizer(
                    batch,
                    batch_size=batch_size,
                    show_progress=progress_bar,
                )  # (N, H, W)
            else:
                # Process individually for variable-size patches
                cluster_maps = []
                patch_iter = patches_array
                if progress_bar:
                    patch_iter = tqdm(
                        patch_iter, total=n_patches, desc="  Patches", leave=False
                    )
                for patch in patch_iter:
                    m = clusterizer(
                        patch[None],  # (1, C, H, W)
                        batch_size=1,
                        show_progress=False,
                    )  # (1, H, W)
                    cluster_maps.append(m[0])
        else:
            cluster_maps = clusterizer(
                patches_array,
                batch_size=batch_size,
                show_progress=progress_bar,
            )  # (N, H, W)

        per_clusterizer_maps.append(cluster_maps)

    # per_clusterizer_maps: K elements, each (N, H, W) array or list of (H, W)
    # Transpose to per-patch view: N elements, each (K, H, W)

    if is_list_input and isinstance(per_clusterizer_maps[0], list):
        # Variable spatial sizes: return list of (K, H, W) uint8
        saliency_list = []
        for i in range(n_patches):
            patch_maps = np.stack(
                [per_clusterizer_maps[k][i] for k in range(n_clusterizers)],
                axis=0
            ).astype(np.uint8)  # (K, H, W)
            saliency_list.append(patch_maps)
        return saliency_list
    else:
        # All maps are (N, H, W) arrays — stack to (K, N, H, W) then transpose
        stacked = np.stack(
            [np.asarray(m) for m in per_clusterizer_maps], axis=0
        )  # (K, N, H, W)
        saliency_array = stacked.transpose(1, 0, 2, 3).astype(np.uint8)  # (N, K, H, W)

        # Confirm all spatial shapes are consistent; warn otherwise
        spatial_shapes = set(
            (saliency_array[i, 0].shape) for i in range(n_patches)
        )
        if len(spatial_shapes) > 1:
            warnings.warn(
                "Saliency maps have inconsistent spatial shapes. "
                "Returning stacked array may be misleading; "
                "consider using list input for variable-size patches.",
                UserWarning,
                stacklevel=2,
            )

        return saliency_array