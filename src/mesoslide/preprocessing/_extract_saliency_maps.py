from typing import TYPE_CHECKING, List, Optional, Union
import warnings
import numpy as np
import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from meson.tools.segmenters import TokenClusterizer

def extract_saliency_maps(
    patches_array: Union[np.ndarray, List[np.ndarray]],
    clusterizers: List["TokenClusterizer"],
    batch_size: int = 16,
    progress_bar: bool = True,
    shared_embedder: bool = False,
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
    shared_embedder : bool, default=False
        If True, assumes all clusterizers use the same embedding model.
        Embeddings are computed once using the first clusterizer's embedder,
        then each clusterizer's _cluster_tokens and _rasterize are called
        directly, avoiding redundant forward passes.

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
    >>> # If all clusterizers share the same embedder, avoid redundant passes:
    >>> maps = extract_saliency_maps(
    ...     patches, clusterizers=[c1, c2], shared_embedder=True
    ... )
    >>>
    >>> # Save for later reuse
    >>> np.save("patches.npy", patches)
    >>> np.save("saliency_maps.npy", maps)

    Notes
    -----
    - Alpha mapping, power transforms, and power transforms, and colourmap
      selection are purely visualization concerns and belong in the plotting
      layer, not here.
    - Each clusterizer's __call__ must return an array of shape (N, H, W)
      with integer cluster labels.
    - When shared_embedder=True, _cluster_tokens and _rasterize are called
      directly on all but the first clusterizer, reusing cached embeddings.
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

    # per_clusterizer_maps[k] will be either an (N, H, W) ndarray or list of (H, W)
    per_clusterizer_maps = [None] * n_clusterizers

    if shared_embedder:
        if is_list_input:
            shapes = [p.shape for p in patches_array]
            all_same_shape = len(set(shapes)) == 1
        else:
            all_same_shape = True

        # Initialise accumulators: list-of-lists per clusterizer
        accum = [[] for _ in range(n_clusterizers)]

        if is_list_input and not all_same_shape:
            # Variable-size patches: process one at a time
            patch_iter = enumerate(patches_array)
            if progress_bar:
                patch_iter = tqdm(
                    patch_iter, total=n_patches, desc="Patches (shared embedder)"
                )
            first_clusterizer = clusterizers[0]
            first_clusterizer.embedder.model.eval()

            for _, patch in patch_iter:
                batch = torch.from_numpy(patch[None]).float() / 255.0  # (1, C, H, W)
                output_size = (patch.shape[1], patch.shape[2])

                with torch.inference_mode():
                    token_embeds = first_clusterizer.embedder.get_token_embeddings(
                        batch.to(first_clusterizer.device), remove_cls=True
                    )  # (1, N_tokens, D)

                for k, clusterizer in enumerate(clusterizers):
                    cluster_map = clusterizer._cluster_tokens(token_embeds)  # (1, g, g)
                    rasterized = clusterizer._rasterize(cluster_map, output_size)  # (1, H, W)
                    accum[k].append(rasterized[0])

            for k in range(n_clusterizers):
                per_clusterizer_maps[k] = accum[k]  # list of (H, W)

        else:
            # Fixed-size patches: process in batches
            if is_list_input:
                patches_tensor = torch.from_numpy(
                    np.stack(patches_array, axis=0)
                ).float() / 255.0
            else:
                patches_tensor = torch.from_numpy(patches_array).float() / 255.0

            output_size = (patches_tensor.shape[2], patches_tensor.shape[3])
            n_batches = int(np.ceil(n_patches / batch_size))
            first_clusterizer = clusterizers[0]
            first_clusterizer.embedder.model.eval()

            batch_iter = range(n_batches)
            if progress_bar:
                batch_iter = tqdm(
                    batch_iter, total=n_batches, desc="Batches (shared embedder)"
                )

            for i in batch_iter:
                batch = patches_tensor[
                    i * batch_size : (i + 1) * batch_size
                ].to(first_clusterizer.device)

                with torch.inference_mode():
                    token_embeds = first_clusterizer.embedder.get_token_embeddings(
                        batch, remove_cls=True
                    )  # (B, N_tokens, D)

                for k, clusterizer in enumerate(clusterizers):
                    if progress_bar and i == 0:
                        tqdm.write(
                            f"  Clusterizer {k + 1}/{n_clusterizers}: "
                            f"{getattr(clusterizer, 'feature_name', str(clusterizer))}"
                        )
                    cluster_map = clusterizer._cluster_tokens(token_embeds)  # (B, g, g)
                    rasterized = clusterizer._rasterize(cluster_map, output_size)  # (B, H, W)
                    accum[k].append(rasterized)

            for k in range(n_clusterizers):
                per_clusterizer_maps[k] = np.concatenate(accum[k], axis=0)  # (N, H, W)

    else:
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
                    cluster_maps = []
                    patch_iter = patches_array
                    if progress_bar:
                        patch_iter = tqdm(
                            patch_iter, total=n_patches, desc="  Patches", leave=False
                        )
                    for patch in patch_iter:
                        m = clusterizer(
                            patch[None],
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

            per_clusterizer_maps[clust_idx] = cluster_maps

    if is_list_input and isinstance(per_clusterizer_maps[0], list):
        # Variable spatial sizes: return list of (K, H, W) uint8
        saliency_list = []
        for i in range(n_patches):
            patch_maps = np.stack(
                [per_clusterizer_maps[k][i] for k in range(n_clusterizers)],
                axis=0,
            ).astype(np.uint8)  # (K, H, W)
            saliency_list.append(patch_maps)
        return saliency_list
    else:
        stacked = np.stack(
            [np.asarray(m) for m in per_clusterizer_maps], axis=0
        )  # (K, N, H, W)
        saliency_array = stacked.transpose(1, 0, 2, 3).astype(np.uint8)  # (N, K, H, W)

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