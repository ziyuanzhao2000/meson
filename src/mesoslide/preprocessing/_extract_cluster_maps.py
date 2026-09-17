import warnings
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from mesoslide.tools.segmenters import TokenClusterizer


CLUSTER_IMG_SUFFIX = "_cluster_img"


def cluster_img_key(clusterizer: "TokenClusterizer") -> str:
    """The `patches.obsm` key a clusterizer's rasterized cluster map is cached under."""
    return f"{clusterizer.feature_name}{CLUSTER_IMG_SUFFIX}"


def extract_cluster_maps(
    patches,
    slides,
    clusterizer: "TokenClusterizer",
    *,
    input_key: Optional[str] = None,
    batch_size: int = 16,
    progress_bar: bool = True,
    cache: bool = False,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Generate a token-cluster map, rasterized to pixel resolution, for a set
    of pre-selected patches, for one `TokenClusterizer`.

    If the cluster map is already cached in
    `patches.obsm[cluster_img_key(clusterizer)]` (e.g. from a previous call
    with `cache=True`), it's returned directly -- no pixel read, no model
    call. Otherwise, this embeds `patches` with the clusterizer's vision
    model (dense, per-token, via `run_model_stages`) and hands the result to
    `clusterizer.transform`, which clusters and rasterizes it up to each
    patch's native pixel size. Renamed from the earlier
    `extract_saliency_maps`: "saliency map" implies a gradient/attribution-
    based importance map (e.g. GradCAM), which this never was -- it's a
    KMeans cluster-ID raster.

    Calling this once per clusterizer (e.g. to build a multi-row gallery,
    see `mesoslide.plotting.plot_patch_gallery_with_saliency`) still only
    embeds a shared underlying vision model once: `run_model_stages` caches
    the embedding stage's output under a key derived from the resolved model
    name, so the second and later clusterizers' calls find that key already
    present in `patches.obsm` and skip straight to their own (cheap)
    `transform` step -- no special handling needed here for that to hold.

    Parameters
    ----------
    patches : AnnData
        Selected tiles, e.g. from :func:`mesoslide.select_top_patches`.
    slides : WSIData, list of WSIData, or {slide_id: WSIData}, optional
        Required to read pixel data, unless the cluster map is already
        cached in `patches.obsm` (see `cache`).
    clusterizer : TokenClusterizer
        Must have a non-empty `feature_name` -- used as this clusterizer's
        own `patches.obsm` cache key.
    input_key : str, optional
        Resume the embedding step from an existing cached dense array
        instead of reading pixels through the vision model -- forwarded to
        `run_model_stages`.
    batch_size : int, default=16
        Batch size for the embedding step.
    progress_bar : bool, default=True
    cache : bool, default=False
        Store the freshly computed rasterized map in
        `patches.obsm[cluster_img_key(clusterizer)]`, so a later call with
        the same clusterizer skips re-computing it. Skipped (with a
        warning) if the result can't be stacked into a single per-patch
        array (inconsistent pixel shapes across patches).

    Returns
    -------
    cluster_map : np.ndarray or list of np.ndarray
        Rasterized cluster-ID map as uint8.
        - If all patches share the same pixel size: `(N, H, W)`.
        - Otherwise: list of length N, each element `(H, W)`.

    Raises
    ------
    ValueError
        If `clusterizer.feature_name` is empty.

    Examples
    --------
    >>> from mesoslide.preprocessing import extract_cluster_maps
    >>> cluster_map = extract_cluster_maps(patches, slides, clusterizer)
    >>> print(cluster_map.shape)   # (N, H, W)  dtype=uint8
    >>>
    >>> # Cache into the patches table, then reuse without slides:
    >>> extract_cluster_maps(patches, slides, clusterizer, cache=True)
    >>> cluster_map = extract_cluster_maps(patches, None, clusterizer)

    Notes
    -----
    Alpha mapping, power transforms, and colourmap selection are purely
    visualization concerns and belong in the plotting layer, not here.
    """
    if not clusterizer.feature_name:
        raise ValueError(
            "clusterizer must have a non-empty feature_name -- it's used as "
            "this clusterizer's patches.obsm cache key."
        )

    key = cluster_img_key(clusterizer)
    if key in patches.obsm:
        return patches.obsm[key]

    from mesoslide.preprocessing._extract_patches import extract_patch_images
    from mesoslide.tools._feature_extraction import run_model_stages
    from mesoslide.tools._model_stage import ImageModelStage

    pixels = extract_patch_images(
        patches, slides, channel_first=True, progress_bar=progress_bar, cache=True,
    )
    is_list_pixels = isinstance(pixels, list)

    # Compose the vision FM embedder with the clusterizer's own `transform`:
    # `run_model_stages` handles the (cacheable) embedding step, and
    # `clusterizer.transform` -- its single public entry point for going
    # from token embeddings to a rasterized cluster map -- handles the rest.
    # `transform` isn't threaded through `as_stage`/`run_model_stages` here
    # because its output_size varies per patch in the list case below, and
    # in the non-list case doing so would duplicate the rasterized array
    # under both the stage's own cache key and `cluster_img_key` below.
    fm_stage = ImageModelStage(clusterizer.model, dense=True, device=clusterizer.device)
    table = run_model_stages(
        patches, [fm_stage], slides=slides,
        input_key=input_key, batch_size=batch_size,
        progress_bar=progress_bar, save=False,
    )
    dense_tokens = table.obsm[fm_stage.name]  # (N, N_tokens, D)

    if is_list_pixels:
        rasterized = [
            clusterizer.transform(dense_tokens[i:i + 1], p.shape[-2:])[0]
            for i, p in enumerate(pixels)
        ]
    else:
        rasterized = clusterizer.transform(dense_tokens, tuple(pixels.shape[-2:]))

    if cache:
        if is_list_pixels:
            warnings.warn(
                f"cache=True has no effect for clusterizer "
                f"'{clusterizer.feature_name}': patches have inconsistent "
                f"pixel shapes and cannot be aligned 1:1 with patches.obs.",
                UserWarning,
                stacklevel=2,
            )
        else:
            patches.obsm[key] = rasterized

    return rasterized
