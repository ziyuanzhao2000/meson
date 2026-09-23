import warnings
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from mesoslide.tools.segmenters import TokenClusterizer
    from mesoslide._patch_data import PatchData


CLUSTER_IMG_SUFFIX = "_cluster_img"


def cluster_img_key(clusterizer: "TokenClusterizer") -> str:
    """The `patches.obsm` key a clusterizer's rasterized cluster map is cached under."""
    return f"{clusterizer.feature_name}{CLUSTER_IMG_SUFFIX}"


def extract_cluster_maps(
    patches: "PatchData",
    clusterizer: "TokenClusterizer",
    model=None,
    *,
    slides=None,
    input_key: Optional[str] = None,
    batch_size: int = 16,
    progress_bar: bool = True,
    cache: bool = True,
    overwrite: bool = False,
    token: Optional[str] = None,
    model_path=None,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Generate a token-cluster map, rasterized to pixel resolution, for a set
    of pre-selected patches, for one `TokenClusterizer`.

    If the cluster map is already cached in
    `patches.obsm[cluster_img_key(clusterizer)]` (e.g. from a previous call
    with `cache=True`), it's returned directly -- no pixel read, no model
    call -- unless `overwrite=True`. Otherwise, this embeds `patches` with a vision model (dense,
    per-token, via `run_model_stages`) and hands the result to
    `clusterizer.transform`, which clusters and rasterizes it up to each
    patch's native pixel size. Renamed from the earlier
    `extract_saliency_maps`: "saliency map" implies a gradient/attribution-
    based importance map (e.g. GradCAM), which this never was -- it's a
    KMeans cluster-ID raster.

    `TokenClusterizer` doesn't hold onto its vision model (see its class
    docstring) -- only `model_name`. When `model` is omitted here (the
    common case), it's resolved from `clusterizer.model_name` via
    `lazyslide_models.MODEL_REGISTRY`, the same way `TokenClusterizer.fit()`
    defaults its own `model`. Pass `model` explicitly instead when calling
    this once per clusterizer for several clusterizers that share one model
    (e.g. to build a multi-row gallery, see
    `mesoslide.plotting.plot_patch_gallery_with_saliency`) -- letting each
    call auto-resolve its own model instance still only runs the embedding
    *computation* once (`run_model_stages` caches that in `patches.obsm`
    under a key derived from the resolved model name, regardless of which
    Python model object triggered it), but it does re-instantiate/reload
    the model object itself on every call; passing one already-resolved
    `model` in avoids that.

    Parameters
    ----------
    patches : PatchData
        Selected tiles, e.g. from :func:`mesoslide.select_top_patches`.
    clusterizer : TokenClusterizer
        Must have a non-empty `feature_name` -- used as this clusterizer's
        own `patches.obsm` cache key.
    model : str or lazyslide_models.ImageModel, optional
        The vision model to embed `patches` with -- must match (or be
        compatible with) the model `clusterizer` was constructed/fit with.
        Forwarded to `ImageModelStage`. Defaults to re-resolving
        `clusterizer.model_name` from `lazyslide_models.MODEL_REGISTRY`.
    slides : WSIData, list of WSIData, or {slide_id: WSIData}, optional
        Forwarded to :func:`extract_patch_images`/`run_model_stages`.
        Not required in the common case -- falls back to
        `patches.obs['_slide_ref']`, populated automatically by
        :func:`mesoslide.select_top_patches` and friends. Pass this
        explicitly to read from slides you already have open without
        relying on that column (or when the cluster map is already cached,
        it's unused either way).
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
    overwrite : bool, default=False
        Recompute even if a cluster map is already cached in
        `patches.obsm[cluster_img_key(clusterizer)]`, replacing the cached
        value (when `cache=True`).
    token, model_path
        Forwarded to model resolution when `model` is omitted.

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
    >>> cluster_map = extract_cluster_maps(patches, clusterizer)
    >>> print(cluster_map.shape)   # (N, H, W)  dtype=uint8
    >>>
    >>> # Cache into the patches table, then reuse later, still no slides needed:
    >>> extract_cluster_maps(patches, clusterizer, cache=True)
    >>> cluster_map = extract_cluster_maps(patches, clusterizer)

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
    if key in patches.obsm and not overwrite:
        return patches.obsm[key]

    from mesoslide.preprocessing._extract_patches import extract_patch_images
    from mesoslide.tools._feature_extraction import run_model_stages
    from mesoslide.tools._model_stage import ImageModelStage, _resolve_model

    pixels = extract_patch_images(
        patches, slides, channel_first=True, progress_bar=progress_bar, cache=True,
    )
    is_list_pixels = isinstance(pixels, list)

    if model is None:
        model, _ = _resolve_model(clusterizer.model_name, model_path=model_path, token=token)

    # Compose the vision FM embedder with the clusterizer's own `transform`:
    # `run_model_stages` handles the (cacheable) embedding step, and
    # `clusterizer.transform` -- its single public entry point for going
    # from token embeddings to a rasterized cluster map -- handles the rest.
    # `transform` isn't threaded through `as_stage`/`run_model_stages` here
    # because its output_size varies per patch in the list case below, and
    # in the non-list case doing so would duplicate the rasterized array
    # under both the stage's own cache key and `cluster_img_key` below.
    fm_stage = ImageModelStage(model, dense=True, device=clusterizer.device)
    table = run_model_stages(
        patches, [fm_stage], slides=slides,
        input_key=input_key, batch_size=batch_size,
        progress_bar=progress_bar, save=False,
    )
    dense_tokens = table.obsm[fm_stage.name]  # (N, N_tokens, D)

    ##  This is for handling mixed patch size
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
