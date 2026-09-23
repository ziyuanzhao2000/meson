import warnings
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

from mesoslide._deprecated import deprecated_kwargs, rename

if TYPE_CHECKING:
    from mesoslide.tools.segmenters import TokenClusterer
    from mesoslide._patch_data import PatchData


CLUSTER_IMG_SUFFIX = "_cluster_img"


def cluster_img_key(clusterer: "TokenClusterer") -> str:
    """The `patches.obsm` key a clusterer's rasterized cluster map is cached under."""
    return f"{clusterer.display_name}{CLUSTER_IMG_SUFFIX}"


@deprecated_kwargs(clusterizer=rename("clusterer"))
def extract_cluster_maps(
    patches: "PatchData",
    clusterer: "TokenClusterer",
    model=None,
    *,
    slides=None,
    input_key: Optional[str] = None,
    batch_size: int = 128,
    device: Optional[str] = None,
    progress_bar: bool = True,
    cache: bool = True,
    overwrite: bool = False,
    token: Optional[str] = None,
    model_path=None,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Generate a token-cluster map, rasterized to pixel resolution, for a set
    of pre-selected patches, for one `TokenClusterer`.

    If the cluster map is already cached in
    `patches.obsm[cluster_img_key(clusterer)]` (e.g. from a previous call
    with `cache=True`), it's returned directly -- no pixel read, no model
    call -- unless `overwrite=True`. Otherwise, this embeds `patches` with a
    vision model (dense, per-token, via `run_model_stages`) and hands the
    result to `clusterer.transform`, which clusters and rasterizes it up to
    each patch's native pixel size.

    When `model` is omitted, it's resolved from `clusterer.model_name_` via
    `lazyslide_models.MODEL_REGISTRY`. Pass `model` explicitly when calling
    this for several clusterers that share one model (e.g. a multi-row
    gallery): the embedding itself is computed once either way
    (`run_model_stages` caches it in `patches.obsm` under a key derived from
    the model name), but passing one resolved instance avoids reloading the
    model on every call.

    Parameters
    ----------
    patches : PatchData
        Selected tiles, e.g. from :func:`mesoslide.select_top_patches`.
    clusterer : TokenClusterer
        Fitted clusterer. Its `display_name` (`name`, else `feature_name_`)
        must be non-empty -- it's the `patches.obsm` cache key.
    model : str or lazyslide_models.ImageModel, optional
        The vision model to embed `patches` with -- must match the model
        `clusterer` was fit with. Defaults to `clusterer.model_name_`.
    slides : WSIData, list of WSIData, or {slide_id: WSIData}, optional
        Forwarded to :func:`extract_patch_images`/`run_model_stages`.
        Not required in the common case -- falls back to
        `patches.obs['_slide_ref']`, populated automatically by
        :func:`mesoslide.select_top_patches` and friends.
    input_key : str, optional
        Resume the embedding step from an existing cached dense array
        instead of reading pixels through the vision model -- forwarded to
        `run_model_stages`.
    batch_size : int, default=128
        Batch size for the embedding step.
    device : str, optional
        Torch device for the vision model. Defaults to "cuda" if available.
    progress_bar : bool, default=True
    cache : bool, default=True
        Store the freshly computed rasterized map in
        `patches.obsm[cluster_img_key(clusterer)]`, so a later call with
        the same clusterer skips re-computing it. Skipped (with a warning)
        if patches have inconsistent pixel shapes.
    overwrite : bool, default=False
        Recompute even if a cluster map is already cached, replacing the
        cached value (when `cache=True`).
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
        If `clusterer.display_name` is empty, or no model can be resolved.

    Examples
    --------
    >>> from mesoslide.preprocessing import extract_cluster_maps
    >>> cluster_map = extract_cluster_maps(patches, clusterer)
    >>> print(cluster_map.shape)   # (N, H, W)  dtype=uint8
    """
    if not clusterer.display_name:
        raise ValueError(
            "clusterer must have a non-empty name (or feature_name_) -- it's used as "
            "this clusterer's patches.obsm cache key."
        )

    key = cluster_img_key(clusterer)
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
        model_name = getattr(clusterer, "model_name_", None)
        if model_name is None:
            raise ValueError("model is required when clusterer has no model_name_.")
        model, _ = _resolve_model(model_name, model_path=model_path, token=token)

    # `run_model_stages` handles the (cacheable) embedding step and
    # `clusterer.transform` the rest. `transform` isn't run as a stage here
    # because output_size varies per patch in the list case, and in the
    # array case the map would be cached twice (stage key and `key`).
    fm_stage = ImageModelStage(model, dense=True, device=device)
    table = run_model_stages(
        patches, [fm_stage], slides=slides,
        input_key=input_key, batch_size=batch_size,
        progress_bar=progress_bar, save=False,
    )
    dense_tokens = table.obsm[fm_stage.name]  # (N, N_tokens, D)

    # Mixed patch sizes
    if is_list_pixels:
        rasterized = [
            clusterer.transform(dense_tokens[i:i + 1], p.shape[-2:])[0]
            for i, p in enumerate(pixels)
        ]
    else:
        rasterized = clusterer.transform(dense_tokens, tuple(pixels.shape[-2:]))

    if cache:
        if is_list_pixels:
            warnings.warn(
                f"cache=True has no effect for clusterer "
                f"'{clusterer.display_name}': patches have inconsistent "
                f"pixel shapes and cannot be aligned 1:1 with patches.obs.",
                UserWarning,
                stacklevel=2,
            )
        else:
            patches.obsm[key] = rasterized

    return rasterized
