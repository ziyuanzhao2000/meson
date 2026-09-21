from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import cv2
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

from mesoslide.tools._model_stage import (
    CallableStage,
    ImageModelStage,
    _canonical_registry_name,
    _require_dense_capable,
    _resolve_model,
)


class TokenClusterizer(TransformerMixin, BaseEstimator):
    """
    Token-level clustering and rasterization for vision transformer embeddings.

    Takes per-token embeddings from a `lazyslide_models` vision foundation
    model, applies KMeans clustering, and upsamples cluster assignments to
    patch image resolution. Its public surface is a single sklearn-style
    `transform()` method -- token embeddings in, rasterized cluster maps out
    -- mirroring how `mesoslide.tools.sparse_coding.SparseAutoencoder` wraps
    a fitted torch model as a plain `transform()`. Embedding patches into
    tokens is delegated to `mesoslide.tools._feature_extraction
    .run_model_stages`, the same engine `feature_extraction` and
    `extract_cluster_maps` use; `extract_cluster_maps` composes
    `transform()` with a vision FM embedding stage (`as_stage()` wraps
    `transform()` as a `ModelStage` for that purpose) rather than
    duplicating clustering/rasterization logic of its own.

    A `TokenClusterizer` does not hold onto the vision model itself -- only
    `grid_size`/`patch_size`/`model_name`, read from it once at construction
    time. Everywhere `transform()`/`as_stage()` are used, only
    already-computed token embeddings are needed, so keeping the (often
    large) model out of `self` keeps a pickled clusterizer small, e.g. for
    saving many per-feature clusterizers that all happen to share the same
    model. `model_name` is kept (a plain string, unlike the model itself) so
    that `fit()` and `extract_cluster_maps` can re-resolve the same model
    from `lazyslide_models.MODEL_REGISTRY` on their own when not given one
    explicitly -- see their docstrings.

    Parameters
    ----------
    model : str or lazyslide_models.ImageModel
        A key into `lazyslide_models.MODEL_REGISTRY` (e.g. "uni2"), an
        arbitrary timm model name, or an already-instantiated
        `lazyslide_models` `ImageModel`. Must be ViT-style (exposes
        `grid_size`, `patch_size`, `encode_image_dense`). Only used here to
        read `grid_size`/`patch_size`/`model_name` -- the model itself is
        not stored.
    kmeans : sklearn.cluster.KMeans or compatible
        Clustering model; fit lazily by `fit()` if not already fitted.
    interpolation : str, default='nearest'
        Interpolation method for upsampling ('nearest' or 'bilinear')
    cluster_order : np.ndarray, optional
        Custom ordering for cluster IDs. If provided, remaps cluster labels
        according to this order before rasterization.
    feature_name : str, optional
        Used as this clusterizer's default cache key when run as a stage
        (see `extract_cluster_maps`) and for row labels in plotting.
    device : str, optional
        Torch device `fit()` runs its vision model on. Defaults to "cuda" if
        available, else "cpu".
    token, model_path
        Forwarded to model resolution (see `mesoslide.tools._feature_extraction
        .feature_extraction`).

    Examples
    --------
    >>> from mesoslide.tools.segmenters import TokenClusterizer
    >>> from sklearn.cluster import KMeans
    >>>
    >>> clusterizer = TokenClusterizer(
    ...     model="uni2", kmeans=KMeans(n_clusters=3), feature_name="my_feature",
    ... )
    >>>
    >>> # Turn a set of selected patches into rasterized cluster maps
    >>> from mesoslide.preprocessing import extract_cluster_maps
    >>> cluster_masks = extract_cluster_maps(patches, clusterizer, model="uni2")
    >>> # Returns: (N, H, W) uint8 array with cluster IDs
    """

    def __init__(
        self,
        model,
        kmeans = None,
        *,
        interpolation: str = 'nearest',
        cluster_order: Optional[np.ndarray] = None,
        feature_name: str = '',
        device: Optional[str] = None,
        token: Optional[str] = None,
        model_path: "str | Path | None" = None,
    ):
        resolved_model, model_name = _resolve_model(model, model_path=model_path, token=token)
        _require_dense_capable(resolved_model, model_name)
        self.grid_size = resolved_model.grid_size
        self.patch_size = resolved_model.patch_size
        self.model_name = _canonical_registry_name(model_name)
        self.kmeans = kmeans
        self.interpolation = interpolation
        self.cluster_order = cluster_order
        self.feature_name = feature_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if interpolation not in ['nearest', 'bilinear']:
            raise ValueError(f"interpolation must be 'nearest' or 'bilinear', got {interpolation}")

        self.cv2_interp = (
            cv2.INTER_NEAREST_EXACT if interpolation == 'nearest'
            else cv2.INTER_LINEAR
        )

    def _cluster_tokens(self, token_embeddings) -> np.ndarray:
        """
        Apply KMeans clustering to token embeddings.

        Parameters
        ----------
        token_embeddings : torch.Tensor or np.ndarray
            Shape (B, N_tokens, embed_dim)

        Returns
        -------
        cluster_maps : np.ndarray
            Shape (B, grid_h, grid_w) with cluster IDs
        """
        if torch.is_tensor(token_embeddings):
            token_embeddings = token_embeddings.detach().cpu().numpy()
        token_embeddings = token_embeddings.astype(np.float64)

        B, N, D = token_embeddings.shape
        gh, gw = self.grid_size
        assert N == gh * gw, f"Expected {gh * gw} tokens ({gh}x{gw} grid), got {N}"

        cluster_maps = []
        for i in range(B):
            labels = self.kmeans.predict(token_embeddings[i])  # (N,)
            cluster_maps.append(labels.reshape(gh, gw))

        return np.array(cluster_maps, dtype=np.uint8)

    def _rasterize(self, cluster_maps: np.ndarray, output_size: tuple) -> np.ndarray:
        """
        Upsample cluster maps to target resolution.

        Parameters
        ----------
        cluster_maps : np.ndarray
            Shape (B, grid_h, grid_w)
        output_size : tuple
            Target (height, width)

        Returns
        -------
        rasterized : np.ndarray
            Shape (B, H, W) with upsampled cluster IDs
        """
        B = cluster_maps.shape[0]
        H, W = output_size

        rasterized = np.zeros((B, H, W), dtype=np.uint8)

        for i in range(B):
            cluster_map = cluster_maps[i]

            if self.cluster_order is not None:
                cluster_map = self.cluster_order[cluster_map]

            upsampled = cv2.resize(
                cluster_map,
                (W, H),
                interpolation=self.cv2_interp
            )
            rasterized[i] = upsampled

        return rasterized

    def transform(self, token_embeddings, output_size: Optional[tuple] = None) -> np.ndarray:
        """
        Cluster token embeddings and rasterize to pixel resolution.

        The only public entry point for turning per-token embeddings into
        cluster maps -- combines KMeans assignment and upsampling so callers
        (`as_stage()`, `extract_cluster_maps`) don't hand-roll that pairing
        themselves.

        Parameters
        ----------
        token_embeddings : torch.Tensor or np.ndarray
            Shape (B, N_tokens, embed_dim)
        output_size : tuple, optional
            Target (height, width). Defaults to the model's native grid
            resolution in pixels (grid_size * patch_size).

        Returns
        -------
        cluster_masks : np.ndarray
            Shape (B, H, W), dtype uint8. Each value is a cluster ID.
        """
        cluster_maps = self._cluster_tokens(token_embeddings)
        if output_size is None:
            gh, gw = self.grid_size
            ph, pw = self.patch_size
            output_size = (gh * ph, gw * pw)
        return self._rasterize(cluster_maps, output_size)

    def as_stage(self, *, output_size: Optional[tuple] = None, name: Optional[str] = None,
                 cache: bool = True, overwrite: bool = False) -> CallableStage:
        """Wrap this clusterizer's `transform` as a `ModelStage`.

        Used by `extract_cluster_maps` to feed into `run_model_stages`'s
        chain, downstream of an `ImageModelStage(dense=True)`. `output_size`
        is fixed for the whole stage since a single `run_model_stages` batch
        shares one target resolution.
        """
        stage_name = name or self.feature_name or f"cluster_{id(self)}"
        return CallableStage(
            lambda token_embeddings: self.transform(token_embeddings, output_size),
            name=stage_name, input_kind="dense",
            output_kind="dense", cache=cache, overwrite=overwrite, device="cpu",
        )

    def fit(
        self,
        slides,
        feature_name: str,
        model=None,
        n_positive: int = 100,
        n_negative: int = 100,
        batch_size: int = 128, # changed from 16 for better perf on cpu
        top_fraction: float = 0.10,
        show_progress: bool = True,
        take_every: Union[int, None] = None,
        tile_key: str = 'tiles',
        image_slides=None,
        token: Optional[str] = None,
        model_path: "str | Path | None" = None,
    ) -> "TokenClusterizer":
        """
        Compute cluster order based on differential abundance between positive and negative patches.

        This method:
        1. Selects positive (high-scoring) and negative (zero-score) patches for a feature
        2. Extracts image patches and computes token embeddings
        3. Predicts cluster labels for all tokens
        4. Computes differential cluster frequencies (positive - negative)
        5. Ranks clusters by differential abundance
        6. Updates self.cluster_order and returns self

        This is useful for identifying which tissue structures (clusters) are
        most enriched in patches where a specific SAE feature is active.

        Parameters
        ----------
        slides : slides_table, AnnData, WSIData, or sequence/mapping of either
            Where to select patches from. See :func:`mesoslide.select_top_patches`.
        feature_name : str
            Feature name to use for patch selection (e.g., 'UNI_SAE_12345')
        model : str or lazyslide_models.ImageModel, optional
            The vision model to embed patches with -- same accepted forms as
            `__init__`'s `model`. Must resolve to the same `grid_size` this
            clusterizer was constructed with. Defaults to re-resolving
            `self.model_name` (set by `__init__`) from
            `lazyslide_models.MODEL_REGISTRY`; pass this explicitly to reuse
            an already-loaded instance instead of resolving one again, or to
            fit against a different (but grid-compatible) model.
        n_positive : int, default=100
            Number of positive patches to sample
        n_negative : int, default=100
            Number of negative patches to sample
        batch_size : int, default=16
            Batch size for processing
        show_progress : bool, default=True
            Whether to show progress bars
        tile_key : str, default='tiles'
        image_slides : WSIData / list / {slide_id: WSIData}, optional
            Slides to read pixels from, with image data attached. Defaults to
            `slides` when that is already a mapping of open slides; otherwise
            required, since a slides_table alone carries no pixels.
        token, model_path
            Forwarded to model resolution.

        Returns
        -------
        self, with `cluster_order` updated.

        Examples
        --------
        >>> # Compute cluster order for a specific SAE feature
        >>> slides = mesoslide.open_slides(manifest)
        >>> clusterizer.fit(
        ...     slides,
        ...     feature_name='UNI_SAE_12345',
        ...     n_positive=100,
        ...     n_negative=100,
        ... )  # model defaults to re-resolving clusterizer.model_name ('uni2' here)
        >>> # Now the clusterizer will use this ordering when rasterizing
        >>> from mesoslide.preprocessing import extract_cluster_maps
        >>> masks = extract_cluster_maps(patches, clusterizer)
        """
        from mesoslide._patch_selector import select_top_patches, select_negative_patches
        from mesoslide.tools._feature_extraction import run_model_stages

        resolved_model, model_name = _resolve_model(
            model if model is not None else self.model_name,
            model_path=model_path, token=token,
        )
        _require_dense_capable(resolved_model, model_name)
        if tuple(resolved_model.grid_size) != tuple(self.grid_size):
            raise ValueError(
                f"fit()'s model has grid_size={tuple(resolved_model.grid_size)}, "
                f"but this clusterizer was constructed with grid_size="
                f"{tuple(self.grid_size)} -- pass the same model (or an "
                f"equivalent one) used to construct it."
            )

        if show_progress:
            print(f"Selecting patches for feature '{feature_name}'...")

        # Positive patches: evenly sampled from high-scoring patches
        positive_patches_anndata = select_top_patches(
            slides,
            feature_name,
            n=n_positive,
            tile_key=tile_key,
            min_score=0,          # Only positive scores
            top_fraction=top_fraction,  # Only top fraction of patches
            take_every=take_every,
        )
        print(top_fraction)

        # Negative patches: evenly sampled from zero-score patches
        negative_patches_anndata = select_negative_patches(
            slides,
            feature_name,
            n=n_negative,
            tile_key=tile_key,
            take_every=None,      # Auto-compute stride
        )

        if show_progress:
            print(f"Extracting {len(positive_patches_anndata)} positive and "
                  f"{len(negative_patches_anndata)} negative patches...")

        dense_key = "_fit_dense"
        fm_stage = ImageModelStage(resolved_model, dense=True, name=dense_key, device=self.device)

        run_model_stages(
            positive_patches_anndata, [fm_stage], slides=image_slides,
            tile_key=tile_key, batch_size=batch_size,
            progress_bar=show_progress, save=False,
        )
        run_model_stages(
            negative_patches_anndata, [fm_stage], slides=image_slides,
            tile_key=tile_key, batch_size=batch_size,
            progress_bar=show_progress, save=False,
        )

        pos_dense = positive_patches_anndata.obsm[dense_key]  # (n_pos, N_tokens, D)
        neg_dense = negative_patches_anndata.obsm[dense_key]  # (n_neg, N_tokens, D)
        positive_tokens = pos_dense.reshape(-1, pos_dense.shape[-1]).astype(np.float64)
        negative_tokens = neg_dense.reshape(-1, neg_dense.shape[-1]).astype(np.float64)

        if show_progress:
            print("Predicting cluster labels...")

        # Fit KMeans if not already fitted
        if not hasattr(self.kmeans, 'cluster_centers_'):
            if show_progress:
                print("Fitting KMeans on all tokens...")
            self.kmeans = KMeans(n_clusters=3, random_state=0).fit(positive_tokens)

        # Predict cluster labels
        positive_labels = self.kmeans.predict(positive_tokens)
        negative_labels = self.kmeans.predict(negative_tokens)

        # Get number of clusters
        n_clusters = len(np.unique(np.concatenate([positive_labels, negative_labels])))
        if hasattr(self.kmeans, 'n_clusters'):
            n_clusters = self.kmeans.n_clusters

        # Compute normalized frequencies
        positive_counts = np.bincount(positive_labels, minlength=n_clusters)
        negative_counts = np.bincount(negative_labels, minlength=n_clusters)

        percentage_positive = positive_counts / (positive_counts.sum() + 1e-12)
        percentage_negative = negative_counts / (negative_counts.sum() + 1e-12)

        # Compute differential abundance
        diff_percentage = percentage_positive - percentage_negative

        # Rank clusters by differential abundance (high to low)
        cluster_order = np.argsort(np.argsort(-diff_percentage))

        # Store and return
        self.cluster_order = cluster_order

        if show_progress:
            print(f"Cluster order computed and stored. Top 3 enriched clusters: {cluster_order[:3]}")
            print(f"Differential abundances: {diff_percentage[cluster_order[:3]]}")

        return self
