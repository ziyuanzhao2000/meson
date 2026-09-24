import zlib
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils.validation import check_is_fitted

from mesoslide.tools._model_stage import (
    CallableStage,
    ImageModelStage,
    _canonical_registry_name,
    _require_dense_capable,
    _resolve_model,
)

_ORDERINGS = ("correlation", "diff_abundance", "ablation")
_INTERPOLATIONS = {"nearest": cv2.INTER_NEAREST_EXACT, "bilinear": cv2.INTER_LINEAR}


def _as_tokens(X) -> np.ndarray:
    """Token embeddings as a float64 (B, N_tokens, D) array."""
    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(f"Expected token embeddings of shape (B, N_tokens, D), got {X.shape}")
    return X


class TokenClusterer(TransformerMixin, BaseEstimator):
    """
    KMeans clustering of ViT token embeddings, ordered by association with a
    feature score, rasterized to per-pixel cluster maps.

    `fit` learns `n_clusters` KMeans centroids over individual tokens, then
    remaps the raw cluster ids into a canonical order (`cluster_order_`) so
    that the highest id is the cluster most associated with the target
    feature score and 0 the least. `predict` returns ordered labels on the
    token grid; `transform` upsamples them to pixel resolution.

    `fit`/`fit_order` operate on already-computed token embeddings. To
    sample patches from slides and embed them with a vision model, use
    :func:`fit_token_clusterer`.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of KMeans clusters (at most 256; maps are uint8).
    ordering : {'correlation', 'diff_abundance', 'ablation'}, default='correlation'
        How clusters are ranked against the feature score:

        - 'correlation': Spearman correlation, across patches, between each
          cluster's per-patch token count and the patch's score (`fit_order`).
        - 'diff_abundance': pooled token frequency in positive (`y > 0`)
          minus negative (`y <= 0`) patches (`fit_order`).
        - 'ablation': how much the score drops when the cluster's tokens are
          removed from the vision model's input, beyond removing the same
          number of random tokens (`fit_order_ablation`; needs pixels, a
          vision model and a scorer, see :func:`fit_token_clusterer`).
    interpolation : {'nearest', 'bilinear'}, default='nearest'
        Upsampling method used by `transform`.
    name : str, optional
        Label for this clusterer: its `patches.obsm` cache key in
        :func:`mesoslide.preprocessing.extract_cluster_maps` and its row
        label in gallery plots. Never changed by `fit`. Defaults to
        `feature_name_` when unset (see `display_name`).
    random_state : int, default=0
        Seed for KMeans.
    n_init : int or 'auto', default=10
        Number of KMeans initializations; the run with the lowest inertia is
        kept. More than one makes the result robust to small changes in the
        input tokens, which can otherwise land a single run in a worse local
        optimum.

    Attributes
    ----------
    kmeans_ : sklearn.cluster.KMeans
        The fitted KMeans. Persist the whole clusterer with joblib.
    cluster_centers_ : ndarray of shape (n_clusters, D)
    cluster_order_ : ndarray of shape (n_clusters,)
        Maps raw KMeans id -> canonical id.
    n_clusters_ : int
    n_features_in_ : int
        Token embedding dimension D.
    grid_size_ : tuple of int
        Token grid (gh, gw). Taken from the vision model by
        `fit_token_clusterer`; inferred as square by `fit` otherwise.
    patch_size_ : tuple of int or None
        ViT patch size in pixels; sets `transform`'s default output size.
    model_name_ : str or None
        `lazyslide_models.MODEL_REGISTRY` key of the vision model, used to
        re-resolve it in `extract_cluster_maps`.
    feature_name_ : str or None
        Feature the cluster order was fit against, set by `fit_token_clusterer`.
    order_scores_ : ndarray of shape (n_clusters,)
        Per-cluster Spearman rho, differential abundance or mean excess drop
        (ablation), in canonical order.
    ablation_summary_ : pd.DataFrame
        Only under ordering='ablation': :func:`mesoslide.tools.summarize_token_ablation`
        output indexed by canonical cluster id.
    ablation_results_ : pd.DataFrame
        Only under ordering='ablation': per-patch :func:`mesoslide.tools.token_ablation`
        output, with a `cluster` column holding canonical cluster ids.
    ablation_seed_ : int
        Only under ordering='ablation': seed of the random-token controls.
    fit_diagnostics_ : ndarray of shape (n_patches, n_clusters + 1)
        Only under ordering='correlation'. Columns 0..n_clusters-1 are each
        patch's token count per canonical cluster; the last column is `y`.

    Notes
    -----
    `fit` reuses already-fitted centroids and only recomputes the cluster
    order. Use `sklearn.base.clone(clusterer)` to start from scratch.

    Examples
    --------
    >>> from mesoslide.tools.segmenters import TokenClusterer, fit_token_clusterer
    >>> clusterer = fit_token_clusterer(slides, "UNI_SAE_12345", model="uni")
    >>> joblib.dump(clusterer, "UNI_SAE_12345.pkl")
    >>> from mesoslide.preprocessing import extract_cluster_maps
    >>> masks = extract_cluster_maps(patches, clusterer)   # (N, H, W) uint8
    """

    def __init__(
        self,
        n_clusters: int = 3,
        *,
        ordering: str = "correlation",
        interpolation: str = "nearest",
        name: Optional[str] = None,
        random_state: Optional[int] = 0,
        n_init: Union[int, str] = 10,
    ):
        self.n_clusters = n_clusters
        self.ordering = ordering
        self.interpolation = interpolation
        self.name = name
        self.random_state = random_state
        self.n_init = n_init

    def __setstate__(self, state):
        # Clusterers pickled before `n_init` existed were fit with sklearn's default.
        state.setdefault("n_init", "auto")
        super().__setstate__(state)

    @property
    def display_name(self) -> str:
        """`name` if set, else `feature_name_`, else ''."""
        return self.name or getattr(self, "feature_name_", None) or ""

    def _validate_params(self):
        if self.ordering not in _ORDERINGS:
            raise ValueError(f"ordering must be one of {_ORDERINGS}, got {self.ordering!r}")
        if self.interpolation not in _INTERPOLATIONS:
            raise ValueError(
                f"interpolation must be one of {tuple(_INTERPOLATIONS)}, got {self.interpolation!r}"
            )
        if not 1 <= self.n_clusters <= 256:
            raise ValueError(f"n_clusters must be in [1, 256], got {self.n_clusters}")

    def _set_grid_size(self, n_tokens: int):
        """Keep a grid_size_ already set from the model; otherwise infer a square grid."""
        grid = getattr(self, "grid_size_", None)
        if grid is not None:
            if grid[0] * grid[1] != n_tokens:
                raise ValueError(f"Expected {grid[0] * grid[1]} tokens ({grid[0]}x{grid[1]} grid), got {n_tokens}")
            return
        side = int(round(np.sqrt(n_tokens)))
        if side * side != n_tokens:
            raise ValueError(
                f"Cannot infer a square token grid from {n_tokens} tokens; set grid_size_ "
                f"(or use fit_token_clusterer, which reads it from the model)."
            )
        self.grid_size_ = (side, side)

    def fit(self, X, y=None, *, X_order=None, y_order=None) -> "TokenClusterer":
        """
        Fit KMeans centroids on `X` (unless already fitted), then the cluster order.

        Parameters
        ----------
        X : array-like or torch.Tensor of shape (n_patches, n_tokens, D)
            Token embeddings to fit KMeans on. Ignored for centroid fitting
            if this clusterer is already fitted.
        y : array-like of shape (n_patches,), optional
            Per-patch feature scores for `X`, used for the cluster order when
            `X_order`/`y_order` are not given.
        X_order, y_order : optional
            A separate set of patches and scores to compute the cluster order
            on (e.g. patches spanning the full score range, while `X` holds
            only top-scoring patches). If neither `y` nor `y_order` is given,
            `cluster_order_` is the identity.

        Returns
        -------
        self
        """
        self._validate_params()
        X = _as_tokens(X)
        _, n_tokens, dim = X.shape
        self._set_grid_size(n_tokens)

        if not hasattr(self, "cluster_centers_"):
            self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state,
                                  n_init=self.n_init)
            self.kmeans_.fit(X.reshape(-1, dim))
            self.cluster_centers_ = self.kmeans_.cluster_centers_
            self.n_clusters_ = self.cluster_centers_.shape[0]
            self.n_features_in_ = dim

        if X_order is None and y_order is None:
            X_order, y_order = X, y
        elif X_order is None or y_order is None:
            raise ValueError("X_order and y_order must be given together.")

        if y_order is None:
            self.cluster_order_ = np.arange(self.n_clusters_)
            return self
        if self.ordering == "ablation":
            raise ValueError(
                "ordering='ablation' needs patch pixels and a vision model: call "
                "fit(X) then fit_order_ablation(...), or use fit_token_clusterer."
            )
        return self.fit_order(X_order, y_order)

    def fit_order(self, X, y) -> "TokenClusterer":
        """
        Recompute `cluster_order_` against feature scores `y`, keeping centroids.

        Parameters
        ----------
        X : array-like or torch.Tensor of shape (n_patches, n_tokens, D)
        y : array-like of shape (n_patches,)
            Per-patch feature scores (under ordering='diff_abundance', patches
            with `y > 0` are positive, the rest negative).

        Returns
        -------
        self
        """
        check_is_fitted(self, "cluster_centers_")
        self._validate_params()
        if self.ordering == "ablation":
            raise ValueError("ordering='ablation' is fit with fit_order_ablation, not fit_order.")
        X = _as_tokens(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        if len(y) != len(X):
            raise ValueError(f"X has {len(X)} patches but y has {len(y)} scores")

        k = self.n_clusters_
        raw = self._raw_labels(X)  # (n_patches, n_tokens)
        counts = np.stack([np.bincount(row, minlength=k) for row in raw]).astype(np.float64)

        if self.ordering == "correlation":
            scores = np.array([spearmanr(counts[:, c], y)[0] for c in range(k)])
        else:
            positive = y > 0
            pos_freq = counts[positive].sum(0) / (counts[positive].sum() + 1e-12)
            neg_freq = counts[~positive].sum(0) / (counts[~positive].sum() + 1e-12)
            scores = pos_freq - neg_freq

        order = self._set_order(scores)

        if self.ordering == "correlation":
            diagnostics = np.zeros((len(X), k + 1), dtype=np.float64)
            diagnostics[:, order] = counts
            diagnostics[:, -1] = y
            self.fit_diagnostics_ = diagnostics
        elif hasattr(self, "fit_diagnostics_"):
            del self.fit_diagnostics_
        return self

    def _set_order(self, scores) -> np.ndarray:
        """Rank raw cluster ids by ascending `scores`; sets `cluster_order_` and `order_scores_`."""
        order = np.argsort(np.argsort(scores))
        self.cluster_order_ = order
        self.order_scores_ = np.empty(len(scores))
        self.order_scores_[order] = scores
        return order

    def fit_order_ablation(self, images, X, model, scorer, *, full_scores=None, device=None,
                           batch_size: int = 64, seed: int = 0,
                           progress_bar: bool = True) -> "TokenClusterer":
        """
        Recompute `cluster_order_` by token ablation, keeping centroids.

        Each cluster is ranked by its mean `excess_drop` over patches (see
        :func:`mesoslide.tools.token_ablation`): how much more the feature
        score falls when the cluster's tokens are removed than when the same
        number of random tokens are removed. Clusters absent from every patch
        rank lowest.

        Parameters
        ----------
        images : array-like of shape (n_patches, C, H, W)
            Patch pixels (uint8).
        X : array-like or torch.Tensor of shape (n_patches, n_tokens, D)
            Token embeddings of the same patches, used to assign clusters.
        model : str or lazyslide_models.ImageModel
            The vision model `X` came from.
        scorer : callable
            Patch embeddings (n, D) -> feature scores (n,), e.g.
            :func:`mesoslide.tools.feature_scorer`.
        full_scores : array-like of shape (n_patches,), optional
            Stored patch scores, reported in `ablation_summary_`.
        device, batch_size, seed, progress_bar
            Forwarded to :func:`mesoslide.tools.token_ablation`.

        Returns
        -------
        self
        """
        from mesoslide.tools._token_ablation import summarize_token_ablation, token_ablation

        check_is_fitted(self, "cluster_centers_")
        self._validate_params()
        results = token_ablation(
            images, self._raw_labels(_as_tokens(X)), model, scorer, full_scores=full_scores,
            device=device, batch_size=batch_size, seed=seed, progress_bar=progress_bar,
        )
        self.ablation_seed_ = seed
        summary = summarize_token_ablation(results)
        scores = np.full(self.n_clusters_, -np.inf)
        scores[summary.index.to_numpy()] = summary["excess_drop"].to_numpy()
        order = self._set_order(scores)

        results.insert(1, "cluster", order[results.pop("label").to_numpy()])
        summary.index = pd.Index(order[summary.index.to_numpy()], name="cluster")
        self.ablation_results_ = results
        self.ablation_summary_ = summary.sort_index()
        if hasattr(self, "fit_diagnostics_"):
            del self.fit_diagnostics_
        return self

    def _raw_labels(self, X: np.ndarray) -> np.ndarray:
        """Nearest-centroid KMeans ids, (B, N_tokens), before `cluster_order_`."""
        B, N, D = X.shape
        gh, gw = self.grid_size_
        if N != gh * gw:
            raise ValueError(f"Expected {gh * gw} tokens ({gh}x{gw} grid), got {N}")
        labels = pairwise_distances_argmin(X.reshape(-1, D), self.cluster_centers_)
        return labels.reshape(B, N)

    def predict(self, X) -> np.ndarray:
        """
        Canonical (ordered) cluster label per token.

        Parameters
        ----------
        X : array-like or torch.Tensor of shape (B, N_tokens, D)

        Returns
        -------
        labels : ndarray of shape (B, grid_h, grid_w), dtype uint8
        """
        check_is_fitted(self, "cluster_order_")
        X = _as_tokens(X)
        gh, gw = self.grid_size_
        return self.cluster_order_[self._raw_labels(X)].reshape(len(X), gh, gw).astype(np.uint8)

    def transform(self, X, output_size: Optional[tuple] = None) -> np.ndarray:
        """
        Canonical cluster labels upsampled to pixel resolution.

        Parameters
        ----------
        X : array-like or torch.Tensor of shape (B, N_tokens, D)
        output_size : tuple, optional
            Target (height, width). Defaults to grid_size_ * patch_size_.

        Returns
        -------
        cluster_masks : ndarray of shape (B, H, W), dtype uint8
        """
        labels = self.predict(X)
        if output_size is None:
            if getattr(self, "patch_size_", None) is None:
                raise ValueError("output_size is required when patch_size_ is unknown.")
            (gh, gw), (ph, pw) = self.grid_size_, self.patch_size_
            output_size = (gh * ph, gw * pw)
        H, W = output_size
        interp = _INTERPOLATIONS[self.interpolation]
        out = np.empty((len(labels), H, W), dtype=np.uint8)
        for i, grid in enumerate(labels):
            out[i] = cv2.resize(grid, (W, H), interpolation=interp)
        return out

    def as_stage(self, *, output_size: Optional[tuple] = None, name: Optional[str] = None,
                 cache: bool = True, overwrite: bool = False) -> CallableStage:
        """Wrap `transform` as a `ModelStage` for `run_model_stages`.

        Chain it downstream of an `ImageModelStage(dense=True)`. `output_size`
        is fixed for the whole stage since one batch shares one resolution.
        """
        stage_name = name or self.display_name or f"cluster_{id(self)}"
        return CallableStage(
            lambda token_embeddings: self.transform(token_embeddings, output_size),
            name=stage_name, input_kind="dense",
            output_kind="dense", cache=cache, overwrite=overwrite, device="cpu",
        )

    def plot_cluster_feature_correlation(self, ax=None):
        """
        Scatter each cluster's per-patch token count against the patch's
        feature score, with each series' Spearman correlation in the legend.

        Requires a fit with ordering='correlation' (uses `fit_diagnostics_`).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. A new figure/axes is created if omitted.

        Returns
        -------
        (fig, ax)
        """
        if not hasattr(self, "fit_diagnostics_"):
            raise RuntimeError(
                "plot_cluster_feature_correlation() requires a fit with "
                "ordering='correlation'; no fit_diagnostics_ found."
            )

        import matplotlib.pyplot as plt

        diagnostics = self.fit_diagnostics_
        n_clusters = diagnostics.shape[1] - 1
        counts = diagnostics[:, :n_clusters]
        feature_scores = diagnostics[:, -1]

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            fig = ax.figure

        colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
        for c in range(n_clusters):
            rho, pval = spearmanr(counts[:, c], feature_scores)
            ax.scatter(
                feature_scores, counts[:, c],
                s=24, color=colors[c], alpha=0.8,
                label=f"Cluster {c} (Spearman ρ={rho:.2f}, p={pval:.2g})",
            )

        ax.set_xlabel("_feature_score")
        ax.set_ylabel("Token count")
        title_name = getattr(self, "feature_name_", None) or self.name
        ax.set_title(f"{title_name}: feature score vs. cluster token count" if title_name
                     else "Feature score vs. cluster token count")
        ax.legend(loc="best", fontsize=8, frameon=False)
        fig.tight_layout()
        return fig, ax

    def plot_cluster_ablation(self, axes=None):
        """
        Per-patch token-ablation diagnostics.

        Left: each patch's score with the cluster's tokens removed vs. with the
        same number of random tokens removed; points below the diagonal mean
        the cluster matters beyond its size. Right: excess drop vs. the
        fraction of the patch the cluster covers, to check that the ranking is
        not driven by cluster size. Requires a fit with ordering='ablation'
        (uses `ablation_results_` and `ablation_summary_`).

        Parameters
        ----------
        axes : sequence of two matplotlib.axes.Axes, optional
            Axes to draw into. A new figure is created if omitted.

        Returns
        -------
        (fig, axes)
        """
        if not hasattr(self, "ablation_results_"):
            raise RuntimeError(
                "plot_cluster_ablation() requires a fit with ordering='ablation'; "
                "no ablation_results_ found."
            )

        import matplotlib.pyplot as plt

        results, summary = self.ablation_results_, self.ablation_summary_
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        else:
            fig = axes[0].figure

        colors = plt.cm.viridis(np.linspace(0, 1, self.n_clusters_))
        top = results[["score_label_removed", "score_random_removed"]].to_numpy().max() * 1.05
        for k in summary.index:
            rows, stats = results[results["cluster"] == k], summary.loc[k]
            axes[0].scatter(
                rows["score_random_removed"], rows["score_label_removed"], s=16, alpha=0.7,
                color=colors[k],
                label=(f"Cluster {k}: excess drop {stats['excess_drop']:.3f} "
                       f"(p={stats['wilcoxon_p']:.2g}, n={int(stats['n_patches'])})"),
            )
            axes[1].scatter(rows["token_fraction"], rows["excess_drop"], s=16, alpha=0.7, color=colors[k])

        axes[0].plot([0, top], [0, top], "k--", lw=1)
        axes[0].set(xlabel="Score, random tokens removed", ylabel="Score, cluster tokens removed",
                    xlim=(0, top), ylim=(0, top))
        axes[0].legend(loc="upper left", fontsize=8, frameon=False)
        axes[1].axhline(0, color="k", lw=1, ls="--")
        axes[1].set(xlabel="Fraction of patch tokens in cluster", ylabel="Excess drop")
        title = self.display_name or "Token ablation"
        fig.suptitle(f"{title}: token ablation (below diagonal = cluster matters beyond its size)")
        fig.tight_layout()
        return fig, axes


def predict_token_labels(clusterers, X, *, chunk_size: int = 50_000) -> np.ndarray:
    """
    Canonical token labels from several clusterers on the same token embeddings.

    Same result as ``np.stack([c.predict(X) for c in clusterers])``, but the
    distances to every clusterer's centroids come from one matrix product per
    chunk of tokens instead of one pass per clusterer.

    Parameters
    ----------
    clusterers : sequence of TokenClusterer
        Fitted clusterers sharing the same `grid_size_` and embedding dimension.
    X : array-like or torch.Tensor of shape (B, N_tokens, D)
    chunk_size : int, default=50000
        Tokens per matrix product, bounding memory at chunk_size x total centroids.

    Returns
    -------
    labels : ndarray of shape (n_clusterers, B, grid_h, grid_w), dtype uint8
    """
    clusterers = list(clusterers)
    if not clusterers:
        raise ValueError("At least one clusterer must be provided.")
    for c in clusterers:
        check_is_fitted(c, "cluster_order_")
    gh, gw = clusterers[0].grid_size_
    dim = clusterers[0].n_features_in_
    for c in clusterers[1:]:
        if tuple(c.grid_size_) != (gh, gw) or c.n_features_in_ != dim:
            raise ValueError("All clusterers must share grid_size_ and n_features_in_.")

    X = _as_tokens(X)
    B, N, D = X.shape
    if N != gh * gw or D != dim:
        raise ValueError(f"Expected tokens of shape (B, {gh * gw}, {dim}), got {X.shape}")
    flat = X.reshape(-1, D)

    centers = np.concatenate([c.cluster_centers_ for c in clusterers])  # (total centroids, D)
    centers_sq = (centers ** 2).sum(1)
    bounds = np.cumsum([0] + [c.n_clusters_ for c in clusterers])

    labels = np.empty((len(clusterers), len(flat)), dtype=np.uint8)
    for start in range(0, len(flat), chunk_size):
        block = flat[start:start + chunk_size]
        # Squared Euclidean distance up to the per-token ||x||^2 term, which doesn't change argmin.
        dist = centers_sq[None, :] - 2.0 * block @ centers.T
        for k, c in enumerate(clusterers):
            raw = dist[:, bounds[k]:bounds[k + 1]].argmin(1)
            labels[k, start:start + len(block)] = c.cluster_order_[raw]
    return labels.reshape(len(clusterers), B, gh, gw)


def _embed_patches(patches, model, *, image_slides, tile_key, batch_size, device, progress_bar):
    """Dense token embeddings and feature scores (None if absent) for selected patches."""
    from mesoslide.tools._feature_extraction import run_model_stages

    key = "_fit_dense"
    stage = ImageModelStage(model, dense=True, name=key, device=device)
    run_model_stages(
        patches, [stage], slides=image_slides, tile_key=tile_key,
        batch_size=batch_size, progress_bar=progress_bar, save=False,
    )
    X = np.asarray(patches.obsm[key])
    obs = patches.obs
    y = obs["_feature_score"].to_numpy().astype(np.float64) if "_feature_score" in obs else None
    return X, y


def fit_token_clusterer(
    slides,
    feature_name: str,
    model=None,
    *,
    clusterer: Optional[TokenClusterer] = None,
    n_patches: int = 100,
    n_negative: int = 100,
    top_fraction: float = 0.10,
    min_score: float = 0.0,
    take_every: Optional[int] = None,
    tile_key: str = "tiles",
    image_slides=None,
    batch_size: int = 128,
    device: Optional[str] = None,
    progress_bar: bool = True,
    scorer=None,
    token: Optional[str] = None,
    model_path: "str | Path | None" = None,
) -> TokenClusterer:
    """
    Sample patches for a feature, embed them, and fit a `TokenClusterer`.

    Patch sampling depends on `clusterer.ordering`:

    - 'correlation': `n_patches` patches from the top `top_fraction` of
      scores. KMeans is fit on them (skipped if `clusterer` is already
      fitted), and clusters are ranked by the Spearman correlation, across
      the same patches, between each cluster's token count and the patch
      score. `n_negative` is unused.
    - 'diff_abundance': `n_patches` top-scoring patches (`top_fraction`) and
      `n_negative` zero-score patches. KMeans is fit on the top-scoring
      patches (if not already fitted); the order compares the two sets.
    - 'ablation': `n_patches` top-scoring patches (`top_fraction`). KMeans is
      fit on them (if not already fitted), then clusters are ranked by token
      ablation on the same patches (see `TokenClusterer.fit_order_ablation`),
      which needs `scorer`. Costs 2 vision-model passes per cluster present
      in each patch. The random-token controls are seeded from `feature_name`
      (CRC32), so refits of the same feature are reproducible.

    Parameters
    ----------
    slides : slides_table, AnnData, WSIData, or sequence/mapping of either
        Where to select patches from. See :func:`mesoslide.select_top_patches`.
    feature_name : str
        Feature whose score ranks clusters (e.g. 'UNI_SAE_12345'). Stored as
        `clusterer.feature_name_`.
    model : str or lazyslide_models.ImageModel, optional
        ViT-style vision model to embed patches with. Defaults to
        `clusterer.model_name_` (required when `clusterer` is None). Pass an
        instance to reuse one already-loaded model across many calls.
    clusterer : TokenClusterer, optional
        Clusterer to fit. Defaults to `TokenClusterer()`. If it is already
        fitted, its centroids are kept and only the cluster order is
        recomputed.
    n_patches : int, default=100
        See above.
    n_negative : int, default=100
        Zero-score patches, only under ordering='diff_abundance'.
    top_fraction : float, default=0.10
        Fraction of qualifying (score > `min_score`) patches that top-scoring
        sets are drawn from.
    min_score : float, default=0.0
    take_every : int, optional
        Forwarded to :func:`mesoslide.select_top_patches`.
    tile_key : str, default='tiles'
    image_slides : WSIData / list / {slide_id: WSIData}, optional
        Slides to read pixels from, with image data attached. Defaults to
        `patches.obs['_slide_ref']` set by the selectors.
    batch_size : int, default=128
    device : str, optional
        Torch device for the vision model. Defaults to "cuda" if available.
    progress_bar : bool, default=True
    scorer : sparse-coding model or callable, optional
        Required for ordering='ablation'. Either a fitted sparse-coding model
        (`SparseAutoencoder`, `LocalityConstrainedCoding`, or anything with the
        same `transform`), scored on `feature_name`'s column with `device`
        (see :func:`mesoslide.tools.feature_scorer`); or any callable mapping
        patch embeddings (n, D) to scores (n,), used as is.
    token, model_path
        Forwarded to model resolution.

    Returns
    -------
    clusterer : TokenClusterer
        Fitted, with `grid_size_`, `patch_size_`, `model_name_` and
        `feature_name_` set.
    """
    from mesoslide._patch_selector import select_negative_patches, select_top_patches

    clusterer = TokenClusterer() if clusterer is None else clusterer
    clusterer._validate_params()
    if model is None:
        model = getattr(clusterer, "model_name_", None)
        if model is None:
            raise ValueError("model is required when clusterer has no model_name_.")

    resolved, resolved_name = _resolve_model(model, model_path=model_path, token=token)
    _require_dense_capable(resolved, resolved_name)
    grid = tuple(resolved.grid_size)
    if getattr(clusterer, "grid_size_", None) is not None and tuple(clusterer.grid_size_) != grid:
        raise ValueError(
            f"model has grid_size={grid}, but clusterer was fit with grid_size_="
            f"{tuple(clusterer.grid_size_)}"
        )
    clusterer.grid_size_ = grid
    clusterer.patch_size_ = tuple(resolved.patch_size)
    clusterer.model_name_ = _canonical_registry_name(resolved_name)

    fitted = hasattr(clusterer, "cluster_centers_")

    def embed(patches):
        return _embed_patches(
            patches, resolved, image_slides=image_slides, tile_key=tile_key,
            batch_size=batch_size, device=device, progress_bar=progress_bar,
        )

    def select_top(fraction):
        return select_top_patches(
            slides, feature_name, n=n_patches, tile_key=tile_key,
            min_score=min_score, top_fraction=fraction, take_every=take_every,
        )

    if clusterer.ordering == "ablation":
        from mesoslide.preprocessing._extract_patches import extract_patch_images
        from mesoslide.tools.sparse_coding._feature_scorer import feature_scorer

        if scorer is None:
            raise ValueError("ordering='ablation' requires scorer (a sparse-coding model or a callable).")
        score_fn = feature_scorer(scorer, feature_name, device=device) if hasattr(scorer, "transform") else scorer
        top = select_top(top_fraction)
        X, y = embed(top)
        # Pixels were cached in `top` by the embedding step above.
        images = extract_patch_images(top, image_slides, tile_key=tile_key, channel_first=True,
                                      progress_bar=False)
        if not fitted:
            clusterer.fit(X)
        clusterer.fit_order_ablation(
            images, X, resolved, score_fn, full_scores=y, device=device,
            batch_size=batch_size, seed=zlib.crc32(feature_name.encode()),
            progress_bar=progress_bar,
        )
        clusterer.feature_name_ = feature_name
        return clusterer

    if clusterer.ordering == "correlation":
        X_fit, y = embed(select_top(top_fraction))
        X_order, y_order = X_fit, y
    else:
        X_pos, _ = embed(select_top(top_fraction))
        X_neg, _ = embed(select_negative_patches(
            slides, feature_name, n=n_negative, tile_key=tile_key, take_every=None,
        ))
        X_fit = X_pos
        X_order = np.concatenate([X_pos, X_neg])
        y_order = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])

    if fitted:
        clusterer.fit_order(X_order, y_order)
    else:
        clusterer.fit(X_fit, X_order=X_order, y_order=y_order)
    clusterer.feature_name_ = feature_name
    return clusterer
