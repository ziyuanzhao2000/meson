import warnings

import numpy as np
from tqdm.auto import tqdm
from scipy.sparse import diags, issparse
from numba import njit, prange
from typing import Optional, List, Union, Sequence

from mesoslide.plotting import plot_clustered_heatmap, plot_feature_gallery


# ── Low-level IoU kernel ────────────────────────────────────────────────────

@njit(parallel=True, fastmath=True)
def _sparse_intersection_kernel(data, indices, indptr, n_feats):
    """Pairwise intersection: sum over rows of min(x_i, x_j).

    Returns the raw intersection matrix rather than IoU. The division by the
    union used to live in this inner loop, which both duplicated work per pair
    and made the result non-additive. Intersection and column sums are each
    additive over row blocks, so keeping them separate is what lets a cohort be
    accumulated one slide at a time (see `iou_from_parts`).
    """
    inter = np.zeros((n_feats, n_feats))
    for i in prange(n_feats):
        si, ei = indptr[i], indptr[i + 1]
        # diagonal: intersection of a column with itself is its own sum
        diag = 0.0
        for p in range(si, ei):
            diag += data[p]
        inter[i, i] = diag
        for j in range(i + 1, n_feats):
            sj, ej = indptr[j], indptr[j + 1]
            intersection = 0.0
            pi, pj = si, sj
            while pi < ei and pj < ej:
                ii, ij = indices[pi], indices[pj]
                if ii == ij:
                    intersection += min(data[pi], data[pj])
                    pi += 1; pj += 1
                elif ii < ij:
                    pi += 1
                else:
                    pj += 1
            inter[i, j] = intersection
            inter[j, i] = intersection
    return inter


def _intersection_and_sums(X):
    """(intersection matrix, per-feature column sums) for one block of rows."""
    if issparse(X):
        X_csc = X.tocsc()
        col_sums = np.array(X_csc.sum(axis=0)).ravel().astype(np.float64)
        inter = _sparse_intersection_kernel(
            X_csc.data.astype(np.float64),
            X_csc.indices,
            X_csc.indptr,
            X_csc.shape[1],
        )
        return inter, col_sums

    X = np.asarray(X, dtype=np.float64)
    col_sums = X.sum(axis=0)
    n_feats = X.shape[1]
    inter = np.zeros((n_feats, n_feats))
    for i in range(n_feats):
        vals = np.minimum(X[:, [i]], X[:, i:]).sum(axis=0)
        inter[i, i:] = vals
        inter[i:, i] = vals
    return inter, col_sums


def iou_from_parts(inter, col_sums):
    """IoU from accumulated intersection and column sums.

    IoU_ij = I_ij / (S_i + S_j - I_ij). Both I and S are sums over rows, so
    this is exact whether the parts came from one matrix or from many blocks
    accumulated in sequence.
    """
    union = col_sums[:, None] + col_sums[None, :] - inter
    iou = np.divide(inter, union, out=np.ones_like(inter), where=union != 0)
    np.fill_diagonal(iou, 1.0)
    return iou


def _weighted_iou(X):
    """Compute (n_features x n_features) weighted pairwise IoU for one matrix."""
    inter, col_sums = _intersection_and_sums(X)
    return iou_from_parts(inter, col_sums)


class FeatureClusterer:
    """
    Computes pairwise IoU between selected sparse-coding features (SAE or LLC), clusters them
    hierarchically, and provides plotting helpers.

    Follows a fit/cluster pattern analogous to FeatureSelector:
    compute_iou() is the expensive step; cluster() and plotting are cheap.

    Parameters
    ----------
    high_activation_threshold : float
        Features with normalised activation above this value are considered
        'strongly active' when building the strict binary IoU matrix used
        for clustering. Default 0.5.

    Examples
    --------
    >>> clusterer = FeatureClusterer()
    >>> clusterer.compute_iou(slides, feature_prefix='UNI_SAE',
    ...                       feature_indices=selected_idx)
    >>> clusterer.cluster(threshold=25, criterion='maxclust')
    >>> clusterer.plot_heatmap()
    >>> clusterer.select_exemplars(slides)
    >>> clusterer.plot_feature_gallery(groups=[1, 2, 3], return_fig=True)
    >>> cluster_ids = clusterer.get_cluster_assignments()
    """

    def __init__(self, high_activation_threshold: float = 0.5):
        self.high_activation_threshold = high_activation_threshold

        # set after compute_iou()
        self.iou_soft_: Optional[np.ndarray] = None   # all-active binary IoU
        self.iou_strict_: Optional[np.ndarray] = None  # threshold-filtered binary IoU
        self.feature_indices_: Optional[np.ndarray] = None
        self.feature_prefix_: Optional[str] = None
        self.tile_key_: str = "tiles"

        # set after cluster()
        self.reordered_idx_: Optional[List[int]] = None
        self.reordered_clusters_: Optional[np.ndarray] = None
        self.linkage_matrix_: Optional[np.ndarray] = None

        # set after select_exemplars()
        self.exemplar_patches_ = None
        self._is_fitted = False
        self._is_clustered = False

    # ── public API ─────────────────────────────────────────────────────────

    def compute_iou(self, slides, feature_prefix: str,
                    feature_indices: np.ndarray, *,
                    tile_key: str = "tiles",
                    progress: bool = True) -> "FeatureClusterer":
        """
        Compute two pairwise IoU matrices over one or more patch tables, one
        slide at a time.

        - iou_soft_   : every nonzero activation counts as active (soft)
        - iou_strict_ : only activations above high_activation_threshold count;
                        used as the clustering distance

        Both quantities involved are sums over rows -- the pairwise
        intersection and the per-feature column sums -- so accumulating them
        slide by slide is exact, not an approximation. Two passes are needed:
        the first for the max-normalisation constant (an associative max over
        slides), the second for the accumulation. Peak memory is one slide's
        ``.X``, regardless of whether `slides` is a single in-memory table or
        a cohort streamed from a manifest.

        Parameters
        ----------
        slides : slides_table, AnnData, WSIData, or sequence/mapping of either
            Patch-level table(s) with sparse SAE/LLC embeddings in .X.
        feature_prefix : str
            Column prefix, e.g. 'UNI_SAE'.
        feature_indices : np.ndarray of int
            Indices of the selected features (output of FeatureSelector).
        tile_key : str, default='tiles'
        progress : bool

        Returns
        -------
        self
        """
        from mesoslide._slides import SlideSource

        source = slides if isinstance(slides, SlideSource) else SlideSource(slides, tile_key=tile_key)
        feature_names = [f'{feature_prefix}_{i}' for i in feature_indices]
        self.feature_prefix_ = feature_prefix
        self.tile_key_ = tile_key

        # Pass 1: max-normalisation constant, as an associative max over slides.
        col_max = None
        it = tqdm(source, desc="IoU pass 1/2 (column max)") if progress else source
        for _, table in it:
            m = self._column_max(table, feature_names)
            col_max = m if col_max is None else np.maximum(col_max, m)
        if col_max is None:
            raise ValueError("No slides to read from.")

        # Pass 2: accumulate intersection and column sums.
        self._start(feature_indices, col_max)
        it = tqdm(source, desc="IoU pass 2/2 (accumulate)") if progress else source
        for _, table in it:
            self._accumulate(table, feature_names)
        return self._finalize()

    # ── accumulation internals ─────────────────────────────────────────────

    @staticmethod
    def _column_max(table, feature_names) -> np.ndarray:
        X = table[:, feature_names].X
        if X.shape[0] == 0:
            return np.zeros(len(feature_names), dtype=np.float64)
        if issparse(X):
            return np.asarray(X.max(axis=0).todense()).ravel().astype(np.float64)
        return np.asarray(X).max(axis=0).astype(np.float64)

    def _start(self, feature_indices, col_max):
        n = len(feature_indices)
        self.feature_indices_ = np.asarray(feature_indices)
        self._col_max = np.where(col_max == 0, 1.0, col_max)
        self._inter_soft = np.zeros((n, n))
        self._sums_soft = np.zeros(n)
        self._inter_strict = np.zeros((n, n))
        self._sums_strict = np.zeros(n)

    def _accumulate(self, table, feature_names):
        X = table[:, feature_names].X
        if X.shape[0] == 0:
            return

        # max-normalise so activations are in [0, 1]
        X_norm = X @ diags(1.0 / self._col_max) if issparse(X) else np.asarray(X) / self._col_max

        if issparse(X_norm):
            X_soft = X_norm.copy()
            X_soft.data[:] = 1.0

            X_strict = X_norm.copy()
            X_strict.data[X_strict.data <= self.high_activation_threshold] = 0
            X_strict.eliminate_zeros()
            X_strict.data[:] = 1.0
        else:
            X_soft = (X_norm > 0).astype(np.float64)
            X_strict = (X_norm > self.high_activation_threshold).astype(np.float64)

        for X_bin, inter_acc, sums_acc in (
            (X_soft, "_inter_soft", "_sums_soft"),
            (X_strict, "_inter_strict", "_sums_strict"),
        ):
            inter, sums = _intersection_and_sums(X_bin)
            setattr(self, inter_acc, getattr(self, inter_acc) + inter)
            setattr(self, sums_acc, getattr(self, sums_acc) + sums)

    def _finalize(self) -> "FeatureClusterer":
        self.iou_soft_ = iou_from_parts(self._inter_soft, self._sums_soft)
        self.iou_strict_ = iou_from_parts(self._inter_strict, self._sums_strict)
        self._is_fitted = True
        return self

    def cluster(
        self,
        threshold: float = 1.0,
        criterion: str = 'distance',
        linkage_method: str = 'average',
    ) -> "FeatureClusterer":
        """
        Hierarchically cluster features using 1 - iou_strict_ as distances.

        Parameters
        ----------
        threshold : float
            Passed to scipy fcluster (meaning depends on criterion).
        criterion : str
            'distance' or 'maxclust'.
        linkage_method : str
            Linkage algorithm, e.g. 'average', 'single', 'complete'.
        """
        self._check_fitted()

        # We call plot_clustered_heatmap internally to get the linkage /
        # reordering — but we do NOT display it here; that is plot_heatmap()'s job.
        # So we call the underlying scipy functions directly to avoid a plot side-effect.
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        dist = 1 - self.iou_strict_
        lm = linkage(squareform(dist), method=linkage_method)
        clusters = fcluster(lm, threshold, criterion)

        # We need the dendrogram order — use seaborn internally via a dry run
        # wrapped in a non-displayed figure.
        import matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns

        with matplotlib.rc_context({'figure.max_open_warning': 0}):
            g = sns.clustermap(dist, row_linkage=lm, col_linkage=lm)
            reordered_idx = g.dendrogram_row.reordered_ind
            plt.close(g.fig)

        reordered_clusters = clusters[reordered_idx]

        # relabel 1..K in appearance order
        seen, new_labels = [], np.zeros_like(reordered_clusters)
        for i, v in enumerate(reordered_clusters):
            if v not in seen:
                seen.append(v)
            new_labels[i] = seen.index(v) + 1

        self.linkage_matrix_ = lm
        self.reordered_idx_ = reordered_idx
        self.reordered_clusters_ = new_labels
        self._cluster_threshold = threshold       
        self._cluster_criterion = criterion       
        self._is_clustered = True
        return self

    def plot_heatmap(self, center: Optional[float] = None, **kwargs):
        """
        Plot the clustered heatmap.

        Clusters on 1 - iou_strict_ (binary, strict threshold),
        displays 1 - iou_soft_ (softer, all-active).

        Returns
        -------
        g : seaborn ClusterGrid
        """
        self._check_clustered()
        g, *_ = plot_clustered_heatmap(
            matrix=1 - self.iou_strict_,
            display_matrix=1 - self.iou_soft_,
            is_distance_matrix=True,
            linkage_method='average',
            threshold=self._cluster_threshold,    
            criterion=self._cluster_criterion,    
            center=center,
            **kwargs,
        )
        return g

    def select_exemplars(
        self,
        slides,
        *,
        n_exemplars: int = 1,
        min_score: float = 0.0,
        tile_key: Optional[str] = None,
    ):
        """
        Select exemplar patches for every clustered feature, in dendrogram order.

        The result is cached on `exemplar_patches_` and used by
        `plot_feature_gallery`. Rows are sorted by (`feature_order`,
        `_feature_rank`), and annotated with these `.obs` columns:

        - feature_id    : global feature index
        - group_id      : cluster label (as in `reordered_clusters_`)
        - feature_order : position in dendrogram order

        Parameters
        ----------
        slides : slides_table, AnnData, WSIData, or sequence/mapping of either
            Where to select exemplars from, as for `select_exemplar_patches`.
        n_exemplars : int, default=1
            Top patches kept per feature. The gallery plots rank 1 only.
        min_score : float, default=0.0
        tile_key : str, optional
            Defaults to the `tile_key` passed to `compute_iou`.

        Returns
        -------
        PatchData
        """
        from mesoslide._patch_selector import select_exemplar_patches

        self._check_clustered()
        tile_key = tile_key or self.tile_key_
        ordered_idx = self.get_reordered_feature_indices()
        names = [f"{self.feature_prefix_}_{idx}" for idx in ordered_idx]
        exemplars = select_exemplar_patches(
            slides, names, n_exemplars=n_exemplars, min_score=min_score,
            tile_key=tile_key,
        )
        if len(exemplars) == 0:
            raise ValueError("No exemplar patches found for any clustered feature.")

        position = {name: i for i, name in enumerate(names)}
        obs = exemplars.obs
        feature_order = obs["_feature_name"].astype(str).map(position).to_numpy().astype(int)
        obs["feature_order"] = feature_order
        obs["feature_id"] = ordered_idx[feature_order]
        obs["group_id"] = self.reordered_clusters_[feature_order]
        exemplars = exemplars[np.lexsort((obs["_feature_rank"].to_numpy(), feature_order))]

        n_missing = len(names) - len(np.unique(feature_order))
        if n_missing:
            warnings.warn(
                f"{n_missing} of {len(names)} features have no exemplar patch "
                f"(no activation above min_score={min_score}); they are left "
                f"out of the gallery.",
                UserWarning,
                stacklevel=2,
            )
        self.exemplar_patches_ = exemplars
        self._exemplar_tile_key = tile_key
        return exemplars

    def plot_feature_gallery(
        self,
        groups: Optional[Sequence[int]] = None,
        *,
        image_slides=None,
        show_labels: bool = True,
        n_cols: int = 10,
        patch_size: float = 1.0,
        border_extend: float = 0.05,
        border_alpha: float = 1.0,
        cmap="tab20",
        fontsize: float = 10,
        output_path: Optional[str] = None,
        dpi: int = 300,
        return_fig: bool = False,
        return_buffer: bool = False,
        progress_bar: bool = True,
    ):
        """
        Plot the top exemplar of each feature, in dendrogram order, with
        borders coloured by cluster.

        Requires `select_exemplars` first. Pixels are read once and cached in
        `exemplar_patches_.obsm`, so repeated calls (e.g. for different
        `groups`) do not re-read slides.

        Parameters
        ----------
        groups : sequence of int, optional
            Cluster labels to include. Defaults to all.
        image_slides : WSIData / list / {slide_id: WSIData}, optional
            Slides to read pixels from. Not needed when the exemplars'
            `_slide_ref` column resolves to slides with image data, or when
            pixels are already cached.
        show_labels : bool, default=True
            Label each patch with its feature_id.
        n_cols, patch_size, border_extend, border_alpha, cmap, fontsize
            Forwarded to `mesoslide.plotting.plot_feature_gallery`. A
            `ListedColormap` is indexed by group id.
        output_path, dpi, return_fig, return_buffer
            Forwarded to `mesoslide.plotting.plot_feature_gallery`.
        progress_bar : bool, default=True

        Returns
        -------
        (fig, axs), a buffer, or None
        """
        from mesoslide.preprocessing._extract_patches import extract_patch_images

        if self.exemplar_patches_ is None:
            raise RuntimeError("Call select_exemplars() before plot_feature_gallery().")
        exemplars = self.exemplar_patches_
        images = extract_patch_images(
            exemplars, image_slides, tile_key=self._exemplar_tile_key,
            channel_first=False, cache=True, progress_bar=progress_bar,
        )
        if len(images) != len(exemplars):
            raise ValueError(
                f"Read {len(images)} images for {len(exemplars)} exemplar patches; "
                f"some reads failed, so images cannot be matched to features."
            )

        obs = exemplars.obs
        keep = (obs["_feature_rank"] == 1).to_numpy().copy()
        if groups is not None:
            keep &= obs["group_id"].isin(list(groups)).to_numpy()
        rows = np.flatnonzero(keep)
        if len(rows) == 0:
            raise ValueError(f"No exemplars belong to groups {list(groups)}.")

        feature_ids = obs["feature_id"].to_numpy()[rows]
        return plot_feature_gallery(
            images=[images[i] for i in rows],
            group_ids=obs["group_id"].to_numpy()[rows].tolist(),
            labels=[str(f) for f in feature_ids] if show_labels else None,
            n_cols=n_cols,
            patch_size=patch_size,
            border_extend=border_extend,
            border_alpha=border_alpha,
            cmap=cmap,
            fontsize=fontsize,
            output_path=output_path,
            dpi=dpi,
            return_fig=return_fig,
            return_buffer=return_buffer,
        )

    def get_cluster_assignments(self) -> np.ndarray:
        """
        Return cluster labels aligned to the dendrogram order.

        Returns
        -------
        np.ndarray of int, shape (n_selected_features,)
        """
        self._check_clustered()
        return self.reordered_clusters_

    def get_reordered_feature_indices(self) -> np.ndarray:
        """Global feature indices in dendrogram order."""
        self._check_clustered()
        return self.feature_indices_[self.reordered_idx_]

    # ── guards ─────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "Call compute_iou() before using this method."
            )

    def _check_clustered(self):
        self._check_fitted()
        if not self._is_clustered:
            raise RuntimeError(
                "Call cluster() before using this method."
            )