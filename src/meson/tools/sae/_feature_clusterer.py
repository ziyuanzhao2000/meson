import numpy as np
from scipy.sparse import diags, issparse
from numba import njit, prange
from typing import Optional, List, Union, Sequence

from meson.plotting import plot_clustered_heatmap, plot_feature_gallery


# ── Low-level IoU kernel ────────────────────────────────────────────────────

@njit(parallel=True, fastmath=True)
def _sparse_iou_kernel(data, indices, indptr, n_feats, col_sums):
    iou = np.eye(n_feats)
    for i in prange(n_feats):
        si, ei = indptr[i], indptr[i + 1]
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
            union = col_sums[i] + col_sums[j] - intersection
            if union > 0:
                v = intersection / union
                iou[i, j] = v
                iou[j, i] = v
    return iou


def _weighted_iou(X):
    """Compute (n_features x n_features) weighted pairwise IoU."""
    if issparse(X):
        X_csc = X.tocsc()
        col_sums = np.array(X_csc.sum(axis=0)).ravel().astype(np.float64)
        return _sparse_iou_kernel(
            X_csc.data.astype(np.float64),
            X_csc.indices,
            X_csc.indptr,
            X_csc.shape[1],
            col_sums,
        )
    else:
        col_sums = X.sum(axis=0)
        n_feats = X.shape[1]
        inter = np.zeros((n_feats, n_feats))
        for i in range(n_feats):
            vals = np.minimum(X[:, [i]], X[:, i:]).sum(axis=0)
            inter[i, i:] = vals
            inter[i:, i] = vals
        union = col_sums[:, None] + col_sums[None, :] - inter
        return np.divide(inter, union,
                         out=np.ones_like(inter),
                         where=union != 0)


class SAEFeatureClusterer:
    """
    Computes pairwise IoU between selected SAE features, clusters them
    hierarchically, and provides plotting helpers.

    Follows a fit/cluster pattern analogous to SAEFeatureSelector:
    compute_iou() is the expensive step; cluster() and plotting are cheap.

    Parameters
    ----------
    high_activation_threshold : float
        Features with normalised activation above this value are considered
        'strongly active' when building the strict binary IoU matrix used
        for clustering. Default 0.5.

    Examples
    --------
    >>> clusterer = SAEFeatureClusterer()
    >>> clusterer.compute_iou(all_patches, feature_prefix='UNI_SAE',
    ...                       feature_indices=selected_idx)
    >>> clusterer.cluster(threshold=25, criterion='maxclust')
    >>> clusterer.plot_heatmap()
    >>> cluster_ids = clusterer.get_cluster_assignments()
    """

    def __init__(self, high_activation_threshold: float = 0.5):
        self.high_activation_threshold = high_activation_threshold

        # set after compute_iou()
        self.iou_soft_: Optional[np.ndarray] = None   # all-active binary IoU
        self.iou_strict_: Optional[np.ndarray] = None  # threshold-filtered binary IoU
        self.feature_indices_: Optional[np.ndarray] = None

        # set after cluster()
        self.reordered_idx_: Optional[List[int]] = None
        self.reordered_clusters_: Optional[np.ndarray] = None
        self.linkage_matrix_: Optional[np.ndarray] = None
        self._is_fitted = False
        self._is_clustered = False

    # ── public API ─────────────────────────────────────────────────────────

    def compute_iou(self, adata, feature_prefix: str,
                    feature_indices: np.ndarray) -> "SAEFeatureClusterer":
        """
        Normalise the embedding matrix and compute two pairwise IoU matrices:

        - iou_soft_   : every nonzero activation counts as active (soft)
        - iou_strict_ : only activations above high_activation_threshold
                        count (strict); used as the clustering distance

        Parameters
        ----------
        adata : AnnData
            Patch-level AnnData with sparse SAE embeddings in .X.
        feature_prefix : str
            Column prefix, e.g. 'UNI_SAE'.
        feature_indices : np.ndarray of int
            Indices of the selected features (output of SAEFeatureSelector).
        """
        feature_names = [f'{feature_prefix}_{i}' for i in feature_indices]
        X = adata[:, feature_names].X

        # max-normalise so activations are in [0, 1]
        col_max = np.array(X.max(axis=0).todense()).ravel()
        col_max[col_max == 0] = 1.0
        X_norm = X @ diags(1.0 / col_max)

        # soft: every nonzero position is active
        X_soft = X_norm.copy()
        X_soft.data[:] = 1.0

        # strict: only strongly active positions
        X_strict = X_norm.copy()
        X_strict.data[X_strict.data <= self.high_activation_threshold] = 0
        X_strict.eliminate_zeros()
        X_strict.data[:] = 1.0

        self.iou_soft_ = _weighted_iou(X_soft)
        self.iou_strict_ = _weighted_iou(X_strict)
        self.feature_indices_ = np.asarray(feature_indices)
        self._is_fitted = True
        return self

    def cluster(
        self,
        threshold: float = 1.0,
        criterion: str = 'distance',
        linkage_method: str = 'average',
    ) -> "SAEFeatureClusterer":
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

    def plot_feature_gallery(
        self,
        exemplar_patches: Optional[dict] = None,
        sdata=None,
        patch_table_names: Union[str, Sequence[str], None] = None,
        feature_prefix: Optional[str] = None,
        show_labels: bool = True,
        n_cols: int = 10,
        patch_size: float = 2.0,
        border_extend: float = 0.05,
        cmap: str = 'tab10',
        fontsize: float = 6,
    ):
        """
        Plot exemplar patches ordered and coloured by cluster assignment.

        Supply either `exemplar_patches` (pre-loaded dict) **or**
        (`sdata`, `patch_table_names`, `feature_prefix`) to extract
        top-1 patches on the fly via select_exemplar_patches.

        Parameters
        ----------
        exemplar_patches : dict, optional
            Mapping global feature index → array (N, H, W, 3). Index [0] is used.
        sdata : SpatialData, optional
            Required when exemplar_patches is None.
        patch_table_names : str or sequence of str, optional
            Table name(s) in sdata, e.g. 'TB001_grid_point_patch' or a list.
            Required when exemplar_patches is None.
        feature_prefix : str, optional
            e.g. 'UNI_SAE'. Required when exemplar_patches is None.
        show_labels, n_cols, patch_size, border_extend, cmap
            Forwarded to plot_feature_gallery.

        Returns
        -------
        fig, axs
        """
        from meson.plotting._patch_gallery import extract_patch_images
        from meson._patch_selector import select_exemplar_patches

        self._check_clustered()
        ordered_idx = self.feature_indices_[self.reordered_idx_]

        if exemplar_patches is not None:
            images = [exemplar_patches[idx][0] for idx in ordered_idx]
        else:
            if sdata is None or patch_table_names is None or feature_prefix is None:
                raise ValueError(
                    "Provide either exemplar_patches, or all of "
                    "(sdata, patch_table_names, feature_prefix)."
                )
            feature_names = [f"{feature_prefix}_{idx}" for idx in ordered_idx]
            exemplar_adata = select_exemplar_patches(
                sdata,
                patch_table_names=patch_table_names,
                feature_names=feature_names,
                n_exemplars=1,
            )
            images = extract_patch_images(sdata, exemplar_adata, progress=True)

        labels = [str(idx) for idx in ordered_idx] if show_labels else None
        group_ids = self.reordered_clusters_.tolist()

        return plot_feature_gallery(
            images=images,
            group_ids=group_ids,
            labels=labels,
            n_cols=n_cols,
            patch_size=patch_size,
            border_extend=border_extend,
            border_alpha=1.0,
            cmap=cmap,
            fontsize=fontsize,
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