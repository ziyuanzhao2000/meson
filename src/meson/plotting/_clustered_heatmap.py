import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from typing import Optional, Tuple


def _reorder_cluster_labels(arr: np.ndarray) -> np.ndarray:
    """Relabel clusters 1..K in the order they first appear in arr."""
    new_arr = np.zeros_like(arr)
    seen = []
    for i, v in enumerate(arr):
        if v not in seen:
            seen.append(v)
        new_arr[i] = seen.index(v) + 1
    return new_arr


def plot_clustered_heatmap(
    matrix: np.ndarray,
    display_matrix: Optional[np.ndarray] = None,
    is_distance_matrix: bool = True,
    metric: str = 'cosine',
    linkage_method: str = 'average',
    threshold: float = 1.0,
    criterion: str = 'distance',
    cmap: str = 'coolwarm_r',
    center: Optional[float] = None,
) -> Tuple:
    """
    Hierarchically cluster a matrix and render a clustermap with cluster
    bounding boxes overlaid.

    When display_matrix is provided, clustering is derived from matrix
    but the heatmap visualises display_matrix — useful for clustering on
    a strict binary IoU while showing the softer weighted IoU.

    Parameters
    ----------
    matrix : np.ndarray
        Square distance (or similarity) matrix used for clustering.
    display_matrix : np.ndarray, optional
        Square matrix to visualise. Defaults to matrix.
    is_distance_matrix : bool
        If True, matrix is treated as a precomputed distance matrix.
        If False, rows are treated as feature vectors and pairwise
        distances are computed using metric.
    metric : str
        Distance metric passed to scipy.spatial.distance.pdist.
        Only used when is_distance_matrix=False.
    linkage_method : str
        Linkage algorithm passed to scipy.cluster.hierarchy.linkage.
    threshold : float
        Cluster threshold passed to fcluster.
    criterion : str
        Criterion passed to fcluster (e.g. 'distance', 'maxclust').
    cmap : str
        Colormap for the heatmap.
    center : float, optional
        Value mapped to the centre of the diverging colormap.
        Defaults to the midpoint of display_matrix.

    Returns
    -------
    g : seaborn.matrix.ClusterGrid
    linkage_matrix : np.ndarray
    reordered_idx : list of int
        Column/row order after dendrogram reordering.
    reordered_clusters : np.ndarray
        Cluster labels aligned to reordered_idx, relabelled 1..K
        in order of appearance.
    reordered_display_matrix : np.ndarray
        display_matrix reordered to match the dendrogram.
    """
    if is_distance_matrix:
        dist_matrix = matrix
        linkage_matrix = linkage(squareform(dist_matrix), method=linkage_method)
    else:
        condensed = pdist(matrix, metric=metric)
        dist_matrix = squareform(condensed)
        linkage_matrix = linkage(condensed, method=linkage_method)

    clusters = fcluster(linkage_matrix, threshold, criterion)

    vis_matrix = display_matrix if display_matrix is not None else dist_matrix

    vmin, vmax = vis_matrix.min(), vis_matrix.max()
    vcenter = center if center is not None else (vmin + vmax) / 2
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    g = sns.clustermap(
        vis_matrix,
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix,
        cmap=cmap,
        norm=norm,
        fmt='.2f',
    )
    g.ax_heatmap.tick_params(
        right=False, bottom=False,
        labelbottom=False, labelright=False
    )

    reordered_idx = g.dendrogram_row.reordered_ind
    reordered_clusters = clusters[reordered_idx]
    reordered_display = vis_matrix[reordered_idx, :][:, reordered_idx]

    n_clusters = len(np.unique(clusters))
    print(f'Number of clusters: {n_clusters}')

    for cluster_id in range(1, n_clusters + 1):
        mask = reordered_clusters == cluster_id
        if not mask.any():
            continue
        start = np.where(mask)[0][0]
        end = np.where(mask)[0][-1]
        rect = mpatches.Rectangle(
            (start, start),
            end - start + 1,
            end - start + 1,
            fill=False,
            edgecolor='black',
            linewidth=2,
        )
        g.ax_heatmap.add_patch(rect)

    return (
        g,
        linkage_matrix,
        reordered_idx,
        _reorder_cluster_labels(reordered_clusters),
        reordered_display,
    )