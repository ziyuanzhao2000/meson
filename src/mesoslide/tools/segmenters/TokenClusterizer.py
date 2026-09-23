"""Deprecated alias of `TokenClusterer`, kept so existing pickles still load.

Pickled `TokenClusterizer` objects reference this module path; `__setstate__`
migrates their attributes to the `TokenClusterer` layout.
"""

import warnings
from typing import Optional

import numpy as np

from ._token_clusterer import TokenClusterer


class TokenClusterizer(TokenClusterer):
    """Deprecated: use :class:`TokenClusterer` and :func:`fit_token_clusterer`."""

    def __init__(
        self,
        n_clusters: int = 3,
        *,
        ordering: str = "correlation",
        interpolation: str = "nearest",
        name: Optional[str] = None,
        random_state: Optional[int] = 0,
    ):
        warnings.warn(
            "TokenClusterizer is deprecated, use TokenClusterer. Its constructor no "
            "longer takes model/kmeans/device; fit from slides with fit_token_clusterer().",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(
            n_clusters, ordering=ordering, interpolation=interpolation,
            name=name, random_state=random_state,
        )

    def __setstate__(self, state):
        if "kmeans" in state:
            state = _migrate_legacy_state(state)
        super().__setstate__(state)


def _migrate_legacy_state(old: dict) -> dict:
    """Map a pre-0.10 TokenClusterizer `__dict__` onto TokenClusterer attributes."""
    kmeans = old["kmeans"]
    centers = kmeans.cluster_centers_
    order = old.get("cluster_order")
    feature_name = old.get("feature_name") or None
    state = {
        "n_clusters": int(kmeans.n_clusters),
        # Old files carrying fit_diagnostics_ were fit with the correlation heuristic.
        "ordering": "correlation" if "fit_diagnostics_" in old else "diff_abundance",
        "interpolation": old.get("interpolation", "nearest"),
        "name": feature_name,
        "random_state": kmeans.random_state,
        "kmeans_": kmeans,
        "cluster_centers_": centers,
        "cluster_order_": np.arange(len(centers)) if order is None else np.asarray(order),
        "n_clusters_": centers.shape[0],
        "n_features_in_": centers.shape[1],
        "grid_size_": tuple(int(v) for v in old["grid_size"]),
        "patch_size_": tuple(int(v) for v in old["patch_size"]),
        "model_name_": old.get("model_name"),
        "feature_name_": feature_name,
    }
    if "fit_diagnostics_" in old:
        state["fit_diagnostics_"] = old["fit_diagnostics_"]
    if "_sklearn_version" in old:
        state["_sklearn_version"] = old["_sklearn_version"]
    return state
