"""Group sparse-coding features (e.g. by cluster) into per-group .obs scores.

A cohort of SAE/LLC features is often clustered (see FeatureClusterer) into a
handful of interpretable groups. Downstream analyses then want one summary
score per group per tile, not per feature -- e.g. the max of a cluster's
member features' (normalised) activations. This module computes that summary
directly from each slide's per-feature scores, following the same two-pass
streaming pattern as FeatureClusterer.compute_iou: pass 1 (only needed when
normalising) finds each feature's cohort-wide max, pass 2 aggregates.
"""

import numpy as np
from tqdm.auto import tqdm
from typing import Optional


class FeatureGroupAggregator:
    """
    Aggregate sparse-coding feature scores into per-group summary columns in
    ``.obs`` (e.g. collapsing SAE features assigned to the same cluster into
    one 'SAE_cluster_3' score per patch).

    Follows the fit/transform pattern used by FeatureClusterer.compute_iou:
    fit() builds the group -> features inverse index and, if normalising,
    each feature's cohort-wide max, in one pass; transform() is a second,
    cheap pass that gathers each group's member scores and aggregates them.

    Parameters
    ----------
    method : {"max", "sum"}, default "max"
        "max" takes the per-tile maximum across a group's member features.
        "sum" takes the per-tile sum.
    normalize : bool or None, default None
        Divide each feature by its cohort-wide max (computed in fit()) before
        aggregating. None defaults to True for method="max" (scores must be
        comparable across differently-scaled features before taking a max)
        and False for method="sum" (summing raw, unnormalised activations of
        differently-scaled features is rarely meaningful, so it is opt-in).

    Attributes
    ----------
    feature_to_group_ : dict[int, int]
        The fitted feature -> group mapping, set after fit().
    group_to_features_ : dict[int, list[int]]
        Inverse of feature_to_group_, built once in fit().
    global_max_ : dict[int, float] or None
        Per-feature cohort-wide max; set in fit() only when normalize=True.

    Examples
    --------
    >>> agg = FeatureGroupAggregator(method="max")
    >>> agg.fit(slides, feature_prefix="UNI_SAE", feature_to_group=feature_to_group)
    >>> agg.transform(slides, group_prefix="SAE_cluster")
    """

    def __init__(self, method: str = "max", normalize: "bool | None" = None):
        if method not in ("max", "sum"):
            raise ValueError(f"method must be 'max' or 'sum', got {method!r}")
        self.method = method
        self.normalize = (method == "max") if normalize is None else normalize

        # set after fit()
        self.feature_to_group_: Optional[dict] = None
        self.group_to_features_: Optional[dict] = None
        self.feature_prefix_: Optional[str] = None
        self.global_max_: Optional[dict] = None
        self._is_fitted = False

    # ── public API ─────────────────────────────────────────────────────────

    def fit(self, slides, feature_prefix: str, feature_to_group: "dict[int, int]", *,
            tile_key: str = "tiles", progress: bool = True) -> "FeatureGroupAggregator":
        """
        Build the group -> features inverse index and, if normalising, each
        feature's cohort-wide max, in one pass over `slides`.

        Parameters
        ----------
        slides : AnnData, WSIData, or sequence/mapping/manifest of either
            Patch-level table(s) with sparse SAE/LLC scores in .X (or copied
            to .obs), as accepted by SlideSource.
        feature_prefix : str
            Prefix of feature var_names, e.g. 'UNI_SAE'.
        feature_to_group : dict[int, int]
            Maps feature_idx -> group_id.
        tile_key : str, default='tiles'
        progress : bool

        Returns
        -------
        self
        """
        from mesoslide._slides import SlideSource
        from mesoslide._utils import get_patch_scores

        self.feature_to_group_ = {int(k): int(v) for k, v in feature_to_group.items()}
        if not self.feature_to_group_:
            raise ValueError("feature_to_group must be non-empty.")
        self.feature_prefix_ = feature_prefix

        # Inverse index, built once here rather than looked up per group per
        # slide -- avoids repeatedly scanning feature_to_group during transform.
        group_to_features: dict = {}
        for feature_idx, group_id in self.feature_to_group_.items():
            group_to_features.setdefault(group_id, []).append(feature_idx)
        self.group_to_features_ = group_to_features

        if self.normalize:
            source = slides if isinstance(slides, SlideSource) else SlideSource(slides, tile_key=tile_key)
            global_max = {f: 0.0 for f in self.feature_to_group_}
            it = tqdm(source, desc="Aggregator fit (global max)") if progress else source
            for _, table in it:
                for f in global_max:
                    scores = get_patch_scores(table, f"{feature_prefix}_{f}")
                    if scores.size:
                        global_max[f] = max(global_max[f], float(scores.max()))
            # guard div-by-zero, same pattern as FeatureClusterer._start
            self.global_max_ = {f: (m if m != 0 else 1.0) for f, m in global_max.items()}
        else:
            self.global_max_ = None

        self._is_fitted = True
        return self

    def transform(self, slides, *, group_prefix: str = "group",
                  tile_key: str = "tiles", write: bool = False,
                  progress: bool = True) -> "dict":
        """
        Gather each group's member feature scores and aggregate them into
        one ``.obs[f"{group_prefix}_{group_id}"]`` column per slide.

        Parameters
        ----------
        slides : AnnData, WSIData, or sequence/mapping/manifest of either
            Same slides fit() was called on, or a new cohort to apply the
            fitted normalisation constants to.
        group_prefix : str, default 'group'
        tile_key : str, default='tiles'
        write : bool, default False
            If True, persist the updated table back to each slide's store
            via `slide.write_element(table_key, overwrite=True)`. Requires
            `slides` to be a WSIData or a list/tuple of WSIData (mirrors
            SparseAutoencoder.transform / LocalityConstrainedCoding.transform).
        progress : bool

        Returns
        -------
        dict[str | None, AnnData]
            slide_id -> mutated patch table, keyed as SlideSource yields
            (None for a single bare AnnData input).
        """
        from mesoslide._slides import SlideSource, tile_table_key
        from mesoslide._utils import get_patch_scores

        self._check_fitted()

        if write:
            slide_list = slides if isinstance(slides, (list, tuple)) else [slides]
            if not all(hasattr(s, "tables") for s in slide_list):
                raise TypeError(
                    "write=True requires `slides` to be a WSIData or list/tuple of WSIData."
                )
            table_key = tile_table_key(tile_key)
            pairs = [(getattr(s, "name", None), s, s.tables[table_key]) for s in slide_list]
        else:
            source = slides if isinstance(slides, SlideSource) else SlideSource(slides, tile_key=tile_key)
            pairs = [(slide_id, None, table) for slide_id, table in source]

        results = {}
        it = tqdm(pairs, desc="Aggregating feature groups") if progress else pairs
        for slide_id, wsi, table in it:
            for group_id, feats in self.group_to_features_.items():
                stacked = np.empty((len(feats), table.n_obs), dtype=np.float64)
                for row, f in enumerate(feats):
                    scores = get_patch_scores(table, f"{self.feature_prefix_}_{f}")
                    if self.normalize:
                        scores = scores / self.global_max_[f]
                    stacked[row] = scores
                agg = stacked.max(axis=0) if self.method == "max" else stacked.sum(axis=0)
                table.obs[f"{group_prefix}_{group_id}"] = agg

            results[slide_id] = table
            if write:
                table_key = tile_table_key(tile_key)
                wsi.tables[table_key] = table
                wsi.write_element(table_key, overwrite=True)

        return results

    def fit_transform(self, slides, feature_prefix: str, feature_to_group: "dict[int, int]", *,
                       group_prefix: str = "group", tile_key: str = "tiles",
                       write: bool = False, progress: bool = True) -> "dict":
        """fit() then transform() over the same `slides`. See fit()/transform() for parameters."""
        return self.fit(slides, feature_prefix, feature_to_group,
                         tile_key=tile_key, progress=progress).transform(
            slides, group_prefix=group_prefix, tile_key=tile_key,
            write=write, progress=progress,
        )

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")


def aggregate_feature_groups(slides, feature_prefix: str, feature_to_group: "dict[int, int]", *,
                              method: str = "max", normalize: "bool | None" = None,
                              group_prefix: str = "group", tile_key: str = "tiles",
                              write: bool = False, progress: bool = True) -> "dict":
    """
    One-call convenience wrapper: fit_transform() a fresh FeatureGroupAggregator.

    See FeatureGroupAggregator for parameter docs.

    Returns
    -------
    dict[str | None, AnnData]
        slide_id -> mutated patch table, same as FeatureGroupAggregator.transform().
    """
    return FeatureGroupAggregator(method=method, normalize=normalize).fit_transform(
        slides, feature_prefix, feature_to_group,
        group_prefix=group_prefix, tile_key=tile_key, write=write, progress=progress,
    )
