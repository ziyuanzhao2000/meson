"""
Patch selection utilities for SAE feature analysis.

All functions return AnnData objects with consistent metadata columns:
    _source_patch_table : str   — which sdata element the patch came from
    _feature_name       : str   — which feature was used for ranking (where applicable)
    _feature_rank       : int   — rank of each patch within its feature (1 = best)
    _feature_score      : float — raw score value
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence, Union

import numpy as np
import anndata as ad

from meson._utils import get_patch_scores


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_table_names(patch_table_names: Union[str, Sequence[str]]) -> list[str]:
    if isinstance(patch_table_names, str):
        return [patch_table_names]
    return list(patch_table_names)


def _empty_result(sdata, table_names: list[str]) -> ad.AnnData:
    return sdata[table_names[0]][[]].copy()


def _build_output(
    sdata,
    table_names: list[str],
    selected_indices: dict[str, list[int]],
    selected_scores: Optional[dict[str, list[float]]] = None,
    extra_obs: Optional[dict[str, dict[str, list]]] = None,
    sort_by_score: bool = False,
) -> ad.AnnData:
    """
    Gather row subsets from sdata tables and concatenate them.

    Parameters
    ----------
    selected_indices : {table_name: [row_idx, ...]}
    selected_scores  : {table_name: [score, ...]}  optional
    extra_obs        : {col_name: {table_name: [value, ...]}}  optional
    sort_by_score    : if True, sort output by _feature_score descending
    """
    subsets = []
    for table_name in table_names:
        idx_list = selected_indices.get(table_name, [])
        if not idx_list:
            continue
        subset = sdata[table_name][np.asarray(idx_list, dtype=np.int64)].copy()
        subset.obs["_source_patch_table"] = table_name

        if selected_scores is not None:
            subset.obs["_feature_score"] = np.asarray(
                selected_scores[table_name], dtype=np.float32
            )

        if extra_obs is not None:
            for col, table_vals in extra_obs.items():
                if table_name in table_vals:
                    subset.obs[col] = table_vals[table_name]

        subsets.append(subset)

    if not subsets:
        return _empty_result(sdata, table_names)

    out = ad.concat(subsets, join="outer", merge="same")

    if sort_by_score and "_feature_score" in out.obs.columns:
        order = np.argsort(-out.obs["_feature_score"].to_numpy())
        out = out[order].copy()

    return out


# ---------------------------------------------------------------------------
# Public selection functions
# ---------------------------------------------------------------------------

def select_random_patches(
    sdata,
    patch_table_names: Union[str, Sequence[str]],
    n: int,
    random_state: Optional[int] = None,
) -> ad.AnnData:
    """
    Randomly sample n patches across one or more patch tables.

    Parameters
    ----------
    sdata : SpatialData
    patch_table_names : str or sequence of str
    n : int
    random_state : int, optional

    Returns
    -------
    AnnData with `.obs['_source_patch_table']`
    """
    table_names = _resolve_table_names(patch_table_names)

    if n < 0:
        raise ValueError("n must be >= 0.")
    if n == 0:
        return _empty_result(sdata, table_names)

    rng = np.random.default_rng(random_state)

    all_candidates: list[tuple[str, int]] = []
    for table_name in table_names:
        n_patches = len(sdata[table_name])
        all_candidates.extend((table_name, i) for i in range(n_patches))

    if not all_candidates:
        return _empty_result(sdata, table_names)

    n_to_sample = min(n, len(all_candidates))
    if n_to_sample < n:
        print(f"Warning: Only {len(all_candidates)} patches available, sampling all.")

    sampled = [
        all_candidates[i]
        for i in rng.choice(len(all_candidates), size=n_to_sample, replace=False)
    ]

    selected_indices: dict[str, list[int]] = defaultdict(list)
    for table_name, row_idx in sampled:
        selected_indices[table_name].append(row_idx)

    return _build_output(sdata, table_names, selected_indices)


def select_patches_for_binary_feature(
    sdata,
    patch_table_names: Union[str, Sequence[str]],
    feature_name: str,
    n: Optional[int] = None,
    random_state: Optional[int] = None,
    deprecated_rng = False
) -> ad.AnnData:
    """
    Sample patches where a binary feature (stored in .obs) equals 1.

    Parameters
    ----------
    sdata : SpatialData
    patch_table_names : str or sequence of str
    feature_name : str
        Column name in .obs
    n : int, optional
        Number to sample; None returns all active patches.
    random_state : int, optional

    Returns
    -------
    AnnData with `.obs['_source_patch_table']`
    """
    table_names = _resolve_table_names(patch_table_names)

    if n is not None and n < 0:
        raise ValueError("n must be >= 0 or None.")
    if n == 0:
        return _empty_result(sdata, table_names)

    all_active: list[tuple[str, int]] = []
    for table_name in table_names:
        patch_table = sdata[table_name]
        if feature_name not in patch_table.obs.columns:
            print(
                f"Warning: Feature '{feature_name}' not found in '{table_name}', "
                "skipping."
            )
            continue
        active_idx = np.where(patch_table.obs[feature_name] == 1)[0]
        all_active.extend((table_name, int(i)) for i in active_idx)

    if not all_active:
        raise ValueError(
            f"No active patches found for feature '{feature_name}' "
            f"in tables: {table_names}"
        )

    if n is None:
        selected = all_active
    else:
        num_active = len(all_active)
        n_to_sample = min(n, num_active)
        if n_to_sample < n:
            print(f"Warning: Only {num_active} active patches, sampling all.")
        # This is because numpy upgraded its random API and the old one is now deprecated, 
        # but we want to keep it around for reproducibility to get same results as in the paper
        if deprecated_rng:
            np.random.seed(random_state)
            indices = np.random.choice(num_active, size=n_to_sample, replace=False)
        else:  
            rng = np.random.default_rng(random_state)
            indices = rng.choice(num_active, size=n_to_sample, replace=False)
        selected = [all_active[i] for i in indices]

    selected_indices: dict[str, list[int]] = defaultdict(list)
    for table_name, row_idx in selected:
        selected_indices[table_name].append(row_idx)

    return _build_output(sdata, table_names, selected_indices)


def select_top_patches(
    sdata,
    patch_table_names: Union[str, Sequence[str]],
    feature_name: str,
    n: Optional[int] = None,
    min_score: Optional[float] = None,
    take_every: int = 1,
) -> ad.AnnData:
    """
    Select top-scoring patches for a feature across one or more patch tables,
    globally sorted by score descending.

    Parameters
    ----------
    sdata : SpatialData
    patch_table_names : str or sequence of str
    feature_name : str
    n : int, optional
        Hard cap on output size. None returns all qualifying patches (after stride).
    min_score : float, optional
        Minimum score threshold; defaults to 0 when n is None, -inf otherwise.
    take_every : int
        Stride through the score-sorted list before applying the n cap.

    Returns
    -------
    AnnData sorted by `_feature_score` descending.
    Adds `.obs['_source_patch_table']`, `.obs['_feature_name']`,
         `.obs['_feature_score']`, `.obs['_feature_rank']`.
    """
    table_names = _resolve_table_names(patch_table_names)

    if n is not None and n < 0:
        raise ValueError("n must be >= 0 or None.")
    if n == 0:
        return _empty_result(sdata, table_names)

    if min_score is None:
        min_score = 0.0 if n is None else float("-inf")

    all_candidates: list[tuple[float, str, int]] = []
    for table_name in table_names:
        patch_table = sdata[table_name]
        try:
            scores = get_patch_scores(patch_table, feature_name)
        except KeyError as exc:
            raise KeyError(f"{exc} (table='{table_name}')") from exc

        keep = np.where(scores > min_score)[0]
        all_candidates.extend(
            (float(scores[i]), table_name, int(i)) for i in keep
        )

    all_candidates.sort(key=lambda x: x[0], reverse=True)
    if take_every is None: 
        take_every = max(1, len(all_candidates) // n) if n is not None else 1
    strided = all_candidates[::take_every]
    selected = strided[:n] if n is not None else strided

    selected_indices: dict[str, list[int]] = defaultdict(list)
    selected_scores: dict[str, list[float]] = defaultdict(list)
    extra_rank: dict[str, list[int]] = defaultdict(list)
    extra_fname: dict[str, list[str]] = defaultdict(list)

    for rank, (score, table_name, row_idx) in enumerate(selected, start=1):
        selected_indices[table_name].append(row_idx)
        selected_scores[table_name].append(score)
        extra_rank[table_name].append(rank)
        extra_fname[table_name].append(feature_name)

    return _build_output(
        sdata,
        table_names,
        selected_indices,
        selected_scores=selected_scores,
        extra_obs={
            "_feature_rank": extra_rank,
            "_feature_name": extra_fname,
        },
        sort_by_score=True,
    )


def select_negative_patches(
    sdata,
    patch_table_names: Union[str, Sequence[str]],
    feature_name: str,
    n: Optional[int] = None,
    take_every: Optional[int] = None,
) -> ad.AnnData:
    """
    Select patches with zero score for a feature.

    Parameters
    ----------
    sdata : SpatialData
    patch_table_names : str or sequence of str
    feature_name : str
    n : int, optional
    take_every : int, optional
        Stride; auto-computed from n if None.

    Returns
    -------
    AnnData with `.obs['_source_patch_table']`
    """
    table_names = _resolve_table_names(patch_table_names)

    if n is not None and n < 0:
        raise ValueError("n must be >= 0 or None.")
    if n == 0:
        return _empty_result(sdata, table_names)

    all_candidates: list[tuple[str, int]] = []
    for table_name in table_names:
        patch_table = sdata[table_name]
        try:
            scores = get_patch_scores(patch_table, feature_name)
        except KeyError as exc:
            raise KeyError(f"{exc} (table='{table_name}')") from exc

        zero_idx = np.where(scores == 0)[0]
        all_candidates.extend((table_name, int(i)) for i in zero_idx)

    if not all_candidates:
        return _empty_result(sdata, table_names)

    if take_every is not None:
        stride = take_every
    elif n is not None:
        stride = max(1, len(all_candidates) // n)
    else:
        stride = 1

    strided = all_candidates[::stride]
    selected = strided[:n] if n is not None else strided

    selected_indices: dict[str, list[int]] = defaultdict(list)
    for table_name, row_idx in selected:
        selected_indices[table_name].append(row_idx)

    return _build_output(sdata, table_names, selected_indices)


# ---------------------------------------------------------------------------
# Exemplar patch selection
# ---------------------------------------------------------------------------

def select_exemplar_patches(
    sdata,
    patch_table_names: Union[str, Sequence[str]],
    feature_names: Sequence[str],
    n_exemplars: int = 1,
    min_score: float = 0.0,
) -> ad.AnnData:
    """
    For each feature, select the top-n_exemplars highest-scoring patches.

    This is the primary entry point for building exemplar galleries.
    Each output row has a `_feature_rank` column (1 = top patch) so that
    callers can filter to rank == 1 for a single representative image per
    feature, or keep all n_exemplars rows.

    Parameters
    ----------
    sdata : SpatialData
    patch_table_names : str or sequence of str
    feature_names : sequence of str
        e.g. ['UNI_SAE_123', 'UNI_SAE_456']
    n_exemplars : int
        Number of top patches to keep per feature. Default 1.
    min_score : float
        Minimum score to be considered as an exemplar. Default 0.

    Returns
    -------
    AnnData
        All exemplar rows concatenated. Columns added to .obs:
            _source_patch_table : str
            _feature_name       : str  — which feature this row was selected for
            _feature_rank       : int  — 1 = best patch for that feature
            _feature_score      : float

    Examples
    --------
    >>> exemplars = select_exemplar_patches(
    ...     sdata,
    ...     [f'{img}_grid_point_patch' for img in image_names],
    ...     feature_names=['UNI_SAE_123', 'UNI_SAE_456'],
    ...     n_exemplars=10,
    ... )
    >>> # get only the single best patch per feature
    >>> top1 = exemplars[exemplars.obs['_feature_rank'] == 1]
    """
    table_names = _resolve_table_names(patch_table_names)

    per_feature_adatas: list[ad.AnnData] = []
    for feature_name in feature_names:
        adata = select_top_patches(
            sdata,
            table_names,
            feature_name=feature_name,
            n=n_exemplars,
            min_score=min_score,
            take_every=1,
        )
        if len(adata) == 0:
            continue
        per_feature_adatas.append(adata)

    if not per_feature_adatas:
        return _empty_result(sdata, table_names)

    return ad.concat(per_feature_adatas, join="outer", merge="same")