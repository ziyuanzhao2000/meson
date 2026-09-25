"""Patch selection for SAE feature analysis.

Each SpatialData now holds one WSI, so slides are no longer picked out by
mangled element names (``{image}_grid_point_patch``). Every selector takes
``slides``: a single tile table, one ``WSIData``, a sequence or mapping of
either, or a cohort manifest DataFrame.

The per-slide loop is deliberate. Selection reads one score vector per slide
(0.39 MB for a 48568-tile slide) and copies only the rows it selects, so a
40-slide cohort costs megabytes. Concatenating the cohort first would copy
~8 GB of embeddings that selection never looks at -- see :mod:`mesoslide._slides`.

All functions return a `PatchData` (a `spatialdata.SpatialData`-backed patch
table; `.obs`/`.obsm`/`.X`/`.var` proxy the same way a bare AnnData already
did) with consistent metadata columns:
    slide_id       : str   -- which slide the patch came from
    _feature_name  : str   -- feature used for ranking (where applicable)
    _feature_rank  : int   -- rank within its feature (1 = best)
    _feature_score : float -- raw score value
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence, Union
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import anndata as ad
import geopandas as gpd
import shapely.geometry
from spatialdata.models import ShapesModel, TableModel

from mesoslide._utils import get_patch_scores
from mesoslide._slides import SLIDE_ID, SLIDE_REF, DEFAULT_TILE_KEY, SlideSource
from mesoslide._patch_data import PatchData, TILES_KEY, TILES_TABLE_KEY
from mesoslide._deprecated import (
    SLIDES_HINT,
    check_not_spatialdata,
    deprecated_kwargs,
    removed,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _source(slides, tile_key: str, func_name: str = "select") -> SlideSource:
    if isinstance(slides, SlideSource):
        return slides
    check_not_spatialdata(slides, func_name)
    return SlideSource(slides, tile_key=tile_key)


def _empty_result(source: SlideSource) -> ad.AnnData:
    """A deliberate exception to "every selector returns PatchData":
    `spatialdata.models.ShapesModel` unconditionally disallows an empty
    shapes element (`len(geometry) == 0` always raises, even via
    `.parse()`), so a genuinely empty selection (nothing matched) cannot be
    represented as a `PatchData` at all. Returns a bare, empty AnnData
    instead -- unaffected by the PatchData migration, since `source.first()`
    already returns a plain per-slide table, not a wrapped object.
    """
    return source.first()[[]].copy()


def _parse_patch_data(table: "ad.AnnData", tiles: "gpd.GeoDataFrame") -> "PatchData":
    """Wrap one row-aligned `(table, tiles)` pair into a validated `PatchData`.

    Adds/refreshes a `_patch_uid` obs column mirroring `table.obs_names`, and
    uses it as SpatialData's `instance_key` (rather than the original
    `tile_id` column) -- `tile_id` restarts at 0 per slide, so after a
    multi-slide `ad.concat(..., index_unique="-")` re-suffixes `obs_names`,
    the raw `tile_id` values would silently stop matching the shapes
    element's (also re-indexed) index. `_patch_uid` is *defined* to equal
    `obs_names`, so it stays correct through any such re-suffixing.
    """
    table = table.copy()
    table.obs["_patch_uid"] = table.obs_names.to_numpy()
    table.obs["library_id"] = pd.Categorical([TILES_KEY] * len(table))
    tiles = tiles.copy()
    tiles.index = table.obs_names
    return PatchData(
        shapes={TILES_KEY: ShapesModel.parse(tiles)},
        tables={TILES_TABLE_KEY: TableModel.parse(
            table, region=TILES_KEY, region_key="library_id", instance_key="_patch_uid",
            overwrite_metadata=True,
        )},
    )


def _concat_patch_data(parts: list) -> "PatchData":
    """Merge single-slide `PatchData` objects into one.

    Tile ids restart at 0 on every slide, so the table half is concatenated
    exactly as a bare AnnData always was (`ad.concat(..., index_unique="-")`
    to avoid duplicate obs_names); the shapes half is `pd.concat`-ed in the
    same part order (so row position stays aligned) and then re-parsed via
    `_parse_patch_data`, which derives a fresh `_patch_uid`/index from the
    table's own post-concat `obs_names` rather than trying to independently
    replicate `ad.concat`'s own suffixing scheme for `tile_id`.
    """
    tables = [p.tables[TILES_TABLE_KEY] for p in parts]
    merged_table = ad.concat(tables, join="outer", merge="same", index_unique="-")
    merged_tiles = gpd.GeoDataFrame(
        pd.concat([p.shapes[TILES_KEY] for p in parts], ignore_index=True),
        geometry="geometry",
    )
    return _parse_patch_data(merged_table, merged_tiles)


def _build_output(
    source: SlideSource,
    selected_indices: dict,
    selected_scores: Optional[dict] = None,
    extra_obs: Optional[dict] = None,
    sort_by_score: bool = False,
):
    """Materialise the selected rows, one slide at a time.

    This is the piece that keeps memory bounded: it copies only selected rows,
    and for a manifest-backed source it re-reads each slide rather than holding
    the cohort.

    Parameters
    ----------
    selected_indices : {slide_id: [row_idx, ...]}
    selected_scores  : {slide_id: [score, ...]}  optional
    extra_obs        : {col_name: {slide_id: [value, ...]}}  optional
    sort_by_score    : sort output by _feature_score descending

    Returns
    -------
    PatchData, or a bare empty AnnData if nothing matched (see `_empty_result`).
    """
    subsets = []
    for slide_id, table, tiles, ref in source.iter_with_ref():
        idx_list = selected_indices.get(slide_id, [])
        if len(idx_list) == 0:
            continue
        idx_arr = np.asarray(idx_list, dtype=np.int64)
        subset_table = table[idx_arr].copy()
        if tiles is not None:
            subset_tiles = tiles.iloc[idx_arr]
        else:
            # A bare AnnData input (no owning WSIData/shapes) carries no
            # known tile size -- fall back to a 1x1-pixel placeholder box at
            # each row's own (x, y), just enough to satisfy PatchData's
            # structural requirement for a shapes element. Not meaningful
            # geometry; pass the owning WSIData/a PatchData instead of a
            # bare table when real tile shapes are needed.
            x = subset_table.obs["x"].to_numpy()
            y = subset_table.obs["y"].to_numpy()
            subset_tiles = gpd.GeoDataFrame(
                {"tile_id": np.arange(len(subset_table))},
                geometry=[shapely.geometry.box(xi, yi, xi + 1, yi + 1) for xi, yi in zip(x, y)],
            )

        # A PatchData input carries slide_id None: it may already have a
        # slide_id column that we must not overwrite.
        if slide_id is not None:
            subset_table.obs[SLIDE_ID] = slide_id

        # Object-dtype Series, not a bare scalar assignment -- a bare `ref`
        # (a WSIData) would otherwise get interpreted as array-like by pandas.
        subset_table.obs[SLIDE_REF] = pd.Series(
            [ref] * len(subset_table), index=subset_table.obs.index, dtype=object,
        )

        if selected_scores is not None:
            subset_table.obs["_feature_score"] = np.asarray(
                selected_scores[slide_id], dtype=np.float32
            )

        if extra_obs is not None:
            for col, slide_vals in extra_obs.items():
                if slide_id in slide_vals:
                    subset_table.obs[col] = slide_vals[slide_id]

        subsets.append(_parse_patch_data(subset_table, subset_tiles))

    if not subsets:
        return _empty_result(source)

    out = subsets[0] if len(subsets) == 1 else _concat_patch_data(subsets)

    if sort_by_score and "_feature_score" in out.obs.columns:
        order = np.argsort(-out.obs["_feature_score"].to_numpy())
        out = out[order]

    return out


def _scored_candidates(source: SlideSource, feature_name: str, keep):
    """Stream slides, applying `keep(scores) -> row indices` to each.

    Returns (scores, slide_ids, row_indices) as parallel arrays. Holding these
    as arrays rather than a list of tuples is what makes a cohort-wide sort
    affordable: ~10 bytes per candidate instead of ~80.
    """
    all_scores, all_slides, all_rows = [], [], []
    slide_order = []
    for slide_id, table in source:
        try:
            scores = get_patch_scores(table, feature_name)
        except KeyError as exc:
            raise KeyError(f"{exc} (slide={slide_id!r})") from exc
        idx = keep(np.asarray(scores))
        if len(idx) == 0:
            continue
        slide_order.append(slide_id)
        all_scores.append(np.asarray(scores, dtype=np.float64)[idx])
        all_rows.append(np.asarray(idx, dtype=np.int64))
        all_slides.append(np.full(len(idx), len(slide_order) - 1, dtype=np.int32))

    if not all_scores:
        return (np.empty(0), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int64), [])
    return (
        np.concatenate(all_scores),
        np.concatenate(all_slides),
        np.concatenate(all_rows),
        slide_order,
    )


def _group(slide_codes, rows, slide_order, scores=None):
    """Regroup flat selection arrays back into per-slide index lists."""
    indices = defaultdict(list)
    values = defaultdict(list)
    for k in range(len(rows)):
        sid = slide_order[slide_codes[k]]
        indices[sid].append(int(rows[k]))
        if scores is not None:
            values[sid].append(float(scores[k]))
    return indices, values


# ---------------------------------------------------------------------------
# Public selection functions
# ---------------------------------------------------------------------------

@deprecated_kwargs(
    patch_table_names=removed(SLIDES_HINT),
    sdata=removed(SLIDES_HINT),
)
def select_random_patches(
    slides,
    n: int,
    *,
    tile_key: str = DEFAULT_TILE_KEY,
    random_state: Optional[int] = None,
) -> "PatchData":
    """
    Randomly sample n patches across one or more slides.

    Parameters
    ----------
    slides : AnnData, WSIData, sequence/mapping of either, or slides_table
    n : int
    tile_key : str
    random_state : int, optional

    Returns
    -------
    AnnData with `.obs['slide_id']`
    """
    source = _source(slides, tile_key, "select_random_patches")

    if n < 0:
        raise ValueError("n must be >= 0.")
    if n == 0:
        return _empty_result(source)

    rng = np.random.default_rng(random_state)

    sizes, slide_order = [], []
    for slide_id, table in source:
        slide_order.append(slide_id)
        sizes.append(len(table))

    total = int(np.sum(sizes))
    if total == 0:
        return _empty_result(source)

    n_to_sample = min(n, total)
    if n_to_sample < n:
        print(f"Warning: Only {total} patches available, sampling all.")

    flat = rng.choice(total, size=n_to_sample, replace=False)
    # Map flat cohort-wide positions back to (slide, row) without materialising
    # a per-patch candidate list.
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    slide_codes = np.searchsorted(offsets, flat, side="right") - 1
    rows = flat - offsets[slide_codes]

    selected_indices, _ = _group(slide_codes, rows, slide_order)
    return _build_output(source, selected_indices)


@deprecated_kwargs(
    patch_table_names=removed(SLIDES_HINT),
    sdata=removed(SLIDES_HINT),
)
def select_patches_for_binary_feature(
    slides,
    feature_name: str,
    n: Optional[int] = None,
    *,
    tile_key: str = DEFAULT_TILE_KEY,
    random_state: Optional[int] = None,
    deprecated_rng: bool = False,
) -> "PatchData":
    """
    Sample patches where a binary feature (stored in .obs) equals 1.

    Parameters
    ----------
    slides : AnnData, WSIData, sequence/mapping of either, or slides_table
    feature_name : str
        Column name in .obs
    n : int, optional
        Number to sample; None returns all active patches.
    tile_key : str
    random_state : int, optional
    deprecated_rng : bool
        Use numpy's legacy global RNG, to reproduce the published results.

    Returns
    -------
    AnnData with `.obs['slide_id']`
    """
    source = _source(slides, tile_key, "select_patches_for_binary_feature")

    if n is not None and n < 0:
        raise ValueError("n must be >= 0 or None.")
    if n == 0:
        return _empty_result(source)

    active_rows, active_codes, slide_order = [], [], []
    for slide_id, table in source:
        if feature_name not in table.obs.columns:
            print(
                f"Warning: Feature '{feature_name}' not found in slide "
                f"{slide_id!r}, skipping."
            )
            continue
        idx = np.where(table.obs[feature_name].to_numpy() == 1)[0]
        if len(idx) == 0:
            continue
        slide_order.append(slide_id)
        active_rows.append(idx.astype(np.int64))
        active_codes.append(np.full(len(idx), len(slide_order) - 1, dtype=np.int32))

    if not active_rows:
        raise ValueError(
            f"No active patches found for feature '{feature_name}' across the given slides."
        )

    rows = np.concatenate(active_rows)
    codes = np.concatenate(active_codes)

    if n is not None:
        num_active = len(rows)
        n_to_sample = min(n, num_active)
        if n_to_sample < n:
            print(f"Warning: Only {num_active} active patches, sampling all.")
        # numpy's legacy global RNG is kept available so published figures stay
        # reproducible against the original results.
        if deprecated_rng:
            np.random.seed(random_state)
            pick = np.random.choice(num_active, size=n_to_sample, replace=False)
        else:
            pick = np.random.default_rng(random_state).choice(
                num_active, size=n_to_sample, replace=False
            )
        rows, codes = rows[pick], codes[pick]

    selected_indices, _ = _group(codes, rows, slide_order)
    return _build_output(source, selected_indices)


@deprecated_kwargs(
    patch_table_names=removed(SLIDES_HINT),
    sdata=removed(SLIDES_HINT),
)
def select_top_patches(
    slides,
    feature_name: str,
    n: Optional[int] = None,
    *,
    tile_key: str = DEFAULT_TILE_KEY,
    min_score: Optional[float] = None,
    take_every: Optional[int] = None,
    top_fraction: Optional[float] = None,
) -> "PatchData":
    """
    Select top-scoring patches for a feature across slides, globally sorted
    by score descending.

    Parameters
    ----------
    slides : AnnData, WSIData, sequence/mapping of either, or slides_table
    feature_name : str
    n : int, optional
        Hard cap on output size. None returns all qualifying patches (after stride).
        Required when top_fraction is set.
    tile_key : str
    min_score : float, optional
        Minimum score threshold; defaults to 0 when n is None or top_fraction is
        set, -inf otherwise.
    take_every : int, optional
        Keep every `take_every`-th patch of the score-sorted pool, then apply
        the n cap (take_every=1 gives the top n). None (default) spreads
        min(n, pool size) picks evenly over the whole pool.
    top_fraction : float, optional
        Restrict the pool to the top fraction (0, 1] of qualifying (score >
        min_score) patches, e.g. 0.1 keeps only the top 10% by score, before
        take_every/n are applied. Requires n.

    Returns
    -------
    AnnData sorted by `_feature_score` descending, with `.obs['slide_id']`,
    `.obs['_feature_name']`, `.obs['_feature_score']`, `.obs['_feature_rank']`.
    """
    source = _source(slides, tile_key, "select_top_patches")

    if n is not None and n < 0:
        raise ValueError("n must be >= 0 or None.")
    if n == 0:
        return _empty_result(source)

    if take_every is not None and take_every < 1:
        raise ValueError("take_every must be >= 1 or None.")
    if top_fraction is not None:
        if not (0 < top_fraction <= 1):
            raise ValueError("top_fraction must be in (0, 1].")
        if n is None:
            raise ValueError("n is required when top_fraction is set.")

    if min_score is None:
        if top_fraction is not None:
            min_score = 0.0
        else:
            min_score = 0.0 if n is None else float("-inf")

    scores, codes, rows, slide_order = _scored_candidates(
        source, feature_name, lambda s: np.where(s > min_score)[0]
    )
    if len(scores) == 0:
        return _empty_result(source)

    order = np.argsort(-scores, kind="stable")
    scores, codes, rows = scores[order], codes[order], rows[order]

    if top_fraction is not None:
        top_count = max(1, int(np.ceil(top_fraction * len(scores))))
        scores, codes, rows = scores[:top_count], codes[:top_count], rows[:top_count]

    if take_every is not None:
        pick = np.arange(0, len(scores), take_every)[:n]
    elif n is not None:
        # Spread picks over the pool; at most one pick per row.
        pick = np.linspace(0, len(scores) - 1, min(n, len(scores))).astype(np.int64)
    else:
        pick = np.arange(len(scores))
    scores, codes, rows = scores[pick], codes[pick], rows[pick]

    selected_indices, selected_scores = _group(codes, rows, slide_order, scores)

    extra_rank, extra_fname = defaultdict(list), defaultdict(list)
    for rank, code in enumerate(codes, start=1):
        sid = slide_order[code]
        extra_rank[sid].append(rank)
        extra_fname[sid].append(feature_name)

    return _build_output(
        source,
        selected_indices,
        selected_scores=selected_scores,
        extra_obs={"_feature_rank": extra_rank, "_feature_name": extra_fname},
        sort_by_score=True,
    )


@deprecated_kwargs(
    patch_table_names=removed(SLIDES_HINT),
    sdata=removed(SLIDES_HINT),
)
def select_negative_patches(
    slides,
    feature_name: str,
    n: Optional[int] = None,
    *,
    tile_key: str = DEFAULT_TILE_KEY,
    take_every: Optional[int] = None,
) -> "PatchData":
    """
    Select patches with zero score for a feature.

    Parameters
    ----------
    slides : AnnData, WSIData, sequence/mapping of either, or slides_table
    feature_name : str
    n : int, optional
    tile_key : str
    take_every : int, optional
        Stride; auto-computed from n if None.

    Returns
    -------
    AnnData with `.obs['slide_id']`
    """
    source = _source(slides, tile_key, "select_negative_patches")

    if n is not None and n < 0:
        raise ValueError("n must be >= 0 or None.")
    if n == 0:
        return _empty_result(source)

    _, codes, rows, slide_order = _scored_candidates(
        source, feature_name, lambda s: np.where(s == 0)[0]
    )
    if len(rows) == 0:
        return _empty_result(source)

    if take_every is not None:
        stride = take_every
    elif n is not None:
        stride = max(1, len(rows) // n)
    else:
        stride = 1

    codes, rows = codes[::stride], rows[::stride]
    if n is not None:
        codes, rows = codes[:n], rows[:n]

    selected_indices, _ = _group(codes, rows, slide_order)
    return _build_output(source, selected_indices)


# ---------------------------------------------------------------------------
# Exemplar patch selection
# ---------------------------------------------------------------------------

@deprecated_kwargs(
    patch_table_names=removed(SLIDES_HINT),
    sdata=removed(SLIDES_HINT),
)
def select_exemplar_patches(
    slides,
    feature_names: Sequence[str],
    n_exemplars: int = 1,
    *,
    tile_key: str = DEFAULT_TILE_KEY,
    min_score: float = 0.0,
) -> "PatchData":
    """
    For each feature, select the top-n_exemplars highest-scoring patches.

    This is the primary entry point for building exemplar galleries. Each output
    row has a `_feature_rank` column (1 = top patch) so callers can filter to
    rank == 1 for a single representative image per feature.

    Parameters
    ----------
    slides : AnnData, WSIData, sequence/mapping of either, or slides_table
    feature_names : sequence of str
        e.g. ['UNI_SAE_123', 'UNI_SAE_456']
    n_exemplars : int
        Number of top patches to keep per feature. Default 1.
    tile_key : str
    min_score : float
        Minimum score to be considered an exemplar. Default 0.

    Returns
    -------
    AnnData
        All exemplar rows concatenated, with .obs columns slide_id,
        _feature_name, _feature_rank, _feature_score.

    Examples
    --------
    >>> exemplars = select_exemplar_patches(
    ...     manifest, feature_names=['UNI_SAE_123', 'UNI_SAE_456'], n_exemplars=10
    ... )
    >>> top1 = exemplars[exemplars.obs['_feature_rank'] == 1]

    Notes
    -----
    A manifest-backed `slides` is re-read once per feature. Pass an in-memory
    mapping (see :func:`mesoslide.open_slides`) when scanning many features.
    """
    source = _source(slides, tile_key, "select_exemplar_patches")

    per_feature = []
    for feature_name in tqdm(feature_names):
        adata = select_top_patches(
            source,
            feature_name,
            n=n_exemplars,
            min_score=min_score,
            take_every=1,
        )
        if len(adata) == 0:
            continue
        per_feature.append(adata)

    if not per_feature:
        return _empty_result(source)

    return per_feature[0] if len(per_feature) == 1 else _concat_patch_data(per_feature)
