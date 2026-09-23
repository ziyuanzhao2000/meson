"""Per-patch cell phenotype extraction, analogous to `extract_patch_cells`."""

from typing import TYPE_CHECKING, Optional

import anndata as ad
import numpy as np
import pandas as pd
from spatialdata.models import TableModel

from mesoslide._slides import DEFAULT_TILE_KEY, SLIDE_ID
from ._extract_patches import _resolve_slides, _resolve_slides_from_ref
from ._extract_cells import extract_patch_cells

if TYPE_CHECKING:
    from mesoslide._patch_data import PatchData


def extract_patch_cell_phenotypes(
    patches: "PatchData",
    slides=None,
    *,
    cells_key: str = "cells",
    table_key: Optional[str] = None,
    tile_key: str = DEFAULT_TILE_KEY,
    patch_id_col: str = "patch_idx",
    progress_bar: bool = True,
    cache: bool = True,
    overwrite: bool = False,
) -> "ad.AnnData":
    """One concatenated AnnData spanning every patch's cells, tagged by which patch each row came from.

    If already cached in `patches.tables[table_key]` (e.g. from a previous
    call with `cache=True`), it's returned directly, unless `overwrite=True`. Otherwise, calls
    :func:`extract_patch_cells` (always with `cache=True`, to warm that cache
    regardless of this function's own `cache` argument -- mirroring
    `extract_cluster_maps`'s own unconditional
    `extract_patch_images(..., cache=True)` call) to determine which
    `cell_id`s belong to each patch, then subsets `wsidata.tables[table_key]`
    (from whichever slide holds `cells_key`) by those ids per patch and
    concatenates the results, following the same recipe
    `mesoslide._slides.concat_slides` uses for cohort-level concatenation
    (one combined `AnnData`, `join="outer"`, a provenance column).

    Parameters
    ----------
    patches : PatchData
        Selected tiles, e.g. from :func:`mesoslide.select_top_patches`.
    slides : WSIData, list of WSIData, or {slide_id: WSIData}, optional
        The slide(s) holding `wsidata.shapes[cells_key]`/`wsidata.tables[table_key]`.
        Defaults to `patches.obs['_slide_ref']`.
    cells_key : str, default='cells'
        Also the `region` this result's table is linked to in `patches.shapes`.
    table_key : str, optional
        `wsidata.tables` key holding the per-cell phenotype/quantification
        table, e.g. written by :func:`mesoslide.tl.add_cell_phenotypes`.
        Defaults to `f"{cells_key}_phenotypes"`, matching that function's own
        default. Also the key this result is cached under in `patches.tables`.
    tile_key : str, default='tiles'
        Forwarded to :func:`extract_patch_cells` for patch sizing.
    patch_id_col : str, default='patch_idx'
        `.obs` column identifying which patch (0-based position in
        `patches.obs`) each row came from.
    progress_bar : bool, default=True
    cache : bool, default=True
        Store the result in `patches.tables[table_key]` -- a real,
        serializable SpatialData table element, linked to
        `patches.shapes[cells_key]` via `region`/`region_key`/`instance_key`.
        Survives `patches.write(...)`/`read_patch_data(...)`.
    overwrite : bool, default=False
        Recompute even if `table_key` is already cached in `patches.tables`,
        replacing the cached value (when `cache=True`). Does not force
        `extract_patch_cells`'s own cache to be recomputed -- pass
        `overwrite=True` to that function directly for that.

    Returns
    -------
    anndata.AnnData
        Every patch's cells' phenotype rows, concatenated, with
        `patch_id_col` identifying provenance. A patch with no cells
        contributes no rows.

    Raises
    ------
    KeyError
        If `table_key` is missing from a touched slide's tables.

    Examples
    --------
    >>> import mesoslide as ms
    >>> table = ms.pp.extract_patch_cell_phenotypes(patches, cells_key="cells")
    >>> table.obs.groupby("patch_idx").size()
    """
    table_key = table_key or f"{cells_key}_phenotypes"
    if table_key in patches.tables and not overwrite:
        return patches.tables[table_key]

    cell_gdf = extract_patch_cells(
        patches, slides, cells_key=cells_key, tile_key=tile_key,
        patch_id_col=patch_id_col, progress_bar=progress_bar, cache=True,
    )

    opened: list = []
    try:
        if slides is not None:
            cells_slide_map = _resolve_slides(slides)
        else:
            cells_slide_map, opened = _resolve_slides_from_ref(patches)
        cells_single = set(cells_slide_map) == {None}
        patch_df = patches.obs

        parts = []
        for patch_id, group in cell_gdf.groupby(patch_id_col):
            cells_slide_id = None if cells_single else patch_df.iloc[patch_id][SLIDE_ID]
            wsi = cells_slide_map[cells_slide_id]
            if table_key not in wsi.tables:
                raise KeyError(
                    f"wsidata.tables has no '{table_key}' for slide '{cells_slide_id}'. "
                    "Run mesoslide.tl.add_cell_phenotypes first."
                )
            table = wsi.tables[table_key]
            cell_ids = group["cell_id"].astype(np.int64).to_numpy()
            mask = table.obs["cell_id"].astype(np.int64).isin(cell_ids)
            sub = table[mask].copy()
            sub.obs[patch_id_col] = patch_id
            parts.append(sub)
    finally:
        for wsi in opened:
            try:
                wsi.close()
            except Exception:
                pass

    if parts:
        result = ad.concat(parts, join="outer", merge="same", index_unique="-")
    else:
        result = ad.AnnData()

    if cache:
        if len(result) == 0:
            raise ValueError(
                "No patch has any cells -- an empty result can't be cached "
                "in patches.tables (spatialdata.models.TableModel expects a "
                "non-empty, index-aligned table). Call with cache=False to "
                "get the empty AnnData back without caching it."
            )
        # A cell straddling a patch boundary appears once per patch it
        # intersects (see extract_patch_cells), so `cell_id` alone can repeat
        # -- TableModel's instance_key must be unique per region, so a fresh
        # per-row id is used instead. `cell_id` stays as a plain column.
        result.obs["cell_id"] = result.obs["cell_id"].astype(np.int64)
        result.obs["_row_uid"] = np.arange(len(result)).astype(str)
        result.obs_names = result.obs["_row_uid"].to_numpy()
        result.obs["_region_key"] = pd.Categorical([cells_key] * len(result))
        patches.tables[table_key] = TableModel.parse(
            result, region=cells_key, region_key="_region_key", instance_key="_row_uid",
        )
        return patches.tables[table_key]
    return result
