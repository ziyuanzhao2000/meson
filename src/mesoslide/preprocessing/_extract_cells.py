"""Per-patch cell polygon extraction, analogous to `extract_patch_images`/`extract_cluster_maps`."""

from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd
from shapely.geometry import box as shapely_box
from spatialdata.models import ShapesModel
from tqdm.auto import tqdm

from mesoslide._slides import DEFAULT_TILE_KEY, SLIDE_ID
from ._extract_patches import _resolve_slides, _resolve_slides_from_ref, _tile_size

if TYPE_CHECKING:
    from mesoslide._patch_data import PatchData


def cells_in_patch(cells_gdf: "gpd.GeoDataFrame", x: int, y: int, w: int, h: int) -> "gpd.GeoDataFrame":
    """`cells_gdf` rows whose geometry intersects the (x, y, w, h) box, in WSI pixel space."""
    box = shapely_box(x, y, x + w, y + h)
    idx = cells_gdf.sindex.query(box, predicate="intersects")
    return cells_gdf.iloc[idx]


def extract_patch_cells(
    patches: "PatchData",
    slides=None,
    *,
    cells_key: str = "cells",
    tile_key: str = DEFAULT_TILE_KEY,
    patch_id_col: str = "patch_idx",
    progress_bar: bool = True,
    cache: bool = True,
    overwrite: bool = False,
) -> "gpd.GeoDataFrame":
    """One row per cell across every patch, tagged by which patch it came from.

    If already cached in `patches.shapes[cells_key]` (e.g. from a previous
    call with `cache=True`), it's returned directly -- no slide reads, no
    spatial filtering -- unless `overwrite=True`. Otherwise, for each patch this filters
    `wsidata.shapes[cells_key]` (from whichever `WSIData` holds it -- may
    differ from the slide the patch table was tiled on, e.g. a
    separately-registered CyCIF segmentation slide) down to the cells whose
    geometry intersects that patch's box, then concatenates every patch's
    subset into one `GeoDataFrame`.

    Coordinates stay in WSI/level-0 pixel space -- shifting to patch-local
    pixel coordinates (`mesoslide.plotting._cell_overlay.translate_to_patch_local`)
    is a plotting-time concern, not an extraction one.

    Parameters
    ----------
    patches : PatchData
        Selected tiles, e.g. from :func:`mesoslide.select_top_patches`.
        Required `.obs` columns: `x`, `y` (level-0 top-left tile corner).
    slides : WSIData, list of WSIData, or {slide_id: WSIData}, optional
        The slide(s) holding `wsidata.shapes[cells_key]`. Defaults to
        `patches.obs['_slide_ref']`.
    cells_key : str, default='cells'
        `wsidata.shapes` key holding cell polygons, e.g. written by
        :func:`mesoslide.tl.add_cell_polygons`. Also the key this result is
        cached under in `patches.shapes`.
    tile_key : str, default='tiles'
        Tile shapes key on the *reference* slide (`patches.obs['_slide_ref']`)
        used to size each patch -- see :func:`extract_patch_images`. This is
        independent of `cells_key`'s own slide, since a cells-owning slide
        (e.g. a segmentation slide ingested separately) is not guaranteed to
        have been tiled itself.
    patch_id_col : str, default='patch_idx'
        Column added to the result identifying which patch (0-based position
        in `patches.obs`) each row came from.
    progress_bar : bool, default=True
    cache : bool, default=True
        Store the result in `patches.shapes[cells_key]` -- a real,
        serializable SpatialData shapes element, unlike the `.uns`-based
        caching `extract_patch_images`/`extract_cluster_maps` use for
        `.obsm`. Survives `patches.write(...)`/`read_patch_data(...)`.
    overwrite : bool, default=False
        Recompute even if `cells_key` is already cached in `patches.shapes`,
        replacing the cached value (when `cache=True`).

    Returns
    -------
    geopandas.GeoDataFrame
        Every patch's intersecting cells, concatenated, with `patch_id_col`
        identifying provenance. A patch with no intersecting cells
        contributes no rows.

    Raises
    ------
    KeyError
        If `cells_key` is missing from a touched slide's shapes.
    ValueError
        If `patches.obs` is missing required columns, or if no patch has any
        intersecting cells (an empty result can't be cached -- SpatialData's
        `ShapesModel` disallows an empty shapes element).

    Examples
    --------
    >>> import mesoslide as ms
    >>> cell_gdf = ms.pp.extract_patch_cells(patches, cells_key="cells")
    >>> cell_gdf.groupby("patch_idx").size()
    """
    if cells_key in patches.shapes and not overwrite:
        return patches.shapes[cells_key]

    opened: list = []
    try:
        if slides is not None:
            cells_slide_map = _resolve_slides(slides)
        else:
            cells_slide_map, opened = _resolve_slides_from_ref(patches)
        cells_single = set(cells_slide_map) == {None}

        ref_slide_map, ref_opened = _resolve_slides_from_ref(patches)
        opened.extend(ref_opened)
        ref_single = set(ref_slide_map) == {None}
        sizes = {sid: _tile_size(wsi, tile_key) for sid, wsi in ref_slide_map.items()}

        patch_df = patches.obs
        required = ["x", "y"]
        if not cells_single or not ref_single:
            required.append(SLIDE_ID)
        missing = [c for c in required if c not in patch_df.columns]
        if missing:
            raise ValueError(f"patches.obs missing required columns: {missing}")

        iterator = tqdm(range(len(patch_df)), desc="Extracting patch cells") if progress_bar else range(len(patch_df))

        parts = []
        for i in iterator:
            patch = patch_df.iloc[i]
            cells_slide_id = None if cells_single else patch[SLIDE_ID]
            ref_slide_id = None if ref_single else patch[SLIDE_ID]
            wsi = cells_slide_map[cells_slide_id]
            if cells_key not in wsi.shapes:
                raise KeyError(
                    f"wsidata.shapes has no '{cells_key}' for slide '{cells_slide_id}'. "
                    "Run mesoslide.tl.add_cell_polygons first."
                )
            h, w = sizes[ref_slide_id]
            x, y = int(patch.x), int(patch.y)

            part = cells_in_patch(wsi.shapes[cells_key], x, y, w, h)
            if len(part) == 0:
                continue
            part = part.copy()
            part[patch_id_col] = i
            parts.append(part)
    finally:
        for wsi in opened:
            try:
                wsi.close()
            except Exception:
                pass

    if parts:
        result = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry")
    else:
        result = gpd.GeoDataFrame(columns=["geometry", patch_id_col])

    if cache:
        if len(result) == 0:
            raise ValueError(
                "No patch has any intersecting cells -- an empty result can't "
                "be cached in patches.shapes (spatialdata.models.ShapesModel "
                "disallows an empty shapes element). Call with cache=False to "
                "get the empty GeoDataFrame back without caching it."
            )
        patches.shapes[cells_key] = ShapesModel.parse(result)
        return patches.shapes[cells_key]
    return result
