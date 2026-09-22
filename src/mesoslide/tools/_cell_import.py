"""Ingest an external cell-segmentation mask and phenotyping table into a WSIData.

Two independent steps -- neither requires the other to have run first, except
that :func:`add_cell_phenotypes` needs cell polygons already in place to
attach phenotypes to:

- :func:`add_cell_polygons` reads a whole-slide instance-labeled OME-TIFF
  (e.g. from an MCMICRO/CyCIF pipeline) and writes cell polygons into
  ``wsidata.shapes[key_added]``, following the same shapes schema
  ``lazyslide.seg.cells`` itself produces (``geometry``, ``cell_id``), so
  downstream lazyslide/wsidata tooling treats it identically.
- :func:`add_cell_phenotypes` joins a per-cell phenotyping/quantification CSV
  (matched by cell ID, the standard MCMICRO convention) onto existing cell
  polygons, writing the phenotype label as a shapes column (what makes it
  visible to ``lazyslide.pl.WSIViewer(...).add_polygons(color_by=...)``) and
  the full table as a separate AnnData in ``wsidata.tables``.

The mask is read through :func:`ezslide.open_slide` / ``WSIData.read_region``
rather than a hand-rolled ``tifffile``/``zarr`` chunk reader, reusing
ezslide's existing lazy, tensorstore-backed reader (the same call the
originating notebook uses: ``ezslide.open_slide(path, attach_images=False)``
+ ``.read_region(x, y, w, h, level=0)``). A whole-slide instance mask is
typically tens of gigabytes at level 0 -- reading it in one call is what
produced a 2.29 TiB allocation and a `MemoryError` in that notebook -- so
large masks are read and polygonized in chunks instead.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import pandas as pd
from shapely.affinity import translate

if TYPE_CHECKING:
    import geopandas as gpd
    from anndata import AnnData
    from wsidata import WSIData

DEFAULT_CELLS_KEY = "cells"


def _chunk_boxes(height: int, width: int, chunk_size: int):
    """Yield (x, y, w, h) boxes tiling a (height, width) extent, row-major."""
    for y in range(0, height, chunk_size):
        h = min(chunk_size, height - y)
        for x in range(0, width, chunk_size):
            w = min(chunk_size, width - x)
            yield x, y, w, h


def _tile_boxes(tiles: "gpd.GeoDataFrame"):
    """Yield (x, y, w, h) boxes from a tiles shapes element's bounds."""
    bounds = tiles.bounds
    for minx, miny, maxx, maxy in bounds[["minx", "miny", "maxx", "maxy"]].to_numpy():
        yield int(minx), int(miny), int(maxx - minx), int(maxy - miny)


def _polygonize_chunk(mask_wsi: "WSIData", x: int, y: int, w: int, h: int, *,
                       halo: int, height: int, width: int,
                       min_area: float, min_hole_area: float, detect_holes: bool):
    """Read one (padded, clipped) chunk and polygonize it in global coordinates."""
    from lazyslide.cv import InstanceMap

    px0 = max(x - halo, 0)
    py0 = max(y - halo, 0)
    px1 = min(x + w + halo, width)
    py1 = min(y + h + halo, height)

    chunk = mask_wsi.read_region(px0, py0, px1 - px0, py1 - py0, level=0)
    if chunk.ndim > 2:
        chunk = chunk[..., 0]
    if not np.issubdtype(chunk.dtype, np.integer):
        chunk = chunk.astype(np.int64)

    if not np.any(chunk):
        return None

    gdf = InstanceMap(chunk).to_polygons(
        min_area=min_area, min_hole_area=min_hole_area, detect_holes=detect_holes,
    )
    if len(gdf) == 0:
        return None
    gdf["geometry"] = gdf["geometry"].apply(lambda p: translate(p, xoff=px0, yoff=py0))
    return gdf


def _polygonize_mask(
    mask_wsi: "WSIData",
    *,
    chunked: bool,
    boxes,
    halo: int,
    min_area: float,
    min_hole_area: float,
    detect_holes: bool,
    progress_bar: bool,
) -> "gpd.GeoDataFrame":
    from lazyslide.cv import InstanceMap
    import geopandas as gpd
    from tqdm.auto import tqdm

    height, width = mask_wsi.properties.shape

    if not chunked:
        arr = mask_wsi.read_region(0, 0, width, height, level=0)
        if arr.ndim > 2:
            arr = arr[..., 0]
        if not np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.int64)
        return InstanceMap(arr).to_polygons(
            min_area=min_area, min_hole_area=min_hole_area, detect_holes=detect_holes,
        )

    boxes = list(boxes)
    iterator = tqdm(boxes, desc="Polygonizing mask chunks") if progress_bar else boxes
    parts = []
    for x, y, w, h in iterator:
        part = _polygonize_chunk(
            mask_wsi, x, y, w, h, halo=halo, height=height, width=width,
            min_area=min_area, min_hole_area=min_hole_area, detect_holes=detect_holes,
        )
        if part is not None:
            parts.append(part)

    if not parts:
        raise ValueError("No cells were found in the mask.")

    merged = pd.concat(parts, ignore_index=True)
    dup_mask = merged["instance_id"].duplicated(keep=False)
    unique_part = merged.loc[~dup_mask]
    dup_part = merged.loc[dup_mask]
    if len(dup_part) == 0:
        return unique_part.reset_index(drop=True)

    merged_dup_geom = dup_part.groupby("instance_id")["geometry"].apply(
        lambda geoms: geoms.union_all().buffer(0)
    )
    dup_gdf = gpd.GeoDataFrame(
        {"instance_id": merged_dup_geom.index.to_numpy()},
        geometry=merged_dup_geom.to_numpy(),
    )
    return pd.concat([unique_part, dup_gdf], ignore_index=True)


def add_cell_polygons(
    wsidata: "WSIData",
    mask_path: str,
    *,
    key_added: str = DEFAULT_CELLS_KEY,
    chunked: Optional[bool] = None,
    tile_key: Optional[str] = None,
    chunk_size: int = 4096,
    halo: int = 32,
    min_area: float = 0.0,
    min_hole_area: float = 0.0,
    detect_holes: bool = True,
    max_inmemory_bytes: int = 2 * 1024**3,
    progress_bar: bool = True,
    overwrite: bool = False,
    save: bool = False,
) -> "WSIData":
    """Read a whole-slide instance-labeled cell mask into ``wsidata.shapes``.

    Writes a ``geometry``/``cell_id`` shapes element matching the schema
    ``lazyslide.seg.cells`` itself produces, so `zs.pl.WSIViewer` and other
    lazyslide/wsidata tooling treat it identically. No phenotype information
    is attached here -- see :func:`add_cell_phenotypes`.

    Parameters
    ----------
    wsidata : WSIData
        Slide to write the cell polygons into. Need not be the same slide
        the mask itself was generated from a copy of -- the mask is opened
        and read independently, and its pixel coordinates are assumed to
        already be in `wsidata`'s own level-0 pixel space (e.g. because the
        mask was produced by an MCMICRO-style pipeline registered to the
        same slide).
    mask_path : str
        Path to an instance-labeled OME-TIFF (or any format `ezslide.open_slide`
        can read), where background is label 0 and every other integer value
        identifies one cell.
    key_added : str, default='cells'
        `wsidata.shapes` key to write to.
    chunked : bool, optional
        Force chunked (True) or whole-array (False) reading. Omit to decide
        automatically from the mask's size vs. `max_inmemory_bytes`.
    tile_key : str, optional
        If given and `wsidata.shapes[tile_key]` exists, chunk on those tile
        boxes instead of a synthetic grid -- reuses geometry the slide
        already has and scopes work to tissue-containing regions. Left
        `None` (the default) when the slide has not been tiled yet, or when
        chunking should be independent of tiling state entirely.
    chunk_size : int, default=4096
        Chunk edge length in pixels, used when `tile_key` is `None` or
        `wsidata.shapes[tile_key]` does not exist.
    halo : int, default=32
        Padding (pixels) added to each chunk before reading, so a cell whose
        true extent crosses a chunk boundary is captured in more than one
        chunk's read and can be reunited by :func:`_polygonize_mask`'s
        instance-id merge, rather than truncated.
    min_area, min_hole_area, detect_holes
        Passed to `lazyslide.cv.InstanceMap.to_polygons`.
    max_inmemory_bytes : int, default=2 GiB
        Threshold (mask height * width * itemsize) above which `chunked`
        auto-decides True. This bounds how much *mask pixel data* is ever
        materialized at once (confirmed empirically: chunked reading keeps
        pixel-read memory in the hundreds of MB regardless of mask size,
        where a naive whole-mask read is what produces a multi-terabyte
        allocation on a real WSI-scale mask). It does **not** bound total
        memory for masks with very large cell counts: every extracted
        polygon is held in memory until the whole mask has been processed
        and written to `wsidata.shapes`, since the shapes element is the
        actual deliverable. On the real ~1.35M-cell mask this feature was
        built against, peak RSS was measured at ~9.8 GB -- well under the
        mask's own ~14 GB raw size (and the notebook's original whole-array
        `MemoryError`), but scaling with cell count, not with this
        parameter.
    progress_bar : bool, default=True
    overwrite : bool, default=False
        Allow replacing an existing `wsidata.shapes[key_added]`.
    save : bool, default=False
        Persist `key_added` via `wsidata.write_element(key_added,
        overwrite=True)`. Defaults to `False` -- unlike
        `run_model_stages`/`feature_extraction`'s `save=True` default,
        ingestion is typically one step of several (e.g. followed by
        :func:`add_cell_phenotypes`) before anything should hit disk, and
        `wsidata` must already be backed by a Zarr store (`wsidata.path`
        set, e.g. via a prior `wsidata.write(store)`) for this to succeed.

    Returns
    -------
    WSIData
        `wsidata`, mutated in place and also returned for chaining.

    Raises
    ------
    ValueError
        If `key_added` already exists and `overwrite=False`, or no cells
        were found in the mask.

    Examples
    --------
    >>> import mesoslide as ms
    >>> wsi = ms.tl.add_cell_polygons(wsi, "cellRing.ome.tif")
    >>> wsi.shapes["cells"].columns
    Index(['geometry', 'cell_id'], dtype='object')
    """
    import ezslide
    from wsidata.io import add_shapes

    if key_added in wsidata.shapes and not overwrite:
        raise ValueError(
            f"wsidata.shapes['{key_added}'] already exists; pass overwrite=True to replace it."
        )

    mask_wsi = ezslide.open_slide(mask_path, attach_images=False)
    try:
        height, width = mask_wsi.properties.shape
        itemsize = np.dtype(mask_wsi.reader.series.dtype).itemsize

        if chunked is None:
            chunked = (height * width * itemsize) > max_inmemory_bytes

        boxes = None
        if chunked:
            tiles = wsidata.shapes.get(tile_key) if tile_key is not None else None
            boxes = _tile_boxes(tiles) if tiles is not None else _chunk_boxes(
                height, width, chunk_size
            )

        gdf = _polygonize_mask(
            mask_wsi, chunked=chunked, boxes=boxes, halo=halo,
            min_area=min_area, min_hole_area=min_hole_area, detect_holes=detect_holes,
            progress_bar=progress_bar,
        )
    finally:
        mask_wsi.close()

    gdf = gdf[gdf["instance_id"] != 0].reset_index(drop=True)
    if len(gdf) == 0:
        raise ValueError(f"No cells found in mask '{mask_path}' (all labels were background).")

    gdf = gdf.rename(columns={"instance_id": "cell_id"})
    gdf["cell_id"] = gdf["cell_id"].astype(np.int64)
    gdf = gdf[["cell_id", "geometry"]]
    # TableModel's region annotation matches instance_key values against the
    # *index* of the shapes element it annotates, not merely a same-named
    # column (see tools/_feature_extraction.py::_build_table_for_slide) -- a
    # later add_cell_phenotypes table with instance_key="cell_id" needs this.
    gdf.index = gdf["cell_id"].to_numpy()

    add_shapes(wsidata, key_added, gdf)
    if save:
        wsidata.write_element(key_added, overwrite=True)
    return wsidata


def add_cell_phenotypes(
    wsidata: "WSIData",
    phenotype_csv: str,
    *,
    cells_key: str = DEFAULT_CELLS_KEY,
    id_col: str = "CellID",
    phenotype_col: Optional[str] = None,
    phenotype_out_col: str = "phenotype",
    marker_cols: Optional[Sequence[str]] = None,
    table_key: Optional[str] = None,
    drop_unmatched: bool = True,
    overwrite: bool = False,
    save: bool = False,
) -> "WSIData":
    """Join a per-cell phenotyping CSV onto existing cell polygons.

    Requires `wsidata.shapes[cells_key]` (from :func:`add_cell_polygons`) to
    already exist -- the two are independent steps by design, so this never
    triggers mask ingestion itself.

    Writes the phenotype label directly as a column on the shapes
    GeoDataFrame (`phenotype_out_col`, default `'phenotype'`) -- what makes
    it visible to `lazyslide.pl.WSIViewer(wsi).add_polygons(cells_key,
    color_by=phenotype_out_col)`, since `WSIViewer` reads `color_by` as a
    literal shapes column, not from a separate table -- and the full CSV as
    an `AnnData` table at `wsidata.tables[table_key]`, joined by `cell_id`.

    Parameters
    ----------
    wsidata : WSIData
    phenotype_csv : str
        Path to a per-cell CSV with an `id_col` column (default `'CellID'`,
        the standard MCMICRO/CyCIF quantification convention) matching the
        integer labels used to build `wsidata.shapes[cells_key]`.
    cells_key : str, default='cells'
    id_col : str, default='CellID'
    phenotype_col : str, optional
        Column in the CSV holding a categorical phenotype/cell-type label.
        If given, written onto the shapes GeoDataFrame as `phenotype_out_col`.
        If omitted, no shapes column is written (only the full table).
    phenotype_out_col : str, default='phenotype'
        Shapes column name for the written phenotype label. Pass `'class'`
        for literal parity with lazyslide's own segmentation-time class
        column, at the cost of conflating a richer phenotyping scheme with
        that narrower concept.
    marker_cols : sequence of str, optional
        CSV columns to route into the table's dense `.X` matrix (with `.var`
        indexed by these names), matching lazyslide's own dense-feature
        convention (`cells_features`) closely enough for `.X`-generic
        tooling. Omit to leave every column, including marker intensities,
        in `.obs` -- simpler, no naming assumptions, but invisible to
        `.X`-based tools.
    table_key : str, optional
        `wsidata.tables` key for the full phenotype/quantification table.
        Defaults to `f"{cells_key}_phenotypes"`.
    drop_unmatched : bool, default=True
        Drop cells with no matching CSV row rather than keeping them with a
        null phenotype.
    overwrite : bool, default=False
        Allow replacing an existing `table_key`.
    save : bool, default=False
        Persist the touched element(s) via `wsidata.write_element(...,
        overwrite=True)`: `table_key` always, plus `cells_key` when
        `phenotype_col` was given (since only then is the shapes element
        mutated). Requires `wsidata` already backed by a Zarr store.

    Returns
    -------
    WSIData
        `wsidata`, mutated in place and also returned for chaining.

    Raises
    ------
    KeyError
        If `cells_key` (or its `cell_id` column) is missing -- run
        :func:`add_cell_polygons` first.
    ValueError
        If `id_col` is missing from the CSV, or `table_key` already exists
        and `overwrite=False`.
    """
    from anndata import AnnData
    from spatialdata.models import TableModel
    from wsidata.io import add_shapes

    if cells_key not in wsidata.shapes:
        raise KeyError(
            f"wsidata.shapes has no '{cells_key}'. Run mesoslide.tl.add_cell_polygons "
            f"first; available shapes: {list(wsidata.shapes)}"
        )
    cells_gdf = wsidata.shapes[cells_key]
    if "cell_id" not in cells_gdf.columns:
        raise KeyError(f"wsidata.shapes['{cells_key}'] has no 'cell_id' column.")

    table_key = table_key or f"{cells_key}_phenotypes"
    if table_key in wsidata.tables and not overwrite:
        raise ValueError(
            f"wsidata.tables['{table_key}'] already exists; pass overwrite=True to replace it."
        )

    df = pd.read_csv(phenotype_csv)
    if id_col not in df.columns:
        raise ValueError(
            f"'{id_col}' not found in {phenotype_csv}; available columns: {list(df.columns)}"
        )
    df = df.rename(columns={id_col: "cell_id"})
    df["cell_id"] = df["cell_id"].astype(np.int64)

    how = "inner" if drop_unmatched else "left"
    n_before = len(cells_gdf)
    joined = cells_gdf.merge(df, on="cell_id", how=how)
    if drop_unmatched and len(joined) < n_before:
        warnings.warn(
            f"{n_before - len(joined)} of {n_before} cells had no matching row in "
            f"{phenotype_csv} and were dropped.",
            UserWarning,
            stacklevel=2,
        )

    if phenotype_col is not None:
        if phenotype_col not in df.columns:
            raise ValueError(
                f"phenotype_col='{phenotype_col}' not found in {phenotype_csv}; "
                f"available columns: {list(df.columns)}"
            )
        updated = cells_gdf.merge(
            df[["cell_id", phenotype_col]], on="cell_id", how="left",
        )
        updated = updated.rename(columns={phenotype_col: phenotype_out_col})
        updated.index = updated["cell_id"].to_numpy()
        add_shapes(wsidata, cells_key, updated)
        if save:
            wsidata.write_element(cells_key, overwrite=True)

    obs_cols = [c for c in df.columns if marker_cols is None or c not in marker_cols]
    obs = joined[obs_cols].copy()
    obs["library_id"] = pd.Categorical([cells_key] * len(obs))
    obs.index = obs["cell_id"].astype(str).to_numpy()

    X, var = None, None
    if marker_cols:
        X = joined[list(marker_cols)].to_numpy(dtype=np.float32)
        var = pd.DataFrame(index=pd.Index(marker_cols, name="marker"))

    table = TableModel.parse(
        AnnData(X=X, obs=obs, var=var),
        region=cells_key, region_key="library_id", instance_key="cell_id",
    )
    wsidata.tables[table_key] = table
    if save:
        wsidata.write_element(table_key, overwrite=True)
    return wsidata


def add_cells(
    wsidata: "WSIData",
    *,
    mask_path: Optional[str] = None,
    phenotype_csv: Optional[str] = None,
    key_added: str = DEFAULT_CELLS_KEY,
    mask_kwargs: Optional[dict] = None,
    phenotype_kwargs: Optional[dict] = None,
) -> "WSIData":
    """Convenience wrapper: run whichever of `add_cell_polygons`/`add_cell_phenotypes` apply.

    Equivalent to calling the two functions directly; kept for callers who
    have both inputs on hand at once. See each function's own docstring for
    parameters -- pass slide-specific overrides via `mask_kwargs`/
    `phenotype_kwargs`.
    """
    if mask_path is None and phenotype_csv is None:
        raise ValueError("At least one of mask_path or phenotype_csv must be given.")
    if mask_path is not None:
        wsidata = add_cell_polygons(wsidata, mask_path, key_added=key_added, **(mask_kwargs or {}))
    if phenotype_csv is not None:
        wsidata = add_cell_phenotypes(
            wsidata, phenotype_csv, cells_key=key_added, **(phenotype_kwargs or {})
        )
    return wsidata
