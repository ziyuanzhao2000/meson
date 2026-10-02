"""Binary tile labels from polygon annotations, by area coverage."""

from typing import TYPE_CHECKING, Optional

import numpy as np

from mesoslide._slides import DEFAULT_TILE_KEY, table_tile_geometries, tile_table_key

if TYPE_CHECKING:
    from wsidata import WSIData


def tile_coverage(wsi: "WSIData", polygons, *, tile_key: str = DEFAULT_TILE_KEY,
                  table_key: Optional[str] = None) -> np.ndarray:
    """
    Fraction of each tile's area covered by the union of `polygons`.

    Parameters
    ----------
    wsi : WSIData
    polygons : GeoDataFrame, GeoSeries or sequence of shapely geometries
        Level-0 pixel coordinates, e.g. from :func:`mesoslide.read_annotations`.
    tile_key : str, default='tiles'
    table_key : str, optional
        Defaults to ``f"{tile_key}_table"``.

    Returns
    -------
    coverage : ndarray of shape (n_tiles,), float64 in [0, 1], in tile-table row order
    """
    import shapely
    from shapely.strtree import STRtree

    tiles = np.asarray(table_tile_geometries(wsi, tile_key, table_key).values)
    geoms = getattr(polygons, "geometry", polygons)
    polys = np.asarray([g for g in geoms if g is not None and not g.is_empty], dtype=object)
    coverage = np.zeros(len(tiles), dtype=np.float64)
    if len(polys) == 0 or len(tiles) == 0:
        return coverage

    tile_idx, poly_idx = STRtree(polys).query(tiles, predicate="intersects")
    if len(tile_idx) == 0:
        return coverage
    order = np.argsort(tile_idx, kind="stable")
    tile_idx, poly_idx = tile_idx[order], poly_idx[order]
    hit_tiles, starts = np.unique(tile_idx, return_index=True)
    # Union of each hit tile's polygons, so overlapping annotations are not counted twice.
    unions = [shapely.union_all(polys[group]) for group in np.split(poly_idx, starts[1:])]
    hit = tiles[hit_tiles]
    coverage[hit_tiles] = shapely.area(shapely.intersection(hit, unions)) / shapely.area(hit)
    return coverage


def label_tiles(
    wsi: "WSIData",
    polygons,
    *,
    coverage_threshold: float,
    tile_key: str = DEFAULT_TILE_KEY,
    table_key: Optional[str] = None,
    key_added: Optional[str] = None,
) -> np.ndarray:
    """
    Label each tile positive when annotations cover more than `coverage_threshold` of it.

    Parameters
    ----------
    wsi : WSIData
    polygons : GeoDataFrame, GeoSeries or sequence of shapely geometries
        Level-0 pixel coordinates, e.g. from :func:`mesoslide.read_annotations`.
    coverage_threshold : float
        A tile is positive when covered area / tile area > this value.
    tile_key : str, default='tiles'
    table_key : str, optional
        Defaults to ``f"{tile_key}_table"``.
    key_added : str, optional
        Also write the labels to ``tiles_table.obs[key_added]``.

    Returns
    -------
    labels : ndarray of shape (n_tiles,), bool, in tile-table row order
    """
    table_key = table_key or tile_table_key(tile_key)
    labels = tile_coverage(wsi, polygons, tile_key=tile_key, table_key=table_key) > coverage_threshold
    if key_added is not None:
        wsi.tables[table_key].obs[key_added] = labels
    return labels
