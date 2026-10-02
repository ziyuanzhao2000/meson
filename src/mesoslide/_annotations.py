"""Read polygon annotations (GeoJSON, QuPath/OMERO point-list CSV, WKT CSV) into a GeoDataFrame."""

from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd

_POINT_COLUMNS = ("all_points", "Points")
_NAME_COLUMNS = ("name", "Text", "Name")


def _polygonal(geom):
    """Repair `geom` and keep only its polygonal parts (Polygon or MultiPolygon), or None."""
    from shapely import make_valid
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import unary_union

    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    parts = [g for g in getattr(geom, "geoms", []) if isinstance(g, (Polygon, MultiPolygon)) and not g.is_empty]
    return unary_union(parts) if parts else None


def _largest_part(geom):
    if geom is not None and geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda p: p.area)
    return geom


def _infer_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".geojson", ".json"):
        return "geojson"
    if suffix == ".csv":
        columns = pd.read_csv(path, nrows=0).columns
        if "geometry" in columns:
            return "csv_wkt"
        if any(c in columns for c in _POINT_COLUMNS):
            return "csv_points"
        raise ValueError(
            f"Cannot tell the CSV layout of {path}: expected a 'geometry' (WKT) column "
            f"or one of {_POINT_COLUMNS}."
        )
    raise ValueError(f"Cannot infer the annotation format of {path}; pass format=.")


def read_annotations(
    path: Union[str, Path],
    *,
    format: Optional[str] = None,
    scale: Tuple[float, float] = (1.0, 1.0),
    name_filter: Optional[Union[str, Callable[[pd.Series], np.ndarray]]] = None,
    name_col: Optional[str] = None,
    swap_xy: bool = False,
    largest_component: bool = False,
):
    """
    Read polygon annotations into a GeoDataFrame in level-0 pixel coordinates.

    Geometries are repaired with ``shapely.make_valid`` and reduced to their
    polygonal parts; empty or non-polygonal annotations are dropped.

    Parameters
    ----------
    path : str or Path
    format : {'geojson', 'csv_points', 'csv_wkt'}, optional
        Inferred from the extension and, for CSV, the columns:

        - 'geojson': any file geopandas reads (e.g. QuPath GeoJSON export).
        - 'csv_points': one row per polygon with a space-separated
          ``"x,y x,y ..."`` vertex list in 'all_points' or 'Points'
          (e.g. OMERO ROI exports, :func:`mesoslide.xml2csv`).
        - 'csv_wkt': one row per polygon with a WKT 'geometry' column.
    scale : (sx, sy), default=(1, 1)
        Multiplies coordinates (about the origin), e.g. 4 for annotations
        drawn on a 4x downsampled image.
    name_filter : str or callable, optional
        Keep rows whose name starts with this string, or for which the
        callable (given the name Series) returns True.
    name_col : str, optional
        Column holding annotation names. Defaults to the first of 'name',
        'Text', 'Name' present; output always has a 'name' column.
    swap_xy : bool, default=False
        Swap x and y after loading (some exports store row/column order).
    largest_component : bool, default=False
        Reduce each MultiPolygon to its largest polygon.

    Returns
    -------
    geopandas.GeoDataFrame
        With 'name' and 'geometry' columns (plus any others from the source),
        index reset.
    """
    import geopandas as gpd
    from shapely import wkt
    from shapely.affinity import scale as scale_geom
    from shapely.ops import transform

    from mesoslide._utils import points2poly

    path = Path(path)
    format = format or _infer_format(path)

    if format == "geojson":
        gdf = gpd.read_file(path)
    elif format in ("csv_points", "csv_wkt"):
        df = pd.read_csv(path)
        if format == "csv_wkt":
            geometry = df.pop("geometry").map(wkt.loads)
        else:
            col = next((c for c in _POINT_COLUMNS if c in df.columns), None)
            if col is None:
                raise ValueError(f"{path} has none of the point-list columns {_POINT_COLUMNS}.")
            geometry = df.pop(col).map(points2poly)
        gdf = gpd.GeoDataFrame(df, geometry=geometry.to_list())
    else:
        raise ValueError(f"format must be 'geojson', 'csv_points' or 'csv_wkt', got {format!r}")

    name_col = name_col or next((c for c in _NAME_COLUMNS if c in gdf.columns), None)
    names = gdf[name_col] if name_col is not None else pd.Series("", index=gdf.index)
    gdf["name"] = names.fillna("").astype(str)

    if name_filter is not None:
        keep = (gdf["name"].str.startswith(name_filter) if isinstance(name_filter, str)
                else np.asarray(name_filter(gdf["name"]), dtype=bool))
        gdf = gdf[keep]

    sx, sy = scale
    geoms = []
    for geom in gdf.geometry:
        if geom is not None and (sx, sy) != (1, 1):
            geom = scale_geom(geom, xfact=sx, yfact=sy, origin=(0, 0))
        if geom is not None and swap_xy:
            geom = transform(lambda x, y, z=None: (y, x), geom)
        geom = _polygonal(geom)
        geoms.append(_largest_part(geom) if largest_component else geom)
    gdf = gdf.set_geometry(gpd.GeoSeries(geoms, index=gdf.index))
    return gdf[gdf.geometry.notna()].reset_index(drop=True)
