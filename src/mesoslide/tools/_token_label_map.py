"""Whole-slide token label map from per-tile labels cached by feature_extraction."""

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from mesoslide._interpolation import assemble_token_map
from mesoslide._slides import DEFAULT_TILE_KEY, table_tile_geometries, tile_table_key

if TYPE_CHECKING:
    from wsidata import WSIData


def token_label_map(
    wsi: "WSIData",
    key: str,
    *,
    tile_key: str = DEFAULT_TILE_KEY,
    table_key: Optional[str] = None,
    grid_size: Optional[Tuple[int, int]] = None,
    downsample: float = 1,
    fill_margins: bool = True,
    background: int = 0,
) -> np.ndarray:
    """
    Stitch the per-tile token labels in ``tiles_table.obsm[key]`` into one map.

    `key` holds one flat row of token labels per tile, as written by
    ``feature_extraction(dense=True, reducer=labeler.reducer(), dense_key_added=key)``
    with a :class:`TokenClassifier` or :class:`TokenClusterer`. Tile size and
    stride are read from ``wsi.tile_spec(tile_key)``; see
    :func:`mesoslide.assemble_token_map` for how overlapping tiles are combined.

    Parameters
    ----------
    wsi : WSIData
    key : str
        obsm key of the per-tile labels, shape (n_tiles, grid_h * grid_w).
    tile_key : str, default='tiles'
    table_key : str, optional
        Defaults to ``f"{tile_key}_table"``.
    grid_size : (grid_h, grid_w), optional
        Token grid; inferred as square when omitted.
    downsample : float, default=1
        Output resolution relative to level 0.
    fill_margins : bool, default=True
    background : int, default=0

    Returns
    -------
    label_map : ndarray of shape (round(H0 / downsample), round(W0 / downsample)), dtype uint8
    """
    table_key = table_key or tile_table_key(tile_key)
    table = wsi.tables[table_key]
    if key not in table.obsm:
        raise KeyError(f"'{key}' not in {table_key}.obsm; available: {list(table.obsm)}")
    labels = np.asarray(table.obsm[key])
    if labels.ndim != 2:
        raise ValueError(f"Expected {table_key}.obsm['{key}'] of shape (n_tiles, n_tokens), got {labels.shape}")
    n_tokens = labels.shape[1]
    if grid_size is None:
        side = int(round(np.sqrt(n_tokens)))
        if side * side != n_tokens:
            raise ValueError(f"Cannot infer a square token grid from {n_tokens} tokens; pass grid_size.")
        grid_size = (side, side)
    if grid_size[0] * grid_size[1] != n_tokens:
        raise ValueError(f"grid_size {grid_size} does not match {n_tokens} tokens per tile.")
    if labels.min() < 0 or labels.max() > 255:
        raise ValueError("Token labels must lie in [0, 255].")

    spec = wsi.tile_spec(tile_key)
    if spec.base_width != spec.base_height or spec.base_stride_width != spec.base_stride_height:
        raise ValueError("token_label_map requires square tiles and equal x/y strides.")

    bounds = table_tile_geometries(wsi, tile_key, table_key).bounds
    tile_xy = bounds[["minx", "miny"]].to_numpy()
    grids = labels.astype(np.uint8).reshape(len(labels), *grid_size)
    return assemble_token_map(
        tile_xy, grids, int(spec.base_width), int(spec.base_stride_width),
        shape=tuple(wsi.properties.shape), downsample=downsample,
        fill_margins=fill_margins, background=background,
    )
