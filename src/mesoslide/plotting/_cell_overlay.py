"""Draw cell polygons onto an already-extracted patch image array.

Unlike `lazyslide.pl.WSIViewer` -- which draws live vector `PathPatch`
artists onto a persistent matplotlib Axes calibrated to WSI pixel space via
`imshow(..., extent=...)` -- `GalleryPlan` works with plain numpy patch
crops that carry no coordinate metadata of their own (see
`preprocessing._extract_patches.extract_patch_images`). These helpers
therefore rasterize polygons into an RGBA numpy overlay and alpha-composite
it onto an existing frame array, rather than drawing vector artists, so a
cell-overlay row is just another `np.ndarray` frame like every other
`GalleryPlan` row -- `render()` needs no changes to display it.
"""

from typing import Dict, Optional, Tuple, Union

import cv2
import distinctipy
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from shapely.affinity import affine_transform
from shapely.geometry import box as shapely_box

#: lazyslide's own default qualitative palette (`lazyslide.plotting._wsi_viewer`),
#: reused here for visual parity with `lazyslide.pl.WSIViewer` at low cardinality.
LAZYSLIDE_PALETTE = (
    "#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5",
    "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0",
)

#: Fixed seed for `distinctipy.get_colors`, used for every categorical
#: palette this module resolves beyond `LAZYSLIDE_PALETTE`'s size -- so a
#: given category count always gets the same colors across calls/sessions,
#: not a fresh random draw each time.
_DISTINCT_COLOR_SEED = 0


def cells_in_patch(cells_gdf, x: int, y: int, w: int, h: int):
    """`cells_gdf` rows whose geometry intersects the (x, y, w, h) box, in WSI pixel space."""
    box = shapely_box(x, y, x + w, y + h)
    idx = cells_gdf.sindex.query(box, predicate="intersects")
    return cells_gdf.iloc[idx]


def translate_to_patch_local(gdf, x: int, y: int, downsample: float = 1.0):
    """Shift `gdf`'s geometry from WSI pixel space to patch-local pixel space.

    `downsample` accounts for a patch frame read at reduced resolution
    relative to the polygons' own (level-0) coordinates.
    """
    scale = 1.0 / downsample
    matrix = [scale, 0, 0, scale, -x * scale, -y * scale]
    out = gdf.copy()
    out["geometry"] = out["geometry"].apply(lambda g: affine_transform(g, matrix))
    return out


def resolve_categorical_palette(values, palette: Optional[Dict] = None) -> Dict:
    """category -> RGBA, for `values` (a column of, typically, string labels).

    Reuses `LAZYSLIDE_PALETTE` for low cardinality (matching lazyslide's own
    default look), then falls back to `distinctipy.get_colors` -- unlike a
    fixed-size discrete colormap (`tab10`/`tab20`), this scales to any
    category count without two categories ever colliding on the same color,
    at the cost of visual distinctness degrading gracefully as the count
    grows very large (an inherent property of the color space, not a bug).
    A fixed seed (`_DISTINCT_COLOR_SEED`) makes the assignment deterministic
    across calls for the same category count.
    """
    categories = pd.Categorical(values)
    cats = list(categories.categories)
    if palette is not None:
        missing = [c for c in cats if c not in palette]
        if missing:
            raise ValueError(f"palette is missing colors for categories: {missing}")
        return {c: to_rgba(palette[c]) for c in cats}
    if len(cats) <= len(LAZYSLIDE_PALETTE):
        return {c: to_rgba(LAZYSLIDE_PALETTE[i]) for i, c in enumerate(cats)}
    colors = distinctipy.get_colors(len(cats), rng=_DISTINCT_COLOR_SEED)
    return {c: to_rgba(colors[i]) for i, c in enumerate(cats)}


def rasterize_cell_polygons(
    frame_shape: Tuple[int, int],
    cells_local_gdf,
    *,
    color_by: Optional[str] = None,
    palette: Optional[Dict] = None,
    fill_alpha: float = 0.35,
    edge_color: Optional[Union[str, Tuple]] = "white",
    edge_only: bool = False,
    linewidth: float = 1.0,
    supersample: int = 4,
) -> np.ndarray:
    """Rasterize `cells_local_gdf` (patch-local pixel coordinates) into an RGBA overlay.

    Draws at `supersample` x `frame_shape` resolution via `cv2.fillPoly`/
    `cv2.polylines`, then downsamples with `cv2.INTER_AREA` -- cheap
    anti-aliasing that keeps edges reasonably crisp without a live vector
    renderer.

    Returns
    -------
    np.ndarray
        `(H, W, 4)` float array in `[0, 1]`, `H, W = frame_shape`.
    """
    h, w = frame_shape
    sh, sw = h * supersample, w * supersample
    overlay = np.zeros((sh, sw, 4), dtype=np.float32)
    if len(cells_local_gdf) == 0:
        return overlay

    resolved_palette = None
    if color_by is not None:
        resolved_palette = resolve_categorical_palette(
            cells_local_gdf[color_by], palette
        )
        fill_colors = [resolved_palette[v] for v in cells_local_gdf[color_by]]
    else:
        single = to_rgba(palette) if isinstance(palette, str) else to_rgba("#FFE31A")
        fill_colors = [single] * len(cells_local_gdf)

    edge_rgba = to_rgba(edge_color) if edge_color is not None else None

    for geom, color in zip(cells_local_gdf.geometry, fill_colors):
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            exterior = (np.asarray(poly.exterior.coords) * supersample).astype(np.int32)
            if not edge_only:
                fc = (*color[:3], fill_alpha)
                cv2.fillPoly(overlay, [exterior], color=fc)
                for ring in poly.interiors:
                    hole = (np.asarray(ring.coords) * supersample).astype(np.int32)
                    cv2.fillPoly(overlay, [hole], color=(0, 0, 0, 0))
            if edge_rgba is not None:
                lw = max(1, int(round(linewidth * supersample)))
                cv2.polylines(overlay, [exterior], isClosed=True, color=edge_rgba, thickness=lw)

    return cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)


def alpha_composite(base_rgb: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    """Standard over-compositing of `overlay_rgba` onto `base_rgb`, both float `[0, 1]`."""
    alpha = overlay_rgba[..., 3:4]
    return base_rgb * (1 - alpha) + overlay_rgba[..., :3] * alpha
