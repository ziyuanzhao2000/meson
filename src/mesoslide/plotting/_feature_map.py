"""Render a per-tile feature over its slide image."""

import math

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.pyplot as plt

from mesoslide._interpolation import interpolate_patch_max
from mesoslide._slides import DEFAULT_TILE_KEY, tile_table_key
from mesoslide._deprecated import ELEMENT_NAME_HINT, deprecated_kwargs, drop, removed

_RENDERING_HINT = (
    "plot_feature_map no longer renders via spatialdata_plot; there is only "
    "one rendering path now (a rasterized max-over-overlap heatmap)."
)


def _axes_device_px(ax):
    """Estimate the axes size in device pixels without requiring a prior draw.

    Uses ``get_position() x figsize x dpi``, valid as soon as the axes exists
    (unlike ``get_window_extent``, which needs a renderer) -- mirrors
    lazyslide's ``WSIViewer._axes_device_px``.
    """
    fig = ax.get_figure()
    if fig is None:
        return None, None
    pos = ax.get_position()
    fw, fh = fig.get_size_inches()
    return max(1.0, pos.width * fw * fig.dpi), max(1.0, pos.height * fh * fig.dpi)


def _render_slide_background(ax, wsi, *, image_size=2000, oversample=1.5):
    """Draw a display-resolution slide image behind the tile overlay.

    Reads via wsi.reader.get_region at the pyramid level that resolves the
    actual rendered axes size (times `oversample`), rather than always
    reading a fixed-size thumbnail regardless of the figure size -- mirrors
    how lazyslide.pl.tiles picks its background resolution. Works whether or
    not the slide was opened with attach_images=True, same as the thumbnail
    path it replaces.
    """
    from ezslide import resolve_display_level

    props = wsi.properties
    h0, w0 = props.shape  # level 0, NOT image.shape
    axes_px_w, axes_px_h = _axes_device_px(ax)
    level, downsample = resolve_display_level(props, axes_px_w, axes_px_h, oversample=oversample)

    while (
        (w0 / downsample > image_size or h0 / downsample > image_size)
        and level < props.n_level - 1
    ):
        level += 1
        downsample = props.level_downsample[level]

    dw = max(1, math.ceil(w0 / downsample))
    dh = max(1, math.ceil(h0 / downsample))
    image = wsi.reader.get_region(0, 0, dw, dh, level=level)

    ax.imshow(image, extent=[0, w0, h0, 0], origin="upper", zorder=-100)
    # Force the full-slide extent regardless of what the tile overlay
    # autoscaled the axes to.
    ax.set_xlim(0, w0)
    ax.set_ylim(h0, 0)


def _tile_centers_and_values(wsi, tile_key, table_key, feature_name):
    """Return (xs, ys, values, patch_size) for the tile overlay.

    xs/ys are level-0 pixel tile-center coordinates (interpolate_patch_max's
    convention); values come straight from the table's .obs column or its
    .X (via a .var name) -- read-only, no copy or mutation of the table,
    since nothing here needs an obs column for a downstream library to find.
    """
    if tile_key not in wsi.shapes:
        raise ValueError(
            f"Slide has no tiles element '{tile_key}'. Available shapes: {list(wsi.shapes)}"
        )

    table = wsi.tables.get(table_key)
    if table is None:
        raise ValueError(
            f"Slide has no table '{table_key}'. Available tables: {list(wsi.tables)}"
        )

    if feature_name in table.obs.columns:
        values = table.obs[feature_name].to_numpy()
    elif feature_name in table.var_names:
        x = table[:, feature_name].X
        values = x.toarray()[:, 0] if hasattr(x, "toarray") else np.asarray(x).reshape(-1)
    else:
        raise KeyError(
            f"Feature '{feature_name}' is in neither {table_key}.obs nor its "
            f".var_names."
        )

    tiles = wsi.shapes[tile_key]
    spec = wsi.tile_spec(tile_key)
    if spec.base_width != spec.base_height:
        raise ValueError(
            f"plot_feature_map requires square tiles; got base_width="
            f"{spec.base_width}, base_height={spec.base_height}."
        )
    patch_size = spec.base_width

    instance_key = table.uns.get("spatialdata_attrs", {}).get("instance_key")
    if instance_key is not None and instance_key in table.obs.columns:
        tile_ids = table.obs[instance_key].to_numpy()
        try:
            tile_ids = tile_ids.astype(tiles.index.dtype)
        except (TypeError, ValueError):
            pass
        bounds = tiles.loc[tile_ids].bounds
    else:
        # No instance_key to join on; fall back to row order matching tiles.
        bounds = tiles.bounds

    xs = bounds["minx"].to_numpy() + patch_size / 2
    ys = bounds["miny"].to_numpy() + patch_size / 2
    return xs, ys, values, patch_size


@deprecated_kwargs(
    image_name=removed(
        "Pass the WSIData itself as the first argument; each store holds one slide."
    ),
    bbox_postfix=drop('_grid_point_bbox', ELEMENT_NAME_HINT),
    patch_postfix=drop('_grid_point_patch', ELEMENT_NAME_HINT),
    method=drop('datashader', _RENDERING_HINT),
    datashader_reduction=drop('max', _RENDERING_HINT),
)
def plot_feature_map(
    wsi,
    feature_name,
    *,
    tile_key=DEFAULT_TILE_KEY,
    table_key=None,
    image_size=2000,
    oversample=1.5,
    cmap=None,
    fill_alpha=1.0,
    figsize=(10, 10),
    colorbar=False,
    title=None,
    norm=None,
    return_ax=False,
):
    """
    Plot a per-tile feature as an overlay on the slide image.

    Renders by rasterizing tile scores onto a display-resolution canvas
    (``interpolate_patch_max``, highest score wins on overlap) rather than
    through spatialdata_plot -- this scales with the rendered figure size,
    not the number of tiles.

    Parameters
    ----------
    wsi : WSIData
        A single slide. The background is read lazily via ``wsi.reader``,
        the same mechanism ``lazyslide.pl.tiles`` uses -- so this works
        whether or not the slide was opened with ``attach_images=True``.
    feature_name : str
        Feature to colour by. Either an .obs column of the tile table or a
        .var name read from its .X.
    tile_key : str, default='tiles'
        Tile shapes element; also what the overlay is drawn from. Tiles
        must be square.
    table_key : str, optional
        Tile table name. Defaults to ``f"{tile_key}_table"``.
    image_size : int, default=2000
        Ceiling on the max pixel dimension of the background image read via
        ``wsi.reader``. The actual read resolution is chosen from the
        rendered figure size (see ``oversample``); this only caps the worst
        case, e.g. for a very large ``figsize``/``dpi``.
    oversample : float, default=1.5
        Read this many times more pixels than the axes occupies, for
        crispness. Larger is sharper but slower.
    cmap : matplotlib colormap, optional
        Defaults to transparent-to-green.
    fill_alpha : float, default=1.0
        Multiplies the colormap's own alpha channel (the default cmap is
        already transparent at low values).
    figsize : tuple, default=(10, 10)
    colorbar : bool, default=False
    title : str, optional
        Defaults to the slide's filename.
    norm : matplotlib Normalize, optional
        Defaults to Normalize(vmin=0, vmax=<feature's max value>), so the
        highest-scoring tile always reaches full color regardless of scale.
    return_ax : bool, default=False

    Returns
    -------
    fig, or (fig, ax) when return_ax=True
    """
    table_key = table_key or tile_table_key(tile_key)
    xs, ys, values, patch_size = _tile_centers_and_values(wsi, tile_key, table_key, feature_name)

    if cmap is None:
        cmap = LinearSegmentedColormap.from_list(
            'transparent_to_green', [(1, 1, 1, 0), (0, 1, 0, 1)], N=256
        )
    if norm is None:
        vmax = float(np.nanmax(values)) if len(values) else 1.0
        norm = Normalize(vmin=0, vmax=vmax if vmax > 0 else 1.0)
    if title is None:
        title = wsi.name

    props = wsi.properties
    h0, w0 = props.shape

    fig, ax = plt.subplots(figsize=figsize)
    axes_px_w, axes_px_h = _axes_device_px(ax)

    # The color canvas is computed directly into an array, not read from the
    # slide file, so -- unlike the background image -- it isn't constrained
    # to an existing pyramid level: use the continuous display-target
    # downsample directly, capped by image_size the same way.
    canvas_downsample = max(1.0, w0 / image_size, h0 / image_size)
    if axes_px_w and axes_px_h:
        canvas_downsample = max(
            canvas_downsample, w0 / (axes_px_w * oversample), h0 / (axes_px_h * oversample)
        )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    # Skip samples whose score maps to a fully transparent color -- with the
    # default transparent-to-green cmap that is every exactly-zero score,
    # which for a sparse feature (e.g. most SAE activations) is most tiles.
    # Safe generally: only skipped when doing so is visually identical to
    # painting it (this cmap already renders a real 0 as invisible).
    zero_is_invisible = sm.to_rgba(0.0)[3] < 1e-9
    keep = (values != 0) if zero_is_invisible else np.ones(len(values), dtype=bool)

    samples = dict(zip(
        zip(xs[keep].astype(int).tolist(), ys[keep].astype(int).tolist()),
        values[keep].tolist(),
    ))
    canvas = interpolate_patch_max(samples, h0, w0, patch_size, downsample=canvas_downsample)

    rgba = sm.to_rgba(canvas)  # float (H, W, 4) in [0, 1]; keeps cmap's own alpha ramp
    rgba[..., 3] *= fill_alpha
    rgba[np.isnan(canvas), 3] = 0  # background is always fully transparent
    ax.imshow(rgba, extent=[0, w0, h0, 0], origin="upper", zorder=-99)
    ax.set_axis_off()  # no ticks, no spines, no frame
    
    if colorbar:
        fig.colorbar(sm, ax=ax)

    _render_slide_background(ax, wsi, image_size=image_size, oversample=oversample)

    ax.set_title(title)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if return_ax:
        return fig, ax
    return fig
