"""Render a per-tile feature over its slide image."""

from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.pyplot as plt

from mesoslide._slides import DEFAULT_TILE_KEY, tile_table_key
from mesoslide._deprecated import ELEMENT_NAME_HINT, deprecated_kwargs, drop, removed

# Private obs column the .X -> .obs bridge writes into, kept distinct from the
# feature name so spatialdata_plot never sees the value in both places.
_RENDER_COLUMN = "_mesoslide_render_value"


def _render_slide_background(ax, wsi, size=2000, norm=None):
    """Draw a downsampled full-slide image behind the tile overlay.

    Reads directly via wsi.reader (WSIData.get_thumbnail wraps
    reader.get_thumbnail), so this works whether or not the slide was opened
    with attach_images=True -- mirroring how lazyslide.pl.tiles reads its
    background (ImageDataSource(wsi.reader), never wsi.images).

    `image` is a small, downsampled array (~`size` px on its long side);
    `extent` is deliberately the *level-0* (full-resolution) size, since
    that's the coordinate space the tile shapes are in (their x/y are
    level-0 pixel origins). imshow stretches `image` to fill `extent`
    regardless of the array's own resolution -- the same trick lazyslide's
    ImageDataSource.get_extent() relies on, returning viewport.w0/h0
    ("invariant to the chosen pyramid level") even though the image it
    pairs with was read at a coarser level.

    `norm`, if given, rescales the RGB array (e.g. Normalize(vmin=0,
    vmax=255) maps uint8 to [0, 1]) -- imshow's own `norm=` argument only
    applies to scalar-mappable data, not RGB, so this is applied by hand.
    """
    image = wsi.get_thumbnail(size=size, as_array=True)
    if norm is not None:
        image = norm(image)
    h0, w0 = wsi.properties.shape  # level 0, NOT image.shape
    ax.imshow(image, extent=[0, w0, h0, 0], origin="upper", zorder=-100)
    # Force the full-slide extent regardless of what .pl.show() autoscaled
    # the axes to from the shapes alone (typically a tighter tiles bbox).
    ax.set_xlim(0, w0)
    ax.set_ylim(h0, 0)


def _align_instance_ids(table, element):
    """Make the table's instance_key dtype match the element's index dtype.

    Stores written before this was fixed hold `tile_id` as str while the tiles
    GeoDataFrame indexes on int, and SpatialData matches the two by value -- so
    without this they look unrelated and rendering is refused. Returns the table
    unchanged when they already agree.
    """
    attrs = table.uns.get("spatialdata_attrs", {})
    key = attrs.get("instance_key")
    if key is None or key not in table.obs.columns:
        return table

    target = element.index.dtype
    if table.obs[key].dtype == target:
        return table
    try:
        aligned = table.obs[key].astype(target)
    except (TypeError, ValueError):
        return table

    view = table.copy()
    view.obs[key] = aligned
    return view


@deprecated_kwargs(
    image_name=removed(
        "Pass the WSIData itself as the first argument; each store holds one slide."
    ),
    bbox_postfix=drop('_grid_point_bbox', ELEMENT_NAME_HINT),
    patch_postfix=drop('_grid_point_patch', ELEMENT_NAME_HINT),
)
def plot_feature_map(
    wsi,
    feature_name,
    *,
    tile_key=DEFAULT_TILE_KEY,
    table_key=None,
    image_size=2000,
    cmap=None,
    fill_alpha=0.7,
    figsize=(10, 10),
    colorbar=False,
    title=None,
    norm=None,
    method='datashader',
    datashader_reduction='max',
    return_ax=False,
):
    """
    Plot a per-tile feature as an overlay on the slide image.

    Parameters
    ----------
    wsi : WSIData
        A single slide. The background is read lazily via ``wsi.reader``
        (``WSIData.get_thumbnail``), the same mechanism ``lazyslide.pl.tiles``
        uses -- so this works whether or not the slide was opened with
        ``attach_images=True``.
    feature_name : str
        Feature to colour by. Either an .obs column of the tile table or a
        .var name in its .X -- in the latter case the scores are copied into
        .obs first, since spatialdata_plot cannot read .X by var name here.
    tile_key : str, default='tiles'
        Tile shapes element; also what the overlay is drawn from.
    table_key : str, optional
        Tile table name. Defaults to ``f"{tile_key}_table"``.
    image_size : int, default=2000
        Max dimension (in pixels) of the background thumbnail read via
        ``wsi.get_thumbnail``. Higher values show more slide detail at the
        cost of a slower, larger read.
    cmap : matplotlib colormap, optional
        Defaults to transparent-to-green.
    fill_alpha : float, default=0.7
    figsize : tuple, default=(10, 10)
    colorbar : bool, default=False
    title : str, optional
        Defaults to the slide's filename.
    norm : matplotlib Normalize, optional
        Defaults to Normalize(vmin=0, vmax=255).
    method : str, default='datashader'
    datashader_reduction : str, default='max'
    return_ax : bool, default=False

    Returns
    -------
    fig, or (fig, ax) when return_ax=True
    """
    from spatialdata_plot import pl  # noqa: F401  (registers the .pl accessor)

    table_key = table_key or tile_table_key(tile_key)

    if tile_key not in wsi.shapes:
        raise ValueError(
            f"Slide has no tiles element '{tile_key}'. Available shapes: {list(wsi.shapes)}"
        )

    table = wsi.tables.get(table_key)
    if table is None:
        raise ValueError(
            f"Slide has no table '{table_key}'. Available tables: {list(wsi.tables)}"
        )

    # spatialdata_plot colours shapes by an .obs column; bridge from .X when the
    # feature is a var name (SAE scores live there). The bridged column gets a
    # private name and lands on a copy: reusing `feature_name` would make the
    # value ambiguous between .obs and .var, and writing it back would mutate
    # the caller's table as a side effect of drawing a picture.
    render_table, color_key = table, feature_name
    if feature_name not in table.obs.columns:
        if feature_name in table.var_names:
            from mesoslide._utils import copy_feature_score_to_obs
            render_table = table.copy()
            color_key = _RENDER_COLUMN
            copy_feature_score_to_obs(render_table, feature_name, obs_colname=color_key)
        else:
            raise KeyError(
                f"Feature '{feature_name}' is in neither {table_key}.obs nor its "
                f".var_names."
            )

    if cmap is None:
        cmap = LinearSegmentedColormap.from_list(
            'transparent_to_green', [(1, 1, 1, 0), (0, 1, 0, 1)], N=256
        )
    if norm is None:
        norm = Normalize(vmin=0, vmax=255)
    if title is None:
        title = wsi.name

    # The background is drawn separately via wsi.reader (see
    # _render_slide_background), so `view` only needs the shapes/table
    # spatialdata_plot renders the tile overlay from.
    import spatialdata

    view = spatialdata.SpatialData(
        shapes={tile_key: wsi.shapes[tile_key]},
        tables={table_key: _align_instance_ids(render_table, wsi.shapes[tile_key])},
    )

    coordinate_system = (
        "global" if "global" in view.coordinate_systems else view.coordinate_systems[0]
    )

    fig, ax = plt.subplots(figsize=figsize)
    view.pl.render_shapes(
        element=tile_key,
        color=color_key,
        cmap=cmap,
        fill_alpha=fill_alpha,
        method=method,
        datashader_reduction=datashader_reduction,
    ).pl.show(coordinate_systems=coordinate_system, title=title,
              colorbar=colorbar, ax=ax)

    _render_slide_background(ax, wsi, size=image_size, norm=norm)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if return_ax:
        return fig, ax
    return fig
