"""Render a per-tile feature over its slide image."""

from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.pyplot as plt

from mesoslide._slides import DEFAULT_TILE_KEY, tile_table_key
from mesoslide._deprecated import ELEMENT_NAME_HINT, deprecated_kwargs, drop, removed

DEFAULT_IMAGE_KEY = "wsi"

# Private obs column the .X -> .obs bridge writes into, kept distinct from the
# feature name so spatialdata_plot never sees the value in both places.
_RENDER_COLUMN = "_mesoslide_render_value"


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
    image_key=DEFAULT_IMAGE_KEY,
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
        A single slide, opened with image data attached:
        ``ezslide.read_wsi(store, attach_images=True)``. A store written by
        ``wsi.write()`` holds shapes and tables but no pixels, so a slide read
        back without ``attach_images`` has nothing to render under the overlay.
    feature_name : str
        Feature to colour by. Either an .obs column of the tile table or a
        .var name in its .X -- in the latter case the scores are copied into
        .obs first, since spatialdata_plot cannot read .X by var name here.
    tile_key : str, default='tiles'
        Tile shapes element; also what the overlay is drawn from.
    table_key : str, optional
        Tile table name. Defaults to ``f"{tile_key}_table"``.
    image_key : str, default='wsi'
        Image element name, as attached by ezslide.
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

    if image_key not in wsi.images:
        raise ValueError(
            f"Slide has no image element '{image_key}', so there is nothing to "
            "render the overlay on. wsi.write() does not persist WSI pixels -- "
            "reopen with ezslide.read_wsi(store, attach_images=True). "
            f"Available images: {list(wsi.images)}"
        )
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
            'transparent_to_green', [(1, 1, 1, 0), (0, 1, 0, 0.5)], N=256
        )
    if norm is None:
        norm = Normalize(vmin=0, vmax=255)
    if title is None:
        title = wsi.name

    # wsidata keeps the WSI image in `_exclude_elements` so wsi.write() does not
    # try to persist the pixels. That also hides it from SpatialData.__getitem__,
    # which is what spatialdata_plot resolves elements through -- so rendering
    # needs a plain SpatialData holding the three elements explicitly.
    import spatialdata

    view = spatialdata.SpatialData(
        images={image_key: wsi.images[image_key]},
        shapes={tile_key: wsi.shapes[tile_key]},
        tables={table_key: _align_instance_ids(render_table, wsi.shapes[tile_key])},
    )

    coordinate_system = (
        "global" if "global" in view.coordinate_systems else view.coordinate_systems[0]
    )

    fig, ax = plt.subplots(figsize=figsize)
    view.pl.render_images(element=image_key, norm=norm) \
        .pl.render_shapes(
            element=tile_key,
            color=color_key,
            cmap=cmap,
            fill_alpha=fill_alpha,
            method=method,
            datashader_reduction=datashader_reduction,
        ) \
        .pl.show(coordinate_systems=coordinate_system, title=title,
                 colorbar=colorbar, ax=ax)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if return_ax:
        return fig, ax
    return fig
