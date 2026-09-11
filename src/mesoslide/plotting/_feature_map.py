"""Render a per-tile feature over its slide image."""

import math

from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.pyplot as plt

from mesoslide._slides import DEFAULT_TILE_KEY, tile_table_key
from mesoslide._deprecated import ELEMENT_NAME_HINT, deprecated_kwargs, drop, removed

# Private obs column the .X -> .obs bridge writes into, kept distinct from the
# feature name so spatialdata_plot never sees the value in both places.
_RENDER_COLUMN = "_mesoslide_render_value"


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


def _render_slide_background(ax, wsi, *, image_size=2000, oversample=1.5, norm=None):
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

    # Safety ceiling on the read's pixel dimensions, mirroring lazyslide's
    # img_bytes_limit role but expressed in the "max dimension" unit this
    # function has always documented.
    while (
        (w0 / downsample > image_size or h0 / downsample > image_size)
        and level < props.n_level - 1
    ):
        level += 1
        downsample = props.level_downsample[level]

    dw = max(1, math.ceil(w0 / downsample))
    dh = max(1, math.ceil(h0 / downsample))
    image = wsi.reader.get_region(0, 0, dw, dh, level=level)

    if norm is not None:
        image = norm(image)
    ax.imshow(image, extent=[0, w0, h0, 0], origin="upper", zorder=-100)
    # Force the full-slide extent regardless of what .pl.show() autoscaled
    # the axes to from the shapes alone (typically a tighter tiles bbox).
    ax.set_xlim(0, w0)
    ax.set_ylim(h0, 0)


def _prepare_render_table(table, tiles, feature_name):
    """Return ``(table_to_render, color_key)``, copying only ``.obs``, only if needed.

    Colouring needs either a ``.var`` name bridged into ``.obs`` (SAE scores
    live in ``.X``) or the table's instance_key dtype aligned with the tiles
    index (older stores hold it as str while the tiles GeoDataFrame indexes
    on int, and SpatialData matches the two by value). Both only ever read
    ``.X``/write ``.obs``, so when either is needed, a single AnnData is
    rebuilt that duplicates just ``.obs`` and shares everything else (``.X``,
    ``.obsm``, ``.varm``, ``.layers``) by reference -- avoiding a full deep
    copy of per-slide embeddings (~0.2 GB) for what is otherwise a picture.
    Returns the original table unchanged (no copy) when neither is needed.
    """
    color_key = feature_name
    needs_bridge = False
    if feature_name not in table.obs.columns:
        if feature_name in table.var_names:
            needs_bridge = True
            color_key = _RENDER_COLUMN
        else:
            raise KeyError(
                f"Feature '{feature_name}' is in neither the tile table's .obs "
                f"nor its .var_names."
            )

    attrs = table.uns.get("spatialdata_attrs", {})
    instance_key = attrs.get("instance_key")
    aligned = None
    if instance_key is not None and instance_key in table.obs.columns:
        target = tiles.index.dtype
        if table.obs[instance_key].dtype != target:
            try:
                aligned = table.obs[instance_key].astype(target)
            except (TypeError, ValueError):
                aligned = None

    if not needs_bridge and aligned is None:
        return table, color_key

    import anndata as ad

    prepared = ad.AnnData(
        X=table.X,
        obs=table.obs.copy(),
        var=table.var,
        uns=table.uns,
        obsm=table.obsm,
        varm=table.varm,
        layers=table.layers,
    )
    if aligned is not None:
        prepared.obs[instance_key] = aligned
    if needs_bridge:
        from mesoslide._utils import copy_feature_score_to_obs
        copy_feature_score_to_obs(prepared, feature_name, obs_colname=color_key)

    return prepared, color_key


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
    oversample=1.5,
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
        A single slide. The background is read lazily via ``wsi.reader``,
        the same mechanism ``lazyslide.pl.tiles`` uses -- so this works
        whether or not the slide was opened with ``attach_images=True``.
    feature_name : str
        Feature to colour by. Either an .obs column of the tile table or a
        .var name in its .X -- in the latter case the scores are copied into
        .obs first, since spatialdata_plot cannot read .X by var name here.
    tile_key : str, default='tiles'
        Tile shapes element; also what the overlay is drawn from.
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

    render_table, color_key = _prepare_render_table(table, wsi.shapes[tile_key], feature_name)

    if cmap is None:
        cmap = LinearSegmentedColormap.from_list(
            'transparent_to_green', [(1, 1, 1, 0), (0, 1, 0, 1)], N=256
        )
    if norm is None:
        norm = Normalize(vmin=0, vmax=255)
    if title is None:
        title = wsi.name

    coordinate_system = (
        "global" if "global" in wsi.coordinate_systems else wsi.coordinate_systems[0]
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Render directly on wsi -- no throwaway SpatialData. A shapes-only
    # render never looks up wsi's image element by name, so the fact that
    # WSIData excludes it from name-based lookup (write() skips the pixels)
    # never comes up here. table_name is explicit because a real wsi --
    # unlike the old throwaway view -- may have more than one table
    # annotating tile_key (e.g. a bridged feature table alongside tiles_table).
    render_kwargs = dict(
        element=tile_key,
        color=color_key,
        table_name=table_key,
        cmap=cmap,
        fill_alpha=fill_alpha,
        method=method,
        datashader_reduction=datashader_reduction,
    )
    show_kwargs = dict(
        coordinate_systems=coordinate_system, title=title, colorbar=colorbar, ax=ax
    )

    if render_table is table:
        wsi.pl.render_shapes(**render_kwargs).pl.show(**show_kwargs)
    else:
        # Temporarily swap in the prepared table so render_shapes sees the
        # bridged/aligned column without mutating the caller's table. Not
        # safe for concurrent calls on the same wsi from multiple threads.
        wsi.tables[table_key] = render_table
        try:
            wsi.pl.render_shapes(**render_kwargs).pl.show(**show_kwargs)
        finally:
            wsi.tables[table_key] = table

    _render_slide_background(ax, wsi, image_size=image_size, oversample=oversample, norm=norm)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if return_ax:
        return fig, ax
    return fig
