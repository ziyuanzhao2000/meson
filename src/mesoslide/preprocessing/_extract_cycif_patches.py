"""Read CyCIF/multiplex channel data behind a set of selected tiles."""

from typing import TYPE_CHECKING, List, Optional, Union
import warnings
import numpy as np
from tqdm.auto import tqdm

from mesoslide._slides import CYCIF_IMG_KEY, DEFAULT_TILE_KEY, SLIDE_ID
from mesoslide.preprocessing._extract_patches import (
    _resolve_slides,
    _resolve_slides_from_ref,
    _tile_size,
)

if TYPE_CHECKING:
    import anndata as ad
    import pandas as pd
    from wsidata import WSIData


def _channel_indices(
    channels: List[str],
    marker_table: Optional["pd.DataFrame"],
    marker_col: str,
) -> List[int]:
    """Resolve channel names to integer indices via a marker table, or pass
    integer channels straight through."""
    if marker_table is None:
        return [int(c) for c in channels]
    marker_ids = dict(zip(marker_table[marker_col], marker_table.index))
    missing = [c for c in channels if c not in marker_ids]
    if missing:
        raise ValueError(
            f"Channels not found in marker_table['{marker_col}']: {missing}"
        )
    return [int(marker_ids[c]) for c in channels]


def extract_cycif_patch_images(
    patches: "ad.AnnData",
    channels: List[str],
    slides=None,
    *,
    tile_key: str = DEFAULT_TILE_KEY,
    marker_table: Optional["pd.DataFrame"] = None,
    marker_col: str = "marker_name",
    progress_bar: bool = True,
    skip_errors: bool = True,
    cache: bool = False,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Read CyCIF channel data for the tiles described by a patch table.

    Mirrors :func:`extract_he_patch_images`: coordinates come from
    ``patches.obs['x']``/``['y']`` (a tile's top-left corner at level 0), and
    pixels are read the same way, through ``wsi.read_region(x, y, w, h)``,
    which returns ``(H, W, C)`` for any channel count -- the requested
    `channels` are then selected from the last axis. Tile size (`w`/`h`)
    comes from ``wsi.tile_spec(tile_key)`` on the `slides` passed in when
    available, else from ``patches.obs['_slide_ref']`` -- a CyCIF slide
    registered to the same pixel grid as the H&E slide is often never itself
    tiled with lazyslide, so `_slide_ref` (typically the H&E slide, which
    *was* tiled) is the usual source in practice.

    If ``patches.obsm[CYCIF_IMG_KEY]`` is already populated (e.g. from a
    previous call with ``cache=True``), it is returned directly and no slide
    is read; note the cache is only valid for the same `channels` it was
    built with.

    Parameters
    ----------
    patches : AnnData
        Selected tiles, e.g. from :func:`mesoslide.select_top_patches`.
        Required .obs columns: 'x', 'y'; plus 'slide_id' when `slides` covers
        more than one slide.
    channels : list of str
        Marker/channel names to extract, in the order they should be
        returned. Looked up via `marker_table`/`marker_col` if given, else
        treated as integer channel indices.
    slides : WSIData, list of WSIData, or {slide_id: WSIData}, optional
        The CyCIF-backed slides to read from, registered to the same level-0
        pixel grid as the H&E slides used to build `patches` -- note this is
        usually a *different* slide than `patches.obs['_slide_ref']` points
        to right after selection (which references the H&E/tiling slide).
        When omitted, falls back to `patches.obs['_slide_ref']` same as
        :func:`extract_he_patch_images`; if that still points at H&E slides,
        retarget it first: `patches = mesoslide.attach_slide_ref(patches,
        cycif_slides, slide_id_map=...)` (needed since slide ids don't
        necessarily match 1:1 between an H&E store and its registered CyCIF
        store). A patch table selected directly against a CyCIF-backed
        manifest needs no retargeting -- `_slide_ref` already points there.
    tile_key : str, default='tiles'
    marker_table : pandas.DataFrame, optional
        Maps channel names to channel indices via `marker_table.index`,
        e.g. loaded from a markers CSV with ``pd.read_csv(...)``.
    marker_col : str, default='marker_name'
        Column in `marker_table` holding the channel name.
    progress_bar : bool, default=True
    skip_errors : bool, default=True
        Skip failed reads instead of raising on the first one.
    cache : bool, default=False
        Store the extracted array in ``patches.obsm[CYCIF_IMG_KEY]``
        (channel-first) so a later call with the same `channels` can skip
        reading from slides. Skipped (with a warning) if `skip_errors` caused
        rows to be dropped, or if patches have inconsistent shapes.

    Returns
    -------
    np.ndarray or list of np.ndarray
        Channel-first (N, len(channels), H, W) if all patches share a shape,
        else a list (with a warning). N may be < len(patches) when
        skip_errors=True.

    Raises
    ------
    ValueError
        If required columns are missing, a needed slide is absent, a
        requested channel is not in `marker_table`, or nothing could be read.
    """
    if CYCIF_IMG_KEY in patches.obsm:
        return patches.obsm[CYCIF_IMG_KEY]

    channel_idx = _channel_indices(channels, marker_table, marker_col)

    opened: list = []
    try:
        if slides is not None:
            slide_map = _resolve_slides(slides)
        else:
            slide_map, opened = _resolve_slides_from_ref(patches)
        single = set(slide_map) == {None}

        required = ["x", "y"] if single else ["x", "y", SLIDE_ID]
        missing = [c for c in required if c not in patches.obs.columns]
        if missing:
            raise ValueError(
                f"patches.obs missing required columns: {missing}. "
                f"got: {list(patches.obs.columns)}"
            )

        patch_df = patches.obs

        # Tile size normally comes from the pixel-reading slides themselves, but a
        # CyCIF slide registered to the same pixel grid is often never itself tiled
        # with lazyslide -- fall back to patches.obs['_slide_ref'] (typically the
        # H&E slide that *was* tiled) in that case.
        try:
            sizes = {sid: _tile_size(wsi, tile_key) for sid, wsi in slide_map.items()}
        except ValueError:
            ref_slide_map, ref_opened = _resolve_slides_from_ref(patches)
            opened.extend(ref_opened)
            ref_sizes = {sid: _tile_size(wsi, tile_key) for sid, wsi in ref_slide_map.items()}
            if single:
                # slide_map has exactly one entry keyed None (a lone WSIData was
                # passed for pixel reads), but ref_slide_map is keyed by
                # patches.obs['slide_id'] when that column exists, regardless of
                # how many pixel-reading slides were passed -- so its keys don't
                # necessarily line up with slide_map's. Every patch looks up
                # `sizes[None]` here (single=True), so re-key onto that; this
                # assumes one tile size across whichever ref slide(s) matched,
                # true in the common single-slide case.
                sizes = {None: next(iter(ref_sizes.values()))}
            else:
                sizes = ref_sizes

        extracted = []
        iterator = patch_df.iterrows()
        if progress_bar:
            iterator = tqdm(iterator, total=len(patch_df), desc="Extracting CyCIF patches")

        for _, patch in iterator:
            slide_id = None if single else patch[SLIDE_ID]
            try:
                wsi = slide_map[slide_id]
            except KeyError:
                msg = (
                    f"Warning: no slide '{slide_id}' resolved "
                    f"(have: {sorted(k for k in slide_map if k is not None)})"
                )
                if skip_errors:
                    tqdm.write(msg) if progress_bar else print(msg)
                    continue
                raise ValueError(msg.removeprefix("Warning: "))

            try:
                h, w = sizes[slide_id]
                arr = wsi.read_region(int(patch.x), int(patch.y), w, h)  # (H, W, C)
                arr = arr[..., channel_idx]
                extracted.append(np.moveaxis(arr, -1, 0))  # (len(channels), H, W)
            except Exception as e:
                if skip_errors:
                    msg = (
                        f"Warning: Failed to read patch at ({patch.x}, {patch.y}) "
                        f"from {slide_id}: {type(e).__name__}: {e}"
                    )
                    tqdm.write(msg) if progress_bar else print(msg)
                    continue
                raise

        if len(extracted) == 0:
            raise ValueError("No patches were successfully extracted")

        shapes = [p.shape for p in extracted]
        if len(set(shapes)) != 1:
            warnings.warn(
                f"Patches have inconsistent shapes ({len(set(shapes))} distinct shapes). "
                "Returning a list instead of a stacked array.",
                UserWarning,
                stacklevel=2,
            )
            if cache:
                warnings.warn(
                    "cache=True has no effect: patches have inconsistent shapes and "
                    "cannot be aligned 1:1 with patches.obs.",
                    UserWarning,
                    stacklevel=2,
                )
            return extracted

        stacked = np.stack(extracted, axis=0)  # channel-first (N, len(channels), H, W)

        if cache:
            if len(extracted) == len(patch_df):
                patches.obsm[CYCIF_IMG_KEY] = stacked
            else:
                warnings.warn(
                    "cache=True has no effect: skip_errors dropped "
                    f"{len(patch_df) - len(extracted)} row(s), so the result cannot "
                    "be aligned 1:1 with patches.obs.",
                    UserWarning,
                    stacklevel=2,
                )

        return stacked
    finally:
        for wsi in opened:
            try:
                wsi.close()
            except Exception:
                pass
