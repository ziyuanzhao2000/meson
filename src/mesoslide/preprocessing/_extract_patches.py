"""Read the image data behind a set of selected tiles."""

from typing import TYPE_CHECKING, List, Optional, Union
import warnings
import numpy as np
from tqdm import tqdm

from mesoslide._slides import (
    DEFAULT_TILE_KEY,
    SLIDE_ID,
    slide_id_from,
    tile_table_key,
)

if TYPE_CHECKING:
    import anndata as ad
    from wsidata import WSIData


def _resolve_slides(slides) -> dict:
    """Normalise `slides` to {slide_id: WSIData}; a lone slide keys on None."""
    from wsidata import WSIData

    if isinstance(slides, WSIData):
        return {None: slides}
    if isinstance(slides, dict):
        return slides
    if isinstance(slides, (list, tuple)):
        return {slide_id_from(w): w for w in slides}
    raise TypeError(
        "slides must be a WSIData, a list of them, or the {slide_id: WSIData} "
        f"mapping returned by mesoslide.open_slides; got {type(slides).__name__}."
    )


def _tile_size(wsi: "WSIData", tile_key: str) -> tuple:
    """Tile height/width at level 0, from the slide's own tile spec."""
    spec = wsi.tile_spec(tile_key)
    if spec is None:
        raise ValueError(
            f"Slide has no tile spec for '{tile_key}'. Tile it first with "
            "lazyslide.pp.tile_tissues."
        )
    return (
        int(getattr(spec, "base_height", spec.height)),
        int(getattr(spec, "base_width", spec.width)),
    )


def extract_patches(
    patches: "ad.AnnData",
    slides,
    *,
    tile_key: str = DEFAULT_TILE_KEY,
    channel_first: bool = True,
    progress_bar: bool = True,
    skip_errors: bool = True,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Read image data for the tiles described by a patch table.

    Coordinates come from ``patches.obs['x']`` / ``['y']`` (a tile's top-left
    corner at level 0) and the tile size from each slide's own
    ``wsi.tile_spec(tile_key)``, rather than from per-row bounds columns.

    Parameters
    ----------
    patches : AnnData
        Selected tiles, e.g. from :func:`mesoslide.select_top_patches`.
        Required .obs columns: 'x', 'y'; plus 'slide_id' when `slides` covers
        more than one slide.
    slides : WSIData, list of WSIData, or {slide_id: WSIData}
        The slides to read from. Use :func:`mesoslide.open_slides` to build the
        mapping from a cohort manifest. Slides must have image data attached
        (``ezslide.read_wsi(store, attach_images=True)``); a store written by
        ``wsi.write()`` holds no pixels on its own.
    tile_key : str, default='tiles'
    channel_first : bool, default=True
        True -> each patch (C, H, W), stacked (N, C, H, W).
        False -> each patch (H, W, C), stacked (N, H, W, C).
    progress_bar : bool, default=True
    skip_errors : bool, default=True
        Skip failed reads instead of raising on the first one.

    Returns
    -------
    np.ndarray or list of np.ndarray
        Stacked if all patches share a shape, else a list (with a warning).
        Note N may be < len(patches) when skip_errors=True.

    Raises
    ------
    ValueError
        If required columns are missing, a needed slide is absent, or nothing
        could be read.

    Examples
    --------
    >>> import mesoslide as ms
    >>> slides = ms.open_slides(manifest)
    >>> top = ms.select_top_patches(manifest, 'UNI_SAE_123', n=100)
    >>> imgs = ms.pp.extract_patches(top, slides, channel_first=False)

    Notes
    -----
    This is for reading a *selected subset*. To iterate every tile of a slide,
    use ``ezslide.tile_images(wsi, tile_key=...)`` (block-deduping, and what
    :func:`mesoslide.tl.embed_patch` uses) or ``wsi.iter.tile_images(key)``.
    """
    slide_map = _resolve_slides(slides)
    single = set(slide_map) == {None}

    required = ["x", "y"] if single else ["x", "y", SLIDE_ID]
    missing = [c for c in required if c not in patches.obs.columns]
    if missing:
        raise ValueError(
            f"patches.obs missing required columns: {missing}. "
            "These come from the tile table written by mesoslide.tl.embed_patch; "
            f"got: {list(patches.obs.columns)}"
        )

    patch_df = patches.obs
    sizes = {sid: _tile_size(wsi, tile_key) for sid, wsi in slide_map.items()}

    extracted = []
    iterator = patch_df.iterrows()
    if progress_bar:
        iterator = tqdm(iterator, total=len(patch_df), desc="Extracting patches")

    for _, patch in iterator:
        slide_id = None if single else patch[SLIDE_ID]
        try:
            wsi = slide_map[slide_id]
        except KeyError:
            msg = (
                f"Warning: no slide '{slide_id}' in `slides` "
                f"(have: {sorted(k for k in slide_map if k is not None)})"
            )
            if skip_errors:
                tqdm.write(msg) if progress_bar else print(msg)
                continue
            raise ValueError(msg.removeprefix("Warning: "))

        try:
            h, w = sizes[slide_id]
            # read_region returns (H, W, C) uint8 at level 0
            arr = wsi.read_region(int(patch.x), int(patch.y), w, h)
            extracted.append(np.moveaxis(arr, -1, 0) if channel_first else arr)
        except Exception as e:
            if skip_errors:
                msg = f"Warning: Failed to read patch at ({patch.x}, {patch.y}) from {slide_id}: {e}"
                tqdm.write(msg) if progress_bar else print(msg)
                continue
            raise

    if len(extracted) == 0:
        raise ValueError("No patches were successfully extracted")

    shapes = [p.shape for p in extracted]
    if len(set(shapes)) == 1:
        return np.stack(extracted, axis=0)

    warnings.warn(
        f"Patches have inconsistent shapes ({len(set(shapes))} distinct shapes). "
        "Returning a list instead of a stacked array.",
        UserWarning,
        stacklevel=2,
    )
    return extracted
