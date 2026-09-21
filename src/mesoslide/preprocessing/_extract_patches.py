"""Read the image data behind a set of selected tiles."""

from typing import TYPE_CHECKING, List, Optional, Union
import warnings
import numpy as np
from tqdm.auto import tqdm

from mesoslide._slides import (
    CYCIF_PATCH_IMG_KEY,
    DEFAULT_TILE_KEY,
    HE_PATCH_IMG_KEY,
    SLIDE_ID,
    SLIDE_REF,
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


def _resolve_slides_from_ref(patches: "ad.AnnData") -> "tuple[dict, list]":
    """Build {slide_id: WSIData} from patches.obs[SLIDE_REF], the fallback
    used when no `slides` argument is given.

    Each unique live `WSIData` reference is reused directly (no open); each
    unique path-string reference is opened exactly once via
    `ezslide.read_slide`, regardless of how many rows share it. Returns
    `(slide_map, opened)` -- `opened` lists the WSIData this function itself
    opened, for the caller to close once done reading.
    """
    if SLIDE_REF not in patches.obs.columns:
        raise ValueError(
            "slides was not given and patches.obs has no "
            f"'{SLIDE_REF}' column -- pass slides= explicitly, or build "
            "patches via mesoslide.select_* (or attach one with "
            "mesoslide.attach_slide_ref) so it carries its own slide "
            "reference."
        )

    from wsidata import WSIData
    import ezslide

    has_slide_id = SLIDE_ID in patches.obs.columns
    slide_ids = patches.obs[SLIDE_ID] if has_slide_id else None
    refs = patches.obs[SLIDE_REF]

    slide_map: dict = {}
    opened: list = []
    opened_by_path: dict = {}
    for i in range(len(patches)):
        slide_id = slide_ids.iat[i] if has_slide_id else None
        if slide_id in slide_map:
            continue
        ref = refs.iat[i]
        if ref is None or (isinstance(ref, float) and np.isnan(ref)):
            continue  # left unresolved; existing KeyError/skip_errors path handles it
        if isinstance(ref, WSIData):
            slide_map[slide_id] = ref
            continue
        path = str(ref)
        if path not in opened_by_path:
            wsi = ezslide.read_slide(path, attach_images=True)
            opened_by_path[path] = wsi
            opened.append(wsi)
        slide_map[slide_id] = opened_by_path[path]

    return slide_map, opened


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


def _resolve_obsm_key_pre_read(
    patches: "ad.AnnData",
    channels: Optional[List[int]],
    obsm_key: Optional[str],
) -> Optional[str]:
    """Which `patches.obsm` key (if any) already holds a cached extraction,
    decided before opening any slide.

    `channels=None` results can land under either the H&E or CyCIF key
    depending on the slide's actual channel count (only known after a read),
    so both are checked in that case; an explicit `channels` can never hit
    the H&E slot (see :func:`extract_patch_images`)."""
    if obsm_key is not None:
        return obsm_key if obsm_key in patches.obsm else None
    if channels is not None:
        return CYCIF_PATCH_IMG_KEY if CYCIF_PATCH_IMG_KEY in patches.obsm else None
    he_hit = HE_PATCH_IMG_KEY in patches.obsm
    cycif_hit = CYCIF_PATCH_IMG_KEY in patches.obsm
    if he_hit and cycif_hit:
        warnings.warn(
            f"Both '{HE_PATCH_IMG_KEY}' and '{CYCIF_PATCH_IMG_KEY}' are cached "
            "in patches.obsm; using the H&E one. Pass obsm_key= to disambiguate.",
            UserWarning,
            stacklevel=3,
        )
        return HE_PATCH_IMG_KEY
    if he_hit:
        return HE_PATCH_IMG_KEY
    if cycif_hit:
        return CYCIF_PATCH_IMG_KEY
    return None


def _resolve_obsm_key_post_read(
    channels: Optional[List[int]],
    obsm_key: Optional[str],
    n_channels: int,
) -> str:
    """Which `patches.obsm` key a freshly extracted array should be cached
    under, now that its actual channel count is known."""
    if obsm_key is not None:
        return obsm_key
    if channels is not None:
        return CYCIF_PATCH_IMG_KEY
    return HE_PATCH_IMG_KEY if n_channels == 3 else CYCIF_PATCH_IMG_KEY


def extract_patch_images(
    patches: "ad.AnnData",
    slides=None,
    *,
    channels: Optional[List[int]] = None,
    tile_key: str = DEFAULT_TILE_KEY,
    channel_first: bool = True,
    progress_bar: bool = True,
    skip_errors: bool = True,
    cache: bool = True,
    obsm_key: Optional[str] = None,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Read image data for the tiles described by a patch table.

    Reads whichever channels are asked for: `channels=None` (the default)
    reads every channel `wsi.read_region` returns, which is the correct
    behavior for an H&E slide (3-channel RGB); `channels=[...]` selects a
    subset by integer index, for CyCIF/multiplex slides with more than 3
    channels. To select by marker name instead of index, resolve indices
    first with :func:`mesoslide.preprocessing.channel_indices_from_markers`.

    If a matching cached array is already populated in `patches.obsm` (e.g.
    from a previous call with `cache=True`), it is returned directly
    (converted to the requested `channel_first` layout) and no slide is
    read -- see `obsm_key` below for which key is checked.

    Otherwise, coordinates come from ``patches.obs['x']`` / ``['y']`` (a
    tile's top-left corner at level 0) and the tile size from each slide's
    own ``wsi.tile_spec(tile_key)``, rather than from per-row bounds columns.

    Parameters
    ----------
    patches : AnnData
        Selected tiles, e.g. from :func:`mesoslide.select_top_patches`.
        Required .obs columns: 'x', 'y'; plus 'slide_id' when `slides` covers
        more than one slide.
    slides : WSIData, list of WSIData, or {slide_id: WSIData}, optional
        The slides to read from. Use :func:`mesoslide.open_slides` to build the
        mapping from a cohort manifest. Slides must have image data attached
        (``ezslide.read_slide(store, attach_images=True)``); a store written by
        ``wsi.write()`` holds no pixels on its own. When omitted, resolved from
        ``patches.obs['_slide_ref']`` instead -- populated automatically by
        :func:`mesoslide.select_top_patches` and friends, or attach one
        manually with :func:`mesoslide.attach_slide_ref`. A slide referenced
        by a live `WSIData` there is reused directly (no re-open); one
        referenced by a store path is opened once (regardless of how many
        rows share it) and closed again before this function returns.
    channels : list of int, optional
        Integer channel indices to select from the last axis of each
        `read_region` result, in the order they should be returned. Omit to
        read every channel (the H&E default).
    tile_key : str, default='tiles'
    channel_first : bool, default=True
        True -> each patch (C, H, W), stacked (N, C, H, W).
        False -> each patch (H, W, C), stacked (N, H, W, C).
    progress_bar : bool, default=True
    skip_errors : bool, default=True
        Skip failed reads instead of raising on the first one.
    cache : bool, default=False
        Store the extracted array (channel-first) in `patches.obsm`, so a
        later call on the same `patches` object can skip reading from
        slides. Skipped (with a warning) if `skip_errors` caused rows to be
        dropped, or if patches have inconsistent shapes, since either case
        cannot be aligned 1:1 with `.obs`.
    obsm_key : str, optional
        `patches.obsm` key to read/write the cache under. Defaults to
        choosing between `mesoslide._slides.HE_PATCH_IMG_KEY` and
        `..._slides.CYCIF_PATCH_IMG_KEY`: the H&E key only when `channels`
        is not given *and* the extracted array turns out to have 3
        channels; the CyCIF key in every other case (including an explicit
        `channels` of length 3). Pass this explicitly to keep several
        distinct extractions (e.g. two different channel subsets) cached
        side by side on the same `patches` object.

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
    >>> imgs = ms.pp.extract_patch_images(top, slides, channel_first=False)
    >>>
    >>> idx = ms.pp.channel_indices_from_markers(['DAPI', 'CD3'], marker_table)
    >>> cycif = ms.pp.extract_patch_images(top, cycif_slides, channels=idx)

    Notes
    -----
    This is for reading a *selected subset*. To iterate every tile of a slide,
    use ``ezslide.tile_images(wsi, tile_key=...)`` (block-deduping, and what
    :func:`mesoslide.tl.feature_extraction` uses) or ``wsi.iter.tile_images(key)``.
    """
    cached_key = _resolve_obsm_key_pre_read(patches, channels, obsm_key)
    if cached_key is not None:
        cached = patches.obsm[cached_key]
        return np.moveaxis(cached, 1, -1) if not channel_first else cached

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
                "These come from the tile table written by mesoslide.tl.feature_extraction; "
                f"got: {list(patches.obs.columns)}"
            )

        patch_df = patches.obs

        # Tile size normally comes from the pixel-reading slides themselves, but a
        # CyCIF slide registered to the same pixel grid is often never itself tiled
        # with lazyslide -- fall back to patches.obs['_slide_ref'] (typically the
        # H&E slide that *was* tiled) in that case. Kept general (not gated on
        # `channels`) since the same mismatch can occur in the opposite direction
        # too -- a CyCIF-selected patch table extracting from a registered H&E slide.
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
            iterator = tqdm(iterator, total=len(patch_df), desc="Extracting patches")

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
                # read_region returns (H, W, C) uint8 at level 0; keep channel-first
                # internally so a cached result is always in the canonical layout.
                arr = wsi.read_region(int(patch.x), int(patch.y), w, h)
                if channels is not None:
                    arr = arr[..., channels]
                extracted.append(np.moveaxis(arr, -1, 0))
            except Exception as e:
                if skip_errors:
                    msg = f"Warning: Failed to read patch at ({patch.x}, {patch.y}) from {slide_id}: {e}"
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
            result = extracted
            return result if channel_first else [np.moveaxis(p, 0, -1) for p in result]

        stacked = np.stack(extracted, axis=0)  # channel-first (N, C, H, W)

        if cache:
            if len(extracted) == len(patch_df):
                key = _resolve_obsm_key_post_read(channels, obsm_key, stacked.shape[1])
                patches.obsm[key] = stacked
            else:
                warnings.warn(
                    "cache=True has no effect: skip_errors dropped "
                    f"{len(patch_df) - len(extracted)} row(s), so the result cannot "
                    "be aligned 1:1 with patches.obs.",
                    UserWarning,
                    stacklevel=2,
                )

        return stacked if channel_first else np.moveaxis(stacked, 1, -1)
    finally:
        for wsi in opened:
            try:
                wsi.close()
            except Exception:
                pass
