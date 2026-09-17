"""Cross-slide access, for a world where one SpatialData holds exactly one WSI.

After the ezslide/lazyslide migration every element name is a constant (``wsi``,
``tiles``, ``tissues``, ``tiles_table``), so a slide is no longer identified by a
mangled element name but by the store it lives in. This module is the single
place that knows how to go from a cohort of stores to per-slide tables.

The default is to **stream**: :func:`iter_slides` opens one slide at a time and
closes it before advancing, so peak memory is one slide regardless of cohort
size. That matters more than it might look. On a 48568-tile slide the UNI
embeddings are 0.2 GB, so a 40-slide cohort is ~8 GB if concatenated up front --
while the score vector that patch selection actually reads is 0.39 MB per slide,
15.5 MB for the cohort. ``anndata.concat`` copies rather than viewing, so
concatenating first pays the 8 GB to read the 15 MB.

:func:`concat_slides` is therefore opt-in, and drops ``obsm`` unless asked for.

Why not ``lazyslide.agg_wsi`` or ``wsidata.io.concat_feature_anndata``: both
assume lazyslide's own feature layout, where features live in ``.X`` of a table
named ``f"{model}_{tile_key}"``. mesoslide keeps embeddings in
``tiles_table.obsm`` with ``.X`` reserved for SAE scores, so those helpers do not
see them. The cohort manifest here is deliberately the same DataFrame shape
``agg_wsi`` accepts, so one manifest serves both.

For interactive work where a single concatenated object is genuinely convenient
and the cohort is small, ``anndata.experimental.read_lazy`` / ``AnnCollection``
are the escape hatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Mapping, Optional, Sequence, Union

# Re-exported for mesoslide's existing public API (ms.iter_slides,
# ms.open_slides) and reused internally by concat_slides/SlideSource below.
# These are generic "walk a cohort of WSI stores" utilities with no
# mesoslide-specific (tile-table/AnnData-schema) assumptions, so they live in
# ezslide now, one layer below mesoslide's schema-specific code.
from ezslide import SLIDE_ID, resolve_manifest, iter_slides, open_slides  # noqa: F401

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData
    from wsidata import WSIData


DEFAULT_TILE_KEY = "tiles"

PATCH_IMG_KEY = "patch_img"
"""obsm key under which extract_patch_images caches extracted pixel data,
channel-first (N, C, H, W)."""


def tile_table_key(tile_key: str = DEFAULT_TILE_KEY) -> str:
    """Name of the AnnData table annotating ``tile_key``.

    Mirrors the default in :func:`mesoslide.tools.feature_extraction`: it cannot be
    ``tile_key`` itself, because SpatialData requires element names to be unique
    across all element types and ``tile_key`` already names the tiles shapes.
    """
    return f"{tile_key}_table"


def slide_id_from(wsi: "WSIData") -> str:
    """Derive a stable slide id from a slide's filename ("LSP24521.ome.tif" -> "LSP24521")."""
    return Path(wsi.name).name.split(".")[0]


def _is_anndata(obj) -> bool:
    from anndata import AnnData
    return isinstance(obj, AnnData)


def _is_wsidata(obj) -> bool:
    from wsidata import WSIData
    return isinstance(obj, WSIData)


def _is_slides_table(obj) -> bool:
    import pandas as pd
    return isinstance(obj, pd.DataFrame)


def concat_slides(
    slides_table: "pd.DataFrame",
    *,
    tile_key: str = DEFAULT_TILE_KEY,
    store_col: str = "store",
    slide_id_col: str = SLIDE_ID,
    obsm_keys: Sequence[str] = (),
    join: str = "outer",
) -> "AnnData":
    """Concatenate a cohort's tile tables into one AnnData. Opt-in; prefer :func:`iter_slides`.

    ``obsm`` entries are **dropped unless named in** ``obsm_keys``. That default
    is the difference between ~15 MB and ~8 GB for a 40-slide cohort, since
    ``anndata.concat`` copies rather than viewing. Ask for embeddings only when
    you are going to read them.

    Parameters
    ----------
    slides_table, store_col, slide_id_col
        As in :func:`iter_slides`.
    tile_key
        Tile shapes element whose table is concatenated.
    obsm_keys
        ``obsm`` entries to carry through, e.g. ``("UNI_embedding",)``. Empty by
        default.
    join
        Passed to ``anndata.concat``.

    Returns
    -------
    AnnData
        Rows from every slide, with ``obs['slide_id']`` and ``obs['store']``
        recording provenance. Row order follows ``slides_table`` order.
    """
    import anndata as ad

    table_key = tile_table_key(tile_key)
    obsm_keys = list(obsm_keys)
    parts = []

    import ezslide

    for slide_id, store in resolve_manifest(slides_table, store_col, slide_id_col):
        wsi = ezslide.read_slide(store)
        try:
            table = _require_table(wsi, table_key, slide_id)
            # Rebuild rather than slice: this is where obsm is deliberately
            # dropped unless the caller named it.
            sub = ad.AnnData(
                X=table.X,
                obs=table.obs.copy(),
                var=table.var.copy(),
                obsm={k: table.obsm[k] for k in obsm_keys if k in table.obsm},
            )
            sub.obs[SLIDE_ID] = slide_id
            sub.obs["store"] = str(store)
            parts.append(sub)
        finally:
            try:
                wsi.close()
            except Exception:
                pass

    if not parts:
        raise ValueError("slides_table is empty; nothing to concatenate.")

    return ad.concat(parts, join=join, merge="same", index_unique="-")


def _require_table(wsi: "WSIData", table_key: str, slide_id: str) -> "AnnData":
    table = wsi.tables.get(table_key)
    if table is None:
        raise KeyError(
            f"Slide '{slide_id}' has no table '{table_key}'. Run "
            f"mesoslide.tl.feature_extraction (or lazyslide.pp.tile_tissues) on it first; "
            f"available tables: {list(wsi.tables)}"
        )
    return table


class SlideSource:
    """Re-iterable, uniform access to a cohort's per-slide tile tables.

    Accepts whatever a caller naturally has -- one table, one slide, a list, a
    ``{slide_id: ...}`` mapping, or a cohort manifest -- and presents them all as
    ``(slide_id, AnnData)`` pairs. Iterating a manifest re-opens the stores, so
    memory stays bounded to one slide; iterating in-memory inputs is free.

    ``slide_id`` is ``None`` for a single bare AnnData, which is how callers know
    not to stamp a provenance column over one that may already be there (e.g. on
    the output of :func:`concat_slides`).
    """

    def __init__(
        self,
        slides,
        *,
        tile_key: str = DEFAULT_TILE_KEY,
        store_col: str = "store",
        slide_id_col: str = SLIDE_ID,
    ):
        self._tile_key = tile_key
        self._table_key = tile_table_key(tile_key)
        self._store_col = store_col
        self._slide_id_col = slide_id_col
        self._manifest = None
        self._items: Optional[list] = None

        if _is_slides_table(slides):
            self._manifest = slides
        elif _is_anndata(slides):
            self._items = [(None, slides)]
        elif _is_wsidata(slides):
            self._items = [(slide_id_from(slides),
                            _require_table(slides, self._table_key, slide_id_from(slides)))]
        elif isinstance(slides, Mapping):
            self._items = [(str(k), self._coerce(v, str(k))) for k, v in slides.items()]
        elif isinstance(slides, Sequence) and not isinstance(slides, (str, bytes)):
            self._items = [
                (self._default_id(v, i), self._coerce(v, self._default_id(v, i)))
                for i, v in enumerate(slides)
            ]
        else:
            raise TypeError(
                "slides must be an AnnData, a WSIData, a sequence or mapping of "
                f"either, or a slides_table DataFrame; got {type(slides).__name__}."
            )

    def _default_id(self, obj, i: int) -> str:
        return slide_id_from(obj) if _is_wsidata(obj) else str(i)

    def _coerce(self, obj, slide_id: str) -> "AnnData":
        if _is_anndata(obj):
            return obj
        if _is_wsidata(obj):
            return _require_table(obj, self._table_key, slide_id)
        raise TypeError(
            f"slides entry '{slide_id}' must be an AnnData or WSIData, "
            f"got {type(obj).__name__}."
        )

    def __iter__(self) -> Iterator[tuple[Optional[str], "AnnData"]]:
        if self._items is not None:
            yield from self._items
            return
        for slide_id, wsi in iter_slides(
            self._manifest,
            store_col=self._store_col,
            slide_id_col=self._slide_id_col,
        ):
            yield slide_id, _require_table(wsi, self._table_key, slide_id)

    def first(self) -> "AnnData":
        """The first slide's table, for deriving an empty result with the right schema."""
        for _, table in self:
            return table
        raise ValueError("No slides to read from.")
