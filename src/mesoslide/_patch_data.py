"""A patch table, as a real `SpatialData` object.

`PatchData` wraps exactly what a bare "patches" `AnnData` already was in
spirit -- a row-subset of a `WSIData`'s own `tiles_table` (a SpatialData
Table annotating a `tiles` Shapes element) -- but makes the shapes side
explicit too, so per-patch data that doesn't fit `.obsm`'s one-array-per-row
contract (cell polygons, phenotype tables) has a genuine, serializable home
(`shapes`/`tables`) instead of being dumped in `.uns`, which can't survive
`write_h5ad`/`write_zarr`.

Mirrors how `wsidata.WSIData` already subclasses `spatialdata.SpatialData`
for a single slide: a typed constructor plus `@property` accessors layered
on a generic base, no metaclass tricks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from spatialdata import SpatialData

if TYPE_CHECKING:
    import anndata as ad
    import geopandas as gpd
    import numpy as np
    import pandas as pd


TILES_KEY = "tiles"
TILES_TABLE_KEY = "tiles_table"


class PatchData(SpatialData):
    """A patch table backed by `shapes["tiles"]` + `tables["tiles_table"]`.

    `.obs`/`.obsm`/`.X`/`.var`/`len(...)` all delegate to `tables["tiles_table"]`
    -- the same AnnData a bare "patches" table already was -- so existing
    code that only reads/writes those (e.g. `extract_patch_images`,
    `extract_cluster_maps`, every `GalleryPlan.add_*_row`) works unchanged.
    Cell-aware code additionally populates `shapes["cells"]`/
    `tables["cells_phenotypes"]` once needed (see
    `mesoslide.preprocessing.extract_patch_cells`/`extract_patch_cell_phenotypes`).
    """

    def __init__(self, *, shapes: dict, tables: dict, **kwargs):
        if TILES_KEY not in shapes:
            raise ValueError(f"PatchData requires a '{TILES_KEY}' shapes element.")
        if TILES_TABLE_KEY not in tables:
            raise ValueError(f"PatchData requires a '{TILES_TABLE_KEY}' tables element.")
        super().__init__(shapes=shapes, tables=tables, **kwargs)

    @property
    def obs(self) -> "pd.DataFrame":
        return self.tables[TILES_TABLE_KEY].obs

    @obs.setter
    def obs(self, value: "pd.DataFrame") -> None:
        self.tables[TILES_TABLE_KEY].obs = value

    @property
    def obsm(self):
        return self.tables[TILES_TABLE_KEY].obsm

    @property
    def n_obs(self) -> int:
        return self.tables[TILES_TABLE_KEY].n_obs

    @property
    def X(self) -> "np.ndarray":
        return self.tables[TILES_TABLE_KEY].X

    @property
    def var(self) -> "pd.DataFrame":
        return self.tables[TILES_TABLE_KEY].var

    @property
    def var_names(self):
        return self.tables[TILES_TABLE_KEY].var_names

    @property
    def obs_names(self):
        return self.tables[TILES_TABLE_KEY].obs_names

    @property
    def uns(self):
        return self.tables[TILES_TABLE_KEY].uns

    def __len__(self) -> int:
        return len(self.tables[TILES_TABLE_KEY])

    def __getitem__(self, key) -> "PatchData":
        """Row-index into the patch table, like AnnData's own `__getitem__`.

        `SpatialData.__getitem__` normally means "look up an element by
        name" (e.g. `wsi["cells"]`), but no current code does that on a bare
        patch table -- it does `patches[bool_mask]`/`patches[[]]` instead
        (`_patch_selector.py`'s `_empty_result`/`select_exemplar_patches`'s
        own docstring example). This delegates the actual key-type handling
        to AnnData's own well-tested `__getitem__`, then recovers which
        positions survived via `obs_names` (unique tile ids) to keep
        `shapes["tiles"]` row-aligned.
        """
        table = self.tables[TILES_TABLE_KEY]
        sub_table = table[key].copy()
        if len(sub_table) == 0:
            # spatialdata.models.ShapesModel hard-forbids an empty shapes
            # element (`len(geometry) == 0` always raises, even via
            # `.parse()`) -- PatchData structurally cannot represent zero
            # patches. Raise a clear message here rather than letting the
            # caller hit a cryptic spatialdata ValidationError three frames
            # down.
            raise ValueError(
                "This selection has zero patches -- PatchData cannot represent "
                "an empty patch table (spatialdata.models.ShapesModel disallows "
                "an empty shapes element). Check len(patches) > 0 first."
            )
        pos = table.obs.index.get_indexer(sub_table.obs.index)
        sub_tiles = self.shapes[TILES_KEY].iloc[pos]
        extra_shapes = {k: v for k, v in self.shapes.items() if k != TILES_KEY}
        extra_tables = {k: v for k, v in self.tables.items() if k != TILES_TABLE_KEY}
        return PatchData(
            shapes={TILES_KEY: sub_tiles, **extra_shapes},
            tables={TILES_TABLE_KEY: sub_table, **extra_tables},
        )

    def copy(self) -> "PatchData":
        """A copy of every element -- `spatialdata.SpatialData` has no `.copy()` at all."""
        return PatchData(
            shapes={k: v.copy() for k, v in self.shapes.items()},
            tables={k: v.copy() for k, v in self.tables.items()},
        )

    def write(self, file_path, **kwargs) -> None:
        """Persist to a Zarr store.

        Automatically strips live `WSIData` references from `_slide_ref`
        first (see `mesoslide.strip_slide_refs`) -- those aren't
        picklable/zarr-writable -- on a copy, so the in-memory `patches`
        object this is called on is never mutated.
        """
        from ._slides import strip_slide_refs

        stripped = strip_slide_refs(self)
        SpatialData.write(stripped, file_path, **kwargs)


def read_patch_data(path, slides=None) -> "PatchData":
    """Read a `PatchData` back from a Zarr store written by `PatchData.write`.

    `_slide_ref` comes back as plain path strings (whatever
    `strip_slide_refs` left before writing). Pass `slides` (anything
    `mesoslide.attach_slide_ref` accepts: a manifest, a single `WSIData`, or
    a sequence/mapping of them) to re-attach live `WSIData` references in
    the same call; omit it to leave `_slide_ref` as path strings, ready for
    a later `attach_slide_ref` call.
    """
    import spatialdata as sd

    from ._slides import attach_slide_ref

    raw = sd.read_zarr(path)
    result = PatchData(shapes=dict(raw.shapes), tables=dict(raw.tables))
    if slides is not None:
        result = attach_slide_ref(result, slides)
    return result
