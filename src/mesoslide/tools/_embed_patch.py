"""Vision foundation-model patch embedding for ezslide/wsidata slides.

Runs a lazyslide-models vision encoder over the tiles of an ezslide/wsidata
`WSIData` slide and writes the resulting embeddings into a shared per-tile-set
AnnData table (`slide.tables[table_key]`, default `f"{tile_key}_table"`), one
`obsm` entry per model. The table is stored under a key distinct from the
tiles shapes element (`tile_key`) because SpatialData requires element names
to be unique across *all* element types, not just within one. `.X` is left
untouched so it stays free for interpretable features (e.g. a sparse
autoencoder) computed downstream, per AnnData's own layering convention --
`sc.pp.neighbors(table, use_rep=key_added)` redirects scanpy's graph/
clustering calls onto a given `obsm` entry without needing them in `.X`.

The SAE and KMeans post-processing branches previously handled by this module
have moved, unchanged, to `mesoslide.tools._legacy._embed_patch`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from spatialdata.models import TableModel
from torch.utils.data import DataLoader
from tqdm import tqdm

if TYPE_CHECKING:
    from lazyslide_models.base import ImageModel
    from wsidata import WSIData


def _resolve_model(model, *, model_path=None, token=None):
    """Resolve a model name/instance to an `(ImageModel, name)` pair.

    Mirrors lazyslide's own `load_models` helper: a registered name is
    instantiated from `lazyslide_models.MODEL_REGISTRY`; an unregistered name
    falls back to a generic timm wrapper; an already-instantiated model is
    used as-is.
    """
    if isinstance(model, str):
        from lazyslide_models import MODEL_REGISTRY
        if model in MODEL_REGISTRY:
            instance, name = MODEL_REGISTRY[model](model_path=model_path, token=token), model
        else:
            from lazyslide_models import TimmModel
            instance, name = TimmModel(model, model_path=model_path, token=token), model
    else:
        instance, name = model, model.name

    from ._timm_transform_patch import patch_transform_if_needed
    patch_transform_if_needed(instance)
    return instance, name


def embed_patch(
    slide: "WSIData",
    model: "str | ImageModel",
    *,
    tile_key: str = "tiles",
    table_key: str | None = None,
    key_added: str | None = None,
    batch_size: int = 32,
    num_workers: int = 0,  # >0 uses multiprocessing_context="spawn" below, since
                            # tensorstore-backed readers aren't fork-safe
    device: str | None = None,
    token: str | None = None,
    model_path: str | Path | None = None,
    block: bool = True,
    cache_size: int = 4,
    amp: bool = False,
    overwrite: bool = False,
    save: bool = True,
) -> "WSIData":
    """Embed every tile of `slide[tile_key]` with a vision foundation model.

    Embeddings are written to `slide.tables[table_key].obsm[key_added]`
    (`key_added` defaults to the resolved model name). Requires
    `slide[tile_key]` to already exist (see `lazyslide.pp.tile_tissues`).

    Parameters
    ----------
    slide
        An ezslide/wsidata `WSIData` slide with a tile shapes element at
        `slide[tile_key]`.
    model
        A key into `lazyslide_models.MODEL_REGISTRY` (e.g. "uni2", "conch"),
        an arbitrary timm model name, or an already-instantiated
        `lazyslide_models` `ImageModel`.
    tile_key
        Name of the tile shapes element.
    table_key
        Name of the AnnData table the embeddings are stored in. Defaults to
        `f"{tile_key}_table"` -- it cannot default to `tile_key` itself, since
        SpatialData requires every element name to be unique across all
        element types, and `tile_key` already names the tiles shapes element.
    key_added
        `obsm` key to write the embeddings under. Defaults to the resolved
        model name.
    device
        Torch device to run the model on. Defaults to "cuda" if available,
        else "cpu".
    block
        Use ezslide's block-deduping tile reader (faster for dense or
        overlapping tile grids). See `ezslide.tile_images`.
    amp
        Run the forward pass under `torch.autocast` (CUDA only).
    overwrite
        Recompute even if `key_added` is already present in the table.
    save
        Persist the updated table back to the slide's Zarr store via
        `slide.write_element`, which requires `slide` to already be backed
        by one (i.e. `slide.write(...)` has been called at least once). Set
        to `False` to only mutate `slide` in memory.
    """
    import ezslide

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, model_name = _resolve_model(model, model_path=model_path, token=token)
    key_added = key_added or model_name
    table_key = table_key or f"{tile_key}_table"

    table = slide.tables.get(table_key)
    if table is not None and not overwrite and key_added in table.obsm:
        return slide

    model.to(device)
    model.model.eval()
    transform = model.get_transform()

    dataset = ezslide.tile_images(
        slide,
        tile_key=tile_key,
        transform=transform,
        block=block,
        num_workers=num_workers,
        cache_size=cache_size,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )

    n_tiles = len(dataset)
    amp_on = bool(amp) and "cuda" in str(device)
    outputs = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Embedding tiles with {model_name}"):
            image = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp_on):
                batch_embedding = model.encode_image(image)
            outputs.append(batch_embedding.float().cpu().numpy())

    embeddings = np.vstack(outputs).astype(np.float32)

    if table is None:
        tiles = slide[tile_key]
        bounds = tiles.bounds
        obs = pd.DataFrame({
            "tile_id": tiles["tile_id"].to_numpy() if "tile_id" in tiles.columns
                       else np.arange(n_tiles),
            "tissue_id": tiles["tissue_id"].to_numpy() if "tissue_id" in tiles.columns
                         else 0,
            "x": bounds["minx"].to_numpy(),
            "y": bounds["miny"].to_numpy(),
            "library_id": pd.Categorical([tile_key] * n_tiles),
        })
        # Index must be str for AnnData, but the tile_id *column* has to keep the
        # tiles element's own dtype: SpatialData matches instance_key values
        # against the element index, and a str/int mismatch makes the table look
        # unrelated to its shapes (spatialdata_plot then refuses to render it).
        # Assign from a bare array so the index inherits no name -- an index
        # named after a column whose values differ is rejected on write.
        # This mirrors wsidata.io.add_features.
        obs.index = obs["tile_id"].astype(str).to_numpy()
        table = TableModel.parse(
            AnnData(obs=obs),
            region=tile_key, region_key="library_id", instance_key="tile_id",
        )
        table.obsm[key_added] = embeddings
        slide.tables[table_key] = table
    else:
        table.obsm[key_added] = embeddings

    if save:
        slide.write_element(table_key, overwrite=True)
    return slide
