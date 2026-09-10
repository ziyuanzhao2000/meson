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

Named `feature_extraction` to match lazyslide's own `zs.tl.feature_extraction`.
`embed_patch` is kept as a deprecated alias.

The SAE and KMeans post-processing branches previously handled by this module
have moved, unchanged, to `mesoslide.tools._legacy._embed_patch`.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Callable

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


def _require_dense_capable(model, model_name: str) -> None:
    """Check the model can produce per-token embeddings for dense=True.

    Raises rather than letting AttributeError surface later mid-loop, and
    names which registered models already work: everything built on
    `lazyslide_models.base.TimmViTModel` (uni, uni2, virchow, virchow2, ...).
    """
    from lazyslide_models.base import ViTModelProtocol

    if not isinstance(model, ViTModelProtocol):
        raise NotImplementedError(
            f"dense=True requires a ViT-style model exposing grid_size, "
            f"patch_size and encode_image_dense (see "
            f"lazyslide_models.base.ViTModelProtocol); '{model_name}' does not. "
            f"Registered models built on lazyslide_models.base.TimmViTModel "
            f"(uni, uni2, virchow, virchow2, ...) support this."
        )


def _validate_reduced_shape(reduced, tokens_shape: tuple) -> torch.Tensor:
    """Check a reducer's output against the (batch, n_tokens) it must produce."""
    b, n, _ = tokens_shape
    if not torch.is_tensor(reduced):
        reduced = torch.as_tensor(reduced)
    if reduced.ndim == 3 and reduced.shape[-1] == 1:
        reduced = reduced[..., 0]
    if tuple(reduced.shape) != (b, n):
        raise ValueError(
            f"reducer must return shape {(b, n)} (batch, n_tokens) -- it should "
            f"collapse the embedding dimension D to one scalar per token; got "
            f"{tuple(reduced.shape)} from an input of shape {tuple(tokens_shape)}."
        )
    return reduced


def feature_extraction(
    slide: "WSIData",
    model: "str | ImageModel",
    *,
    tile_key: str = "tiles",
    table_key: str | None = None,
    key_added: str | None = None,
    dense: bool = False,
    reducer: "Callable[[torch.Tensor], torch.Tensor] | None" = None,
    dense_key_added: str | None = None,
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

    The pooled embedding is written to `slide.tables[table_key].obsm[key_added]`
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
        `obsm` key to write the pooled embeddings under. Defaults to the
        resolved model name.
    dense
        Also compute a reduced per-token map for each tile, via `reducer`.
        Requires a ViT-style model (`grid_size`, `patch_size`,
        `encode_image_dense`) -- true today for every `MODEL_REGISTRY` entry
        built on `lazyslide_models.base.TimmViTModel` (uni, uni2, virchow,
        virchow2, ...).
    reducer
        Required when `dense=True` and the dense map still needs computing.
        Maps a batch of per-token embeddings, `(B, N_tokens, D)`, to one score
        per token, `(B, N_tokens)` (a trailing size-1 axis is also accepted
        and squeezed). Called once per DataLoader batch inside the same
        `inference_mode`/`autocast` context as the forward pass, on a tensor
        still on `device` -- the reducer decides whether to move it to CPU.
        Unlike lazyslide's own `dense=True` (which stores every unreduced
        token vector as a new table row, `(n_tiles * N_tokens, D)`), reducing
        first keeps the result to a single flat `obsm` entry the same size as
        any other feature: `(n_tiles, N_tokens)`.
    dense_key_added
        `obsm` key for the reduced dense map. Defaults to `f"{key_added}_dense"`,
        mirroring lazyslide's own dense-key suffix. The map is flat,
        `(n_tiles, grid_h * grid_w)`; reshape a row to `(grid_h, grid_w)` using
        the model's own `grid_size` when a spatial layout is needed -- token
        order is row-major (`k = row * grid_w + col`), matching timm's own
        patch-token flattening.
    device
        Torch device to run the model on. Defaults to "cuda" if available,
        else "cpu".
    block
        Use ezslide's block-deduping tile reader (faster for dense or
        overlapping tile grids). See `ezslide.tile_images`.
    amp
        Run the forward pass under `torch.autocast` (CUDA only).
    overwrite
        Recompute even if a key is already present in the table. Applies
        independently to `key_added` and `dense_key_added`: an already-cached
        one is left untouched unless `overwrite=True`, even while the other is
        being (re)computed.
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
    dense_key = dense_key_added or f"{key_added}_dense"

    table = slide.tables.get(table_key)
    need_pooled = overwrite or table is None or key_added not in table.obsm
    need_dense = dense and (overwrite or table is None or dense_key not in table.obsm)

    if not need_pooled and not need_dense:
        return slide

    if need_dense:
        if reducer is None:
            raise ValueError(
                "dense=True requires a reducer callable that maps per-token "
                "embeddings (B, N_tokens, D) to per-token scalars (B, N_tokens); "
                "pass reducer=..."
            )
        _require_dense_capable(model, model_name)

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
    pooled_outputs, dense_outputs = [], []

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Embedding tiles with {model_name}"):
            image = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp_on):
                if need_pooled:
                    pooled_outputs.append(model.encode_image(image).float().cpu().numpy())
                if need_dense:
                    patch_tokens = model.encode_image_dense(image).patch_tokens
                    reduced = _validate_reduced_shape(
                        reducer(patch_tokens), patch_tokens.shape
                    )
                    dense_outputs.append(reduced.float().cpu().numpy())

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
        slide.tables[table_key] = table

    if need_pooled:
        table.obsm[key_added] = np.vstack(pooled_outputs).astype(np.float32)
    if need_dense:
        table.obsm[dense_key] = np.vstack(dense_outputs).astype(np.float32)

    if save:
        slide.write_element(table_key, overwrite=True)
    return slide


def embed_patch(*args, **kwargs):
    """Deprecated alias for `feature_extraction`."""
    warnings.warn(
        "mesoslide.tl.embed_patch is deprecated, use "
        "mesoslide.tl.feature_extraction instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return feature_extraction(*args, **kwargs)
