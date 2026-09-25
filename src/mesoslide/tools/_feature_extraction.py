"""Vision foundation-model patch embedding for ezslide/wsidata slides.

Runs a lazyslide-models vision encoder over the tiles of an ezslide/wsidata
`WSIData` slide and writes the resulting embeddings into a shared per-tile-set
AnnData table (`slide.tables[table_key]`, default `f"{tile_key}_table"`), one
`obsm` entry per model. The table is stored under a key distinct from the
tiles shapes element (`tile_key`) because SpatialData requires element names
to be unique across *all* element types, not just within one. `.X` holds a
sparse feature matrix derived from a pooled embedding via `sparse=True` (e.g.
a sparse autoencoder), per AnnData's own layering convention --
`sc.pp.neighbors(table, use_rep=key_added)` redirects scanpy's graph/
clustering calls onto a given `obsm` entry without needing them in `.X`.

Named `feature_extraction` to match lazyslide's own `zs.tl.feature_extraction`.
`embed_patch` is kept as a deprecated alias.

Internally, `feature_extraction` builds a `ModelStage` chain and delegates to
`run_model_stages` (see `mesoslide.tools._model_stage`), the same engine
`fit_token_clusterer`/`extract_cluster_maps` use at patch-table scope. A single
call computes exactly one chain from the image -- pooled (optionally
followed by `sparse_transform`), or dense (followed by the required
`reducer`) -- since these are two independent branches off the same image
batch, not a linear chain; `dense=True` and `sparse=True` together raise.

The KMeans post-processing branch previously handled by this module has
moved, unchanged, to `mesoslide.tools._legacy._embed_patch`.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
)

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy.sparse import csr_matrix, hstack, issparse, vstack
from spatialdata.models import TableModel
from tqdm.auto import tqdm

from ._model_stage import (
    CallableStage,
    ImageModelStage,
    ModelStage,
    _model_name,
    iter_array_batches,
    to_numpy,
)

if TYPE_CHECKING:
    import scipy.sparse as sp
    from lazyslide_models.base import ImageModel
    from wsidata import WSIData


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


def _validate_sparse_shape(matrix, n_tiles: int):
    """Check a sparse_transform's output has one row per tile."""
    if matrix.shape[0] != n_tiles:
        raise ValueError(
            f"sparse_transform must return a matrix of shape ({n_tiles}, M) "
            f"-- one row per tile -- got {tuple(matrix.shape)}."
        )


def _write_sparse_features(table: AnnData, prefix: str, matrix) -> AnnData:
    """Write `matrix` into table.X under var names f"{prefix}_0", f"{prefix}_1", ...

    Any existing columns under this prefix are replaced; columns from other
    prefixes (other sparse_transform calls, or other keys) are preserved.
    Rebuilds the AnnData rather than mutating in place, since X and var_names
    must change together -- AnnData validates each against the other's
    current shape, so assigning them one at a time is rejected.
    """
    matrix = matrix if issparse(matrix) else csr_matrix(matrix)
    new_vars = [f"{prefix}_{i}" for i in range(matrix.shape[1])]

    existing_vars = list(table.var_names)
    diff_vars = [v for v in existing_vars if not v.startswith(f"{prefix}_")]

    base = table.X
    base = csr_matrix((table.n_obs, 0)) if base is None else csr_matrix(base)
    if len(diff_vars) < len(existing_vars):
        keep_cols = [i for i, v in enumerate(existing_vars) if v in diff_vars]
        base = base[:, keep_cols]

    return AnnData(
        X=hstack([base, matrix]).tocsr(),
        obs=table.obs,
        var=pd.DataFrame(index=pd.Index(diff_vars + new_vars)),
        obsm=dict(table.obsm),
        uns=dict(table.uns),
    )


# ---------------------------------------------------------------------------
# run_model_stages: the general chain-of-stages engine
# ---------------------------------------------------------------------------

_PROVENANCE_KEY = "model_stages"


def _stage_key_exists(table: AnnData, stage: ModelStage) -> bool:
    if stage.output_kind == "sparse":
        return any(v.startswith(f"{stage.name}_") for v in table.var_names)
    return stage.name in table.obsm


def _stage_key_shape_ok(table: AnnData, stage: ModelStage) -> bool:
    if stage.output_kind == "sparse":
        return True
    arr = table.obsm[stage.name]
    return arr.shape[0] == table.n_obs


def _stage_provenance(stage: ModelStage) -> dict:
    fn = getattr(stage, "provenance", None)
    return fn() if callable(fn) else {}


def _check_device_available(device: Optional[str], stage_index: int, stage_name: str) -> None:
    if device is None or "cuda" not in device:
        return
    if not torch.cuda.is_available():
        raise ValueError(
            f"stage {stage_index} ('{stage_name}'): requested device '{device}' "
            f"but CUDA is not available."
        )
    if ":" in device:
        idx = int(device.split(":", 1)[1])
        if idx >= torch.cuda.device_count():
            raise ValueError(
                f"stage {stage_index} ('{stage_name}'): requested device "
                f"'{device}' but only {torch.cuda.device_count()} CUDA "
                f"device(s) are visible."
            )


def _preflight(
    table: AnnData,
    stages: List[ModelStage],
    *,
    overwrite: Optional[bool],
    devices: Optional[List[str]],
    input_key: Optional[str],
    has_pixels: bool,
) -> Optional[int]:
    """Validate the whole chain before anything runs.

    Returns the index of the first stage that actually needs to execute, or
    None if every stage is already cached and nothing needs to run.
    """
    n = len(stages)
    seen: Dict[str, int] = {}
    exists = [False] * n
    eff_overwrite = [False] * n

    for i, stage in enumerate(stages):
        if stage.cache and stage.name in seen:
            raise ValueError(
                f"stage {i} ('{stage.name}'): duplicate name, already used by "
                f"stage {seen[stage.name]} in this call."
            )
        if stage.cache:
            seen[stage.name] = i

        eff_overwrite[i] = stage.overwrite if overwrite is None else overwrite
        eff_device = devices[i] if devices is not None else stage.device
        _check_device_available(eff_device, i, stage.name)

        exists[i] = bool(stage.cache) and _stage_key_exists(table, stage)
        if exists[i]:
            if not _stage_key_shape_ok(table, stage):
                raise ValueError(
                    f"stage {i} ('{stage.name}'): existing cached entry is "
                    f"structurally incompatible with output_kind="
                    f"'{stage.output_kind}' (e.g. a row-count mismatch). "
                    f"Refusing to overwrite automatically -- inspect or "
                    f"remove it manually."
                )
            recorded = table.uns.get(_PROVENANCE_KEY, {}).get(stage.name)
            current = _stage_provenance(stage)
            if recorded is not None and current and recorded != current and not eff_overwrite[i]:
                raise ValueError(
                    f"stage {i} ('{stage.name}'): existing cache was produced "
                    f"by a different configuration ({recorded!r} vs "
                    f"{current!r}). Pass overwrite=True to replace it."
                )

    # Scan backward: a cacheable, still-valid stage is a safe resume point
    # (everything before it can stay skipped), so we stop extending run_from
    # further back once we hit one. A cache=False stage has no persisted
    # state of its own -- it only needs to (re)run when something after it
    # does, so it never independently starts a run_from extension.
    run_from = None
    if any(stage.cache for stage in stages):
        for i in range(n - 1, -1, -1):
            stage = stages[i]
            if stage.cache:
                dirty = (not exists[i]) or eff_overwrite[i]
                if dirty:
                    run_from = i
                elif run_from is not None:
                    break
            elif run_from is not None:
                run_from = i
    else:
        # No stage in the chain persists anything -- there is nothing to
        # resume from, so the whole chain always runs.
        run_from = 0

    if run_from is not None:
        first = stages[run_from]
        if run_from == 0:
            if first.input_kind == "image":
                if not has_pixels:
                    raise ValueError(
                        f"stage 0 ('{first.name}'): input_kind='image' but no "
                        f"`slides` was provided and patches.obs['_slide_ref'] "
                        f"isn't populated either -- pass slides=, or build "
                        f"patches via mesoslide.select_* (or attach one with "
                        f"mesoslide.attach_slide_ref) so it can read pixels "
                        f"on its own."
                    )
            else:
                if input_key is None:
                    raise ValueError(
                        f"stage 0 ('{first.name}'): input_kind="
                        f"'{first.input_kind}' has no pixel source; pass "
                        f"input_key= naming an existing cached array."
                    )
                if input_key not in table.obsm:
                    raise ValueError(
                        f"stage 0 ('{first.name}'): input_key='{input_key}' "
                        f"not found in obsm."
                    )
        else:
            prev = stages[run_from - 1]
            if prev.output_kind != first.input_kind:
                raise ValueError(
                    f"stage {run_from} ('{first.name}'): input_kind="
                    f"'{first.input_kind}' has no source -- stage "
                    f"{run_from - 1} ('{prev.name}') produces output_kind="
                    f"'{prev.output_kind}'."
                )
            if not exists[run_from - 1]:
                raise ValueError(
                    f"stage {run_from} ('{first.name}'): input_kind="
                    f"'{first.input_kind}' has no source -- stage "
                    f"{run_from - 1} ('{prev.name}') produces output_kind="
                    f"'{prev.output_kind}' but its key '{prev.name}' does not "
                    f"exist yet and stage {run_from - 1} is not scheduled to run."
                )

    return run_from


def _build_table_for_slide(slide: "WSIData", tile_key: str) -> AnnData:
    """Bootstrap a fresh tile table when `slide` has none yet for `tile_key`."""
    tiles = slide[tile_key]
    bounds = tiles.bounds
    n_tiles = len(tiles)
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
    return TableModel.parse(
        AnnData(obs=obs),
        region=tile_key, region_key="library_id", instance_key="tile_id",
    )


def run_model_stages(
    slide_or_patches,
    stages: Union[Sequence[ModelStage], ModelStage],
    *,
    slides=None,
    input_key: Optional[str] = None,
    tile_key: str = "tiles",
    table_key: Optional[str] = None,
    overwrite: Optional[bool] = None,
    device: Union[str, Sequence[str], None] = None,
    batch_size: Optional[int] = 32,
    num_workers: int = 0,
    block: bool = True,
    cache_size: int = 4,
    save: bool = True,
    progress_bar: bool = True,
) -> Union["WSIData", AnnData]:
    """Run an ordered chain of `ModelStage`s against a slide or a patch table.

    Parameters
    ----------
    slide_or_patches : WSIData or AnnData
        A whole-slide `WSIData` (tiles read via `ezslide.tile_images`,
        results cached in `slide.tables[table_key]` and persisted via
        `slide.write_element` when `save=True`), or a patch-table `AnnData`
        (tiles read via `mesoslide.pp.extract_patch_images(slide_or_patches,
        slides)`, results cached directly in `.obsm` -- no disk persistence).
    stages : ModelStage or sequence of ModelStage
        A linear chain: stage i's output feeds stage i+1's input. The first
        stage need not consume images -- see `input_key`.
    slides : optional
        Needed when `slide_or_patches` is a patch-table AnnData and the
        first stage that actually needs to run consumes images -- unless
        `slide_or_patches.obs['_slide_ref']` is already populated (set
        automatically by `mesoslide.select_top_patches` and friends), in
        which case pixels are read from that instead and `slides` can be
        omitted.
    input_key : str, optional
        Name of an existing `.obsm` entry to start the chain from, when the
        first stage that needs to run doesn't consume images (e.g. resuming
        with a new downstream stage against a previously cached embedding).
    overwrite : bool, optional
        None (default) respects each stage's own `.overwrite`; a bool
        overrides every stage's setting for this call only.
    device : str, sequence of str, optional
        None (default) respects each stage's own `.device`; a single string
        overrides every stage for this call; a sequence the same length as
        `stages` overrides positionally.
    batch_size : int or None, default=32
        Rows per batch, shared by every stage in the chain. None passes the
        whole `obsm` array as one batch; only allowed when the chain starts
        from `input_key` rather than images.
    num_workers, block, cache_size
        Forwarded to the pixel-reading path (whole-slide `ezslide.tile_images`
        / `DataLoader`, or `extract_patch_images` at patch-table scale).
    save : bool, default=True
        Whole-slide only: persist the updated table via `slide.write_element`.

    Returns
    -------
    The same `slide_or_patches` object (mutated in place for the whole-slide
    case; also mutated in place for the patch-table case, except a stage
    with `output_kind="sparse"` at patch-table scale, which is unsupported --
    see Raises).

    Raises
    ------
    ValueError
        If the chain fails preflight validation (see module docstring):
        unresolvable input, duplicate stage names, an incompatible existing
        cache entry, or an unavailable device.
    NotImplementedError
        A patch-table `AnnData` with a stage declaring `output_kind="sparse"`
        -- writing sparse features requires rebuilding `.X`/`.var` together
        into a *new* AnnData (see `_write_sparse_features`), which can't be
        reflected back onto a caller-owned object the way whole-slide's
        `slide.tables[table_key] = table` reassignment can.
    """
    stages = list(stages) if isinstance(stages, (list, tuple)) else [stages]
    if not stages:
        raise ValueError("stages must contain at least one ModelStage.")

    is_patch_table = hasattr(slide_or_patches, "obs")

    if is_patch_table and any(s.output_kind == "sparse" for s in stages):
        raise NotImplementedError(
            "output_kind='sparse' stages are not supported when running "
            "against a patch-table AnnData directly (sparse output requires "
            "rebuilding X/var together into a new AnnData object, which "
            "can't be reflected back onto a caller-owned reference)."
        )

    devices: Optional[List[str]] = None
    if device is not None:
        devices = [device] * len(stages) if isinstance(device, str) else list(device)
        if len(devices) != len(stages):
            raise ValueError(
                f"device sequence must have length {len(stages)} (one per "
                f"stage), got {len(devices)}."
            )
        for stage, d in zip(stages, devices):
            stage.to(d)

    if is_patch_table:
        table = slide_or_patches
        from mesoslide._slides import SLIDE_REF
        has_pixels = slides is not None or (
            SLIDE_REF in table.obs.columns and table.obs[SLIDE_REF].notna().any()
        )
    else:
        table_key = table_key or f"{tile_key}_table"
        table = slide_or_patches.tables.get(table_key)
        if table is None:
            table = _build_table_for_slide(slide_or_patches, tile_key)
        has_pixels = True

    run_from = _preflight(
        table, stages, overwrite=overwrite, devices=devices,
        input_key=input_key, has_pixels=has_pixels,
    )
    if run_from is None:
        return slide_or_patches

    active = stages[run_from:]
    first = active[0]

    if first.input_kind == "image":
        if batch_size is None:
            raise ValueError(
                f"batch_size=None is only supported when the chain starts "
                f"from an obsm array (input_key); stage '{first.name}' reads "
                f"images."
            )
        if is_patch_table:
            from mesoslide.preprocessing._extract_patches import extract_patch_images
            pixels = extract_patch_images(
                table, slides, tile_key=tile_key, channel_first=True,
                progress_bar=progress_bar,
            )
            batches = iter_array_batches(pixels, batch_size)
        else:
            import ezslide
            from torch.utils.data import DataLoader

            dataset = ezslide.tile_images(
                slide_or_patches, tile_key=tile_key, transform=None,
                block=block, num_workers=num_workers, cache_size=cache_size,
            )
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers,
                multiprocessing_context="spawn" if num_workers > 0 else None,
            )
            # ezslide/wsidata's tile datasets hand back channel-last (H, W, C)
            # images (see ezslide.dataset.patch's docstring); every ModelStage
            # expects channel-first (B, C, H, W), same as the patch-table
            # branch above gets from extract_patch_images(channel_first=True).
            batches = (b["image"].permute(0, 3, 1, 2).contiguous() for b in loader)
    else:
        start_key = input_key if run_from == 0 else stages[run_from - 1].name
        start = table.obsm[start_key]
        batches = iter_array_batches(start, batch_size or max(len(start), 1))

    accum: Dict[int, list] = {j: [] for j in range(len(active)) if active[j].cache}

    iterator = tqdm(batches, desc="Running model stages") if progress_bar else batches
    for batch in iterator:
        x = batch
        for j, stage in enumerate(active):
            x = stage(x)
            if not stage.cache:
                continue
            if stage.output_kind == "sparse":
                accum[j].append(x if issparse(x) else csr_matrix(x))
            else:
                accum[j].append(to_numpy(x))

    for j, stage in enumerate(active):
        if not stage.cache:
            continue
        if stage.output_kind == "sparse":
            result = vstack(accum[j]).tocsr()
            table = _write_sparse_features(table, stage.name, result)
        else:
            result = np.concatenate(accum[j], axis=0)
            table.obsm[stage.name] = result
        table.uns.setdefault(_PROVENANCE_KEY, {})[stage.name] = _stage_provenance(stage)

    if is_patch_table:
        return table

    slide_or_patches.tables[table_key] = table
    if save:
        slide_or_patches.write_element(table_key, overwrite=True)
    return slide_or_patches


# ---------------------------------------------------------------------------
# feature_extraction: the friendly, whole-slide-oriented wrapper
# ---------------------------------------------------------------------------

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
    sparse: bool = False,
    sparse_transform: "Callable[[np.ndarray], sp.spmatrix] | None" = None,
    sparse_key_added: str | None = None,
    sparse_batch_size: int | None = None,
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
    progress_bar: bool = True,
) -> "WSIData":
    """Embed every tile of `slide[tile_key]` with a vision foundation model.

    A single call computes exactly one chain from the image: the pooled
    embedding (optionally followed by `sparse_transform`), or, when
    `dense=True`, the dense per-token embedding (followed by the required
    `reducer`) -- these are two independent branches off the same image
    batch, not a linear chain, so `dense=True` and `sparse=True` together
    raise `ValueError`; call `feature_extraction` twice, once per chain, to
    get both. The pooled embedding is written to
    `slide.tables[table_key].obsm[key_added]` (`key_added` defaults to the
    resolved model name). Requires `slide[tile_key]` to already exist (see
    `lazyslide.pp.tile_tissues`).

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
        resolved model name. Ignored when `dense=True` (nothing pooled is
        computed in that call).
    dense
        Compute a reduced per-token map for each tile, via `reducer`, instead
        of the pooled embedding. Requires a ViT-style model (`grid_size`,
        `patch_size`, `encode_image_dense`) -- true today for every
        `MODEL_REGISTRY` entry built on `lazyslide_models.base.TimmViTModel`
        (uni, uni2, virchow, virchow2, ...).
    reducer
        Required when `dense=True` and the dense map still needs computing.
        Maps a batch of per-token embeddings, `(B, N_tokens, D)`, to one score
        per token, `(B, N_tokens)` (a trailing size-1 axis is also accepted
        and squeezed). Called once per batch on a tensor still on `device` --
        the reducer decides whether to move it to CPU. The raw, unreduced
        per-token tensor is never cached; only `reducer`'s output is.
    dense_key_added
        `obsm` key for the reduced dense map. Defaults to `f"{key_added}_dense"`,
        mirroring lazyslide's own dense-key suffix. The map is flat,
        `(n_tiles, grid_h * grid_w)`; reshape a row to `(grid_h, grid_w)` using
        the model's own `grid_size` when a spatial layout is needed -- token
        order is row-major (`k = row * grid_w + col`), matching timm's own
        patch-token flattening.
    sparse
        Also derive a sparse feature matrix from the pooled embedding
        (`table.obsm[key_added]`) via `sparse_transform`, and write it into
        `table.X`. Mutually exclusive with `dense=True`.
    sparse_transform
        Required when `sparse=True` and the result still needs computing.
        Maps the entire table's pooled embedding, `(N_tiles, D)`, to a sparse
        feature matrix, `(N_tiles, M)` (e.g. a sparse autoencoder's
        `.transform()`). Runs after the pooled embedding is cached, reading it
        back from `obsm` in chunks of `sparse_batch_size` rows. Unlike
        `reducer`, it receives a numpy array, and owns any device placement
        it needs internally.
    sparse_batch_size
        Rows per `sparse_transform` call, independent of `batch_size`. None
        (default) calls it once on the full pooled array, leaving batching to
        the transform itself (e.g. `SparseAutoencoder.batch_size`).
    sparse_key_added
        Prefix for the new `var` names, written as `f"{sparse_key_added}_{i}"`
        for `i` in `range(M)`. Defaults to `f"{key_added}_sparse"`, mirroring
        `dense_key_added`'s `f"{key_added}_dense"` default.
    device
        Torch device to run the model on. Defaults to "cuda" if available,
        else "cpu".
    block
        Use ezslide's block-deduping tile reader (faster for dense or
        overlapping tile grids). See `ezslide.tile_images`.
    amp
        Run the forward pass under `torch.autocast` (CUDA only).
    overwrite
        Recompute even if a key is already present in the table.
    save
        Persist the updated table back to the slide's Zarr store via
        `slide.write_element`, which requires `slide` to already be backed
        by one (i.e. `slide.write(...)` has been called at least once). Set
        to `False` to only mutate `slide` in memory.
    """
    if dense and sparse:
        raise ValueError(
            "dense=True and sparse=True can't be combined in one call: "
            "sparse_transform reads the pooled embedding while dense/reducer "
            "reads the dense (per-token) embedding -- two different chains "
            "from the image. Call feature_extraction twice, once per chain."
        )
    if dense and reducer is None:
        raise ValueError(
            "dense=True requires a reducer callable that maps per-token "
            "embeddings (B, N_tokens, D) to per-token scalars (B, N_tokens); "
            "pass reducer=..."
        )
    if sparse and sparse_transform is None:
        raise ValueError(
            "sparse=True requires a sparse_transform callable that maps the "
            "pooled embedding (N_tiles, D) to a sparse feature matrix "
            "(N_tiles, M); pass sparse_transform=..."
        )

    key_added = key_added or (model if isinstance(model, str) else _model_name(model))

    if dense:
        dense_key = dense_key_added or f"{key_added}_dense"

        def _validated_reducer(patch_tokens):
            return _validate_reduced_shape(reducer(patch_tokens), patch_tokens.shape)

        stages: List[ModelStage] = [
            ImageModelStage(
                model, dense=True, name=dense_key, cache=False,
                overwrite=overwrite, device=device, amp=amp,
                model_path=model_path, token=token,
            ),
            CallableStage(
                _validated_reducer, name=dense_key, input_kind="dense",
                output_kind="dense", cache=True, overwrite=overwrite,
            ),
        ]
    else:
        stages = [
            ImageModelStage(
                model, dense=False, name=key_added, cache=True,
                overwrite=overwrite, device=device,
                model_path=model_path, token=token,
            ),
        ]

    slide = run_model_stages(
        slide, stages, tile_key=tile_key, table_key=table_key,
        batch_size=batch_size, num_workers=num_workers, block=block,
        cache_size=cache_size, save=save, progress_bar=progress_bar,
    )
    if not sparse:
        return slide

    # Separate call so sparse coding reads the cached pooled embedding with
    # its own batch size instead of running inside the image batch loop.
    def _validated_sparse_transform(pooled):
        pooled = to_numpy(pooled)
        matrix = sparse_transform(pooled)
        _validate_sparse_shape(matrix, pooled.shape[0])
        return matrix

    sparse_stage = CallableStage(
        _validated_sparse_transform, name=sparse_key_added or f"{key_added}_sparse",
        input_kind="pooled", output_kind="sparse", cache=True, overwrite=overwrite,
    )
    return run_model_stages(
        slide, [sparse_stage], input_key=key_added, tile_key=tile_key,
        table_key=table_key, batch_size=sparse_batch_size, save=save,
        progress_bar=progress_bar,
    )


def embed_patch(*args, **kwargs):
    """Deprecated alias for `feature_extraction`."""
    warnings.warn(
        "mesoslide.tl.embed_patch is deprecated, use "
        "mesoslide.tl.feature_extraction instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return feature_extraction(*args, **kwargs)
