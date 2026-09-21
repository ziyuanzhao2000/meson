"""A chainable, self-describing stage in a patch/tile feature pipeline.

A `ModelStage` wraps one model (a vision foundation model, a fitted KMeans, a
sparse-coding transform, ...) and declares what it consumes, what it
produces, and where its own output should be cached -- so a pipeline of them
can be validated and run generically by `mesoslide.tools._feature_extraction
.run_model_stages`, instead of each caller (feature_extraction,
TokenClusterizer, extract_cluster_maps) hand-rolling its own batching loop
against a different model interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Optional, Protocol, runtime_checkable

import numpy as np
import torch

if TYPE_CHECKING:
    from lazyslide_models.base import ImageModel


def iter_array_batches(x, batch_size: int):
    """Yield `x` in row-chunks of `batch_size`, as torch tensors.

    `x` may be a numpy array, a torch tensor, or anything indexable with
    integer slices. Uint8 input is scaled to float [0, 1]; other dtypes are
    passed through as float.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    n = len(x)
    for i in range(0, n, batch_size):
        batch = x[i:i + batch_size]
        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
        if batch.dtype == torch.uint8:
            batch = batch.float() / 255.0
        else:
            batch = batch.float()
        yield batch


def to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


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


def _canonical_registry_name(name: str) -> str:
    """Case-insensitively match `name` against `MODEL_REGISTRY` keys.

    An already-instantiated model's own `.name` doesn't necessarily match its
    registry key's casing (e.g. UNI's `.name` is "UNI" but its registry key
    is "uni") -- callers that persist a resolved name (e.g.
    `TokenClusterizer.model_name`) and later feed it back into
    `_resolve_model` need the registry key, not whatever casing `.name`
    happened to report, or they'll silently miss the registry entry and fall
    through to the generic (and here, wrong) TimmModel path. Names with no
    matching key (TimmModel-resolved names, which were never registry keys to
    begin with) are returned unchanged.
    """
    from lazyslide_models import MODEL_REGISTRY

    if name in MODEL_REGISTRY:
        return name
    for key in MODEL_REGISTRY:
        if key.lower() == name.lower():
            return key
    return name


def _require_dense_capable(model, model_name: str) -> None:
    """Check the model can produce per-token embeddings (encode_image_dense).

    Raises rather than letting AttributeError surface later mid-loop, and
    names which registered models already work: everything built on
    `lazyslide_models.base.TimmViTModel` (uni, uni2, virchow, virchow2, ...).
    """
    from lazyslide_models.base import ViTModelProtocol

    if not isinstance(model, ViTModelProtocol):
        raise NotImplementedError(
            f"This requires a ViT-style model exposing grid_size, patch_size "
            f"and encode_image_dense (see lazyslide_models.base."
            f"ViTModelProtocol); '{model_name}' does not. Registered models "
            f"built on lazyslide_models.base.TimmViTModel (uni, uni2, "
            f"virchow, virchow2, ...) support this."
        )


@runtime_checkable
class ModelStage(Protocol):
    """A single step in a `run_model_stages` chain.

    `input_kind`/`output_kind` describe what a stage consumes/produces so the
    chain can be validated before anything runs: "image" is a raw pixel
    batch; "pooled" is one vector per patch, (B, D); "dense" is one
    per-token/per-patch-position vector, (B, N_tokens, D) when unreduced or
    (B, N_tokens) once reduced to a scalar (e.g. a cluster label);
    "sparse" is a sparse feature matrix.
    """

    name: str
    input_kind: Literal["image", "pooled", "dense"]
    output_kind: Literal["pooled", "dense", "sparse"]
    cache: bool
    overwrite: bool
    device: str

    def __call__(self, x):
        ...

    def to(self, device: str) -> "ModelStage":
        ...


class ImageModelStage:
    """Wraps a `lazyslide_models.ImageModel` as the first stage of a chain.

    Applies the model's own `get_transform()` to a raw uint8/float pixel
    batch before running it through `encode_image` (pooled) or
    `encode_image_dense` (dense, per-token, unreduced -- CLS/register tokens
    already stripped).
    """

    input_kind: Literal["image"] = "image"

    def __init__(
        self,
        model,
        *,
        dense: bool = False,
        name: Optional[str] = None,
        cache: bool = True,
        overwrite: bool = False,
        device: Optional[str] = None,
        amp: bool = False,
        model_path: "str | Path | None" = None,
        token: Optional[str] = None,
    ):
        self.model, self.model_name = (
            _resolve_model(model, model_path=model_path, token=token)
            if isinstance(model, str) or not hasattr(model, "encode_image")
            else (model, getattr(model, "name", type(model).__name__))
        )
        self.dense = dense
        if dense:
            _require_dense_capable(self.model, self.model_name)
        self.output_kind: Literal["pooled", "dense"] = "dense" if dense else "pooled"
        self.name = name or (f"{self.model_name}_dense" if dense else self.model_name)
        self.cache = cache
        self.overwrite = overwrite
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = amp
        self._moved = False

    def to(self, device: str) -> "ImageModelStage":
        self.device = device
        self._moved = False
        return self

    def provenance(self) -> dict:
        return {
            "stage": type(self).__name__,
            "model_name": self.model_name,
            "dense": self.dense,
            "input_kind": self.input_kind,
            "output_kind": self.output_kind,
        }

    def __call__(self, image_batch: torch.Tensor) -> torch.Tensor:
        if not self._moved:
            self.model.to(self.device)
            self.model.model.eval()
            self._moved = True

        transform = self.model.get_transform()
        batch = transform(image_batch) if transform is not None else image_batch

        amp_on = bool(self.amp) and "cuda" in str(self.device)
        with torch.inference_mode():
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=amp_on):
                batch = batch.to(self.device, non_blocking=True)
                if self.dense:
                    return self.model.encode_image_dense(batch).patch_tokens
                return self.model.encode_image(batch)


class CallableStage:
    """Wraps an already-fitted, stateless, row-wise transform as a stage.

    Fits `sklearn`-style estimators (`kmeans.predict`), a `sparse_transform`
    function, or a future SAE's `.transform` -- anything that maps one batch
    of rows to one batch of outputs with no dependency on other rows, so it
    can be called per-batch exactly like any other stage in the chain.
    """

    def __init__(
        self,
        fn: Callable,
        *,
        name: str,
        input_kind: Literal["image", "pooled", "dense"],
        output_kind: Literal["pooled", "dense", "sparse"],
        cache: bool = True,
        overwrite: bool = False,
        device: str = "cpu",
    ):
        self.fn = fn
        self.name = name
        self.input_kind = input_kind
        self.output_kind = output_kind
        self.cache = cache
        self.overwrite = overwrite
        self.device = device

    def to(self, device: str) -> "CallableStage":
        self.device = device
        return self

    def provenance(self) -> dict:
        return {
            "stage": type(self).__name__,
            "fn": getattr(self.fn, "__qualname__", repr(self.fn)),
            "input_kind": self.input_kind,
            "output_kind": self.output_kind,
        }

    def __call__(self, x):
        # No implicit tensor<->numpy conversion here: `fn` knows what it
        # wants (e.g. feature_extraction's reducer is documented to receive
        # a live torch.Tensor still on device; sklearn-style estimators want
        # numpy) -- callers wrap `fn` with whatever conversion it needs.
        return self.fn(x)
