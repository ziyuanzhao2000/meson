"""A chainable, self-describing stage in a patch/tile feature pipeline.

A `ModelStage` wraps one model (a vision foundation model, a fitted KMeans, a
sparse-coding transform, ...) and declares what it consumes, what it
produces, and where its own output should be cached -- so a pipeline of them
can be validated and run generically by `mesoslide.tools._feature_extraction
.run_model_stages`, instead of each caller (feature_extraction,
fit_token_clusterer, extract_cluster_maps) hand-rolling its own batching loop
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
    integer slices. Uint8 input (pixels) is kept as uint8, matching the
    slide-level tile loader, so the model's transform sees the same input on
    both paths; other dtypes are cast to float.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    n = len(x)
    for i in range(0, n, batch_size):
        batch = x[i:i + batch_size]
        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
        yield batch if batch.dtype == torch.uint8 else batch.float()


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
    `TokenClusterer.model_name_`) and later feed it back into
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


# lazyslide-models registry key -> (model class name, dense-capable), for
# the image encoders. Hard-coded so a model name can be checked and named
# without importing lazyslide_models (~5 s) or loading weights. The class
# name is what an instance's `.name` reports, so provenance matches caches
# written with a loaded model. Dense-capable means instances satisfy
# `lazyslide_models.base.ViTModelProtocol` (grid_size, patch_size,
# encode_image_dense).
_IMAGE_MODELS: dict = {
    "biomedclip": ("BiomedCLIP", False),
    "conch": ("CONCH", False),
    "medsiglip": ("MedSigLip", False),
    "musk": ("MUSK", False),
    "omiclip": ("OmiCLIP", False),
    "plip": ("PLIP", False),
    "quiltnet-b32": ("QuiltNetB32", False),
    "quiltnet-b16": ("QuiltNetB16", False),
    "quiltnet-b16-pmb": ("QuiltNetB16PMB", False),
    "titan": ("Titan", False),
    "conch_v1.5": ("Titan", False),
    "chief": ("CHIEF", False),
    "ctranspath": ("CTransPath", False),
    "genbio-pathfm": ("GenBioPathFM", True),
    "gigapath": ("GigaPath", True),
    "gpfm": ("GPFM", False),
    "h-optimus-0": ("HOptimus0", True),
    "h-optimus-1": ("HOptimus1", True),
    "h0-mini": ("H0Mini", True),
    "hibou-b": ("HibouB", False),
    "hibou-l": ("HibouL", False),
    "lunit-bt": ("LunitResNet50BT", False),
    "lunit-mocov2": ("LunitResNet50MoCoV2", False),
    "lunit-swav": ("LunitResNet50SwAV", False),
    "lunit-dino-s8": ("LunitDINOPatch8", True),
    "lunit-dino-s16": ("LunitDINOPatch16", True),
    "midnight": ("Midnight", True),
    "open-midnight": ("OpenMidnight", True),
    "path_orchestra": ("PathOrchestra", False),
    "phikon": ("Phikon", False),
    "phikonv2": ("PhikonV2", False),
    "uni": ("UNI", True),
    "uni2": ("UNI2", True),
    "virchow": ("Virchow", True),
    "virchow2": ("Virchow2", True),
}


def _registry_entry(name: str):
    """`(class name, dense-capable)` for a model name, or None if unregistered."""
    entry = _IMAGE_MODELS.get(name)
    if entry is not None:
        return entry
    from lazyslide_models import MODEL_REGISTRY
    from lazyslide_models.base import TimmViTModel

    cls = MODEL_REGISTRY.get(name)
    if cls is None:
        return None
    dense = isinstance(cls, type) and issubclass(cls, TimmViTModel)
    return cls.__name__, dense


def _model_name(model) -> str:
    """The `.name` a model name/instance has once loaded, without loading it."""
    if isinstance(model, str):
        entry = _registry_entry(model)
        # Unregistered names load as a generic TimmModel.
        return entry[0] if entry is not None else "TimmModel"
    return getattr(model, "name", type(model).__name__)


def _require_dense_capable(model, model_name: str) -> None:
    """Check the model can produce per-token embeddings (encode_image_dense).

    A model name is looked up in `_IMAGE_MODELS` (no loading); an
    instance is checked against `lazyslide_models.base.ViTModelProtocol`.
    Raises rather than letting AttributeError surface later mid-loop.
    """
    if isinstance(model, str):
        entry = _registry_entry(model)
        capable = entry is not None and entry[1]
    else:
        from lazyslide_models.base import ViTModelProtocol
        capable = isinstance(model, ViTModelProtocol)

    if not capable:
        raise NotImplementedError(
            f"This requires a ViT-style model exposing grid_size, patch_size "
            f"and encode_image_dense (see lazyslide_models.base."
            f"ViTModelProtocol); '{model_name}' does not. Supported registered "
            f"models: {', '.join(k for k, (_, d) in _IMAGE_MODELS.items() if d)}."
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
        # Loaded on first access of `self.model`, so a stage skipped by the
        # cache never loads its weights.
        self._model_spec = model
        self._model_path = model_path
        self._token = token
        self._model = (
            model if not isinstance(model, str) and hasattr(model, "encode_image") else None
        )
        self.model_name = _model_name(model)
        self.dense = dense
        if dense:
            _require_dense_capable(model, self.model_name)
        self.output_kind: Literal["pooled", "dense"] = "dense" if dense else "pooled"
        self.name = name or (f"{self.model_name}_dense" if dense else self.model_name)
        self.cache = cache
        self.overwrite = overwrite
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = amp
        self._moved = False

    @property
    def model(self):
        if self._model is None:
            self._model, _ = _resolve_model(
                self._model_spec, model_path=self._model_path, token=self._token,
            )
        return self._model

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

    def _prepare(self, image_batch: torch.Tensor) -> torch.Tensor:
        """Move the model to `device` once, then apply the model's own transform."""
        if not self._moved:
            self.model.to(self.device)
            self.model.model.eval()
            self._moved = True
        transform = self.model.get_transform()
        return transform(image_batch) if transform is not None else image_batch

    def _encode(self, batch: torch.Tensor) -> torch.Tensor:
        amp_on = bool(self.amp) and "cuda" in str(self.device)
        with torch.inference_mode():
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=amp_on):
                batch = batch.to(self.device, non_blocking=True)
                if self.dense:
                    return self.model.encode_image_dense(batch).patch_tokens
                return self.model.encode_image(batch)

    def __call__(self, image_batch: torch.Tensor) -> torch.Tensor:
        return self._encode(self._prepare(image_batch))

    def encode_kept_tokens(self, image_batch: torch.Tensor, keep_idx) -> torch.Tensor:
        """Encode images using only a subset of their patch tokens.

        Same output as `__call__`, but each image is run on the prefix tokens
        (CLS/registers) plus the patch tokens in its row of `keep_idx`; all
        other patch tokens are removed right after the position embedding. The
        model's own pooling applies to what remains (e.g. CLS for UNI, CLS
        concatenated with the mean of the kept patch tokens for Virchow).
        Requires a timm VisionTransformer backbone (`self.model.model`).

        Parameters
        ----------
        image_batch : torch.Tensor of shape (B, C, H, W)
        keep_idx : array-like of int, shape (B, n_keep)
            Row-major patch-token indices to keep, per image.
        """
        from timm.models import VisionTransformer

        vit = getattr(self.model, "model", None)
        if not isinstance(vit, VisionTransformer):
            raise NotImplementedError(
                f"encode_kept_tokens requires a timm VisionTransformer backbone; "
                f"'{self.model_name}' does not have one."
            )
        batch = self._prepare(image_batch)
        keep_idx = torch.as_tensor(np.asarray(keep_idx), dtype=torch.long, device=self.device)
        original = vit.patch_drop
        vit.patch_drop = _KeepPatchTokens(keep_idx, vit.num_prefix_tokens)
        try:
            return self._encode(batch)
        finally:
            vit.patch_drop = original


class _KeepPatchTokens(torch.nn.Module):
    """Stand-in for timm's `patch_drop`: keeps prefix tokens and selected patch tokens."""

    def __init__(self, keep_idx: torch.Tensor, num_prefix_tokens: int):
        super().__init__()
        self.keep_idx = keep_idx
        self.num_prefix_tokens = num_prefix_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prefix, patches = x[:, :self.num_prefix_tokens], x[:, self.num_prefix_tokens:]
        idx = self.keep_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        return torch.cat([prefix, patches.gather(1, idx)], dim=1)


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
