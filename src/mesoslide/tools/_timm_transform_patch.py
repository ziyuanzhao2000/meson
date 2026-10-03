"""The paper's preprocessing for UNI/UNI2, kept identical across lazyslide-models versions.

The SAE, MP-SAE and token clusterers were fit on UNI embeddings computed
with the checkpoint's own timm `pretrained_cfg` recipe: bilinear resize to
224 px on uint8 pixels, then cast, no center crop (identical to the original
`UNIEmbedder`: `Resize(224)` on uint8). lazyslide-models >= 0.1.0 also uses
bilinear for UNI/UNI2, but casts to float before resizing, which shifts
embeddings slightly (cosine down to ~0.995 on some tiles). To keep embeddings
comparable with those models, UNI and UNI2 get the recipe below, installed
per instance without touching lazyslide_models' global state.

Every other encoder uses lazyslide-models' own `get_transform`, which since
0.1.0 follows each model's upstream recipe (`TimmModel.transform_kws`).
"""

from __future__ import annotations

import types

import torch

#: Class names of the encoders whose embeddings back mesoslide's fitted models.
PAPER_RECIPE_MODELS = frozenset({"UNI", "UNI2"})


def legacy_uni_transform(model):
    """Build the v2 Compose transform from `model.model.pretrained_cfg`, resizing on uint8.

    `model` is a `lazyslide_models.base.TimmModel` (or subclass) instance;
    `model.model` is the underlying raw timm nn.Module, which carries
    `.pretrained_cfg` -- the object `timm.data.resolve_data_config` expects.
    """
    from timm.data import resolve_data_config
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import (
        CenterCrop,
        Compose,
        Normalize,
        Resize,
        ToDtype,
        ToImage,
    )

    cfg = resolve_data_config(model=model.model)
    _, h, w = cfg["input_size"]
    interpolation = getattr(
        InterpolationMode, cfg["interpolation"].upper(), InterpolationMode.BILINEAR,
    )
    crop_pct = cfg.get("crop_pct") or 1.0

    # Resize on raw uint8 pixels, then cast, as the original UNIEmbedder did.
    steps = [ToImage(), Resize((h, w), interpolation=interpolation, antialias=True)]
    if crop_pct < 1.0:
        steps.append(CenterCrop((h, w)))
    steps += [ToDtype(torch.float32, scale=True), Normalize(mean=cfg["mean"], std=cfg["std"])]
    return Compose(steps)


def uses_paper_recipe(model) -> bool:
    """True if `model` gets `legacy_uni_transform` from `patch_transform_if_needed`."""
    from lazyslide_models.base import TimmModel

    return (
        isinstance(model, TimmModel)
        and type(model).__name__ in PAPER_RECIPE_MODELS
        and type(model).get_transform is TimmModel.get_transform
    )


def patch_transform_if_needed(model):
    """Install `legacy_uni_transform` as `model.get_transform` for UNI/UNI2; no-op otherwise.

    Skipped if the class overrides `get_transform` itself, so a future
    lazyslide-models UNI with its own transform is left alone.
    """
    if uses_paper_recipe(model):
        model.get_transform = types.MethodType(lambda self: legacy_uni_transform(self), model)
