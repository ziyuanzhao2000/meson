"""Monkeypatch for lazyslide_models' hardcoded TimmModel.get_transform.

lazyslide_models.base.TimmModel.get_transform() unconditionally returns a
hardcoded bicubic+CenterCrop transform for every timm-backed model it wraps
(UNI, UNI2, GigaPath, Virchow, Virchow2, the Lunit variants), regardless of
what that checkpoint's own timm `pretrained_cfg` declares (e.g. UNI's
pretrained_cfg says bilinear interpolation, crop_pct=1 -- no crop at all).
This builds a corrected, config-driven replacement and applies it per-model-
instance to models resolved by mesoslide.tools.embed_patch, without mutating
lazyslide_models' global state or touching models that already override
get_transform themselves (HOptimus, H0Mini, PathOrchestra).
"""

from __future__ import annotations

import types

import torch


def corrected_transform(model):
    """Build a v2 Compose transform driven by `model.model.pretrained_cfg`.

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

    # Resize on raw uint8 pixels, then cast -- matches the legacy shim and
    # standard PIL/ToTensor practice, rather than lazyslide_models' current
    # cast-before-resize order.
    steps = [ToImage(), Resize((h, w), interpolation=interpolation, antialias=True)]
    if crop_pct < 1.0:
        steps.append(CenterCrop((h, w)))
    steps += [ToDtype(torch.float32, scale=True), Normalize(mean=cfg["mean"], std=cfg["std"])]
    return Compose(steps)


def patch_transform_if_needed(model):
    """Install a per-instance corrected `get_transform` if `model` would
    otherwise fall through to lazyslide_models' buggy TimmModel default.

    No-op for models that override `get_transform` themselves and for
    non-TimmModel models (e.g. CONCH) -- checked via identity against the
    exact base-class method object, not a class-hierarchy isinstance check,
    so it stays self-limiting to exactly the currently-broken models and
    automatically safe if lazyslide_models adds correctly-implemented
    TimmModel subclasses later.
    """
    from lazyslide_models.base import TimmModel

    if not isinstance(model, TimmModel):
        return
    if type(model).get_transform is not TimmModel.get_transform:
        return  # subclass already overrides it; leave alone

    model.get_transform = types.MethodType(lambda self: corrected_transform(self), model)
