"""Encoder preprocessing and the image-model table against lazyslide-models."""

import pytest

lazyslide_models = pytest.importorskip("lazyslide_models")

from mesoslide.tools._model_stage import _IMAGE_MODELS, ImageModelStage  # noqa: E402
from mesoslide.tools._timm_transform_patch import (  # noqa: E402
    PAPER_RECIPE_MODELS,
    patch_transform_if_needed,
    uses_paper_recipe,
)


def test_image_model_table_matches_the_registry():
    from lazyslide_models import MODEL_REGISTRY

    for key, (class_name, _) in _IMAGE_MODELS.items():
        assert key in MODEL_REGISTRY, key
        assert MODEL_REGISTRY[key].__name__ == class_name, key


def test_only_uni_models_get_the_paper_recipe():
    from lazyslide_models import MODEL_REGISTRY
    from lazyslide_models.base import TimmModel

    assert {MODEL_REGISTRY[k].__name__ for k in ("uni", "uni2")} == PAPER_RECIPE_MODELS
    for key in ("uni", "uni2"):
        cls = MODEL_REGISTRY[key]
        assert cls.get_transform is TimmModel.get_transform  # inherited, so the patch applies
    for key in ("virchow2", "gigapath", "h-optimus-0"):
        model = MODEL_REGISTRY[key].__new__(MODEL_REGISTRY[key])  # no weights needed
        assert not uses_paper_recipe(model)
        patch_transform_if_needed(model)
        assert "get_transform" not in vars(model)


def test_provenance_marks_upstream_preprocessing_but_not_uni():
    assert "transform" not in ImageModelStage("uni").provenance()
    assert ImageModelStage("virchow2").provenance()["transform"] == "lazyslide-models"
