"""The old API must fail loudly and name its replacement, not fail obscurely."""

import warnings

import numpy as np
import pytest

import mesoslide as ms

SELECTORS = [
    (ms.select_random_patches, (5,)),
    (ms.select_top_patches, ("score",)),
    (ms.select_negative_patches, ("score",)),
    (ms.select_patches_for_binary_feature, ("flag",)),
]


@pytest.mark.parametrize("fn,args", SELECTORS, ids=lambda v: getattr(v, "__name__", ""))
@pytest.mark.parametrize("kw", ["patch_table_names", "sdata"])
def test_removed_selector_kwargs_point_at_slides(fn, args, kw, one_table):
    with pytest.raises(TypeError, match="has been removed"):
        fn(one_table, *args, **{kw: "anything"})


@pytest.mark.parametrize("fn,args", SELECTORS, ids=lambda v: getattr(v, "__name__", ""))
def test_removed_kwargs_name_the_replacement(fn, args, one_table):
    with pytest.raises(TypeError, match="mesoslide.iter_slides"):
        fn(one_table, *args, patch_table_names="x_grid_point_patch")


def test_a_bare_spatialdata_is_rejected_with_an_explanation():
    """The old positional form was select_top_patches(sdata, patch_table_names, ...)."""
    import spatialdata as sd

    with pytest.raises(TypeError, match="no longer accepted"):
        ms.select_top_patches(sd.SpatialData(), "x_grid_point_patch")


def test_plot_feature_map_image_name_removed(one_table):
    with pytest.raises(TypeError, match="image_name.*has been removed"):
        ms.plotting.plot_feature_map(one_table, "score", image_name="SYNTH00")


def test_element_name_kwarg_warns_at_its_old_default(one_slide):
    """Harmless to accept while the value is one that is now implied anyway."""
    with pytest.warns(DeprecationWarning, match="bbox_postfix"):
        try:
            ms.plotting.plot_feature_map(one_slide, "score",
                                         bbox_postfix="_grid_point_bbox")
        except Exception:
            pass  # only the warning is under test here


def test_element_name_kwarg_rejects_a_value_it_cannot_honour(one_slide):
    with pytest.raises(ValueError, match="can no longer be honoured"):
        ms.plotting.plot_feature_map(one_slide, "score", bbox_postfix="_something_else")


def test_show_image_names_forwards_to_show_slide_ids(one_table):
    images = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    with pytest.warns(DeprecationWarning, match="use 'show_slide_ids'"):
        with pytest.raises(ValueError, match="slide_id"):
            ms.plotting.plot_patch_gallery(one_table[:2], patches_array=images,
                                           show_image_names=True, progress_bar=False)


class TestFeatureReports:
    @pytest.mark.parametrize("kw,val", [("image_names", ["a"]),
                                        ("feature_prefix", "UNI_SAE"),
                                        ("feature_idx", 3)])
    def test_removed_kwargs(self, kw, val):
        fn = ms.plotting.plot_feature_spatial_distribution
        with pytest.raises(TypeError, match="has been removed"):
            fn([], "score", **{kw: val})

    def test_feature_prefix_explains_the_collapse(self):
        fn = ms.plotting.plot_feature_spatial_distribution
        with pytest.raises(TypeError, match="collapse to a single feature_name"):
            fn([], "score", feature_prefix="UNI_SAE")

    def test_point_name_warns_at_its_default(self):
        fn = ms.plotting.plot_feature_spatial_distribution
        with pytest.warns(DeprecationWarning, match="point_name"):
            with pytest.raises(Exception):
                fn([], "score", point_name="grid_point")

    def test_point_name_rejects_other_values(self):
        fn = ms.plotting.plot_feature_spatial_distribution
        with pytest.raises(ValueError, match="can no longer be honoured"):
            fn([], "score", point_name="something_else")


def test_current_api_emits_no_deprecation_warnings(manifest):
    """The replacement path must itself be warning-free."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        ms.select_top_patches(manifest, "score", n=5)


def test_embed_patch_is_a_deprecated_alias_with_identical_output(one_slide, stub_encoder):
    """embed_patch -> feature_extraction: same warning pattern, bit-identical output."""
    with pytest.warns(DeprecationWarning, match="feature_extraction"):
        ms.tl.embed_patch(
            one_slide, stub_encoder, key_added="via_alias",
            batch_size=8, device="cpu", save=False,
        )
    ms.tl.feature_extraction(
        one_slide, stub_encoder, key_added="via_current",
        batch_size=8, device="cpu", save=False,
    )
    table = one_slide.tables["tiles_table"]
    assert np.array_equal(table.obsm["via_alias"], table.obsm["via_current"])
