"""Rendering paths, including the two SpatialData quirks they have to work around."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import mesoslide as ms


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestFeatureMap:
    def test_renders_an_obs_column(self, one_slide):
        fig = ms.plotting.plot_feature_map(one_slide, "score")
        assert fig.axes and fig.axes[0].images

    def test_titles_with_the_slide_name(self, one_slide):
        fig = ms.plotting.plot_feature_map(one_slide, "score")
        assert fig.axes[0].get_title() == one_slide.name

    def test_explicit_title_wins(self, one_slide):
        fig = ms.plotting.plot_feature_map(one_slide, "score", title="custom")
        assert fig.axes[0].get_title() == "custom"

    def test_bridges_a_var_name_in_X_into_obs(self, one_slide):
        """spatialdata_plot colours by an .obs column; SAE scores live in .X."""
        import anndata as ad
        from scipy.sparse import csr_matrix

        table = one_slide.tables["tiles_table"]
        n = table.n_obs
        sae = ad.AnnData(
            X=csr_matrix(np.random.default_rng(0).random((n, 2)).astype("float32")),
            obs=table.obs[["tile_id", "tissue_id", "x", "y", "library_id"]].copy(),
        )
        sae.var_names = ["UNI_SAE_0", "UNI_SAE_1"]
        sae.uns["spatialdata_attrs"] = table.uns["spatialdata_attrs"]
        one_slide.tables["sae_table"] = sae

        fig = ms.plotting.plot_feature_map(one_slide, "UNI_SAE_1", table_key="sae_table")
        assert fig.axes and fig.axes[0].images
        # the bridge must not write back into the caller's table
        assert "UNI_SAE_1" not in one_slide.tables["sae_table"].obs.columns

    def test_missing_pixels_names_attach_images(self, cohort):
        """wsi.write() persists no pixels, so this is the common mistake."""
        import ezslide

        bare = ezslide.read_wsi(cohort[0])
        with pytest.raises(ValueError, match="attach_images=True"):
            ms.plotting.plot_feature_map(bare, "score")

    def test_unknown_feature_is_reported(self, one_slide):
        with pytest.raises(KeyError, match="neither"):
            ms.plotting.plot_feature_map(one_slide, "no_such_feature")

    def test_unknown_table_is_reported(self, one_slide):
        with pytest.raises(ValueError, match="no table"):
            ms.plotting.plot_feature_map(one_slide, "score", table_key="nope")

    def test_renders_even_though_the_image_is_excluded_from_the_sdata(self, one_slide):
        """wsidata keeps the image in _exclude_elements so write() skips the pixels.

        That also hides it from SpatialData.__getitem__, which is how
        spatialdata_plot resolves elements -- hence the internal view object.
        """
        with pytest.raises(KeyError):
            one_slide["wsi"]
        assert "wsi" in one_slide.images
        assert ms.plotting.plot_feature_map(one_slide, "score") is not None


class TestSpatialDistribution:
    def test_renders_a_grid_over_a_mapping(self, open_cohort, tmp_path):
        fig = ms.plotting.plot_feature_spatial_distribution(
            open_cohort, "score", ncols=3, figsize_per_image=(3, 2), dpi=40,
            return_fig=True, output_path=str(tmp_path), show_titles=True,
        )
        assert len(fig.axes) == 3
        assert (tmp_path / "score_spatial_distribution.png").exists()

    def test_streams_a_manifest(self, manifest, tmp_path):
        fig = ms.plotting.plot_feature_spatial_distribution(
            manifest, "score", ncols=2, figsize_per_image=(3, 2), dpi=40,
            return_fig=True,
        )
        assert fig is not None

    def test_raises_when_nothing_could_be_rendered(self, open_cohort):
        with pytest.raises(ValueError, match="No slide could be rendered"):
            ms.plotting.plot_feature_spatial_distribution(open_cohort, "no_such_feature")

    def test_imports_without_reportlab(self):
        """Only create_feature_pdf needs it, and it is not a declared dependency."""
        import inspect

        from mesoslide.plotting import _feature_reports as mod

        assert "reportlab" not in inspect.getsource(mod).split("def create_feature_pdf")[0]


class TestPatchGallery:
    def test_writes_a_gallery_from_slides(self, manifest, open_cohort, tmp_path):
        sel = ms.select_random_patches(manifest, 6, random_state=3)
        ms.plotting.plot_patch_gallery(
            sel, slides=open_cohort, output_path=str(tmp_path),
            filename_prefix="g", patches_per_row=3, show_slide_ids=True,
            progress_bar=False, dpi=40,
        )
        assert list(tmp_path.glob("g_samples_*.png"))

    def test_accepts_pre_extracted_images(self, manifest, open_cohort, tmp_path):
        sel = ms.select_random_patches(manifest, 4, random_state=4)
        imgs = ms.pp.extract_patches(sel, open_cohort, channel_first=False,
                                     progress_bar=False)
        ms.plotting.plot_patch_gallery(
            sel, patches_array=imgs, output_path=str(tmp_path),
            filename_prefix="pre", patches_per_row=2, progress_bar=False, dpi=40,
        )
        assert list(tmp_path.glob("pre_samples_*.png"))

    def test_needs_slides_or_images(self, one_table):
        with pytest.raises(ValueError, match="Either slides or patches_array"):
            ms.plotting.plot_patch_gallery(one_table[:2], progress_bar=False)
