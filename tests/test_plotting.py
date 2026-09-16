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
        fig, ax = ms.plotting.plot_feature_map(one_slide, "score", return_fig=True)
        assert fig.axes and fig.axes[0].images

    def test_titles_with_the_slide_name(self, one_slide):
        fig, ax = ms.plotting.plot_feature_map(one_slide, "score", return_fig=True)
        assert fig.axes[0].get_title() == one_slide.name

    def test_explicit_title_wins(self, one_slide):
        fig, ax = ms.plotting.plot_feature_map(one_slide, "score", title="custom", return_fig=True)
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

        fig, ax = ms.plotting.plot_feature_map(one_slide, "UNI_SAE_1", table_key="sae_table",
                                                return_fig=True)
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
        assert ms.plotting.plot_feature_map(one_slide, "score", return_fig=True) is not None


class TestFeatureGrid:
    """`_render_feature_grid` -- create_feature_report's internal top panel."""

    def test_renders_a_grid_over_a_mapping(self, open_cohort):
        from mesoslide.plotting._feature_reports import _render_feature_grid

        fig = _render_feature_grid(
            open_cohort, "score", ncols=3, figsize_per_image=(3, 2), dpi=40,
            show_titles=True,
        )
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_streams_a_manifest(self, manifest):
        from mesoslide.plotting._feature_reports import _render_feature_grid

        fig = _render_feature_grid(
            manifest, "score", ncols=2, figsize_per_image=(3, 2), dpi=40,
        )
        assert fig is not None
        plt.close(fig)

    def test_raises_when_nothing_could_be_rendered(self, open_cohort):
        from mesoslide.plotting._feature_reports import _render_feature_grid

        with pytest.raises(ValueError, match="No slide could be rendered"):
            _render_feature_grid(open_cohort, "no_such_feature")


class TestPatchGallery:
    def test_writes_a_gallery_from_slides(self, manifest, open_cohort, tmp_path):
        sel = ms.select_random_patches(manifest, 6, random_state=3)
        out = tmp_path / "g.png"
        ms.plotting.plot_patch_gallery(
            sel, slides=open_cohort, output_path=str(out),
            patches_per_row=3, show_slide_ids=True,
            progress_bar=False, dpi=40,
        )
        assert out.exists()

    def test_accepts_pre_cached_images(self, manifest, open_cohort, tmp_path):
        sel = ms.select_random_patches(manifest, 4, random_state=4)
        ms.pp.extract_patch_images(sel, open_cohort, channel_first=False,
                                    progress_bar=False, cache=True)
        out = tmp_path / "pre.png"
        ms.plotting.plot_patch_gallery(
            sel, output_path=str(out),  # no slides -- reads patches.obsm['patch_img']
            patches_per_row=2, progress_bar=False, dpi=40,
        )
        assert out.exists()

    def test_needs_slides_or_images(self, one_table):
        with pytest.raises(ValueError, match="slides is required"):
            ms.plotting.plot_patch_gallery(one_table[:2], progress_bar=False)

    def test_return_buffer_gives_a_list_of_png_buffers(self, manifest, open_cohort):
        from PIL import Image

        sel = ms.select_random_patches(manifest, 4, random_state=4)
        bufs = ms.plotting.plot_patch_gallery(
            sel, slides=open_cohort, return_buffer=True, progress_bar=False, dpi=40,
        )
        assert isinstance(bufs, list) and len(bufs) == 1
        assert Image.open(bufs[0]).size[0] > 0

    def test_multi_page_return_buffer_without_output_path(self, manifest, open_cohort):
        sel = ms.select_random_patches(manifest, 6, random_state=3)
        bufs = ms.plotting.plot_patch_gallery(
            sel, slides=open_cohort, samples_per_figure=2, return_buffer=True,
            progress_bar=False, dpi=40,
        )
        assert isinstance(bufs, list) and len(bufs) == 3

    def test_multi_page_needs_an_output(self, manifest, open_cohort):
        sel = ms.select_random_patches(manifest, 6, random_state=3)
        with pytest.raises(ValueError, match="output_path"):
            ms.plotting.plot_patch_gallery(
                sel, slides=open_cohort, samples_per_figure=2, progress_bar=False,
            )


class TestPatchGalleryWithSaliency:
    @staticmethod
    def _clusterizer(feature_name):
        from sklearn.cluster import KMeans
        from mesoslide.tools.segmenters import TokenClusterizer
        from tests.conftest import StubViTEncoder

        rng = np.random.default_rng(0)
        kmeans = KMeans(n_clusters=3, random_state=0).fit(
            rng.random((30, StubViTEncoder.embed_dim))
        )
        return TokenClusterizer(model=StubViTEncoder(), kmeans=kmeans,
                                 feature_name=feature_name, device="cpu")

    def test_writes_a_gallery_from_slides(self, manifest, open_cohort, tmp_path):
        sel = ms.select_random_patches(manifest, 4, random_state=4)
        out = tmp_path / "sal.png"
        ms.plotting.plot_patch_gallery_with_saliency(
            sel, clusterizers=[self._clusterizer("c1"), self._clusterizer("c2")],
            slides=open_cohort, output_path=str(out),
            patches_per_row=2, progress_bar=False, dpi=40,
        )
        assert out.exists()

    def test_requires_at_least_one_clusterizer(self, manifest, open_cohort):
        sel = ms.select_random_patches(manifest, 2, random_state=0)
        with pytest.raises(ValueError, match="At least one clusterizer"):
            ms.plotting.plot_patch_gallery_with_saliency(
                sel, clusterizers=[], slides=open_cohort, progress_bar=False,
            )

    def test_rejects_duplicate_feature_names(self, manifest, open_cohort):
        sel = ms.select_random_patches(manifest, 2, random_state=0)
        with pytest.raises(ValueError, match="distinct feature_name"):
            ms.plotting.plot_patch_gallery_with_saliency(
                sel, clusterizers=[self._clusterizer("dup"), self._clusterizer("dup")],
                slides=open_cohort, progress_bar=False,
            )

    def test_needs_slides_unless_fully_cached(self, manifest, open_cohort, tmp_path):
        sel = ms.select_random_patches(manifest, 2, random_state=0)
        clusterizer = self._clusterizer("c1")
        with pytest.raises(ValueError, match="slides is required"):
            ms.plotting.plot_patch_gallery_with_saliency(
                sel, clusterizers=[clusterizer], progress_bar=False,
            )

        # Once everything is cached, slides is no longer needed.
        ms.plotting.plot_patch_gallery_with_saliency(
            sel, clusterizers=[clusterizer], slides=open_cohort, cache=True,
            output_path=str(tmp_path / "sal1.png"), progress_bar=False, dpi=40,
        )
        ms.plotting.plot_patch_gallery_with_saliency(
            sel, clusterizers=[clusterizer],
            output_path=str(tmp_path / "sal2.png"), progress_bar=False, dpi=40,
        )
        assert (tmp_path / "sal2.png").exists()


class TestCreateFeaturePdf:
    def test_imports_without_reportlab(self):
        """Only create_feature_report needs it, and it is not a declared dependency."""
        import inspect

        from mesoslide.plotting import _feature_reports as mod

        assert "reportlab" not in inspect.getsource(mod).split("def create_feature_report")[0]

    def test_requires_an_output(self, manifest, open_cohort):
        with pytest.raises(ValueError, match="output_path"):
            ms.plotting.create_feature_report(open_cohort, "score")

    def test_full_pipeline_saves_a_valid_pdf(self, manifest, open_cohort, tmp_path):
        out = tmp_path / "report.pdf"
        ms.plotting.create_feature_report(
            open_cohort, "score", output_path=str(out),
            gallery_nrows=1, patches_per_row=4, top_fraction=0.5,
            ncols=2, figsize_per_image=(3, 2), image_dpi=40, margin_dots=10,
            progress_bar=False,
        )
        assert out.exists()
        assert out.read_bytes()[:4] == b"%PDF"

    def test_return_buffer_with_no_output_path(self, manifest, open_cohort, tmp_path):
        buf = ms.plotting.create_feature_report(
            open_cohort, "score", return_buffer=True,
            gallery_nrows=1, patches_per_row=4, top_fraction=0.5,
            ncols=2, figsize_per_image=(3, 2), image_dpi=40, margin_dots=10,
            progress_bar=False,
        )
        assert buf.getvalue()[:4] == b"%PDF"
        assert not any(tmp_path.iterdir())  # nothing written anywhere
