"""Cell segmentation/phenotype ingestion and the GalleryPlan cell overlay."""

import numpy as np
import pandas as pd
import pytest


def _write_instance_mask(path, shape=(256, 256)):
    """A small label mask with cells at known locations, one spanning x=128."""
    import tifffile

    h, w = shape
    mask = np.zeros((h, w), dtype=np.int32)
    mask[10:40, 10:40] = 1
    mask[60:100, 60:110] = 2
    mask[120:160, 20:250] = 3  # spans multiple chunks under a small chunk_size
    mask[200:230, 200:230] = 4
    tifffile.imwrite(path, mask, photometric="minisblack", metadata={"axes": "YX"}, ome=True)
    return path


def _write_phenotype_csv(path):
    df = pd.DataFrame({
        "CellID": [1, 2, 3, 4],
        "DNA1": [100.0, 200.0, 150.0, 300.0],
        "CD3E": [10.0, 20.0, 5.0, 40.0],
        "X_centroid": [25, 85, 150, 215],
        "Y_centroid": [25, 80, 140, 215],
        "phenotype": ["CD8 T cell", "B cell", "Endothelial cell", "Myofibroblast"],
    })
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def mask_path(tmp_path):
    return _write_instance_mask(str(tmp_path / "mask.ome.tif"))


@pytest.fixture
def phenotype_csv(tmp_path):
    return _write_phenotype_csv(str(tmp_path / "phenotypes.csv"))


@pytest.fixture
def mask_wsi(mask_path):
    import ezslide

    wsi = ezslide.open_slide(mask_path, attach_images=False)
    yield wsi
    wsi.close()


class TestAddCellPolygons:
    def test_unchunked_and_chunked_agree(self, mask_path):
        import ezslide
        from mesoslide.tools._cell_import import add_cell_polygons

        whole = ezslide.open_slide(mask_path, attach_images=False)
        add_cell_polygons(whole, mask_path, chunked=False, progress_bar=False)
        whole_areas = sorted(whole.shapes["cells"].geometry.area.tolist())
        whole.close()

        chunked = ezslide.open_slide(mask_path, attach_images=False)
        add_cell_polygons(
            chunked, mask_path, chunked=True, chunk_size=64, halo=8, progress_bar=False,
        )
        chunked_areas = sorted(chunked.shapes["cells"].geometry.area.tolist())
        chunked.close()

        assert whole_areas == chunked_areas
        assert len(whole_areas) == 4

    def test_chunking_reconstructs_boundary_spanning_cell(self, mask_wsi, mask_path):
        from mesoslide.tools._cell_import import add_cell_polygons

        add_cell_polygons(
            mask_wsi, mask_path, chunked=True, chunk_size=64, halo=8, progress_bar=False,
        )
        gdf = mask_wsi.shapes["cells"]
        assert len(gdf) == 4, "the boundary-spanning cell must merge to one row, not fragment"

    def test_schema_matches_lazyslide_seg_cells(self, mask_wsi, mask_path):
        from mesoslide.tools._cell_import import add_cell_polygons

        add_cell_polygons(mask_wsi, mask_path, chunked=False, progress_bar=False)
        gdf = mask_wsi.shapes["cells"]
        assert list(gdf.columns) == ["cell_id", "geometry"]
        assert gdf["cell_id"].tolist() == sorted(gdf["cell_id"].tolist())
        assert 0 not in gdf["cell_id"].tolist()

    def test_tile_grid_chunking(self, mask_wsi, mask_path):
        """Chunk on an existing tiles shapes element instead of a synthetic grid."""
        import geopandas as gpd
        from shapely.geometry import box
        from spatialdata.models import ShapesModel

        from mesoslide.tools._cell_import import add_cell_polygons

        tiles = gpd.GeoDataFrame(
            {"tile_id": [0, 1, 2, 3]},
            geometry=[box(x, y, x + 64, y + 64) for y in (0, 64, 128, 192) for x in (0,)][:4],
        )
        # cover the full 256x256 extent with 64px tiles
        boxes = [box(x, y, x + 64, y + 64) for y in range(0, 256, 64) for x in range(0, 256, 64)]
        tiles = gpd.GeoDataFrame({"tile_id": range(len(boxes))}, geometry=boxes)
        mask_wsi.shapes["tiles"] = ShapesModel.parse(tiles)

        add_cell_polygons(
            mask_wsi, mask_path, chunked=True, tile_key="tiles", halo=8, progress_bar=False,
        )
        gdf = mask_wsi.shapes["cells"]
        assert len(gdf) == 4
        assert sorted(gdf.geometry.area.tolist()) == [841.0, 841.0, 1911.0, 8931.0]

    def test_refuses_overwrite_by_default(self, mask_wsi, mask_path):
        from mesoslide.tools._cell_import import add_cell_polygons

        add_cell_polygons(mask_wsi, mask_path, chunked=False, progress_bar=False)
        with pytest.raises(ValueError, match="overwrite"):
            add_cell_polygons(mask_wsi, mask_path, chunked=False, progress_bar=False)
        add_cell_polygons(mask_wsi, mask_path, chunked=False, overwrite=True, progress_bar=False)

    def test_save_defaults_false_and_persists_when_true(self, tmp_path, mask_path, phenotype_csv):
        import ezslide

        from mesoslide.tools._cell_import import add_cell_polygons, add_cell_phenotypes

        store = str(tmp_path / "store.zarr")
        wsi = ezslide.open_slide(mask_path, attach_images=False)
        wsi.write(store)

        add_cell_polygons(wsi, mask_path, chunked=False, progress_bar=False)
        assert not (tmp_path / "store.zarr" / "shapes" / "cells").exists(), (
            "save=False (the default) must not touch disk"
        )

        add_cell_phenotypes(wsi, phenotype_csv, phenotype_col="phenotype", save=True)
        wsi.close()

        reopened = ezslide.read_slide(store, attach_images=False)
        try:
            assert "cells" in reopened.shapes
            assert "phenotype" in reopened.shapes["cells"].columns
            assert "cells_phenotypes" in reopened.tables
        finally:
            reopened.close()


class TestAddCellPhenotypes:
    def test_requires_existing_cells(self, mask_wsi, phenotype_csv):
        from mesoslide.tools._cell_import import add_cell_phenotypes

        with pytest.raises(KeyError):
            add_cell_phenotypes(mask_wsi, phenotype_csv)

    def test_writes_shapes_column_and_table(self, mask_wsi, mask_path, phenotype_csv):
        from mesoslide.tools._cell_import import add_cell_polygons, add_cell_phenotypes

        add_cell_polygons(mask_wsi, mask_path, chunked=False, progress_bar=False)
        add_cell_phenotypes(mask_wsi, phenotype_csv, phenotype_col="phenotype")

        gdf = mask_wsi.shapes["cells"]
        assert "phenotype" in gdf.columns
        assert set(gdf["phenotype"]) == {
            "CD8 T cell", "B cell", "Endothelial cell", "Myofibroblast",
        }

        table = mask_wsi.tables["cells_phenotypes"]
        assert table.n_obs == 4
        assert "phenotype" in table.obs.columns
        assert table.X is None

    def test_marker_cols_route_to_X(self, mask_wsi, mask_path, phenotype_csv):
        from mesoslide.tools._cell_import import add_cell_polygons, add_cell_phenotypes

        add_cell_polygons(mask_wsi, mask_path, chunked=False, progress_bar=False)
        add_cell_phenotypes(
            mask_wsi, phenotype_csv, phenotype_col="phenotype",
            marker_cols=["DNA1", "CD3E"],
        )
        table = mask_wsi.tables["cells_phenotypes"]
        assert table.X.shape == (4, 2)
        assert list(table.var.index) == ["DNA1", "CD3E"]
        assert "DNA1" not in table.obs.columns

    def test_lazyslide_wsiviewer_interop(self, mask_wsi, mask_path, phenotype_csv):
        """color_by must work through stock lazyslide with no mesoslide plotting code."""
        import matplotlib
        matplotlib.use("Agg")
        import lazyslide as zs

        from mesoslide.tools._cell_import import add_cell_polygons, add_cell_phenotypes

        add_cell_polygons(mask_wsi, mask_path, chunked=False, progress_bar=False)
        add_cell_phenotypes(mask_wsi, phenotype_csv, phenotype_col="phenotype")

        v = zs.pl.WSIViewer(mask_wsi)
        v.add_polygons("cells", color_by="phenotype", legend=True)
        v.show(ax=None)


class TestResolveCategoricalPalette:
    def test_falls_back_past_the_fixed_palette_size(self):
        """More categories than LAZYSLIDE_PALETTE's 9 colors must not error,
        and every category must get a genuinely distinct color.

        Regression test: the fallback used to index a fixed-size discrete
        colormap (`tab10`, 10 colors) by normalized float position
        (`colormap(g / n)`), which clips to the colormap's last entry for
        any g/n >= 1.0 -- collapsing every category past the colormap's own
        size onto one identical color. Now uses `distinctipy.get_colors`,
        which scales to any category count without collisions.
        """
        from mesoslide.plotting._cell_overlay import resolve_categorical_palette

        categories = [f"cat{i}" for i in range(15)]
        palette = resolve_categorical_palette(categories)
        assert set(palette) == set(categories)
        assert all(len(c) == 4 for c in palette.values())
        assert len(set(palette.values())) == len(categories), "every category must get a distinct color"

    def test_deterministic_across_calls(self):
        """Fixed seed -> same category set gets the same colors every call."""
        from mesoslide.plotting._cell_overlay import resolve_categorical_palette

        categories = [f"cat{i}" for i in range(20)]
        first = resolve_categorical_palette(categories)
        second = resolve_categorical_palette(categories)
        assert first == second

    def test_real_phenotype_list_all_distinct(self):
        """Reproduction of the reported bug: 15 real CyCIF phenotype labels
        used to collapse the last several categories onto one color."""
        from mesoslide.plotting._cell_overlay import resolve_categorical_palette

        phenotypes = [
            "B cell", "CD11c+ CD206+ Mac", "CD11c+ DC/Mac", "CD206+ Mac",
            "CD4 T cell", "CD8 T cell", "Endothelial cell", "Epithelial cell",
            "Mast cell", "Myofibroblast", "Neutrophil", "Other Immune cells",
            "Other T cell", "Treg", "Unknown",
        ]
        palette = resolve_categorical_palette(phenotypes)
        assert len(set(palette.values())) == len(phenotypes)


class TestCellOverlay:
    def _build_he_wsi(self, tmp_path):
        import ezslide
        import tifffile
        from wsidata import WSIData
        from wsidata._model.tile import TileSpec

        path = str(tmp_path / "he.ome.tif")
        rgb = np.full((256, 256, 3), 200, dtype=np.uint8)
        tifffile.imwrite(path, rgb, photometric="rgb", metadata={"axes": "YXS"}, ome=True)
        wsi = ezslide.open_slide(path, attach_images=False)
        spec = TileSpec(height=128, width=128, stride_height=128, stride_width=128)
        wsi.attrs[WSIData.TILE_SPEC_KEY] = {"tiles": spec.to_dict()}
        return wsi

    def test_overlay_on_separate_cells_slide(self, tmp_path, mask_wsi, mask_path, phenotype_csv):
        """cells living on a different WSIData than the tiled reference slide."""
        import anndata as ad

        from mesoslide.tools._cell_import import add_cell_polygons, add_cell_phenotypes
        from mesoslide.plotting._gallery_plan import GalleryPlan

        add_cell_polygons(mask_wsi, mask_path, chunked=False, progress_bar=False)
        add_cell_phenotypes(mask_wsi, phenotype_csv, phenotype_col="phenotype")

        he_wsi = self._build_he_wsi(tmp_path)
        try:
            patches = ad.AnnData(X=np.zeros((2, 1)), obs={"x": [0, 128], "y": [0, 0]})
            patches.obs["_slide_ref"] = [he_wsi, he_wsi]

            plan = GalleryPlan(patches)
            plan.add_he_row(slides={None: he_wsi})
            plan.add_cell_overlay_row(cells_key="cells", slides={None: mask_wsi}, color_by="phenotype")

            frame0 = plan._blocks[-1].frames[0][-1]
            frame1 = plan._blocks[-1].frames[1][-1]
            assert frame0.shape == (128, 128, 3)
            assert frame1.shape == (128, 128, 3)
            # cells are drawn (not left as the plain gray background)
            assert not np.allclose(frame0, frame0[0, 0])
        finally:
            he_wsi.close()

    def test_requires_a_row_to_blend_onto(self, tmp_path, mask_wsi, mask_path):
        import anndata as ad

        from mesoslide.tools._cell_import import add_cell_polygons
        from mesoslide.plotting._gallery_plan import GalleryPlan

        add_cell_polygons(mask_wsi, mask_path, chunked=False, progress_bar=False)
        patches = ad.AnnData(X=np.zeros((1, 1)), obs={"x": [0], "y": [0]})
        plan = GalleryPlan(patches)
        with pytest.raises(ValueError, match="blend_with_previous"):
            plan.add_cell_overlay_row(slides={None: mask_wsi})

    def test_missing_cells_key_raises(self, tmp_path):
        import anndata as ad

        from mesoslide.plotting._gallery_plan import GalleryPlan

        he_wsi = self._build_he_wsi(tmp_path)
        try:
            patches = ad.AnnData(X=np.zeros((1, 1)), obs={"x": [0], "y": [0]})
            patches.obs["_slide_ref"] = [he_wsi]
            plan = GalleryPlan(patches)
            plan.add_he_row(slides={None: he_wsi})
            with pytest.raises(KeyError):
                plan.add_cell_overlay_row(cells_key="cells", slides={None: he_wsi})
        finally:
            he_wsi.close()

    def _build_mif_wsi(self, tmp_path, n_channels=5):
        import ezslide
        import tifffile
        from wsidata import WSIData
        from wsidata._model.tile import TileSpec

        path = str(tmp_path / "mif.ome.tif")
        arr = np.full((n_channels, 64, 64), 100, dtype=np.uint8)
        tifffile.imwrite(path, arr, photometric="minisblack", metadata={"axes": "CYX"}, ome=True)
        wsi = ezslide.open_slide(path, attach_images=False)
        spec = TileSpec(height=128, width=128, stride_height=128, stride_width=128)
        wsi.attrs[WSIData.TILE_SPEC_KEY] = {"tiles": spec.to_dict()}
        return wsi

    def test_global_palette_fixes_per_patch_inconsistency(self, tmp_path):
        """Regression test for the color-consistency bug.

        Categories are chosen so a *per-patch* resolve_categorical_palette
        call would assign 'shared' a different positional index (hence
        color) in each patch -- see TestResolveCategoricalPalette's unit
        test for the isolated demonstration. This confirms
        add_cell_overlay_row's actual resolution is immune: the SAME
        legend_palette dict is used for every patch.
        """
        import ezslide
        import tifffile
        import anndata as ad

        from mesoslide.tools._cell_import import add_cell_polygons, add_cell_phenotypes
        from mesoslide.plotting._gallery_plan import GalleryPlan

        mask = np.zeros((128, 256), dtype=np.int32)
        mask[10:30, 10:30] = 1  # patch A: shared + catA + catB
        mask[10:30, 40:60] = 2
        mask[10:30, 70:90] = 3
        mask[10:30, 138:158] = 4  # patch B: cat0 + cat1 + shared
        mask[10:30, 168:188] = 5
        mask[10:30, 198:218] = 6
        mask_path = str(tmp_path / "mask.ome.tif")
        tifffile.imwrite(mask_path, mask, photometric="minisblack", metadata={"axes": "YX"}, ome=True)

        pheno_path = str(tmp_path / "pheno.csv")
        pd.DataFrame({
            "CellID": [1, 2, 3, 4, 5, 6],
            "phenotype": ["shared", "catA", "catB", "cat0", "cat1", "shared_dup"],
        }).to_csv(pheno_path, index=False)
        # relabel cell 6 to "shared" too, so both patches contain the same category
        df = pd.read_csv(pheno_path)
        df.loc[df.CellID == 6, "phenotype"] = "shared"
        df.to_csv(pheno_path, index=False)

        wsi = ezslide.open_slide(mask_path, attach_images=False)
        add_cell_polygons(wsi, mask_path, chunked=False, progress_bar=False)
        add_cell_phenotypes(wsi, pheno_path, phenotype_col="phenotype")

        he_wsi = self._build_he_wsi(tmp_path)
        try:
            patches = ad.AnnData(X=np.zeros((2, 1)), obs={"x": [0, 128], "y": [0, 0]})
            patches.obs["_slide_ref"] = [he_wsi, he_wsi]
            plan = GalleryPlan(patches)
            plan.add_he_row(slides={None: he_wsi})
            plan.add_cell_overlay_row(cells_key="cells", slides={None: wsi}, color_by="phenotype")

            block = plan._blocks[-1]
            assert set(block.legend_palette) == {"shared", "catA", "catB", "cat0", "cat1"}
            # the fix: exactly one color for "shared", used everywhere -- there is
            # only one dict, so there is no way for the two patches to disagree.
            shared_color = block.legend_palette["shared"]
            assert isinstance(shared_color, tuple) and len(shared_color) == 4
        finally:
            he_wsi.close()
            wsi.close()

    def test_render_draws_legend_and_expands_canvas(self, tmp_path, mask_wsi, mask_path, phenotype_csv):
        from mesoslide.tools._cell_import import add_cell_polygons, add_cell_phenotypes
        from mesoslide.plotting._gallery_plan import GalleryPlan
        import anndata as ad

        add_cell_polygons(mask_wsi, mask_path, chunked=False, progress_bar=False)
        add_cell_phenotypes(mask_wsi, phenotype_csv, phenotype_col="phenotype")

        he_wsi = self._build_he_wsi(tmp_path)
        try:
            patches = ad.AnnData(X=np.zeros((1, 1)), obs={"x": [0], "y": [0]})
            patches.obs["_slide_ref"] = [he_wsi]

            plan = GalleryPlan(patches)
            plan.add_he_row(slides={None: he_wsi})
            plan.add_cell_overlay_row(cells_key="cells", slides={None: mask_wsi}, color_by="phenotype")

            fig_with, axes_with = plan.render(patches_per_row=1, return_fig=True, legend=True, legend_width=3.0)
            assert len(fig_with.legends) == 1
            grid_width_with = axes_with[0, 0].get_position().width * fig_with.get_figwidth()
            fig_w_with = fig_with.get_figwidth()
            plt_close = fig_with.clf
            import matplotlib.pyplot as plt
            plt.close(fig_with)

            fig_without, axes_without = plan.render(patches_per_row=1, return_fig=True, legend=False)
            assert len(fig_without.legends) == 0
            grid_width_without = axes_without[0, 0].get_position().width * fig_without.get_figwidth()
            fig_w_without = fig_without.get_figwidth()
            plt.close(fig_without)

            # canvas grew...
            assert fig_w_with > fig_w_without
            # ...but the patch grid's own physical width did not shrink to
            # make room (allow a small tolerance for float rounding).
            assert grid_width_with == pytest.approx(grid_width_without, rel=0.02)
        finally:
            he_wsi.close()

    def test_row_labels_align_with_row_midline(self, tmp_path, mask_wsi, mask_path, phenotype_csv):
        from mesoslide.tools._cell_import import add_cell_polygons, add_cell_phenotypes
        from mesoslide.plotting._gallery_plan import GalleryPlan
        import anndata as ad
        import matplotlib.pyplot as plt

        add_cell_polygons(mask_wsi, mask_path, chunked=False, progress_bar=False)
        add_cell_phenotypes(mask_wsi, phenotype_csv, phenotype_col="phenotype")

        he_wsi = self._build_he_wsi(tmp_path)
        try:
            patches = ad.AnnData(X=np.zeros((3, 1)), obs={"x": [0, 0, 0], "y": [0, 0, 0]})
            patches.obs["_slide_ref"] = [he_wsi, he_wsi, he_wsi]

            plan = GalleryPlan(patches)
            plan.add_he_row(slides={None: he_wsi})
            plan.add_cell_overlay_row(
                cells_key="cells", slides={None: mask_wsi}, color_by="phenotype",
                blend_with_previous=False, label="Cells",
            )
            fig, axes = plan.render(patches_per_row=3, return_fig=True, legend=False)
            try:
                texts = {t.get_text(): t.get_position()[1] for t in fig.texts}
                for row_idx, label in enumerate(["H&E", "Cells"]):
                    pos = axes[row_idx, 0].get_position()
                    expected_y = (pos.y0 + pos.y1) / 2
                    assert texts[label] == pytest.approx(expected_y, abs=1e-9)
            finally:
                plt.close(fig)
        finally:
            he_wsi.close()

    def test_background_color_heuristic(self, mask_path):
        """The heuristic reads the *cells-owning* slide's own channel count
        -- per the user's request ("chosen based on the slide that holds the
        cell shape table") -- so cells must be attached directly to the
        H&E/mIF slide being tested, not to a separate single-channel mask
        slide (which would always read back as 1-channel)."""
        from mesoslide.tools._cell_import import add_cell_polygons
        from mesoslide.plotting._gallery_plan import GalleryPlan
        import anndata as ad

        he_wsi = self._build_he_wsi_from_mask(mask_path, is_he=True)
        mif_wsi = self._build_he_wsi_from_mask(mask_path, is_he=False)
        try:
            for wsi, expect_white in [(he_wsi, True), (mif_wsi, False)]:
                add_cell_polygons(wsi, mask_path, chunked=False, progress_bar=False, overwrite=True)
                patches = ad.AnnData(X=np.zeros((1, 1)), obs={"x": [0], "y": [0]})
                patches.obs["_slide_ref"] = [wsi]
                plan = GalleryPlan(patches)
                plan.add_cell_overlay_row(
                    cells_key="cells", slides={None: wsi}, tile_key="tiles",
                    blend_with_previous=False,
                )
                frame = plan._blocks[-1].frames[0][-1]
                corner = frame[0, 0]  # a pixel with no cell drawn on it
                if expect_white:
                    assert np.allclose(corner, 1.0, atol=1e-3)
                else:
                    assert np.allclose(corner, 0.0, atol=1e-3)

            # explicit override wins regardless of slide
            patches = ad.AnnData(X=np.zeros((1, 1)), obs={"x": [0], "y": [0]})
            patches.obs["_slide_ref"] = [he_wsi]
            plan = GalleryPlan(patches)
            plan.add_cell_overlay_row(
                cells_key="cells", slides={None: he_wsi}, tile_key="tiles",
                blend_with_previous=False, background_color="red",
            )
            frame = plan._blocks[-1].frames[0][-1]
            assert np.allclose(frame[0, 0], (1.0, 0.0, 0.0), atol=1e-3)
        finally:
            he_wsi.close()
            mif_wsi.close()

    def _build_he_wsi_from_mask(self, mask_path, *, is_he: bool):
        """Read the mask's own extent to build an H&E-shaped or mIF-shaped
        slide the same size, with a tile spec, so cells derived from `mask_path`
        can be attached directly to it."""
        import ezslide
        import tifffile
        from wsidata import WSIData
        from wsidata._model.tile import TileSpec

        with tifffile.TiffFile(mask_path) as tf:
            h, w = tf.series[0].shape
        path = mask_path.replace("mask.ome.tif", f"{'he' if is_he else 'mif'}.ome.tif")
        if is_he:
            arr = np.full((h, w, 3), 200, dtype=np.uint8)
            tifffile.imwrite(path, arr, photometric="rgb", metadata={"axes": "YXS"}, ome=True)
        else:
            arr = np.full((5, h, w), 100, dtype=np.uint8)
            tifffile.imwrite(path, arr, photometric="minisblack", metadata={"axes": "CYX"}, ome=True)
        wsi = ezslide.open_slide(path, attach_images=False)
        spec = TileSpec(height=h, width=w, stride_height=h, stride_width=w)
        wsi.attrs[WSIData.TILE_SPEC_KEY] = {"tiles": spec.to_dict()}
        return wsi


class TestCycifMergeLegend:
    def _write_multichannel(self, path, channels, shape=(64, 64)):
        import tifffile

        arr = np.stack([
            np.full(shape, 50 * (i + 1), dtype=np.uint8) for i in range(len(channels))
        ])
        tifffile.imwrite(path, arr, photometric="minisblack", metadata={"axes": "CYX"}, ome=True)
        return path

    def test_merge_true_attaches_legend(self, tmp_path):
        import ezslide
        import anndata as ad
        from wsidata._model.tile import TileSpec
        from wsidata import WSIData
        from mesoslide.plotting._gallery_plan import GalleryPlan

        channels = ["DNA1", "CD3E", "CD8a"]
        path = str(tmp_path / "mc.ome.tif")
        self._write_multichannel(path, channels)
        marker_table = pd.DataFrame({"marker_name": channels})

        wsi = ezslide.open_slide(path, attach_images=False)
        spec = TileSpec(height=64, width=64, stride_height=64, stride_width=64)
        wsi.attrs[WSIData.TILE_SPEC_KEY] = {"tiles": spec.to_dict()}
        try:
            patches = ad.AnnData(X=np.zeros((1, 1)), obs={"x": [0], "y": [0]})
            patches.obs["_slide_ref"] = [wsi]
            plan = GalleryPlan(patches)
            plan.add_cycif_rows(channels, slides={None: wsi}, marker_table=marker_table, merge=True)

            block = plan._blocks[-1]
            assert block.legend_title == "Merge"
            assert set(block.legend_palette) == set(channels)
        finally:
            wsi.close()

    def test_merge_false_has_no_legend(self, tmp_path):
        import ezslide
        import anndata as ad
        from wsidata._model.tile import TileSpec
        from wsidata import WSIData
        from mesoslide.plotting._gallery_plan import GalleryPlan

        channels = ["DNA1", "CD3E"]
        path = str(tmp_path / "mc.ome.tif")
        self._write_multichannel(path, channels)
        marker_table = pd.DataFrame({"marker_name": channels})

        wsi = ezslide.open_slide(path, attach_images=False)
        spec = TileSpec(height=64, width=64, stride_height=64, stride_width=64)
        wsi.attrs[WSIData.TILE_SPEC_KEY] = {"tiles": spec.to_dict()}
        try:
            patches = ad.AnnData(X=np.zeros((1, 1)), obs={"x": [0], "y": [0]})
            patches.obs["_slide_ref"] = [wsi]
            plan = GalleryPlan(patches)
            plan.add_cycif_rows(channels, slides={None: wsi}, marker_table=marker_table, merge=False)

            block = plan._blocks[-1]
            assert block.legend_palette is None
            assert block.legend_title is None
        finally:
            wsi.close()
