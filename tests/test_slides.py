"""The cross-slide seam: iter_slides / open_slides / concat_slides."""

import numpy as np
import pandas as pd
import pytest

import mesoslide as ms
from mesoslide._slides import SlideSource, slide_id_from, tile_table_key


def test_tile_table_key():
    assert tile_table_key() == "tiles_table"
    assert tile_table_key("dense") == "dense_table"


def test_slide_id_strips_extensions(one_slide):
    assert one_slide.name.endswith(".ome.tif")
    assert slide_id_from(one_slide) == one_slide.name.split(".")[0]


def test_iter_slides_yields_every_slide(manifest):
    seen = [(sid, len(wsi.tables["tiles_table"])) for sid, wsi in ms.iter_slides(manifest)]
    assert len(seen) == len(manifest)
    assert len({sid for sid, _ in seen}) == len(manifest), "slide ids must be distinct"
    assert all(n > 0 for _, n in seen)


def test_iter_slides_is_reiterable(manifest):
    first = [sid for sid, _ in ms.iter_slides(manifest)]
    second = [sid for sid, _ in ms.iter_slides(manifest)]
    assert first == second


def test_iter_slides_round_trips_the_written_store(manifest):
    """A store written by wsi.write() keeps shapes, tables and its source pointer."""
    for _, wsi in ms.iter_slides(manifest, attach_images=True):
        assert "tiles" in wsi.shapes and "tissues" in wsi.shapes
        assert "tiles_table" in wsi.tables
        assert "wsi_source" in wsi.attrs
        assert "wsi" in wsi.images, "attach_images=True must restore the pixels"
        spec = wsi.tile_spec("tiles")
        assert spec.height == spec.width > 0


def test_written_store_has_no_pixels_without_attach(manifest):
    """The premise the plotting/extraction contract rests on."""
    for _, wsi in ms.iter_slides(manifest, attach_images=False):
        assert "wsi" not in wsi.images


def test_iter_slides_uses_slide_id_column_when_present(cohort):
    named = pd.DataFrame({"store": cohort, "slide_id": ["a", "b", "c"]})
    assert [sid for sid, _ in ms.iter_slides(named)] == ["a", "b", "c"]


def test_iter_slides_rejects_a_manifest_without_stores(cohort):
    with pytest.raises(ValueError, match="store_col"):
        list(ms.iter_slides(pd.DataFrame({"path": cohort})))


def test_open_slides_returns_a_mapping_with_pixels(manifest):
    slides = ms.open_slides(manifest)
    try:
        assert len(slides) == len(manifest)
        assert all("wsi" in w.images for w in slides.values())
    finally:
        for w in slides.values():
            w.close()


class TestConcatSlides:
    def test_row_count_is_the_sum(self, manifest):
        expected = sum(len(w.tables["tiles_table"]) for _, w in ms.iter_slides(manifest))
        assert ms.concat_slides(manifest).n_obs == expected

    def test_records_provenance(self, manifest):
        out = ms.concat_slides(manifest)
        assert out.obs["slide_id"].nunique() == len(manifest)
        assert out.obs["store"].nunique() == len(manifest)

    def test_obs_names_are_unique(self, manifest):
        """Tile ids restart at 0 per slide, so concat must disambiguate."""
        assert ms.concat_slides(manifest).obs_names.is_unique

    def test_drops_obsm_by_default(self, manifest):
        """The default that keeps a 40-slide cohort in megabytes."""
        assert list(ms.concat_slides(manifest).obsm) == []

    def test_carries_obsm_when_asked(self, manifest):
        out = ms.concat_slides(manifest, obsm_keys=["stub_embedding"])
        assert out.obsm["stub_embedding"].shape[0] == out.n_obs

    def test_rejects_empty_manifest(self):
        with pytest.raises(ValueError, match="empty"):
            ms.concat_slides(pd.DataFrame({"store": []}))


class TestSlideSource:
    """The adapter every selector funnels its `slides` argument through."""

    def test_single_table_has_no_slide_id(self, one_table):
        items = list(SlideSource(one_table))
        assert len(items) == 1
        assert items[0][0] is None, "a bare table must not be stamped with a slide id"

    def test_single_wsidata_is_identified(self, one_slide):
        (sid, table), = list(SlideSource(one_slide))
        assert sid == slide_id_from(one_slide)
        assert table is one_slide.tables["tiles_table"]

    def test_mapping_preserves_keys(self, open_cohort):
        assert [sid for sid, _ in SlideSource(open_cohort)] == list(open_cohort)

    def test_sequence_of_wsidata_derives_ids(self, open_cohort):
        wsis = list(open_cohort.values())
        assert [sid for sid, _ in SlideSource(wsis)] == [slide_id_from(w) for w in wsis]

    def test_manifest_streams(self, manifest):
        assert len(list(SlideSource(manifest))) == len(manifest)

    def test_rejects_unusable_input(self):
        with pytest.raises(TypeError, match="slides must be"):
            SlideSource(42)

    def test_missing_table_names_the_fix(self, one_slide):
        del one_slide.tables["tiles_table"]
        with pytest.raises(KeyError, match="feature_extraction"):
            list(SlideSource(one_slide))
