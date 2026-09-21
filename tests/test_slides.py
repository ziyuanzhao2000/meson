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

    def test_iter_with_ref_matches_iter_for_slide_id_and_table(self, manifest):
        """Additive: the 3-tuple's first two elements must match __iter__'s pair."""
        plain = list(SlideSource(manifest))
        with_ref = list(SlideSource(manifest).iter_with_ref())
        assert [sid for sid, _ in plain] == [sid for sid, _, _ in with_ref]
        assert [len(table) for _, table in plain] == [len(table) for _, table, _ in with_ref]

    def test_iter_with_ref_mapping_yields_the_live_objects(self, open_cohort):
        for slide_id, _, ref in SlideSource(open_cohort).iter_with_ref():
            assert ref is open_cohort[slide_id]

    def test_iter_with_ref_manifest_yields_store_paths(self, manifest):
        stores = set(manifest["store"])
        for _, _, ref in SlideSource(manifest).iter_with_ref():
            assert isinstance(ref, str) and ref in stores

    def test_iter_with_ref_bare_table_yields_none(self, one_table):
        (_, _, ref), = list(SlideSource(one_table).iter_with_ref())
        assert ref is None


class TestAttachSlideRef:
    def test_rehydrates_a_path_string_to_a_live_object(self, manifest, open_cohort):
        from mesoslide._slides import SLIDE_REF, attach_slide_ref

        sel = ms.select_top_patches(manifest, "score", n=6)
        assert all(isinstance(v, str) for v in sel.obs[SLIDE_REF])

        out = attach_slide_ref(sel, open_cohort)
        for slide_id, ref in zip(out.obs["slide_id"], out.obs[SLIDE_REF]):
            assert ref is open_cohort[slide_id]
        # Original is untouched.
        assert all(isinstance(v, str) for v in sel.obs[SLIDE_REF])

    def test_retargets_via_an_explicit_id_map(self, manifest, open_cohort):
        """Cross-modality hand-over: the target slide set is keyed differently."""
        from mesoslide._slides import SLIDE_REF, attach_slide_ref

        sel = ms.select_top_patches(manifest, "score", n=6)
        fake_cycif = {f"cycif_{sid}": wsi for sid, wsi in open_cohort.items()}
        id_map = {sid: f"cycif_{sid}" for sid in open_cohort}

        out = attach_slide_ref(sel, fake_cycif, slide_id_map=id_map)
        for slide_id, ref in zip(out.obs["slide_id"], out.obs[SLIDE_REF]):
            assert ref is open_cohort[slide_id]

    def test_missing_mapped_id_raises(self, manifest):
        from mesoslide._slides import attach_slide_ref

        sel = ms.select_top_patches(manifest, "score", n=2)
        with pytest.raises(ValueError, match="No entry in `slides`"):
            attach_slide_ref(sel, {"nonexistent": None})


class TestStripSlideRefs:
    def test_replaces_live_objects_with_their_path(self, manifest, open_cohort):
        from mesoslide._slides import SLIDE_REF, attach_slide_ref, strip_slide_refs

        sel = ms.select_top_patches(manifest, "score", n=6)
        rehydrated = attach_slide_ref(sel, open_cohort)

        stripped = strip_slide_refs(rehydrated)
        for orig_path, ref in zip(sel.obs[SLIDE_REF], stripped.obs[SLIDE_REF]):
            assert ref == orig_path
        # Original is untouched (still live objects).
        assert all(hasattr(v, "read_region") for v in rehydrated.obs[SLIDE_REF])

    def test_leaves_path_strings_and_nulls_untouched(self, manifest, one_table):
        from mesoslide._slides import SLIDE_REF, strip_slide_refs

        sel = ms.select_top_patches(manifest, "score", n=4)
        stripped = strip_slide_refs(sel)
        assert list(stripped.obs[SLIDE_REF]) == list(sel.obs[SLIDE_REF])

        bare = ms.select_top_patches(one_table, "score", n=4)
        stripped_bare = strip_slide_refs(bare)
        assert stripped_bare.obs[SLIDE_REF].isna().all()
