"""Reading tile pixels through wsi.read_region and wsi.tile_spec."""

import numpy as np
import pytest

import mesoslide as ms
from tests.conftest import TILE_PX


@pytest.fixture
def selection(manifest):
    return ms.select_random_patches(manifest, 8, random_state=7)


def test_stacks_channel_first(selection, open_cohort):
    out = ms.pp.extract_patch_images(selection, open_cohort, progress_bar=False)
    assert out.shape == (8, 3, TILE_PX, TILE_PX)
    assert out.dtype == np.uint8


def test_stacks_channel_last(selection, open_cohort):
    out = ms.pp.extract_patch_images(selection, open_cohort, channel_first=False,
                                progress_bar=False)
    assert out.shape == (8, TILE_PX, TILE_PX, 3)


def test_the_two_layouts_agree(selection, open_cohort):
    cf = ms.pp.extract_patch_images(selection, open_cohort, progress_bar=False)
    cl = ms.pp.extract_patch_images(selection, open_cohort, channel_first=False,
                               progress_bar=False)
    assert np.array_equal(np.moveaxis(cl, -1, 1), cf)


def test_matches_read_region_exactly(selection, open_cohort):
    """Ground truth: the same bytes the slide reader would hand back."""
    out = ms.pp.extract_patch_images(selection, open_cohort, channel_first=False,
                                progress_bar=False)
    for i in range(len(selection)):
        row = selection.obs.iloc[i]
        wsi = open_cohort[row["slide_id"]]
        ref = wsi.read_region(int(row.x), int(row.y), TILE_PX, TILE_PX)
        assert np.array_equal(out[i], ref), f"row {i} differs from read_region"


def test_tile_size_comes_from_the_slide_not_the_table(selection, open_cohort):
    """No per-row bounds columns are needed any more."""
    trimmed = selection[:, []].copy()
    trimmed.obs = selection.obs[["x", "y", "slide_id"]].copy()
    out = ms.pp.extract_patch_images(trimmed, open_cohort, progress_bar=False)
    assert out.shape[-2:] == (TILE_PX, TILE_PX)


def test_single_slide_needs_no_slide_id(one_slide, one_table):
    sel = ms.select_random_patches(one_table, 4, random_state=0)
    assert "slide_id" not in sel.obs.columns
    assert ms.pp.extract_patch_images(sel, one_slide, progress_bar=False).shape[0] == 4


def test_missing_columns_are_named(one_slide, one_table):
    sel = ms.select_random_patches(one_table, 2, random_state=0)
    del sel.obs["x"]
    with pytest.raises(ValueError, match="missing required columns.*x"):
        ms.pp.extract_patch_images(sel, one_slide, progress_bar=False)


def test_unknown_slide_is_skipped_or_raised(selection, open_cohort):
    sel = selection.copy()
    sel.obs["slide_id"] = "not_a_slide"
    with pytest.raises(ValueError, match="no slide"):
        ms.pp.extract_patch_images(sel, open_cohort, progress_bar=False, skip_errors=False)


def test_rejects_an_unusable_slides_argument(selection):
    with pytest.raises(TypeError, match="open_slides"):
        ms.pp.extract_patch_images(selection, "not a slide", progress_bar=False)


def test_cache_writes_channel_first_array_to_obsm(selection, open_cohort):
    out = ms.pp.extract_patch_images(selection, open_cohort, progress_bar=False,
                                      cache=True)
    assert "he_patch_img" in selection.obsm
    assert np.array_equal(selection.obsm["he_patch_img"], out)


def test_cache_hit_skips_slide_reads(selection, open_cohort):
    ms.pp.extract_patch_images(selection, open_cohort, progress_bar=False, cache=True)
    # A bogus `slides` argument would raise on any real slide lookup, so a
    # successful call here proves the cached obsm array was used instead.
    out = ms.pp.extract_patch_images(selection, {"not_a_slide": None}, progress_bar=False)
    assert out.shape == (8, 3, TILE_PX, TILE_PX)


def test_cache_hit_respects_requested_layout(selection, open_cohort):
    ms.pp.extract_patch_images(selection, open_cohort, progress_bar=False, cache=True)
    cl = ms.pp.extract_patch_images(selection, open_cohort, channel_first=False,
                                     progress_bar=False)
    assert np.array_equal(np.moveaxis(cl, -1, 1), selection.obsm["he_patch_img"])


def test_cache_skipped_when_rows_are_dropped(selection, open_cohort):
    sel = selection.copy()
    sel.obs["slide_id"] = sel.obs["slide_id"].astype(str)
    sel.obs.iloc[0, sel.obs.columns.get_loc("slide_id")] = "not_a_slide"
    with pytest.warns(UserWarning, match="cache=True has no effect"):
        ms.pp.extract_patch_images(sel, open_cohort, progress_bar=False, cache=True)
    assert "he_patch_img" not in sel.obsm


# --- SLIDE_REF fallback: no `slides=` argument at all ------------------------

def test_no_slides_falls_back_to_slide_ref_manifest_backed(manifest, open_cohort):
    """A manifest-backed selection carries store paths in SLIDE_REF; with no
    `slides=`, extraction must reopen from those and still match ground truth.
    """
    sel = ms.select_random_patches(manifest, 8, random_state=7)
    out = ms.pp.extract_patch_images(sel, progress_bar=False)
    ref = ms.pp.extract_patch_images(sel, open_cohort, progress_bar=False)
    assert np.array_equal(out, ref)


def test_no_slides_opens_each_manifest_slide_exactly_once(manifest, monkeypatch):
    """Opening must be grouped by unique slide, not once per row."""
    import ezslide

    sel = ms.select_random_patches(manifest, 20, random_state=1)
    n_unique_slides = sel.obs["slide_id"].nunique()
    assert n_unique_slides < len(sel), "test needs >1 row per slide to be meaningful"

    open_count = {"n": 0}
    original_read_slide = ezslide.read_slide

    def _counting_read_slide(*args, **kwargs):
        open_count["n"] += 1
        return original_read_slide(*args, **kwargs)

    monkeypatch.setattr(ezslide, "read_slide", _counting_read_slide)

    ms.pp.extract_patch_images(sel, progress_bar=False)
    assert open_count["n"] == n_unique_slides


def test_no_slides_reuses_already_open_wsidata_with_zero_reopens(open_cohort, monkeypatch):
    """When SLIDE_REF holds live WSIData (selection ran against open_slides
    output), extraction must reuse those objects directly -- no reopening.
    """
    import ezslide

    sel = ms.select_random_patches(open_cohort, 10, random_state=1)

    open_count = {"n": 0}
    original_read_slide = ezslide.read_slide

    def _counting_read_slide(*args, **kwargs):
        open_count["n"] += 1
        return original_read_slide(*args, **kwargs)

    monkeypatch.setattr(ezslide, "read_slide", _counting_read_slide)

    out = ms.pp.extract_patch_images(sel, progress_bar=False)
    assert open_count["n"] == 0
    assert out.shape[0] == 10


def test_no_slides_and_no_slide_ref_raises(one_slide, one_table):
    from mesoslide._slides import SLIDE_REF

    sel = ms.select_random_patches(one_table, 2, random_state=0)
    del sel.obs[SLIDE_REF]
    with pytest.raises(ValueError, match="slides was not given"):
        ms.pp.extract_patch_images(sel, progress_bar=False)
