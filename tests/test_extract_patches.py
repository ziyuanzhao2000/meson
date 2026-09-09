"""Reading tile pixels through wsi.read_region and wsi.tile_spec."""

import numpy as np
import pytest

import mesoslide as ms
from tests.conftest import TILE_PX


@pytest.fixture
def selection(manifest):
    return ms.select_random_patches(manifest, 8, random_state=7)


def test_stacks_channel_first(selection, open_cohort):
    out = ms.pp.extract_patches(selection, open_cohort, progress_bar=False)
    assert out.shape == (8, 3, TILE_PX, TILE_PX)
    assert out.dtype == np.uint8


def test_stacks_channel_last(selection, open_cohort):
    out = ms.pp.extract_patches(selection, open_cohort, channel_first=False,
                                progress_bar=False)
    assert out.shape == (8, TILE_PX, TILE_PX, 3)


def test_the_two_layouts_agree(selection, open_cohort):
    cf = ms.pp.extract_patches(selection, open_cohort, progress_bar=False)
    cl = ms.pp.extract_patches(selection, open_cohort, channel_first=False,
                               progress_bar=False)
    assert np.array_equal(np.moveaxis(cl, -1, 1), cf)


def test_matches_read_region_exactly(selection, open_cohort):
    """Ground truth: the same bytes the slide reader would hand back."""
    out = ms.pp.extract_patches(selection, open_cohort, channel_first=False,
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
    out = ms.pp.extract_patches(trimmed, open_cohort, progress_bar=False)
    assert out.shape[-2:] == (TILE_PX, TILE_PX)


def test_single_slide_needs_no_slide_id(one_slide, one_table):
    sel = ms.select_random_patches(one_table, 4, random_state=0)
    assert "slide_id" not in sel.obs.columns
    assert ms.pp.extract_patches(sel, one_slide, progress_bar=False).shape[0] == 4


def test_missing_columns_are_named(one_slide, one_table):
    sel = ms.select_random_patches(one_table, 2, random_state=0)
    del sel.obs["x"]
    with pytest.raises(ValueError, match="missing required columns.*x"):
        ms.pp.extract_patches(sel, one_slide, progress_bar=False)


def test_unknown_slide_is_skipped_or_raised(selection, open_cohort):
    sel = selection.copy()
    sel.obs["slide_id"] = "not_a_slide"
    with pytest.raises(ValueError, match="no slide"):
        ms.pp.extract_patches(sel, open_cohort, progress_bar=False, skip_errors=False)


def test_rejects_an_unusable_slides_argument(selection):
    with pytest.raises(TypeError, match="open_slides"):
        ms.pp.extract_patches(selection, "not a slide", progress_bar=False)
