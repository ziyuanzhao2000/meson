"""Patch selection across one slide, several slides, and a streamed manifest."""

import numpy as np
import pytest

import mesoslide as ms


# --- accepted input shapes --------------------------------------------------

def test_accepts_a_bare_table(one_table):
    out = ms.select_top_patches(one_table, "score", n=5)
    assert out.n_obs == 5


def test_bare_table_is_not_stamped_with_a_slide_id(one_table):
    """A single table may already carry provenance; don't overwrite it."""
    assert "slide_id" not in ms.select_top_patches(one_table, "score", n=5).obs.columns


def test_accepts_a_wsidata(one_slide):
    out = ms.select_top_patches(one_slide, "score", n=5)
    assert out.obs["slide_id"].nunique() == 1


def test_accepts_a_manifest_and_streams_it(manifest):
    out = ms.select_top_patches(manifest, "score", n=30)
    assert out.obs["slide_id"].nunique() == len(manifest)


def test_accepts_a_mapping(open_cohort):
    out = ms.select_top_patches(open_cohort, "score", n=30)
    assert set(out.obs["slide_id"]) <= set(open_cohort)


# --- select_top_patches -----------------------------------------------------

class TestSelectTop:
    def test_is_globally_sorted_descending(self, manifest):
        scores = ms.select_top_patches(manifest, "score", n=40).obs["_feature_score"]
        assert np.all(np.diff(scores.to_numpy()) <= 0)

    def test_really_is_the_global_top(self, manifest, open_cohort):
        """The whole point of streaming: the answer must match a full sort."""
        every = np.concatenate(
            [w.tables["tiles_table"].obs["score"].to_numpy() for w in open_cohort.values()]
        )
        expected = np.sort(every)[::-1][:10]
        got = ms.select_top_patches(
            manifest, "score", n=10, take_every=1
        ).obs["_feature_score"].to_numpy()
        assert np.allclose(got, expected)

    def test_annotates_rank_and_feature(self, manifest):
        out = ms.select_top_patches(manifest, "score", n=12)
        assert out.obs["_feature_rank"].min() == 1
        assert list(out.obs["_feature_name"].unique()) == ["score"]

    def test_obs_names_stay_unique_across_slides(self, manifest):
        """Tile ids restart at 0 on every slide."""
        assert ms.select_top_patches(manifest, "score", n=40).obs_names.is_unique

    def test_n_none_returns_everything_above_min_score(self, one_table):
        out = ms.select_top_patches(one_table, "sparse_score", n=None)
        assert out.n_obs == int((one_table.obs["sparse_score"] > 0).sum())

    def test_take_every_strides(self, one_table):
        dense = ms.select_top_patches(one_table, "score", n=None, min_score=-1)
        strided = ms.select_top_patches(one_table, "score", n=None, min_score=-1, take_every=2)
        assert strided.n_obs == len(range(0, dense.n_obs, 2))

    def test_n_zero_is_empty_but_keeps_the_schema(self, one_table):
        out = ms.select_top_patches(one_table, "score", n=0)
        assert out.n_obs == 0
        assert list(out.obs.columns) == list(one_table.obs.columns)

    def test_negative_n_rejected(self, one_table):
        with pytest.raises(ValueError, match="n must be"):
            ms.select_top_patches(one_table, "score", n=-1)

    def test_unknown_feature_names_the_slide(self, manifest):
        with pytest.raises(KeyError, match="slide="):
            ms.select_top_patches(manifest, "no_such_feature", n=5)

    def test_top_fraction_restricts_to_top_percent(self, one_table):
        dense = ms.select_top_patches(one_table, "score", n=None, min_score=-1, take_every=1)
        num_qualifying = dense.n_obs
        out = ms.select_top_patches(one_table, "score", n=5, top_fraction=0.1)
        top_count = max(1, int(np.ceil(0.1 * num_qualifying)))
        assert out.n_obs == min(5, top_count)
        threshold = np.sort(dense.obs["_feature_score"].to_numpy())[::-1][top_count - 1]
        assert np.all(out.obs["_feature_score"].to_numpy() >= threshold)

    def test_top_fraction_requires_n(self, one_table):
        with pytest.raises(ValueError, match="n is required"):
            ms.select_top_patches(one_table, "score", n=None, top_fraction=0.1)

    def test_top_fraction_rejects_take_every(self, one_table):
        with pytest.raises(ValueError, match="mutually exclusive"):
            ms.select_top_patches(one_table, "score", n=5, top_fraction=0.1, take_every=2)

    def test_top_fraction_out_of_range_rejected(self, one_table):
        with pytest.raises(ValueError, match="top_fraction must be"):
            ms.select_top_patches(one_table, "score", n=5, top_fraction=0)


# --- the other selectors ----------------------------------------------------

def test_select_random_is_reproducible(manifest):
    a = ms.select_random_patches(manifest, 20, random_state=0)
    b = ms.select_random_patches(manifest, 20, random_state=0)
    assert list(a.obs_names) == list(b.obs_names)


def test_select_random_spans_slides(manifest):
    out = ms.select_random_patches(manifest, 40, random_state=0)
    assert out.n_obs == 40
    assert out.obs["slide_id"].nunique() > 1


def test_select_random_caps_at_whats_available(one_table):
    out = ms.select_random_patches(one_table, 10 ** 6, random_state=0)
    assert out.n_obs == one_table.n_obs


def test_select_negative_finds_only_zeros(manifest, open_cohort):
    out = ms.select_negative_patches(manifest, "sparse_score", n=10)
    assert out.n_obs <= 10
    lookup = {sid: w.tables["tiles_table"] for sid, w in open_cohort.items()}
    for sid, tid in zip(out.obs["slide_id"], out.obs["tile_id"]):
        table = lookup[sid]
        assert table.obs.loc[table.obs["tile_id"] == tid, "sparse_score"].iloc[0] == 0


def test_select_binary_feature(manifest):
    out = ms.select_patches_for_binary_feature(manifest, "flag", n=15)
    assert out.n_obs == 15


def test_select_binary_feature_all_when_n_is_none(one_table):
    out = ms.select_patches_for_binary_feature(one_table, "flag", n=None)
    assert out.n_obs == int((one_table.obs["flag"] == 1).sum())


def test_select_binary_feature_deprecated_rng_is_reproducible(one_table):
    """Kept so published figures reproduce against the original results."""
    a = ms.select_patches_for_binary_feature(one_table, "flag", n=5, random_state=1,
                                             deprecated_rng=True)
    b = ms.select_patches_for_binary_feature(one_table, "flag", n=5, random_state=1,
                                             deprecated_rng=True)
    assert list(a.obs_names) == list(b.obs_names)


def test_select_binary_feature_raises_when_nothing_is_active(one_table):
    with pytest.raises(ValueError, match="No active patches"):
        ms.select_patches_for_binary_feature(one_table, "tissue_id", n=5)


class TestExemplars:
    def test_one_row_per_feature_per_rank(self, manifest):
        out = ms.select_exemplar_patches(manifest, ["score", "sparse_score"], n_exemplars=3)
        assert out.n_obs == 6
        assert set(out.obs["_feature_name"]) == {"score", "sparse_score"}
        assert set(out.obs["_feature_rank"]) == {1, 2, 3}

    def test_rank_one_is_the_maximum(self, manifest, open_cohort):
        top1 = ms.select_exemplar_patches(manifest, ["score"], n_exemplars=1)
        best = max(w.tables["tiles_table"].obs["score"].max() for w in open_cohort.values())
        assert np.isclose(top1.obs["_feature_score"].iloc[0], best)
