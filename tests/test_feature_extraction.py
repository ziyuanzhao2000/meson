"""feature_extraction's table contract, and its dense/reducer and sparse modes."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import mesoslide as ms


def test_table_has_the_columns_downstream_expects(one_table):
    for col in ("tile_id", "tissue_id", "x", "y", "library_id"):
        assert col in one_table.obs.columns


def test_tile_id_dtype_matches_the_tiles_element(one_slide):
    """SpatialData matches instance_key values against the element index.

    A str/int mismatch makes the table look unrelated to its shapes, and
    spatialdata_plot then refuses to render the overlay.
    """
    table = one_slide.tables["tiles_table"]
    tiles = one_slide.shapes["tiles"]
    assert table.obs["tile_id"].dtype == tiles["tile_id"].dtype


def test_obs_index_is_str(one_table):
    """AnnData requires string obs names."""
    assert all(isinstance(i, str) for i in one_table.obs_names)


def test_obs_index_carries_no_colliding_name(one_table):
    """Regression: an index named after a column whose values differ fails on write.

    The column is int and the index is str, so if the index inherits the name
    'tile_id' anndata rejects the whole table at write time -- which only shows
    up with save=True.
    """
    assert one_table.obs.index.name != "tile_id"


def test_table_round_trips_through_a_written_store(cohort):
    """The fixtures write with save=True, so reaching this at all exercises it."""
    import ezslide

    wsi = ezslide.read_slide(cohort[0])
    table = wsi.tables["tiles_table"]
    assert table.n_obs == len(wsi.shapes["tiles"])
    assert "stub_embedding" in table.obsm
    assert table.obsm["stub_embedding"].shape == (table.n_obs, 8)


def test_whole_slide_batches_are_channel_first_for_the_model_transform(
    one_slide, real_transform_stub_encoder
):
    """Regression: the whole-slide DataLoader path must hand ModelStage
    channel-first (B, C, H, W) batches, matching what the patch-table path
    already gets from extract_patch_images(channel_first=True).

    ezslide/wsidata's tile datasets hand back channel-last (H, W, C) images;
    RealTransformStubEncoder's real ToImage/Normalize transform chain (unlike
    StubEncoder's get_transform() -> None) fails on channel-last input, so
    this exercises the conversion that StubEncoder-based tests skip entirely.
    """
    table = one_slide.tables["tiles_table"]
    n = table.n_obs
    ms.tl.feature_extraction(
        one_slide, real_transform_stub_encoder, key_added="real_transform",
        batch_size=8, num_workers=0, device="cpu", save=False,
    )
    assert table.obsm["real_transform"].shape == (n, 8)


def test_x_y_are_tile_origins(one_slide):
    """extract_patch_images reads from these, so they must be the level-0 top-left."""
    table = one_slide.tables["tiles_table"]
    bounds = one_slide.shapes["tiles"].bounds
    assert np.array_equal(table.obs["x"].to_numpy(), bounds["minx"].to_numpy())
    assert np.array_equal(table.obs["y"].to_numpy(), bounds["miny"].to_numpy())


class TestDenseMode:
    """feature_extraction(dense=True, reducer=...).

    StubViTEncoder.encode_image_dense is deliberately image-independent: token
    k's embedding is a constant-k vector across every embedding dimension, so
    a mean-reducer recovers exactly k -- letting these tests assert on exact
    values rather than only shapes.
    """

    N_TOKENS = 4  # StubViTEncoder.grid_size == (2, 2)

    @staticmethod
    def _mean_reducer(patch_tokens):
        return patch_tokens.mean(-1)

    def test_dense_map_shape_and_values(self, one_slide, vit_stub_encoder):
        table = one_slide.tables["tiles_table"]
        n = table.n_obs
        ms.tl.feature_extraction(
            one_slide, vit_stub_encoder, key_added="vitstub", dense=True,
            reducer=self._mean_reducer, batch_size=16, device="cpu", save=False,
        )
        assert "vitstub" not in table.obsm, "dense=True no longer also computes pooled"
        dense = table.obsm["vitstub_dense"]
        assert dense.shape == (n, self.N_TOKENS)
        expected = np.tile(np.arange(self.N_TOKENS, dtype=np.float32), (n, 1))
        assert np.allclose(dense, expected)

    def test_dense_key_added_overrides_the_default_suffix(self, one_slide, vit_stub_encoder):
        table = one_slide.tables["tiles_table"]
        ms.tl.feature_extraction(
            one_slide, vit_stub_encoder, key_added="vitstub", dense=True,
            reducer=self._mean_reducer, dense_key_added="custom_dense",
            batch_size=16, device="cpu", save=False,
        )
        assert "custom_dense" in table.obsm
        assert "vitstub_dense" not in table.obsm

    def test_already_cached_keys_are_left_untouched(self, one_slide, vit_stub_encoder):
        table = one_slide.tables["tiles_table"]
        n = table.n_obs
        ms.tl.feature_extraction(
            one_slide, vit_stub_encoder, key_added="vitstub", dense=True,
            reducer=self._mean_reducer, batch_size=16, device="cpu", save=False,
        )
        dense_sentinel = np.full((n, self.N_TOKENS), -2.0, dtype=np.float32)
        table.obsm["vitstub_dense"] = dense_sentinel

        ms.tl.feature_extraction(
            one_slide, vit_stub_encoder, key_added="vitstub", dense=True,
            reducer=self._mean_reducer, batch_size=16, device="cpu", save=False,
        )
        assert np.array_equal(table.obsm["vitstub_dense"], dense_sentinel)

    def test_overwrite_replaces_the_dense_key(self, one_slide, vit_stub_encoder):
        table = one_slide.tables["tiles_table"]
        n = table.n_obs
        ms.tl.feature_extraction(
            one_slide, vit_stub_encoder, key_added="vitstub", dense=True,
            reducer=self._mean_reducer, batch_size=16, device="cpu", save=False,
        )
        dense_sentinel = np.full((n, self.N_TOKENS), -2.0, dtype=np.float32)
        table.obsm["vitstub_dense"] = dense_sentinel

        ms.tl.feature_extraction(
            one_slide, vit_stub_encoder, key_added="vitstub", dense=True,
            reducer=self._mean_reducer, batch_size=16, device="cpu", save=False,
            overwrite=True,
        )
        assert not np.array_equal(table.obsm["vitstub_dense"], dense_sentinel)

    def test_recomputes_only_the_missing_key(self, one_slide, vit_stub_encoder):
        """Pooled cached + dense missing must recompute dense alone, and vice versa."""
        table = one_slide.tables["tiles_table"]
        n = table.n_obs
        ms.tl.feature_extraction(
            one_slide, vit_stub_encoder, key_added="vitstub", dense=True,
            reducer=self._mean_reducer, batch_size=16, device="cpu", save=False,
        )
        pooled_sentinel = np.full((n, vit_stub_encoder.embed_dim), -1.0, dtype=np.float32)
        table.obsm["vitstub"] = pooled_sentinel
        del table.obsm["vitstub_dense"]

        ms.tl.feature_extraction(
            one_slide, vit_stub_encoder, key_added="vitstub", dense=True,
            reducer=self._mean_reducer, batch_size=16, device="cpu", save=False,
        )
        assert np.array_equal(table.obsm["vitstub"], pooled_sentinel), "pooled must be untouched"
        assert table.obsm["vitstub_dense"].shape == (n, self.N_TOKENS)

    def test_requires_a_reducer(self, one_slide, vit_stub_encoder):
        with pytest.raises(ValueError, match="reducer"):
            ms.tl.feature_extraction(
                one_slide, vit_stub_encoder, key_added="vitstub_noreducer",
                dense=True, batch_size=16, device="cpu", save=False,
            )

    def test_requires_a_dense_capable_model(self, one_slide, stub_encoder):
        with pytest.raises(NotImplementedError, match="uni"):
            ms.tl.feature_extraction(
                one_slide, stub_encoder, key_added="stub_dense",
                dense=True, reducer=self._mean_reducer,
                batch_size=16, device="cpu", save=False,
            )

    def test_reducer_output_shape_is_validated(self, one_slide, vit_stub_encoder):
        bad_reducer = lambda patch_tokens: patch_tokens[:, :1, :]  # wrong n_tokens
        with pytest.raises(ValueError, match="reducer must return shape"):
            ms.tl.feature_extraction(
                one_slide, vit_stub_encoder, key_added="vitstub_badshape",
                dense=True, reducer=bad_reducer,
                batch_size=16, device="cpu", save=False,
            )

    def test_reducer_may_return_a_trailing_singleton_dim(self, one_slide, vit_stub_encoder):
        """(B, N_tokens, 1) is accepted and squeezed, matching a linear-probe reducer."""
        table = one_slide.tables["tiles_table"]
        squeeze_reducer = lambda patch_tokens: patch_tokens.mean(-1, keepdim=True)
        ms.tl.feature_extraction(
            one_slide, vit_stub_encoder, key_added="vitstub_squeeze", dense=True,
            reducer=squeeze_reducer, batch_size=16, device="cpu", save=False,
        )
        assert table.obsm["vitstub_squeeze_dense"].shape == (table.n_obs, self.N_TOKENS)

    def test_dense_mode_does_not_compute_pooled(self, one_slide, vit_stub_encoder):
        """dense=True computes only the dense chain; call it separately for pooled."""
        table = one_slide.tables["tiles_table"]
        ms.tl.feature_extraction(
            one_slide, vit_stub_encoder, key_added="pooled_only", dense=True,
            reducer=self._mean_reducer, batch_size=16, device="cpu", save=False,
        )
        assert "pooled_only" not in table.obsm
        assert "pooled_only_dense" in table.obsm

    def test_dense_and_sparse_together_raise(self, one_slide, vit_stub_encoder):
        with pytest.raises(ValueError, match="dense=True and sparse=True"):
            ms.tl.feature_extraction(
                one_slide, vit_stub_encoder, key_added="vitstub_both", dense=True,
                reducer=self._mean_reducer, sparse=True,
                sparse_transform=lambda pooled: csr_matrix(pooled[:, :2]),
                batch_size=16, device="cpu", save=False,
            )


class TestSparseMode:
    """feature_extraction(sparse=True, sparse_transform=...)."""

    @staticmethod
    def _double_transform(pooled):
        return csr_matrix(pooled[:, :2] * 2)

    def test_sparse_features_shape_and_values(self, one_slide, stub_encoder):
        ms.tl.feature_extraction(
            one_slide, stub_encoder, key_added="stubsparse",
            sparse=True, sparse_transform=self._double_transform,
            batch_size=16, device="cpu", save=False,
        )
        table = one_slide.tables["tiles_table"]
        pooled = table.obsm["stubsparse"]
        assert table.X.shape == (table.n_obs, 2)
        assert list(table.var_names) == ["stubsparse_sparse_0", "stubsparse_sparse_1"]
        assert np.allclose(table.X.toarray(), pooled[:, :2] * 2)

    def test_sparse_key_added_overrides_the_default_prefix(self, one_slide, stub_encoder):
        ms.tl.feature_extraction(
            one_slide, stub_encoder, key_added="stubsparse2",
            sparse=True, sparse_transform=self._double_transform,
            sparse_key_added="custom_prefix",
            batch_size=16, device="cpu", save=False,
        )
        table = one_slide.tables["tiles_table"]
        assert list(table.var_names) == ["custom_prefix_0", "custom_prefix_1"]

    def test_already_cached_prefix_is_left_untouched(self, one_slide, stub_encoder):
        ms.tl.feature_extraction(
            one_slide, stub_encoder, key_added="stubsparse3",
            sparse=True, sparse_transform=self._double_transform,
            batch_size=16, device="cpu", save=False,
        )
        table = one_slide.tables["tiles_table"]
        sentinel = csr_matrix(np.full((table.n_obs, 2), -1.0, dtype=np.float32))
        table.X = sentinel

        ms.tl.feature_extraction(
            one_slide, stub_encoder, key_added="stubsparse3",
            sparse=True, sparse_transform=self._double_transform,
            batch_size=16, device="cpu", save=False,
        )
        # need_sparse is False (already cached), so feature_extraction returns
        # before ever touching the table -- the same object, sentinel intact.
        table = one_slide.tables["tiles_table"]
        assert np.array_equal(table.X.toarray(), sentinel.toarray())

    def test_overwrite_replaces_only_its_own_prefix(self, one_slide, stub_encoder):
        ms.tl.feature_extraction(
            one_slide, stub_encoder, key_added="stubsparse4",
            sparse=True, sparse_transform=self._double_transform,
            sparse_key_added="prefix_a",
            batch_size=16, device="cpu", save=False,
        )
        ms.tl.feature_extraction(
            one_slide, stub_encoder, key_added="stubsparse4",
            sparse=True, sparse_transform=lambda pooled: csr_matrix(pooled[:, :1] * 3),
            sparse_key_added="prefix_b",
            batch_size=16, device="cpu", save=False,
        )
        table = one_slide.tables["tiles_table"]
        assert set(table.var_names) == {"prefix_a_0", "prefix_a_1", "prefix_b_0"}

        cols_b = [i for i, v in enumerate(table.var_names) if v.startswith("prefix_b_")]
        prefix_b_before = table.X[:, cols_b].toarray()

        ms.tl.feature_extraction(
            one_slide, stub_encoder, key_added="stubsparse4",
            sparse=True, sparse_transform=self._double_transform,
            sparse_key_added="prefix_a",
            batch_size=16, device="cpu", save=False,
            overwrite=True,
        )
        table = one_slide.tables["tiles_table"]
        cols_b_after = [i for i, v in enumerate(table.var_names) if v.startswith("prefix_b_")]
        assert np.array_equal(table.X[:, cols_b_after].toarray(), prefix_b_before)
        assert {v for v in table.var_names if v.startswith("prefix_a_")} == {"prefix_a_0", "prefix_a_1"}

    def test_requires_a_sparse_transform(self, one_slide, stub_encoder):
        with pytest.raises(ValueError, match="sparse_transform"):
            ms.tl.feature_extraction(
                one_slide, stub_encoder, key_added="stubsparse_notransform",
                sparse=True, batch_size=16, device="cpu", save=False,
            )

    def test_transform_output_shape_is_validated(self, one_slide, stub_encoder):
        bad_transform = lambda pooled: csr_matrix(pooled[:1, :2])  # wrong n_tiles
        with pytest.raises(ValueError, match="sparse_transform must return"):
            ms.tl.feature_extraction(
                one_slide, stub_encoder, key_added="stubsparse_badshape",
                sparse=True, sparse_transform=bad_transform,
                batch_size=16, device="cpu", save=False,
            )

    def test_pooled_result_is_unaffected_by_sparse_mode(self, one_slide, stub_encoder):
        table = one_slide.tables["tiles_table"]
        ms.tl.feature_extraction(
            one_slide, stub_encoder, key_added="pooled_only_sparse",
            batch_size=16, device="cpu", save=False,
        )
        pooled_alone = table.obsm["pooled_only_sparse"].copy()
        del table.obsm["pooled_only_sparse"]

        ms.tl.feature_extraction(
            one_slide, stub_encoder, key_added="pooled_only_sparse", sparse=True,
            sparse_transform=self._double_transform,
            batch_size=16, device="cpu", save=False,
        )
        table = one_slide.tables["tiles_table"]
        assert np.array_equal(table.obsm["pooled_only_sparse"], pooled_alone)

    def test_sparse_only_call_skips_the_model_when_pooled_is_cached(self, one_slide, stub_encoder):
        """A sparse-only call against an already-embedded slide must not re-run the encoder."""
        ms.tl.feature_extraction(
            one_slide, stub_encoder, key_added="cached_pooled",
            batch_size=16, device="cpu", save=False,
        )

        calls = []
        orig_encode = stub_encoder.encode_image
        stub_encoder.encode_image = lambda *a, **k: calls.append(1) or orig_encode(*a, **k)
        try:
            ms.tl.feature_extraction(
                one_slide, stub_encoder, key_added="cached_pooled", sparse=True,
                sparse_transform=self._double_transform,
                batch_size=16, device="cpu", save=False,
            )
        finally:
            stub_encoder.encode_image = orig_encode
        assert calls == []
