"""Streaming SAE statistics must equal the concatenated answer, exactly.

These use synthetic sparse matrices rather than real slides: nothing in the live
API populates tiles_table.X until the SAE itself is migrated out of _legacy/, and
the property under test is arithmetic, not I/O.
"""

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from mesoslide.tools.sae import SAEFeatureClusterer, SAEFeatureSelector
from mesoslide.tools.sae._feature_clusterer import (
    _intersection_and_sums,
    _weighted_iou,
    iou_from_parts,
)

PREFIX = "UNI_SAE"
N_FEATURES = 40


def _part(n_obs, seed, n_features=N_FEATURES, density=0.05):
    X = sp.random(n_obs, n_features, density=density, format="csr", random_state=seed)
    a = ad.AnnData(X=X)
    a.var_names = [f"{PREFIX}_{i}" for i in range(n_features)]
    a.obs_names = [f"s{seed}_t{i}" for i in range(n_obs)]
    return a


@pytest.fixture(scope="module")
def parts():
    """Several slides, including the edge cases that break naive accumulators."""
    return [_part(300, 0), _part(1, 1), _part(0, 2), _part(457, 3)]


@pytest.fixture(scope="module")
def batch(parts):
    return ad.concat(parts, index_unique="-")


class TestSelector:
    def test_pct_active_matches_batch(self, parts, batch):
        b = SAEFeatureSelector(n_chunks=7).compute_activation_stats(batch, PREFIX, N_FEATURES)
        s = SAEFeatureSelector(n_chunks=7).fit_slides(parts, PREFIX, N_FEATURES, progress=False)
        assert np.allclose(b.pct_active_, s.pct_active_)

    def test_max_score_matches_batch(self, parts, batch):
        b = SAEFeatureSelector().compute_activation_stats(batch, PREFIX, N_FEATURES)
        s = SAEFeatureSelector().fit_slides(parts, PREFIX, N_FEATURES, progress=False)
        assert np.allclose(b.max_score_, s.max_score_)

    def test_selected_indices_match_batch(self, parts, batch):
        b = SAEFeatureSelector().compute_activation_stats(batch, PREFIX, N_FEATURES)
        s = SAEFeatureSelector().fit_slides(parts, PREFIX, N_FEATURES, progress=False)
        assert np.array_equal(b.get_selected_indices(), s.get_selected_indices())

    def test_pct_active_is_a_true_fraction(self, batch):
        """Not a mean of per-chunk means, which is only right at equal chunk sizes."""
        sel = SAEFeatureSelector(n_chunks=7).compute_activation_stats(batch, PREFIX, N_FEATURES)
        X = batch[:, [f"{PREFIX}_{i}" for i in range(N_FEATURES)]].X
        expected = np.asarray((X > 0).sum(axis=0)).ravel() / batch.n_obs
        assert np.allclose(sel.pct_active_, expected)

    def test_chunking_does_not_change_the_answer(self, batch):
        a = SAEFeatureSelector(n_chunks=3).compute_activation_stats(batch, PREFIX, N_FEATURES)
        b = SAEFeatureSelector(n_chunks=97).compute_activation_stats(batch, PREFIX, N_FEATURES)
        assert np.allclose(a.pct_active_, b.pct_active_)
        assert np.allclose(a.max_score_, b.max_score_)

    def test_accumulating_nothing_is_an_error(self):
        with pytest.raises(ValueError, match="No patches"):
            SAEFeatureSelector().start(N_FEATURES).finalize()


class TestClusterer:
    @pytest.mark.parametrize("matrix", ["iou_soft_", "iou_strict_"])
    def test_iou_matches_batch(self, parts, batch, matrix):
        idx = np.arange(N_FEATURES)
        b = SAEFeatureClusterer().compute_iou(batch, PREFIX, idx)
        s = SAEFeatureClusterer().fit_slides(parts, PREFIX, idx, progress=False)
        assert np.allclose(getattr(b, matrix), getattr(s, matrix))

    def test_slide_order_does_not_matter(self, parts):
        """Sums are commutative; the streamed result must be too."""
        idx = np.arange(N_FEATURES)
        a = SAEFeatureClusterer().fit_slides(parts, PREFIX, idx, progress=False)
        b = SAEFeatureClusterer().fit_slides(parts[::-1], PREFIX, idx, progress=False)
        assert np.allclose(a.iou_soft_, b.iou_soft_)

    def test_clustering_runs_on_streamed_matrices(self, parts):
        c = SAEFeatureClusterer().fit_slides(parts, PREFIX, np.arange(N_FEATURES),
                                             progress=False)
        c.cluster(threshold=4, criterion="maxclust")
        assert len(c.get_cluster_assignments()) == N_FEATURES
        assert len(c.get_reordered_feature_indices()) == N_FEATURES

    def test_guards_before_fitting(self):
        with pytest.raises(RuntimeError, match="compute_iou"):
            SAEFeatureClusterer().get_cluster_assignments()


class TestKernelSplit:
    """The union division was lifted out of the numba inner loop to allow streaming."""

    def test_matches_a_dense_reference(self):
        X = sp.random(200, 25, density=0.08, format="csc", random_state=1)
        X.data[:] = 1.0
        dense = np.asarray(X.todense())
        inter = np.stack([np.minimum(dense[:, [i]], dense).sum(axis=0) for i in range(25)])
        assert np.allclose(_weighted_iou(X), iou_from_parts(inter, dense.sum(axis=0)))

    def test_intersection_is_additive_over_row_blocks(self):
        X = sp.random(300, 20, density=0.1, format="csr", random_state=2)
        whole, whole_sums = _intersection_and_sums(X)
        a, a_sums = _intersection_and_sums(X[:120])
        b, b_sums = _intersection_and_sums(X[120:])
        assert np.allclose(whole, a + b)
        assert np.allclose(whole_sums, a_sums + b_sums)

    def test_sparse_and_dense_paths_agree(self):
        X = sp.random(150, 15, density=0.15, format="csr", random_state=3)
        assert np.allclose(_weighted_iou(X), _weighted_iou(np.asarray(X.todense())))

    def test_iou_diagonal_is_one(self):
        X = sp.random(100, 10, density=0.2, format="csr", random_state=4)
        assert np.allclose(np.diag(_weighted_iou(X)), 1.0)
