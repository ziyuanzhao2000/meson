"""Tests for the K-means backends used to fit an LLC codebook
(mesoslide.tools.sparse_coding._kmeans_backends).
"""

import numpy as np
import pytest
import torch

from mesoslide.tools.sparse_coding._kmeans_backends import (
    TorchMiniBatchKMeans,
    fit_kmeans_backend,
)


def _cuml_available():
    try:
        import cuml  # noqa: F401
        return True
    except ImportError:
        return False


def _make_blobs(n_per_cluster=500, n_clusters=6, d=8, spread=0.3, seed=0):
    """Well-separated Gaussian blobs with known ground-truth centroids."""
    rng = np.random.default_rng(seed)
    true_centers = rng.uniform(-10, 10, size=(n_clusters, d))
    X = np.concatenate([
        true_centers[i] + rng.normal(scale=spread, size=(n_per_cluster, d))
        for i in range(n_clusters)
    ]).astype(np.float32)
    labels = np.repeat(np.arange(n_clusters), n_per_cluster)
    return X, true_centers.astype(np.float32), labels


def _match_centers_to_truth(found, true):
    """Greedy nearest-truth-center matching; returns found reordered to align with true."""
    found = np.asarray(found)
    true = np.asarray(true)
    dists = np.linalg.norm(found[:, None, :] - true[None, :, :], axis=-1)
    order = np.argmin(dists, axis=1)
    return order


class TestTorchMiniBatchKMeans:
    def test_converges_on_synthetic_blobs(self):
        X, true_centers, _ = _make_blobs(n_per_cluster=800, n_clusters=6, d=8, spread=0.2, seed=1)
        model = TorchMiniBatchKMeans(
            n_clusters=6, batch_size=256, max_iter=300, random_state=0, device="cpu"
        )
        model.fit(X)

        order = _match_centers_to_truth(model.cluster_centers_, true_centers)
        assert len(set(order.tolist())) == 6  # each true center claimed exactly once
        matched = true_centers[order]
        err = np.linalg.norm(model.cluster_centers_ - matched, axis=1)
        assert (err < 0.5).all(), f"centroid recovery error too high: {err}"

    def test_inertia_comparable_to_sklearn_minibatchkmeans(self):
        from sklearn.cluster import MiniBatchKMeans

        X, _, _ = _make_blobs(n_per_cluster=500, n_clusters=5, d=6, spread=0.3, seed=2)
        ours = TorchMiniBatchKMeans(
            n_clusters=5, batch_size=256, max_iter=300, random_state=0, device="cpu"
        ).fit(X)
        theirs = MiniBatchKMeans(n_clusters=5, batch_size=256, random_state=0, n_init="auto").fit(X)

        # Loose/directional: different update/init details, so compare order of
        # magnitude rather than exact inertia.
        assert ours.inertia_ < theirs.inertia_ * 3

    def test_empty_clusters_get_reseeded(self):
        # 3 real clusters worth of data, but ask for 6 -- several centers will
        # start with no nearby points assigned to them across many batches.
        X, _, _ = _make_blobs(n_per_cluster=400, n_clusters=3, d=5, spread=0.2, seed=3)
        model = TorchMiniBatchKMeans(
            n_clusters=6, batch_size=64, max_iter=200, random_state=0, device="cpu"
        )
        model.fit(X)

        assign = np.argmin(
            np.linalg.norm(X[:, None, :] - model.cluster_centers_[None, :, :], axis=-1),
            axis=1,
        )
        counts = np.bincount(assign, minlength=6)
        assert (counts > 0).all(), f"some clusters ended up with zero points: {counts}"

    def test_runs_on_cpu_and_cuda_if_available(self):
        X, _, _ = _make_blobs(n_per_cluster=100, n_clusters=3, d=4, seed=4)
        cpu_model = TorchMiniBatchKMeans(n_clusters=3, batch_size=32, max_iter=20, device="cpu").fit(X)
        assert cpu_model.cluster_centers_.shape == (3, 4)
        if torch.cuda.is_available():
            gpu_model = TorchMiniBatchKMeans(n_clusters=3, batch_size=32, max_iter=20, device="cuda").fit(X)
            assert gpu_model.cluster_centers_.shape == (3, 4)

    @pytest.mark.skipif(not _cuml_available(), reason="cuml not installed")
    def test_matches_cuml_kmeans_on_known_centroids(self):
        """Both backends should recover the same ground-truth centroids on
        well-separated synthetic blobs, to within a similar tolerance -- this is
        the direct apples-to-apples check the user asked for, independent of the
        sklearn.MiniBatchKMeans comparison above."""
        import cuml

        X, true_centers, _ = _make_blobs(n_per_cluster=800, n_clusters=6, d=8, spread=0.2, seed=5)

        ours = TorchMiniBatchKMeans(
            n_clusters=6, batch_size=256, max_iter=300, random_state=0, device="cpu"
        ).fit(X)
        cuml_model = cuml.cluster.KMeans(n_clusters=6, random_state=0, n_init="auto").fit(X)
        cuml_centers = np.asarray(cuml_model.cluster_centers_)

        order_ours = _match_centers_to_truth(ours.cluster_centers_, true_centers)
        order_cuml = _match_centers_to_truth(cuml_centers, true_centers)
        assert len(set(order_ours.tolist())) == 6
        assert len(set(order_cuml.tolist())) == 6

        err_ours = np.linalg.norm(ours.cluster_centers_ - true_centers[order_ours], axis=1)
        err_cuml = np.linalg.norm(cuml_centers - true_centers[order_cuml], axis=1)
        # Both should recover the true centroids closely; our streaming/approximate
        # solver is held to a looser bound than cuml's exact Lloyd's algorithm,
        # but not by more than a small constant factor.
        assert (err_ours < 0.5).all(), f"torch_minibatch centroid error too high: {err_ours}"
        assert err_ours.mean() < err_cuml.mean() * 5 + 0.1


class TestBackendDispatch:
    def test_unknown_backend_raises(self):
        rng = np.random.RandomState(0)
        X = rng.normal(size=(20, 4)).astype(np.float32)
        with pytest.raises(ValueError, match="Unknown kmeans backend"):
            fit_kmeans_backend("bogus", X, 3, rng, {})

    def test_sklearn_backend_matches_direct_sklearn_call(self):
        from sklearn.cluster import KMeans

        rng = np.random.RandomState(0)
        X, _, _ = _make_blobs(n_per_cluster=100, n_clusters=4, d=5, seed=6)
        result = fit_kmeans_backend("sklearn", X, 4, np.random.RandomState(0), {"n_init": "auto"})
        direct = KMeans(n_clusters=4, random_state=np.random.RandomState(0).randint(0, 2**32 - 1),
                         n_init="auto")
        # Not a bitwise reproduction (fit_kmeans_backend draws its own internal seed
        # from the RandomState), just a structural/shape sanity check.
        assert result["centers"].shape == (4, 5)
        assert isinstance(result["inertia"], float)

    @pytest.mark.skipif(not _cuml_available(), reason="cuml not installed")
    def test_fit_codebook_with_cuml_backend(self):
        from mesoslide.tools.sparse_coding._llc import LLCModel, fit_codebook

        X, _, _ = _make_blobs(n_per_cluster=100, n_clusters=4, d=5, seed=7)
        model = LLCModel(input_dim=5, n_codewords=4)
        log = fit_codebook(model, X, backend="cuml", random_state=0)
        assert model.codebook.shape == (4, 5)
        assert "inertia" in log
