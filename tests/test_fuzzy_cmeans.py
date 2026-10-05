"""FuzzyCMeans: reference agreement, CPU/GPU agreement, transform contract, edge cases."""

import numpy as np
import pytest
import torch

from mesoslide.tools.sparse_coding import FuzzyCMeans


# --- reference: soft-clustering FuzzyCMeans (soft_clustering/_fcm.py, _base.ratio_memberships; MIT),
# vendored verbatim in its numerics; seeding via initial centers so both start identically.
def _ref_dist2(X, centers):
    x_norm = np.sum(X * X, axis=1, keepdims=True)
    c_norm = np.sum(centers * centers, axis=1, keepdims=True).T
    d = x_norm + c_norm - 2.0 * (X @ centers.T)
    np.maximum(d, 0.0, out=d)
    return d


def _ref_normalize(U, eps=1e-12):
    U = np.maximum(U, eps)
    return U / (U.sum(axis=1, keepdims=True) + eps)


def _ref_ratio(distances, exponent):
    log_d = np.log(np.maximum(distances, np.finfo(np.float64).tiny))
    scores = -exponent * (log_d - log_d.min(axis=1, keepdims=True))
    np.exp(scores, out=scores)
    total = scores.sum(axis=1, keepdims=True)
    return scores / np.where(total > 0, total, 1.0)


def _ref_fit(X, centers, m, max_iter, tol):
    obj_prev = np.inf
    for it in range(max_iter):
        U = _ref_normalize(_ref_ratio(_ref_dist2(X, centers) + 1e-12, 1.0 / (m - 1.0)))
        Um = U ** m
        centers = (Um.T @ X) / (Um.sum(axis=0, keepdims=True).T + 1e-12)
        obj = float(np.sum(Um * _ref_dist2(X, centers)))
        if abs(obj_prev - obj) <= tol:
            break
        obj_prev = obj
    return _ref_normalize(U), centers, it + 1


def _blobs(n=600, d=16, k=4, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.normal(scale=6.0, size=(k, d))
    labels = rng.integers(0, k, n)
    return means[labels] + rng.normal(size=(n, d)), labels


@pytest.mark.parametrize("m", [1.25, 2.0])
def test_matches_reference_from_same_initial_centers(m):
    X, _ = _blobs()
    init = X[[0, 1, 2, 3]].copy()
    # tol=-1 disables early stopping in both, so the comparison does not hinge on an exact float tie
    U_ref, C_ref, n_ref = _ref_fit(X, init, m, max_iter=25, tol=-1.0)
    fcm = FuzzyCMeans(n_components=4, m=m, max_iter=25, tol=-1.0, init=init, backend="numpy", batch_size=97).fit(X)
    np.testing.assert_allclose(fcm.cluster_centers_, C_ref, atol=1e-8)
    assert fcm.n_iter_ == n_ref == 25
    # transform uses the final centers, i.e. one more membership update than U_ref
    U_next = _ref_normalize(_ref_ratio(_ref_dist2(X, C_ref) + 1e-12, 1.0 / (m - 1.0)))
    np.testing.assert_allclose(fcm.transform(X), U_next, atol=1e-6)


def test_memberships_recover_blobs_and_sum_to_one():
    X, labels = _blobs()
    fcm = FuzzyCMeans(n_components=4, m=1.5, random_state=0, backend="numpy").fit(X)
    U = fcm.transform(X)
    assert U.shape == (len(X), 4) and U.dtype == np.float32
    np.testing.assert_allclose(U.sum(1), 1.0, atol=1e-5)
    pred = fcm.predict(X)
    # each true blob maps to one cluster
    assert all(len(np.unique(pred[labels == k])) == 1 for k in range(4))
    assert fcm._training_log["mean_max_membership"] > 0.9


def test_relative_tolerance_stops_early_and_column_selection():
    X, _ = _blobs()
    fcm = FuzzyCMeans(n_components=4, m=1.5, tol=1e-6, max_iter=300, random_state=0, backend="numpy").fit(X)
    assert fcm.n_iter_ < 300
    np.testing.assert_array_equal(fcm.transform(X, column_keep_indices=[2, 0]), fcm.transform(X)[:, [2, 0]])


def test_seeded_fit_is_reproducible_and_scale_invariant():
    X, _ = _blobs()
    a = FuzzyCMeans(n_components=4, m=1.5, random_state=3, backend="numpy").fit(X).transform(X)
    b = FuzzyCMeans(n_components=4, m=1.5, random_state=3, backend="numpy").fit(X).transform(X)
    c = FuzzyCMeans(n_components=4, m=1.5, random_state=3, backend="numpy", scale=True).fit(X).transform(X)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a, c, atol=1e-5)  # memberships do not depend on a uniform rescaling


def test_invalid_parameters_and_point_on_center():
    X, _ = _blobs(n=50)
    with pytest.raises(ValueError, match="m must be > 1"):
        FuzzyCMeans(n_components=2, m=1.0, backend="numpy").fit(X)
    with pytest.raises(ValueError, match="n_components"):
        FuzzyCMeans(n_components=60, backend="numpy").fit(X)
    fcm = FuzzyCMeans(n_components=3, m=1.5, random_state=0, backend="numpy").fit(X)
    U = fcm.transform(fcm.cluster_centers_.astype(np.float64))
    np.testing.assert_allclose(U.max(1), 1.0, atol=1e-6)  # a sample on a center is one-hot


def test_torch_cpu_matches_numpy():
    X, _ = _blobs()
    kw = dict(n_components=4, m=1.5, max_iter=30, tol=-1.0, random_state=0)
    np_fit = FuzzyCMeans(backend="numpy", **kw).fit(X)
    t_fit = FuzzyCMeans(backend="torch", device="cpu", dtype="float64", **kw).fit(X)
    np.testing.assert_allclose(t_fit.cluster_centers_, np_fit.cluster_centers_, atol=1e-8)
    np.testing.assert_allclose(t_fit.transform(X, device="cpu"), np_fit.transform(X), atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_matches_cpu():
    X, _ = _blobs(n=5000, d=64, k=6)
    kw = dict(n_components=6, m=1.25, max_iter=50, tol=-1.0, random_state=1)
    cpu = FuzzyCMeans(backend="numpy", **kw).fit(X)
    g64 = FuzzyCMeans(backend="torch", device="cuda", dtype="float64", **kw).fit(X)
    g32 = FuzzyCMeans(backend="torch", device="cuda", **kw).fit(X)
    np.testing.assert_allclose(g64.cluster_centers_, cpu.cluster_centers_, atol=1e-8)
    U = cpu.transform(X)
    np.testing.assert_allclose(g32.transform(X, device="cuda"), U, atol=1e-4)
    np.testing.assert_array_equal(g32.predict(X, device="cuda"), cpu.predict(X))


def test_feature_extraction_contract():
    """transform output can be written as table features like a sparse-coding model's."""
    import scipy.sparse as sp
    from mesoslide.tools._feature_extraction import _validate_sparse_shape

    X, _ = _blobs()
    fcm = FuzzyCMeans(n_components=4, m=1.5, random_state=0, backend="numpy").fit(X)
    M = sp.csr_matrix(fcm.transform(X))
    _validate_sparse_shape(M, len(X))
    assert M.shape == (len(X), 4)
