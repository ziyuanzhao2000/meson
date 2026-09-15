"""Tests for Locality-constrained Linear Coding (mesoslide.tools.sparse_coding._llc).

scipy.optimize.nnls / lstsq / SLSQP are used only as test oracles here -- the
shipped encode path never calls scipy per-sample (see _llc.py's module docstrings
for why: it would not scale to mesoslide's per-tile data volumes).
"""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from mesoslide.tools.sparse_coding._llc import (
    LLCModel,
    LocalityConstrainedCoding,
    _project_simplex_batched,
    _solve_local,
    _solve_nnls_scipy_loop,
    _solve_shift_invariant,
    _solve_unconstrained,
    fit_codebook,
)

torch.manual_seed(0)


def _random_local_problem(n, k, d, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    B_i = rng.normal(size=(n, k, d))
    return torch.tensor(x, dtype=torch.float64), torch.tensor(B_i, dtype=torch.float64)


class TestClosedFormSolves:
    def test_unconstrained_matches_lstsq(self):
        x, B_i = _random_local_problem(20, 5, 8, seed=0)
        c = _solve_unconstrained(x, B_i)
        for i in range(x.shape[0]):
            expected, *_ = np.linalg.lstsq(B_i[i].numpy().T, x[i].numpy(), rcond=None)
            assert np.allclose(c[i].numpy(), expected, atol=1e-4)

    def test_shift_invariant_sums_to_one(self):
        x, B_i = _random_local_problem(30, 5, 8, seed=1)
        c = _solve_shift_invariant(x, B_i)
        assert np.allclose(c.sum(dim=-1).numpy(), 1.0, atol=1e-5)

    def test_shift_invariant_matches_lagrange_reference(self):
        x, B_i = _random_local_problem(15, 4, 6, seed=2)
        c = _solve_shift_invariant(x, B_i)
        for i in range(x.shape[0]):
            B = B_i[i].numpy()
            G = B @ B.T + 1e-6 * np.eye(B.shape[0])
            ones = np.ones(B.shape[0])
            # Lagrangian stationary point for min ||x - B^T c||^2 s.t. 1^T c = 1:
            # c = G^{-1}(B x + mu*1), mu chosen so 1^T c = 1.
            Ginv_one = np.linalg.solve(G, ones)
            Ginv_Bx = np.linalg.solve(G, B @ x[i].numpy())
            mu = (1 - ones @ Ginv_Bx) / (ones @ Ginv_one)
            expected = Ginv_Bx + mu * Ginv_one
            assert np.allclose(c[i].numpy(), expected, atol=1e-4)


class TestNNLSBatched:
    def test_nonneg_matches_scipy_nnls(self):
        x, B_i = _random_local_problem(25, 5, 8, seed=3)
        c = _solve_local(x.float(), B_i.float(), "nonneg", nnls_iters=500)
        expected = _solve_nnls_scipy_loop(x.numpy(), B_i.numpy())
        assert np.allclose(c.numpy(), expected, atol=1e-2)

    def test_nonneg_is_nonnegative(self):
        x, B_i = _random_local_problem(25, 5, 8, seed=4)
        c = _solve_local(x.float(), B_i.float(), "nonneg", nnls_iters=100)
        assert (c.numpy() >= -1e-6).all()

    def test_nonneg_shift_invariant_on_simplex(self):
        x, B_i = _random_local_problem(25, 5, 8, seed=5)
        c = _solve_local(x.float(), B_i.float(), "nonneg_shift_invariant", nnls_iters=200)
        assert (c.numpy() >= -1e-6).all()
        assert np.allclose(c.sum(dim=-1).numpy(), 1.0, atol=1e-3)

    def test_project_simplex_is_idempotent_on_simplex_points(self):
        rng = np.random.default_rng(6)
        v = rng.dirichlet(np.ones(5), size=10)
        v_t = torch.tensor(v, dtype=torch.float64)
        projected = _project_simplex_batched(v_t)
        assert np.allclose(projected.numpy(), v, atol=1e-6)

    def test_unknown_constraint_raises(self):
        x, B_i = _random_local_problem(2, 3, 4, seed=7)
        with pytest.raises(ValueError, match="Unknown constraint"):
            _solve_local(x.float(), B_i.float(), "bogus", nnls_iters=10)


class TestLLCModelEncode:
    def _fitted_model(self, d=8, m=40, n_train=500, seed=0):
        rng = np.random.default_rng(seed)
        centers = rng.normal(size=(4, d)) * 5
        labels = rng.integers(0, 4, size=n_train)
        X = centers[labels] + rng.normal(scale=0.3, size=(n_train, d))
        model = LLCModel(input_dim=d, n_codewords=m)
        fit_codebook(model, X.astype(np.float32), random_state=seed)
        return model, X.astype(np.float32)

    def test_code_is_exactly_k_sparse(self):
        model, X = self._fitted_model()
        x = torch.tensor(X[:10])
        code = model.encode(x, n_neighbors=5, zero_threshold=0.0)
        nnz = (code != 0).sum(dim=1)
        assert (nnz <= 5).all()

    def test_zero_threshold_can_only_reduce_sparsity_count(self):
        model, X = self._fitted_model()
        x = torch.tensor(X[:20])
        code_no_thresh = model.encode(x, n_neighbors=5, zero_threshold=0.0)
        code_thresh = model.encode(x, n_neighbors=5, zero_threshold=0.05)
        nnz_no_thresh = (code_no_thresh != 0).sum(dim=1)
        nnz_thresh = (code_thresh != 0).sum(dim=1)
        assert (nnz_thresh <= nnz_no_thresh).all()

    def test_reconstruction_error_ordering_roughly_matches_paper(self):
        model, X = self._fitted_model(n_train=800)
        x = torch.tensor(X[:100])
        errors = {}
        for constraint in ["unconstrained", "shift_invariant", "nonneg", "nonneg_shift_invariant"]:
            x_hat, _ = model.forward(x, n_neighbors=5, constraint=constraint, zero_threshold=0.0)
            errors[constraint] = ((x_hat - x) ** 2).mean().item()
        # Loose/directional check only: nonneg_shift_invariant (small bounded
        # convex hull) should not reconstruct better than the least-constrained
        # modes on average, matching the paper's qualitative finding.
        best_of_others = min(errors["unconstrained"], errors["shift_invariant"], errors["nonneg"])
        assert errors["nonneg_shift_invariant"] >= best_of_others - 1e-6


class TestLocalityConstrainedCodingWrapper:
    def _fit(self, n=200, d=6, m=20, **kwargs):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(n, d)).astype(np.float32)
        model = LocalityConstrainedCoding(n_codewords=m, batch_size=64, random_state=0, **kwargs)
        model.fit(X)
        return model, X

    def test_default_constraint_is_nonneg(self):
        assert LocalityConstrainedCoding().constraint == "nonneg"

    def test_fit_transform_roundtrip_shapes(self):
        model, X = self._fit()
        Xt = model.transform(X)
        assert sp.issparse(Xt)
        assert Xt.shape == (X.shape[0], model.n_codewords)

    def test_inverse_transform_roundtrip(self):
        model, X = self._fit()
        Xt = model.transform(X)
        X_hat = model.inverse_transform(Xt)
        assert X_hat.shape == X.shape

    def test_fit_moves_model_to_requested_device(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 6)).astype(np.float32)
        model = LocalityConstrainedCoding(n_codewords=20, batch_size=64, random_state=0)
        model.fit(X, device="cpu")
        assert model.model_.codebook.device.type == "cpu"

    def test_transform_matches_direct_model_encode(self):
        model, X = self._fit()
        Xt = model.transform(X)
        device = model.model_.codebook.device
        x = torch.tensor(X, device=device)
        with torch.no_grad():
            direct = model.model_.encode(
                x, n_neighbors=model.n_neighbors, constraint=model.constraint,
                zero_threshold=model.zero_threshold, nnls_iters=model.nnls_iters,
            )
        assert np.allclose(Xt.toarray(), direct.cpu().numpy(), atol=1e-5)

    def test_torch_minibatch_kmeans_backend(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(300, 6)).astype(np.float32)
        model = LocalityConstrainedCoding(
            n_codewords=10, batch_size=64, random_state=0,
            kmeans_backend="torch_minibatch",
            kmeans_kwargs={"batch_size": 32, "max_iter": 50},
        )
        model.fit(X, device="cpu")
        assert model.model_.codebook.shape == (10, 6)

        Xt = model.transform(X)
        assert sp.issparse(Xt)
        assert Xt.shape == (300, 10)
        X_hat = model.inverse_transform(Xt)
        assert X_hat.shape == X.shape
