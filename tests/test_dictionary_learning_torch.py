"""Tests for the torch backend of MiniBatchDictionaryCoding
(mesoslide.tools.sparse_coding._dictionary_learning_torch).

sklearn's sparse_encode / _update_dict / MiniBatchDictionaryLearning serve as
oracles. lasso_cd (tol=1e-8) is the reference code: positive lasso_lars can
return suboptimal codes on some rows, which FISTA must not be worse than.
Set MESOSLIDE_TEST_DEVICE=cuda to run the torch-side checks on GPU.
"""

import os

import numpy as np
import pytest
import torch
from sklearn.decomposition import MiniBatchDictionaryLearning, sparse_encode
from sklearn.decomposition._dict_learning import _update_dict

from mesoslide.tools.sparse_coding import MiniBatchDictionaryCoding
from mesoslide.tools.sparse_coding._dictionary_learning_torch import (
    _update_dict_torch,
    fista_lasso,
    fit_dictionary_torch,
)

DEVICE = os.environ.get("MESOSLIDE_TEST_DEVICE", "cpu")


def _objective(X, C, D, alpha):
    return 0.5 * ((X - C @ D) ** 2).sum(1) + alpha * np.abs(C).sum(1)


def _lasso_problem(n, d, k, seed):
    rng = np.random.default_rng(seed)
    D = rng.standard_normal((k, d))
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    X = 2 * rng.standard_normal((n, d))
    return X, D


def _fista(X, D, alpha, positive, dtype=torch.float64, **kwargs):
    code = fista_lasso(torch.tensor(X, dtype=dtype, device=DEVICE),
                       torch.tensor(D, dtype=dtype, device=DEVICE),
                       alpha, positive=positive, **kwargs)
    return code.cpu().double().numpy()


@pytest.mark.parametrize("k", [32, 128])  # undercomplete and 2x overcomplete, d=64
@pytest.mark.parametrize("positive", [True, False])
def test_fista_matches_sklearn_lasso(k, positive):
    alpha = 0.5
    X, D = _lasso_problem(200, 64, k, seed=k)
    ref = sparse_encode(X, D, algorithm="lasso_cd", alpha=alpha, positive=positive, max_iter=5000)
    lars = sparse_encode(X, D, algorithm="lasso_lars", alpha=alpha, positive=positive, max_iter=5000)
    C = _fista(X, D, alpha, positive, tol=1e-6)
    f, f_ref = _objective(X, C, D, alpha), _objective(X, ref, D, alpha)
    np.testing.assert_allclose(f, f_ref, rtol=1e-6)
    np.testing.assert_allclose(C, ref, atol=1e-3)
    assert np.all(f <= _objective(X, lars, D, alpha) * (1 + 1e-6))
    if positive:
        assert C.min() >= 0


def test_fista_float32_objective():
    alpha = 0.5
    X, D = _lasso_problem(200, 64, 128, seed=1)
    ref = sparse_encode(X, D, algorithm="lasso_cd", alpha=alpha, positive=True, max_iter=5000)
    C = _fista(X, D, alpha, True, dtype=torch.float32)
    np.testing.assert_allclose(_objective(X, C, D, alpha), _objective(X, ref, D, alpha), rtol=1e-4)


def test_fista_default_tol_objective_gap():
    """Default fista_tol=1e-4 reaches the sklearn objective to ~1e-7 relative."""
    alpha = 0.2
    X, D = _lasso_problem(200, 64, 256, seed=2)
    ref = sparse_encode(X, D, algorithm="lasso_cd", alpha=alpha, positive=True, max_iter=5000)
    C = _fista(X, D, alpha, True)
    np.testing.assert_allclose(_objective(X, C, D, alpha), _objective(X, ref, D, alpha), rtol=1e-6)


def test_update_dict_matches_sklearn():
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((50, 16))
    code = np.abs(rng.standard_normal((50, 8)))  # every atom used
    D0 = rng.standard_normal((8, 16))
    A, B = code.T @ code / 50, Y.T @ code / 50
    expected = D0.copy()
    _update_dict(expected, Y, code.copy(), A, B, random_state=0)

    D = torch.tensor(D0, device=DEVICE)
    n_unused = _update_dict_torch(D, torch.tensor(Y, device=DEVICE), torch.tensor(A, device=DEVICE),
                                  torch.tensor(B, device=DEVICE), positive=False,
                                  generator=torch.Generator().manual_seed(0))
    assert n_unused == 0
    np.testing.assert_allclose(D.cpu().numpy(), expected, atol=1e-10)


def test_update_dict_resamples_unused_atom():
    rng = np.random.default_rng(0)
    Y = torch.tensor(rng.standard_normal((50, 16)), device=DEVICE)
    code = np.abs(rng.standard_normal((50, 8)))
    code[:, 3] = 0
    A = torch.tensor(code.T @ code / 50, device=DEVICE)
    B = Y.T @ torch.tensor(code, device=DEVICE) / 50
    D = torch.zeros(8, 16, dtype=torch.float64, device=DEVICE)
    n_unused = _update_dict_torch(D, Y, A, B, positive=True, generator=torch.Generator().manual_seed(0))
    assert n_unused == 1
    norms = torch.linalg.vector_norm(D, dim=1)
    assert 0 < norms[3] <= 1 + 1e-12
    assert D.min() >= 0


def test_fit_dictionary_matches_sklearn_steps():
    """Same init, no shuffle, 5 minibatch steps: the online updates agree."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5 * 20, 12))
    dict_init = rng.standard_normal((6, 12))
    dict_init /= np.linalg.norm(dict_init, axis=1, keepdims=True)
    params = dict(n_components=6, alpha=0.05, batch_size=20, max_iter=1,
                  positive_code=False, positive_dict=False)
    sk = MiniBatchDictionaryLearning(fit_algorithm="cd", dict_init=dict_init, shuffle=False,
                                     random_state=0, **params).fit(X)
    D, log = fit_dictionary_torch(X, dict_init=dict_init, shuffle=False, device=DEVICE,
                                  fista_tol=1e-8, fista_max_iter=20000, **params)
    assert log["n_steps"] == sk.n_steps_ == 5
    np.testing.assert_allclose(D, sk.components_, atol=1e-5)


def _mean_objective(model, X):
    alpha = model.alpha
    C = model.transform(X, device="cpu").toarray().astype(np.float64)
    return _objective(X - model.mean_, C, model.components_.astype(np.float64), alpha).mean()


def _structured_data(n, seed):
    """Sparse nonnegative combinations of 12 signed atoms in 20 dims, plus noise."""
    rng = np.random.default_rng(seed)
    atoms = rng.standard_normal((12, 20))
    codes = rng.random((n, 12)) * (rng.random((n, 12)) < 0.2)
    return (codes @ atoms + 0.05 * rng.standard_normal((n, 20)) + 1.0).astype(np.float64)


def test_torch_and_sklearn_backends_reach_comparable_objective():
    X_train, X_test = _structured_data(2000, 0), _structured_data(500, 1)
    params = dict(n_components=12, alpha=0.1, batch_size=100, max_iter=20, n_jobs=1,
                  transform_algorithm="lasso_cd", random_state=0)
    sk = MiniBatchDictionaryCoding(backend="sklearn", **params).fit(X_train, device="cpu")
    th = MiniBatchDictionaryCoding(backend="torch", **params).fit(X_train, device=DEVICE)
    assert sk._training_log["backend"] == "sklearn" and th._training_log["backend"] == "torch"
    f_sk, f_th = _mean_objective(sk, X_test), _mean_objective(th, X_test)
    assert f_th <= f_sk * 1.02, (f_th, f_sk)
    assert abs(th._training_log["mean_l0"] - sk._training_log["mean_l0"]) <= 0.25 * sk._training_log["mean_l0"]


def test_cross_backend_transform_same_objective():
    X = _structured_data(300, 2)
    model = MiniBatchDictionaryCoding(n_components=12, alpha=0.1, batch_size=100, max_iter=5,
                                      n_jobs=1, transform_algorithm="lasso_cd", fista_tol=1e-6,
                                      backend="sklearn", random_state=0).fit(X, device="cpu")
    C_sk = model.transform(X, device="cpu").toarray()
    model.set_params(backend="torch")
    C_th = model.transform(X, device=DEVICE).toarray()
    D = model.components_
    np.testing.assert_allclose(_objective(X - model.mean_, C_th, D, 0.1),
                               _objective(X - model.mean_, C_sk, D, 0.1), rtol=1e-5)
    # Block and column slicing on the torch path.
    model.set_params(transform_batch_size=37)
    np.testing.assert_allclose(model.transform(X, [4, 1], device=DEVICE).toarray(), C_th[:, [4, 1]],
                               atol=1e-6)


def test_backend_resolution():
    model = MiniBatchDictionaryCoding()
    assert model._resolve_backend("cpu") == ("cpu", "sklearn")
    assert model._resolve_backend("cuda:0") == ("cuda:0", "torch")
    assert MiniBatchDictionaryCoding(device="cpu")._resolve_backend(None) == ("cpu", "sklearn")
    assert MiniBatchDictionaryCoding(backend="torch")._resolve_backend("cpu") == ("cpu", "torch")
    with pytest.raises(ValueError, match="backend"):
        MiniBatchDictionaryCoding(backend="jax")._resolve_backend("cpu")
    with pytest.raises(ValueError, match="lasso"):
        MiniBatchDictionaryCoding(backend="torch", transform_algorithm="omp")._resolve_backend("cpu")


# ── working-set coder ────────────────────────────────────────────────────────

def _ws(X, D, alpha, positive, **kwargs):
    from mesoslide.tools.sparse_coding._dictionary_learning_torch import lasso_working_set
    code = lasso_working_set(torch.tensor(X, device=DEVICE), torch.tensor(D, device=DEVICE),
                             alpha, positive=positive, **kwargs)
    return code.cpu().double().numpy()


@pytest.mark.parametrize("positive", [True, False])
@pytest.mark.parametrize("m", [2, 8, 32])   # 2 forces growth, 8 forces swaps, 32 mostly fits
def test_working_set_reaches_optimum(positive, m):
    """Objective vs sklearn lasso_cd; codes vs tightly converged full FISTA (lasso_cd may stop early)."""
    alpha = 0.5
    X, D = _lasso_problem(60, 32, 96, seed=3)            # 3x overcomplete
    ref = sparse_encode(X, D, algorithm="lasso_cd", alpha=alpha, positive=positive, max_iter=5000)
    full = _fista(X, D, alpha, positive, tol=1e-8, max_iter=50000)
    C = _ws(X, D, alpha, positive, working_set_size=m, tol=1e-8, max_iter=50000, row_chunk=32)
    assert np.all(_objective(X, C, D, alpha) <= _objective(X, ref, D, alpha) * (1 + 1e-6))
    np.testing.assert_allclose(_objective(X, C, D, alpha), _objective(X, full, D, alpha), rtol=1e-7)
    np.testing.assert_allclose(C, full, atol=1e-4)
    if positive:
        assert C.min() >= 0


def test_working_set_matches_full_fista_default_tol():
    alpha = 0.5
    X, D = _lasso_problem(300, 64, 256, seed=4)
    full = _fista(X, D, alpha, True)
    ws = _ws(X, D, alpha, True, working_set_size=32)
    np.testing.assert_allclose(_objective(X, ws, D, alpha), _objective(X, full, D, alpha), rtol=1e-6)


def test_solve_lasso_dispatch():
    from mesoslide.tools.sparse_coding._dictionary_learning_torch import solve_lasso
    X, D = _lasso_problem(50, 16, 64, seed=5)
    Xt, Dt = torch.tensor(X, device=DEVICE), torch.tensor(D, device=DEVICE)
    a = solve_lasso(Xt, Dt, 0.3, positive=True, method="auto", working_set_size=16, tol=1e-6, max_iter=5000)
    b = solve_lasso(Xt, Dt, 0.3, positive=True, method="full", tol=1e-6, max_iter=5000)
    np.testing.assert_allclose(_objective(X, a.cpu().numpy(), D, 0.3), _objective(X, b.cpu().numpy(), D, 0.3),
                               rtol=1e-6)
    with pytest.raises(ValueError, match="method"):
        solve_lasso(Xt, Dt, 0.3, positive=True, method="lars")


def test_fit_working_set_matches_full():
    """Same init and batches: training with either coder gives the same dictionary."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4 * 50, 24))
    dict_init = rng.standard_normal((48, 24))
    dict_init /= np.linalg.norm(dict_init, axis=1, keepdims=True)
    params = dict(n_components=48, alpha=0.1, batch_size=50, max_iter=1, tol=0.0, max_no_improvement=None,
                  positive_code=True, dict_init=dict_init, shuffle=False, device=DEVICE,
                  fista_tol=1e-8, fista_max_iter=20000)
    D_full, _ = fit_dictionary_torch(X, lasso_method="full", **params)
    D_ws, _ = fit_dictionary_torch(X, lasso_method="working_set", working_set_size=8, **params)
    np.testing.assert_allclose(D_ws, D_full, atol=1e-5)
