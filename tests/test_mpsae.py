"""MatchingPursuitSAE: fit/transform contract and equivalence to the original MP-SAE encoder."""

import joblib
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from mesoslide.tools.sparse_coding import MatchingPursuitDictionary, MatchingPursuitSAE


def _legacy_get_acts(W, x, threshold):
    """Encoder of the original mpsae.MatchingPursuitSAE.get_acts, verbatim."""
    residual = x.clone()
    batch_size = x.shape[0]
    n_latents = W.shape[0]
    z = torch.zeros(batch_size, n_latents)
    prev_support = torch.zeros_like(z).bool()
    done = torch.zeros(batch_size, dtype=torch.bool)
    while not done.all():
        WTr = residual @ W.T
        values, indices = torch.max(torch.relu(WTr), dim=1, keepdim=True)
        z_ = torch.zeros_like(z)
        z_.scatter_(1, indices, values)
        z = torch.where(done.unsqueeze(1), z, z + z_)
        update = torch.matmul(z_, W)
        residual = torch.where(done.unsqueeze(1), residual, residual - update)
        support = z != 0
        converged = (support == prev_support).all(dim=1) | (residual.norm(dim=1) < threshold)
        done = done | converged
        prev_support = support
    return z


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    return rng.normal(size=(300, 16)).astype(np.float32)


@pytest.fixture(scope="module")
def fitted(data):
    return MatchingPursuitSAE(expansion_factor=4, num_steps=20, batch_size=16,
                              transform_batch_size=64, random_state=0).fit(data, device="cpu")


def test_transform_shape_and_sparsity(fitted, data):
    Z = fitted.transform(data, device="cpu", progress_bar=False)
    assert sp.issparse(Z) and Z.shape == (300, 64)
    assert Z.nnz > 0 and (Z.data > 0).all()
    assert Z.nnz < Z.shape[0] * Z.shape[1]


def test_transform_matches_legacy_encoder(fitted, data):
    Z = fitted.transform(data, device="cpu", progress_bar=False).toarray()
    W = fitted.model_.W.detach()
    ref = _legacy_get_acts(W, torch.as_tensor(data), fitted.threshold).numpy()
    np.testing.assert_allclose(Z, ref, rtol=1e-5, atol=1e-6)


def test_batching_does_not_change_the_answer(fitted, data):
    a = fitted.transform(data, device="cpu", progress_bar=False).toarray()
    fitted.transform_batch_size = 7
    try:
        b = fitted.transform(data, device="cpu", progress_bar=False).toarray()
    finally:
        fitted.transform_batch_size = 64
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_column_keep_indices(fitted, data):
    keep = [3, 10, 40]
    full = fitted.transform(data, device="cpu", progress_bar=False).toarray()
    sub = fitted.transform(data, keep, device="cpu", progress_bar=False).toarray()
    np.testing.assert_allclose(sub, full[:, keep])


def test_training_reduces_reconstruction_error(data):
    """Same seed gives the same initial dictionary; 0 steps is the untrained baseline."""
    def recon_err(model):
        Z = model.transform(data, device="cpu", progress_bar=False).toarray()
        return float(((Z @ model.components_ - data) ** 2).sum(axis=1).mean())

    kw = dict(expansion_factor=4, batch_size=32, learning_rate=1e-2, random_state=0)
    untrained = MatchingPursuitSAE(num_steps=0, **kw).fit(data, device="cpu")
    trained = MatchingPursuitSAE(num_steps=300, **kw).fit(data, device="cpu")
    assert len(trained._training_log["mse"]) == 300
    assert recon_err(trained) < recon_err(untrained)


def test_replay_gradient_matches_full_pursuit(data):
    """Training step through the recorded replay equals backprop through the whole pursuit."""
    torch.manual_seed(0)
    model = MatchingPursuitDictionary(16, 64, threshold=1e-2).double()
    x = torch.as_tensor(data[:50], dtype=torch.float64)

    z_full = model._pursuit(x)                      # graph through every iteration
    loss_full = ((z_full @ model.W - x) ** 2).sum(-1).mean()
    grad_full, = torch.autograd.grad(loss_full, model.W)

    x_hat, z = model(x)                             # no-grad pursuit + replay
    loss = ((x_hat - x) ** 2).sum(-1).mean()
    grad, = torch.autograd.grad(loss, model.W)

    assert torch.equal(z != 0, z_full.detach() != 0)
    torch.testing.assert_close(loss, loss_full, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(grad, grad_full, rtol=1e-8, atol=1e-10)


def test_max_iter_bounds_l0(fitted, data):
    capped = MatchingPursuitSAE(expansion_factor=4, max_iter=3)
    capped.model_ = MatchingPursuitDictionary(16, 64, threshold=fitted.threshold, max_iter=3)
    capped.model_.load_state_dict(fitted.model_.state_dict())
    capped.embed_dim_, capped.n_features_in_ = 64, 16
    Z = capped.transform(data, device="cpu", progress_bar=False)
    assert np.diff(Z.indptr).max() <= 3
    # same first atoms as the uncapped pursuit
    ref = _legacy_get_acts(fitted.model_.W.detach(), torch.as_tensor(data), fitted.threshold).numpy()
    assert ((Z.toarray() != 0) <= (ref != 0)).all()


def test_large_max_iter_equals_uncapped(fitted, data):
    model = MatchingPursuitDictionary(16, 64, threshold=fitted.threshold, max_iter=10_000)
    model.load_state_dict(fitted.model_.state_dict())
    x = torch.as_tensor(data)
    assert torch.equal(model.encode(x), fitted.model_.encode(x))


def test_replay_gradient_matches_with_max_iter(data):
    torch.manual_seed(0)
    model = MatchingPursuitDictionary(16, 64, threshold=1e-2, max_iter=4).double()
    x = torch.as_tensor(data[:50], dtype=torch.float64)
    z_full = model._pursuit(x)
    grad_full, = torch.autograd.grad(((z_full @ model.W - x) ** 2).sum(-1).mean(), model.W)
    x_hat, _ = model(x)
    grad, = torch.autograd.grad(((x_hat - x) ** 2).sum(-1).mean(), model.W)
    torch.testing.assert_close(grad, grad_full, rtol=1e-8, atol=1e-10)


def test_model_pickled_without_max_iter_runs_uncapped(fitted, data):
    """Models saved before max_iter existed (e.g. the converted legacy model) load uncapped."""
    import copy
    import pickle
    old = copy.deepcopy(fitted)
    del old.__dict__["max_iter"]
    del old.model_.__dict__["max_iter"]
    old = pickle.loads(pickle.dumps(old))
    assert old.max_iter is None and old.model_.max_iter is None
    a = old.transform(data, device="cpu", progress_bar=False)
    b = fitted.transform(data, device="cpu", progress_bar=False)
    assert (a != b).nnz == 0


def test_fractional_expansion_factor(data):
    model = MatchingPursuitSAE(expansion_factor=0.25, num_steps=3, batch_size=8, max_iter=2,
                               random_state=0).fit(data, device="cpu")
    assert model.embed_dim_ == 4 and model.components_.shape == (4, 16)
    assert model.transform(data, device="cpu", progress_bar=False).shape == (300, 4)


def test_checkpoints(data, tmp_path):
    model = MatchingPursuitSAE(expansion_factor=2, num_steps=7, batch_size=8, random_state=0)
    model.fit(data, device="cpu", checkpoint_every=3, checkpoint_dir=tmp_path)
    paths = sorted(p.name for p in tmp_path.iterdir())
    assert paths == ["step_00003.joblib", "step_00006.joblib", "step_00007.joblib"]
    ckpt = joblib.load(tmp_path / "step_00003.joblib")
    assert ckpt.checkpoint_step_ == 3 and len(ckpt._training_log["mse"]) == 3
    assert ckpt.transform(data[:5], device="cpu", progress_bar=False).shape == (5, 32)
    last = joblib.load(tmp_path / "step_00007.joblib")
    assert torch.equal(last.model_.W, model.model_.W)
    assert model.model_.W.device.type == "cpu"


def test_components_shape(fitted):
    assert fitted.components_.shape == (64, 16)


def test_joblib_round_trip(fitted, data, tmp_path):
    path = tmp_path / "mpsae.joblib"
    joblib.dump(fitted, path)
    loaded = joblib.load(path)
    a = fitted.transform(data, device="cpu", progress_bar=False)
    b = loaded.transform(data, device="cpu", progress_bar=False)
    assert (a != b).nnz == 0


def test_from_state_dict(fitted, data):
    """A model assembled from a saved state_dict, as the legacy conversion does."""
    state = {"W": fitted.model_.W.detach().clone()}
    model = MatchingPursuitSAE(expansion_factor=4)
    model.model_ = MatchingPursuitDictionary(16, 64, threshold=fitted.threshold)
    model.model_.load_state_dict(state)
    model.embed_dim_ = 64
    model.n_features_in_ = 16
    a = fitted.transform(data, device="cpu", progress_bar=False)
    b = model.transform(data, device="cpu", progress_bar=False)
    assert (a != b).nnz == 0


def test_feature_extraction_writes_prefixed_columns(one_slide, stub_encoder):
    import mesoslide as ms

    table = one_slide.tables["tiles_table"]
    emb = np.asarray(table.obsm["stub_embedding"], dtype=np.float32)
    existing = list(table.var_names)
    model = MatchingPursuitSAE(expansion_factor=2, num_steps=5, batch_size=8,
                               random_state=0).fit(emb, device="cpu")
    # stub_embedding is already cached, so the stub encoder is not run
    ms.tl.feature_extraction(
        one_slide, stub_encoder, tile_key="tiles", key_added="stub_embedding",
        sparse=True,
        sparse_transform=lambda X: model.transform(X, device="cpu", progress_bar=False),
        sparse_key_added="STUB_MPSAE", save=False,
    )
    table = one_slide.tables["tiles_table"]
    new = [v for v in table.var_names if v.startswith("STUB_MPSAE_")]
    assert len(new) == emb.shape[1] * 2
    assert all(v in table.var_names for v in existing)
    expected = model.transform(emb, device="cpu", progress_bar=False).toarray()
    np.testing.assert_allclose(table[:, new].X.toarray(), expected)
