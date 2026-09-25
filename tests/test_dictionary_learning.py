"""Tests for MiniBatchDictionaryCoding (mesoslide.tools.sparse_coding._dictionary_learning)."""

import numpy as np
import pytest
import scipy.sparse as sp

from mesoslide.tools.sparse_coding import MiniBatchDictionaryCoding


def _data(n=200, d=16, seed=0):
    return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32) + 3.0


def _fitted(X, fraction=1.0, sample_size=None, **kwargs):
    params = dict(n_components=8, alpha=0.1, batch_size=32, max_iter=5, n_jobs=1, random_state=0)
    params.update(kwargs)
    return MiniBatchDictionaryCoding(**params).fit(X, fraction=fraction, sample_size=sample_size)


def test_fit_transform_shapes_and_nonneg():
    X = _data()
    dl = _fitted(X)
    codes = dl.transform(X)
    assert sp.issparse(codes) and codes.format == "csr"
    assert codes.shape == (len(X), 8)
    assert dl.embed_dim_ == 8
    assert dl.components_.shape == (8, X.shape[1])
    assert codes.min() >= 0
    assert dl._training_log["mean_l0"] > 0


def test_centering_and_matches_sklearn_transform():
    X = _data()
    dl = _fitted(X)
    np.testing.assert_allclose(dl.mean_, X.mean(0), rtol=1e-5)
    expected = dl.model_.transform(X - dl.mean_)
    np.testing.assert_allclose(dl.transform(X).toarray(), expected, rtol=1e-5, atol=1e-6)


def test_no_centering_uses_zero_mean():
    X = _data()
    dl = _fitted(X, center=False)
    assert np.all(dl.mean_ == 0)


def test_block_transform_matches_single_block():
    X = _data()
    dl = _fitted(X)
    full = dl.transform(X).toarray()
    dl.set_params(transform_batch_size=37)
    np.testing.assert_allclose(dl.transform(X).toarray(), full, rtol=1e-6, atol=1e-7)


def test_column_keep_indices_slices_full_output():
    X = _data()
    dl = _fitted(X)
    full = dl.transform(X).toarray()
    cols = [5, 0, 3]
    np.testing.assert_allclose(dl.transform(X, column_keep_indices=cols).toarray(), full[:, cols])


def test_inverse_transform_beats_mean_baseline():
    X = _data()
    dl = _fitted(X, max_iter=20)
    X_hat = dl.inverse_transform(dl.transform(X))
    assert X_hat.shape == X.shape
    err = np.mean((X - X_hat) ** 2)
    baseline = np.mean((X - X.mean(0)) ** 2)
    assert err < baseline


def test_subsampling_and_invalid_fraction():
    X = _data(n=100)
    dl = _fitted(X, fraction=0.5)
    assert dl.model_.n_features_in_ == X.shape[1]
    dl = _fitted(X, sample_size=40)
    # Mean comes from the 40-row subsample, not the full data.
    assert not np.allclose(dl.mean_, X.mean(0))
    with pytest.raises(ValueError, match="fraction"):
        _fitted(X, fraction=0.0)
