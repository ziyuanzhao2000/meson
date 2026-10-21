"""Tests for mesoslide.tools.sparse_coding.FrequencyRankedKMeans."""

import pickle

import numpy as np

from mesoslide.tools.sparse_coding import FrequencyRankedKMeans


def _blobs(sizes=(600, 300, 100), d=8, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-10, 10, size=(len(sizes), d))
    X = np.concatenate([c + rng.normal(scale=0.2, size=(n, d)) for c, n in zip(centers, sizes)])
    return X.astype(np.float32)


def test_labels_ranked_by_cluster_size():
    X = _blobs()
    labels = FrequencyRankedKMeans(n_clusters=3, normalize=False, random_state=0).fit_transform(X)
    np.testing.assert_array_equal(np.bincount(labels), [600, 300, 100])


def test_predict_matches_fit_labels_and_sorted_centroids():
    X = _blobs()
    model = FrequencyRankedKMeans(n_clusters=3, normalize=False, random_state=0).fit(X)
    labels = model.predict(X)
    centroids = model.get_sorted_centroids()
    nearest = np.argmin(((X[:, None, :] - centroids[None]) ** 2).sum(-1), axis=1)
    np.testing.assert_array_equal(labels, nearest)


def test_legacy_import_path_and_pickle_roundtrip():
    from mesoslide.tools.embedders._legacy import FrequencyRankedKMeans as Legacy

    assert Legacy is FrequencyRankedKMeans
    X = _blobs()
    model = FrequencyRankedKMeans(n_clusters=3, random_state=0).fit(X)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
