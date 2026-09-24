"""TokenClusterer: fit/predict/transform, fit_token_clusterer, persistence, extract_cluster_maps."""

import pickle

import numpy as np
import pytest
import torch
from sklearn.base import clone
from sklearn.cluster import KMeans

import mesoslide as ms
from mesoslide.preprocessing import extract_cluster_maps
from mesoslide.tools._model_stage import ImageModelStage
from mesoslide.tools.segmenters import TokenClusterer, fit_token_clusterer, predict_token_labels
from tests.conftest import StubViTEncoder, TILE_PX


class NonSquareStub(StubViTEncoder):
    """Same contract as StubViTEncoder, but a non-square token grid."""
    grid_size = (2, 3)


def _tokens(n_patches=8, n_tokens=4, dim=StubViTEncoder.embed_dim, seed=0):
    return np.random.default_rng(seed).random((n_patches, n_tokens, dim))


def fitted_clusterer(name=None, seed=0, encoder=StubViTEncoder, **params):
    """A fitted clusterer carrying the model metadata fit_token_clusterer would set."""
    c = TokenClusterer(name=name, **params)
    c.grid_size_ = tuple(encoder.grid_size)
    c.patch_size_ = tuple(encoder.patch_size)
    c.model_name_ = encoder.name
    n_tokens = encoder.grid_size[0] * encoder.grid_size[1]
    return c.fit(_tokens(n_tokens=n_tokens, seed=seed))


def test_transform_uses_the_models_own_grid_size():
    """A non-square grid_size must not trip a square-only assumption."""
    encoder = NonSquareStub()
    clusterer = fitted_clusterer(encoder=NonSquareStub)
    images = torch.randint(0, 255, (5, 3, 64, 64), dtype=torch.uint8)
    token_embeddings = ImageModelStage(encoder, dense=True, device="cpu")(images)
    masks = clusterer.transform(token_embeddings, output_size=(64, 64))
    assert masks.shape == (5, 64, 64)
    assert masks.dtype == np.uint8
    # Stub tokens are image-independent, so every patch clusters identically.
    assert np.array_equal(masks[0], masks[1])
    assert clusterer.predict(token_embeddings).shape == (5, 2, 3)


def test_fit_infers_square_grid_and_rejects_non_square():
    assert TokenClusterer().fit(_tokens(n_tokens=9)).grid_size_ == (3, 3)
    with pytest.raises(ValueError, match="square token grid"):
        TokenClusterer().fit(_tokens(n_tokens=6))


def test_n_clusters_and_random_state_are_honored():
    X = _tokens(n_patches=20, n_tokens=9, seed=1)
    c = TokenClusterer(n_clusters=5, random_state=7).fit(X)
    assert c.n_clusters_ == 5
    assert c.kmeans_.random_state == 7
    assert c.cluster_order_.tolist() == list(range(5))  # no y -> identity order
    assert c.kmeans_.n_init == 10   # default: best of 10 initializations
    ref = KMeans(n_clusters=5, random_state=7, n_init=10).fit(X.reshape(-1, X.shape[-1]))
    np.testing.assert_allclose(c.cluster_centers_, ref.cluster_centers_)
    assert TokenClusterer(n_init=1).fit(X).kmeans_.n_init == 1


def test_clusterer_pickled_without_n_init_loads_with_sklearn_default():
    c = TokenClusterer().fit(_tokens(n_tokens=9))
    state = c.__getstate__()
    del state["n_init"]
    old = TokenClusterer.__new__(TokenClusterer)
    old.__setstate__(state)
    assert old.get_params()["n_init"] == "auto"


def test_predict_matches_kmeans_predict():
    X = _tokens(n_patches=10, n_tokens=16, seed=2)
    y = np.random.default_rng(3).random(10)
    c = TokenClusterer(n_clusters=4).fit(X, y)
    raw = c.kmeans_.predict(X.reshape(-1, X.shape[-1])).reshape(10, 4, 4)
    assert np.array_equal(c.predict(X), c.cluster_order_[raw])


def test_predict_token_labels_matches_per_clusterer_predict():
    X = _tokens(n_patches=10, n_tokens=16, seed=12)
    y = np.random.default_rng(4).random(10)
    clusterers = [
        TokenClusterer(n_clusters=k, random_state=s).fit(_tokens(n_patches=10, n_tokens=16, seed=s), y)
        for k, s in [(3, 0), (4, 1), (2, 2)]
    ]
    expected = np.stack([c.predict(X) for c in clusterers])
    out = predict_token_labels(clusterers, X, chunk_size=37)  # uneven chunks
    assert out.shape == (3, 10, 4, 4) and out.dtype == np.uint8
    assert np.array_equal(out, expected)
    with pytest.raises(ValueError, match="share grid_size_"):
        predict_token_labels([clusterers[0], TokenClusterer().fit(_tokens(n_tokens=9))], X)


def test_refit_keeps_centroids_and_only_reorders():
    X = _tokens(n_patches=12, n_tokens=16, seed=4)
    c = TokenClusterer().fit(X, np.random.default_rng(0).random(12))
    centers = c.cluster_centers_.copy()
    c.fit(_tokens(n_patches=12, n_tokens=16, seed=5), np.random.default_rng(1).random(12))
    np.testing.assert_array_equal(c.cluster_centers_, centers)
    c.fit_order(X, -np.arange(12.0))
    np.testing.assert_array_equal(c.cluster_centers_, centers)
    assert sorted(c.cluster_order_.tolist()) == [0, 1, 2]
    assert c.fit_diagnostics_.shape == (12, 4)


def test_diff_abundance_ordering_drops_correlation_diagnostics():
    X = _tokens(n_patches=12, n_tokens=16, seed=6)
    y = np.r_[np.ones(6), np.zeros(6)]
    c = TokenClusterer(ordering="diff_abundance").fit(X, y)
    assert not hasattr(c, "fit_diagnostics_")
    assert c.order_scores_.shape == (3,)
    assert np.all(np.diff(c.order_scores_) >= 0)  # canonical order is ascending


def test_clone_needs_no_model():
    c = fitted_clusterer(name="c", n_clusters=4)
    fresh = clone(c)
    assert fresh.get_params() == c.get_params()
    assert not hasattr(fresh, "cluster_centers_")


def test_legacy_pickle_is_migrated():
    """A pre-0.10 TokenClusterizer pickle loads as a working TokenClusterer."""
    from mesoslide.tools.segmenters.TokenClusterizer import TokenClusterizer

    rng = np.random.default_rng(0)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(rng.random((30, 4)))
    old = TokenClusterizer.__new__(TokenClusterizer)
    old.__dict__.update(
        grid_size=(2, 2), patch_size=(64, 64), model_name="vit-stub", kmeans=kmeans,
        interpolation="nearest", cluster_order=np.array([2, 0, 1]),
        feature_name="UNI_SAE_7", device="cuda:0", cv2_interp=6,
    )
    new = pickle.loads(pickle.dumps(old))
    assert isinstance(new, TokenClusterer)
    assert new.name == new.feature_name_ == "UNI_SAE_7"
    assert new.ordering == "diff_abundance"
    assert not hasattr(new, "device") and not hasattr(new, "cluster_order")
    X = _tokens(seed=11)
    raw = kmeans.predict(X.reshape(-1, 4)).reshape(8, 2, 2)
    assert np.array_equal(new.predict(X), np.array([2, 0, 1])[raw])
    assert new.transform(X).shape == (8, 128, 128)


def test_fit_token_clusterer_correlation(manifest, open_cohort):
    encoder = StubViTEncoder()
    c = fit_token_clusterer(
        manifest, "sparse_score", encoder, clusterer=TokenClusterer(name="label"),
        n_patches=6, top_fraction=0.5, progress_bar=False, image_slides=open_cohort, device="cpu",
    )
    assert c.name == "label"  # never overwritten by fit
    assert c.feature_name_ == "sparse_score"
    assert c.grid_size_ == (2, 2) and c.patch_size_ == (64, 64)
    assert c.model_name_ == "vit-stub"
    assert sorted(c.cluster_order_.tolist()) == [0, 1, 2]
    assert c.fit_diagnostics_.shape[1] == 4


def test_fit_token_clusterer_diff_abundance(manifest, open_cohort):
    c = fit_token_clusterer(
        manifest, "sparse_score", StubViTEncoder(),
        clusterer=TokenClusterer(ordering="diff_abundance", name="label"),
        n_patches=6, n_negative=6, top_fraction=0.5, progress_bar=False, image_slides=open_cohort, device="cpu",
    )
    assert c.name == "label"
    assert c.feature_name_ == "sparse_score"
    assert sorted(c.cluster_order_.tolist()) == [0, 1, 2]


def test_model_name_is_canonicalized_to_the_registry_key(manifest, open_cohort, monkeypatch):
    """model_name_ stores the registry key, not the instance's self-reported casing."""
    from lazyslide_models import MODEL_REGISTRY

    class LoudNameStub(StubViTEncoder):
        name = "VIT-STUB-LOUD"

    monkeypatch.setitem(MODEL_REGISTRY, "vit-stub-loud", lambda **kw: LoudNameStub())
    c = fit_token_clusterer(
        manifest, "sparse_score", LoudNameStub(), n_patches=6, top_fraction=0.5,
        progress_bar=False, image_slides=open_cohort, device="cpu",
    )
    assert c.model_name_ == "vit-stub-loud"


def test_extract_cluster_maps_shares_embedding_across_calls(manifest, open_cohort):
    """Calling extract_cluster_maps once per clusterer embeds a shared model once,
    via run_model_stages' per-key caching in patches.obsm."""
    selection = ms.select_random_patches(manifest, 6, random_state=3)
    encoder = StubViTEncoder()

    call_count = {"n": 0}
    original = encoder.encode_image_dense

    def _counting_encode_image_dense(batch, *args, **kwargs):
        call_count["n"] += 1
        return original(batch, *args, **kwargs)

    encoder.encode_image_dense = _counting_encode_image_dense

    c1 = fitted_clusterer(name="c1", seed=1)
    c2 = fitted_clusterer(name="c2", seed=2)

    map1 = extract_cluster_maps(selection, c1, encoder, slides=open_cohort, progress_bar=False, device="cpu")
    map2 = extract_cluster_maps(selection, c2, encoder, slides=open_cohort, progress_bar=False, device="cpu")

    assert map1.shape == (6, TILE_PX, TILE_PX)
    assert map2.shape == (6, TILE_PX, TILE_PX)
    assert map1.dtype == np.uint8
    assert call_count["n"] == 1


def test_extract_cluster_maps_auto_resolves_model_from_model_name(manifest, open_cohort, monkeypatch):
    """With `model` omitted, the model is pulled from the registry via model_name_."""
    from lazyslide_models import MODEL_REGISTRY

    selection = ms.select_random_patches(manifest, 2, random_state=0)
    clusterer = fitted_clusterer(name="auto_resolve")
    monkeypatch.setitem(
        MODEL_REGISTRY, "vit-stub",
        lambda model_path=None, token=None: StubViTEncoder(),
    )
    cluster_map = extract_cluster_maps(selection, clusterer, slides=open_cohort, progress_bar=False, device="cpu")
    assert cluster_map.shape == (2, TILE_PX, TILE_PX)
    assert cluster_map.dtype == np.uint8


def test_extract_cluster_maps_accepts_deprecated_clusterizer_kwarg(manifest, open_cohort):
    selection = ms.select_random_patches(manifest, 2, random_state=0)
    with pytest.warns(DeprecationWarning, match="'clusterizer' is deprecated"):
        out = extract_cluster_maps(
            selection, clusterizer=fitted_clusterer(name="old_kw"), model=StubViTEncoder(),
            slides=open_cohort, progress_bar=False, device="cpu",
        )
    assert out.shape == (2, TILE_PX, TILE_PX)


def test_extract_cluster_maps_rejects_empty_name(manifest, open_cohort):
    selection = ms.select_random_patches(manifest, 2, random_state=0)
    with pytest.raises(ValueError, match="non-empty name"):
        extract_cluster_maps(selection, fitted_clusterer(), StubViTEncoder(), slides=open_cohort)


def test_extract_cluster_maps_cache_avoids_recompute(manifest, open_cohort):
    from mesoslide.preprocessing._extract_cluster_maps import cluster_img_key

    selection = ms.select_random_patches(manifest, 4, random_state=2)
    encoder = StubViTEncoder()

    call_count = {"n": 0}
    original = encoder.encode_image_dense

    def _counting_encode_image_dense(batch, *args, **kwargs):
        call_count["n"] += 1
        return original(batch, *args, **kwargs)

    encoder.encode_image_dense = _counting_encode_image_dense

    clusterer = fitted_clusterer(name="cached_cluster")
    first = extract_cluster_maps(
        selection, clusterer, encoder, slides=open_cohort, progress_bar=False, cache=True, device="cpu",
    )
    assert cluster_img_key(clusterer) in selection.obsm
    assert call_count["n"] == 1

    # Cached: neither the model nor slides are touched.
    second = extract_cluster_maps(selection, clusterer, None, slides=None, progress_bar=False)
    assert call_count["n"] == 1
    assert np.array_equal(first, second)
