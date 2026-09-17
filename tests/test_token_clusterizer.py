"""TokenClusterizer: transform(), fit(), and extract_cluster_maps."""

import numpy as np
import pytest
import torch
from sklearn.cluster import KMeans

import mesoslide as ms
from mesoslide.preprocessing import extract_cluster_maps
from mesoslide.tools._model_stage import ImageModelStage
from mesoslide.tools.segmenters import TokenClusterizer
from tests.conftest import StubViTEncoder, TILE_PX


class NonSquareStub(StubViTEncoder):
    """Same contract as StubViTEncoder, but a non-square token grid."""
    grid_size = (2, 3)


def _fitted_kmeans(n_clusters=3, seed=0):
    rng = np.random.default_rng(seed)
    return KMeans(n_clusters=n_clusters, random_state=0).fit(
        rng.random((30, StubViTEncoder.embed_dim))
    )


def test_transform_uses_the_models_own_grid_size():
    """A non-square grid_size must not trip the old square-only assumption."""
    encoder = NonSquareStub()
    clusterizer = TokenClusterizer(model=encoder, kmeans=_fitted_kmeans(), device="cpu")
    images = torch.randint(0, 255, (5, 3, 64, 64), dtype=torch.uint8)
    fm_stage = ImageModelStage(encoder, dense=True, device="cpu")
    token_embeddings = fm_stage(images)
    masks = clusterizer.transform(token_embeddings, output_size=(64, 64))
    assert masks.shape == (5, 64, 64)
    assert masks.dtype == np.uint8
    # StubViTEncoder's tokens are deterministic (token k's embedding is a
    # constant-k vector), so every patch clusters identically here.
    assert np.array_equal(masks[0], masks[1])


def test_fit_produces_a_valid_cluster_order(manifest, open_cohort):
    encoder = StubViTEncoder()
    clusterizer = TokenClusterizer(model=encoder, kmeans=_fitted_kmeans(), device="cpu")
    clusterizer.fit(
        manifest, feature_name="sparse_score", n_positive=6, n_negative=6,
        show_progress=False, image_slides=open_cohort,
    )
    n_clusters = clusterizer.kmeans.n_clusters
    assert clusterizer.cluster_order.shape == (n_clusters,)
    assert set(clusterizer.cluster_order.tolist()) == set(range(n_clusters))


def test_extract_cluster_maps_shares_embedding_across_calls(manifest, open_cohort):
    """extract_cluster_maps takes one clusterizer at a time; calling it once per
    clusterizer (as plot_patch_gallery_with_saliency does) still only embeds a
    shared underlying model once -- via run_model_stages' own per-key caching
    in patches.obsm, not anything specific to a since-removed list signature.
    """
    selection = ms.select_random_patches(manifest, 6, random_state=3)
    encoder = StubViTEncoder()

    call_count = {"n": 0}
    original = encoder.encode_image_dense

    def _counting_encode_image_dense(batch, *args, **kwargs):
        call_count["n"] += 1
        return original(batch, *args, **kwargs)

    encoder.encode_image_dense = _counting_encode_image_dense

    c1 = TokenClusterizer(model=encoder, kmeans=_fitted_kmeans(seed=1),
                           feature_name="c1", device="cpu")
    c2 = TokenClusterizer(model=encoder, kmeans=_fitted_kmeans(seed=2),
                           feature_name="c2", device="cpu")

    map1 = extract_cluster_maps(selection, open_cohort, c1, progress_bar=False)
    map2 = extract_cluster_maps(selection, open_cohort, c2, progress_bar=False)

    assert map1.shape == (6, TILE_PX, TILE_PX)
    assert map2.shape == (6, TILE_PX, TILE_PX)
    assert map1.dtype == np.uint8
    # One embedding pass for the whole batch (num batches with batch_size=16
    # and 6 patches is 1), shared across both clusterizers' calls.
    assert call_count["n"] == 1


def test_extract_cluster_maps_rejects_empty_feature_name(manifest, open_cohort):
    selection = ms.select_random_patches(manifest, 2, random_state=0)
    encoder = StubViTEncoder()
    c1 = TokenClusterizer(model=encoder, kmeans=_fitted_kmeans(), device="cpu")  # feature_name=''
    with pytest.raises(ValueError, match="non-empty feature_name"):
        extract_cluster_maps(selection, open_cohort, c1)


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

    clusterizer = TokenClusterizer(model=encoder, kmeans=_fitted_kmeans(),
                                    feature_name="cached_cluster", device="cpu")

    first = extract_cluster_maps(
        selection, open_cohort, clusterizer, progress_bar=False, cache=True,
    )
    assert cluster_img_key(clusterizer) in selection.obsm
    assert call_count["n"] == 1

    # Second call: neither the model nor slides should be touched at all.
    second = extract_cluster_maps(
        selection, None, clusterizer, progress_bar=False,
    )
    assert call_count["n"] == 1
    assert np.array_equal(first, second)
