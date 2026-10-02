"""TokenClassifier, the shared token-labeler base, and token map assembly."""

import glob
import os
import pickle

import joblib
import numpy as np
import pytest
import torch
from sklearn.neighbors import KNeighborsClassifier

import mesoslide as ms
from mesoslide import assemble_token_map
from mesoslide.tools.segmenters import TokenClassifier, predict_token_labels
from tests.conftest import StubViTEncoder, TILE_PX
from tests.test_token_clusterer import fitted_clusterer


def _knn(metric="cosine", n_ref=60, dim=16, n_classes=5, seed=0):
    rng = np.random.default_rng(seed)
    X, y = rng.normal(size=(n_ref, dim)), rng.integers(1, n_classes + 1, n_ref)
    return KNeighborsClassifier(n_neighbors=1, metric=metric).fit(X, y)


def _queries(n=500, dim=16, seed=1):
    return np.random.default_rng(seed).normal(size=(n, dim))


# ---------------------------------------------------------------------------
# TokenClassifier
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", ["cosine", "euclidean", "minkowski"])
def test_from_sklearn_matches_knn_predict(metric):
    knn = _knn(metric)
    clf = TokenClassifier.from_sklearn(knn, device="cpu", dtype="float64")
    X = _queries()
    assert np.array_equal(clf.predict_flat(X), knn.predict(X))
    assert clf.metric == ("cosine" if metric == "cosine" else "euclidean")


def test_float32_agrees_with_float64():
    knn = _knn()
    X = _queries(n=2000)
    f64 = TokenClassifier.from_sklearn(knn, device="cpu", dtype="float64").predict_flat(X)
    f32 = TokenClassifier.from_sklearn(knn, device="cpu", dtype="float32").predict_flat(X)
    assert (f64 == f32).mean() > 0.995


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_matches_cpu():
    knn = _knn()
    X = _queries(n=2000)
    cpu = TokenClassifier.from_sklearn(knn, device="cpu", dtype="float64").predict_flat(X)
    gpu = TokenClassifier.from_sklearn(knn, device="cuda", dtype="float64").predict_flat(X)
    assert np.array_equal(cpu, gpu)


def test_from_sklearn_rejects_unsupported_models():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(10, 4)), rng.integers(0, 2, 10)
    with pytest.raises(NotImplementedError, match="n_neighbors=1"):
        TokenClassifier.from_sklearn(KNeighborsClassifier(n_neighbors=3).fit(X, y))
    with pytest.raises(NotImplementedError, match="metric"):
        TokenClassifier.from_sklearn(KNeighborsClassifier(n_neighbors=1, metric="manhattan").fit(X, y))
    with pytest.raises(TypeError):
        TokenClassifier.from_sklearn(object())


def test_fit_validates_labels():
    X = np.zeros((3, 4))
    with pytest.raises(ValueError, match="0, 255"):
        TokenClassifier().fit(X, np.array([0, 1, 300]))
    with pytest.raises(ValueError, match="0, 255"):
        TokenClassifier().fit(X, np.array([0.5, 1.0, 2.0]))


def test_predict_grid_transform_and_n_labels():
    knn = _knn()
    clf = TokenClassifier.from_sklearn(knn, device="cpu", grid_size=(2, 3), patch_size=(8, 8))
    X = np.random.default_rng(2).normal(size=(4, 6, 16))
    labels = clf.predict(X)
    assert labels.shape == (4, 2, 3) and labels.dtype == np.uint8
    assert np.array_equal(labels.reshape(-1), knn.predict(X.reshape(-1, 16)))
    masks = clf.transform(X)
    assert masks.shape == (4, 16, 24)
    assert np.array_equal(masks[:, ::8, ::8], labels)
    assert clf.n_labels_ == int(knn.classes_.max()) + 1
    with pytest.raises(ValueError, match="tokens"):
        clf.predict(np.zeros((1, 5, 16)))


def test_square_grid_is_inferred_when_unset():
    clf = TokenClassifier.from_sklearn(_knn(), device="cpu")
    assert clf.predict(np.random.default_rng(0).normal(size=(2, 9, 16))).shape == (2, 3, 3)


def test_save_load_and_pickle_round_trip(tmp_path):
    knn = _knn()
    clf = TokenClassifier.from_sklearn(knn, name="tissue", device="cpu", grid_size=(14, 14),
                                       patch_size=(16, 16), model_name="uni")
    X = _queries()
    expected = clf.predict_flat(X)  # also populates the tensor cache

    clf.save(tmp_path / "clf.npz")
    loaded = TokenClassifier.load(tmp_path / "clf.npz", device="cpu")
    assert np.array_equal(loaded.predict_flat(X), expected)
    assert (loaded.name, loaded.grid_size_, loaded.patch_size_, loaded.model_name_) == (
        "tissue", (14, 14), (16, 16), "uni")

    state = pickle.loads(pickle.dumps(clf))
    assert getattr(state, "_tensor_cache", None) is None
    assert np.array_equal(state.predict_flat(X), expected)


def test_refit_invalidates_cached_prototypes():
    clf = TokenClassifier(device="cpu")
    clf.fit(np.eye(2), np.array([1, 2]))
    assert clf.predict_flat(np.array([[1.0, 0.0]]))[0] == 1
    clf.fit(np.eye(2), np.array([7, 8]))
    assert clf.predict_flat(np.array([[1.0, 0.0]]))[0] == 7


def test_predict_token_labels_accepts_mixed_labelers():
    clusterer = fitted_clusterer(name="c")
    dim = StubViTEncoder.embed_dim
    rng = np.random.default_rng(3)
    clf = TokenClassifier("euclidean", name="k", device="cpu").fit(rng.normal(size=(6, dim)), np.arange(6))
    X = rng.normal(size=(5, 4, dim))
    out = predict_token_labels([clusterer, clf], X)
    assert np.array_equal(out, np.stack([clusterer.predict(X), clf.predict(X)]))


# ---------------------------------------------------------------------------
# Real classifiers (opt-in: MESOSLIDE_TEST_TOKEN_CLASSIFIERS=<glob>[:<glob>...] of joblib files)
# ---------------------------------------------------------------------------

def _real_classifier_paths():
    patterns = os.environ.get("MESOSLIDE_TEST_TOKEN_CLASSIFIERS", "")
    return sorted(p for pattern in patterns.split(os.pathsep) if pattern for p in glob.glob(pattern))


@pytest.mark.parametrize("path", _real_classifier_paths() or [None])
def test_real_classifier_matches_sklearn(path):
    if path is None:
        pytest.skip("set MESOSLIDE_TEST_TOKEN_CLASSIFIERS to a glob of KNeighborsClassifier joblib files")
    knn = joblib.load(path)
    rng = np.random.default_rng(0)
    # Queries near the reference tokens, as real tokens are.
    ref = np.asarray(knn._fit_X)
    X = ref[rng.integers(0, len(ref), 4000)] + rng.normal(scale=ref.std(), size=(4000, ref.shape[1]))
    expected = knn.predict(X)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        f64 = TokenClassifier.from_sklearn(knn, device=device, dtype="float64").predict_flat(X)
        assert np.array_equal(f64, expected), device
        f32 = TokenClassifier.from_sklearn(knn, device=device, dtype="float32").predict_flat(X)
        assert (f32 == expected).mean() > 0.999, device


# ---------------------------------------------------------------------------
# Token map assembly
# ---------------------------------------------------------------------------

def test_assemble_without_overlap_tiles_the_grids():
    grids = np.arange(4 * 4, dtype=np.uint8).reshape(4, 2, 2)
    xy = np.array([[0, 0], [8, 0], [0, 8], [8, 8]])
    out = assemble_token_map(xy, grids, tile_px=8, stride_px=8)
    expected = np.block([[np.kron(grids[0], np.ones((4, 4))), np.kron(grids[1], np.ones((4, 4)))],
                         [np.kron(grids[2], np.ones((4, 4))), np.kron(grids[3], np.ones((4, 4)))]])
    assert out.shape == (16, 16)
    assert np.array_equal(out, expected)


@pytest.mark.parametrize("tile_px,stride_px,grid", [(10, 6, 3), (555, 256, 14), (189, 128, 14)])
def test_assemble_overlapping_tiles_use_central_windows(tile_px, stride_px, grid):
    """Each pixel of the interior comes from the tile whose central window contains it."""
    n = 4
    xy = np.array([(i * stride_px, j * stride_px) for j in range(n) for i in range(n)])
    tile_ids = np.arange(len(xy), dtype=np.uint8) + 1
    grids = np.broadcast_to(tile_ids[:, None, None], (len(xy), grid, grid)).copy()
    out = assemble_token_map(xy, grids, tile_px, stride_px)

    margin = (tile_px - stride_px) / 2
    for k, (x, y) in enumerate(xy):
        cx, cy = int(x + margin + stride_px / 2), int(y + margin + stride_px / 2)
        assert out[cy, cx] == tile_ids[k]
    # Fully covered: no background anywhere inside the tiled extent.
    assert (out > 0).all()

    no_margins = assemble_token_map(xy, grids, tile_px, stride_px, fill_margins=False)
    m = int(np.floor(margin))
    if m > 0:
        assert (no_margins[:m, :] == 0).all() and (no_margins[:, :m] == 0).all()


def test_assemble_maps_tokens_by_pixel_centre_for_non_divisible_tiles():
    grids = np.arange(9, dtype=np.uint8).reshape(1, 3, 3)
    out = assemble_token_map(np.array([[0, 0]]), grids, tile_px=10, stride_px=10)
    # Token boundaries at 10/3 and 20/3: pixel centres 0.5..2.5 -> 0, 3.5..6.5 -> 1, 7.5..9.5 -> 2.
    assert np.array_equal(out[0], [0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    assert np.array_equal(out[:, 0], [0, 0, 0, 3, 3, 3, 3, 6, 6, 6])


def test_assemble_downsample_and_shape():
    grids = np.ones((1, 2, 2), dtype=np.uint8)
    out = assemble_token_map(np.array([[4, 4]]), grids, tile_px=8, stride_px=8, shape=(20, 30), downsample=2)
    assert out.shape == (10, 15)
    assert out[2:6, 2:6].all() and out.sum() == 16


def test_token_label_map_from_feature_extraction(one_slide):
    """Labels written by feature_extraction(dense=True, reducer=clf.reducer()) stitch back per tile."""
    encoder = StubViTEncoder()
    dim = encoder.embed_dim
    # Stub token k is a constant-k vector; euclidean 1-NN to constant prototypes recovers k.
    clf = TokenClassifier("euclidean", device="cpu").fit(
        np.repeat(np.arange(4.0)[:, None], dim, axis=1), np.arange(4) + 10)
    ms.tl.feature_extraction(one_slide, encoder, key_added="vitstub", dense=True,
                             reducer=clf.reducer(), dense_key_added="labels",
                             batch_size=16, device="cpu", save=False)
    table = one_slide.tables["tiles_table"]
    assert np.array_equal(table.obsm["labels"][0], [10, 11, 12, 13])

    label_map = ms.tl.token_label_map(one_slide, "labels")
    assert label_map.shape == tuple(one_slide.properties.shape)
    xy = ms.table_tile_geometries(one_slide).bounds[["minx", "miny"]].to_numpy().astype(int)
    half = TILE_PX // 2
    x, y = xy[0]
    # Tiles are laid out with stride == tile size here, so each tile keeps its whole grid.
    assert label_map[y + half // 2, x + half // 2] == 10
    assert label_map[y + half // 2, x + half + half // 2] == 11
    assert label_map[y + half + half // 2, x + half // 2] == 12
    assert label_map[y + half + half // 2, x + half + half // 2] == 13
    assert set(np.unique(label_map)) <= {0, 10, 11, 12, 13}


def test_select_region_patches_and_extract_cluster_maps(one_slide):
    """A FOV tiled on its own grid: pixels match a direct read, and a TokenClassifier labels every patch."""
    from mesoslide.preprocessing import extract_cluster_maps, extract_patch_images

    region = (100, 150, 2 * TILE_PX, TILE_PX + 10)  # height rounds up to 2 rows
    patches = ms.select_region_patches(one_slide, region)
    assert patches.uns["grid_shape"] == (2, 2)
    assert list(patches.obs["x"]) == [100, 100 + TILE_PX, 100, 100 + TILE_PX]
    assert list(patches.obs["y"]) == [150, 150, 150 + TILE_PX, 150 + TILE_PX]

    pixels = extract_patch_images(patches, channel_first=False, progress_bar=False)
    direct = one_slide.reader.get_region(100 + TILE_PX, 150 + TILE_PX, TILE_PX, TILE_PX, level=0)
    assert np.array_equal(np.asarray(pixels[3])[..., :3], np.asarray(direct)[..., :3])

    encoder = StubViTEncoder()
    clf = TokenClassifier("euclidean", name="stub", device="cpu").fit(
        np.repeat(np.arange(4.0)[:, None], encoder.embed_dim, axis=1), np.arange(4) + 10)
    maps = extract_cluster_maps(patches, clf, encoder, device="cpu", progress_bar=False)
    assert maps.shape == (4, TILE_PX, TILE_PX)
    half = TILE_PX // 2
    assert maps[0, 0, 0] == 10 and maps[0, 0, half] == 11 and maps[0, half, 0] == 12 and maps[0, -1, -1] == 13
