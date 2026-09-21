"""Shared fixtures.

The suite is self-contained: it builds small synthetic slides through the real
ezslide/lazyslide pipeline rather than depending on any particular dataset, so
it runs anywhere and in seconds. Tests that want the project's actual TB cohort
can opt in by setting MESOSLIDE_TEST_STORES to a glob of .zarr stores; they skip
otherwise.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import pytest
import torch

TILE_PX = 128
SLIDE_SHAPE = (1024, 1536)  # (h, w)
MPP = 0.5


# ---------------------------------------------------------------------------
# Synthetic slides
# ---------------------------------------------------------------------------

def _write_slide(path, seed: int, shape=SLIDE_SHAPE):
    """Write a small RGB OME-TIFF with one tissue-like blob on a bright ground."""
    import tifffile

    h, w = shape
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 240, np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    blob = ((yy - h / 2) / (0.37 * h)) ** 2 + ((xx - w / 2) / (0.37 * w)) ** 2 < 1
    img[blob] = rng.integers(60, 190, size=(int(blob.sum()), 3), dtype=np.uint8)
    tifffile.imwrite(
        path, img, photometric="rgb", tile=(256, 256), ome=True,
        metadata={"axes": "YXS"},
    )
    return path


def _build_store(tmpdir, name: str, seed: int, shape=SLIDE_SHAPE):
    """Run a synthetic slide through open -> find_tissues -> tile -> write."""
    import ezslide
    import lazyslide as zs

    slide_path = os.path.join(tmpdir, f"{name}.ome.tif")
    _write_slide(slide_path, seed, shape)

    # Pin the reader, as the real pipeline does. Auto-detection picks
    # OpenSlideReader for a plain OME-TIFF, and that reader has no pyramid
    # chunk path, so rendering fails on it.
    wsi = ezslide.open_slide(slide_path, attach_images=True, reader="tifffile_zarr")
    wsi.set_mpp(MPP)  # otherwise tissue segmentation warns and tile specs lack scale
    zs.pp.find_tissues(wsi, detect_holes=False)
    zs.pp.tile_tissues(wsi, tile_px=TILE_PX, stride_px=TILE_PX, background_filter=False)

    store = os.path.join(tmpdir, f"{name}.zarr")
    wsi.write(store)
    return store


class StubEncoder:
    """Minimal ImageModel: enough surface for feature_extraction, no downloads."""

    name = "stub"

    def __init__(self):
        self.model = torch.nn.Identity()

    def to(self, device):
        return self

    def get_transform(self):
        return None

    def encode_image(self, batch):
        flat = batch.reshape(batch.shape[0], -1).float()
        return flat[:, :8] / 255.0


class RealTransformStubEncoder:
    """A stub encoder whose ``get_transform()`` is a real ``ImageModel``-style
    Compose (ToImage/ToDtype/Resize/Normalize), unlike ``StubEncoder``'s
    ``None``.

    Exists to catch channel-order bugs: ``Normalize`` only tolerates
    channel-first ``(B, C, H, W)`` input, so a stage fed channel-last tiles
    fails inside the transform rather than silently producing wrong output.
    """

    name = "real-transform-stub"

    def __init__(self):
        self.model = torch.nn.Identity()

    def to(self, device):
        return self

    def get_transform(self):
        from torchvision.transforms.v2 import Compose, Normalize, Resize, ToDtype, ToImage

        return Compose([
            ToImage(),
            ToDtype(dtype=torch.float32, scale=True),
            Resize(size=(64, 64), antialias=False),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def encode_image(self, batch):
        assert batch.shape[1] == 3, f"expected channel-first input, got shape {tuple(batch.shape)}"
        return batch.reshape(batch.shape[0], -1)[:, :8]


class StubViTEncoder:
    """A dense-capable stub: grid_size/patch_size/encode_image_dense, no downloads.

    encode_image_dense is deterministic and independent of image content --
    token k's embedding is a constant-k vector across every embedding
    dimension, so a mean-reducer recovers exactly k, and tests can assert
    against that without depending on real pixel values.
    """

    name = "vit-stub"
    grid_size = (2, 2)
    patch_size = (64, 64)
    num_prefix_tokens = 1
    embed_dim = 4

    def __init__(self):
        self.model = torch.nn.Identity()

    def to(self, device):
        return self

    def try_compile(self, **kwargs):
        pass

    def get_transform(self):
        return None

    def encode_image(self, batch):
        flat = batch.reshape(batch.shape[0], -1).float()
        return flat[:, :self.embed_dim] / 255.0

    def encode_image_dense(self, batch):
        from lazyslide_models.base import DenseTokens

        b = batch.shape[0]
        n_tokens = self.grid_size[0] * self.grid_size[1]
        token_idx = torch.arange(n_tokens, dtype=torch.float32)
        patch_tokens = (
            token_idx.view(1, n_tokens, 1)
            .expand(b, n_tokens, self.embed_dim)
            .clone()
        )
        cls_token = torch.zeros(b, self.embed_dim)
        return DenseTokens(cls_token=cls_token, patch_tokens=patch_tokens)


def _add_features(store, seed: int):
    """Attach a deterministic per-tile embedding and score columns to a store.

    Goes through mesoslide.tl.feature_extraction with a stub encoder so the
    table is built by the real code path, then persists it.
    """
    import ezslide

    import mesoslide as ms

    wsi = ezslide.read_slide(store, attach_images=True)
    ms.tl.feature_extraction(
        wsi, StubEncoder(), key_added="stub_embedding",
        batch_size=8, num_workers=0, device="cpu", save=False,
    )
    table = wsi.tables["tiles_table"]
    rng = np.random.default_rng(seed)
    n = table.n_obs
    table.obs["score"] = rng.random(n).astype("float32")
    # a sparse-ish score, so select_negative_patches has real zeros to find
    binary = np.zeros(n, dtype="float32")
    binary[rng.choice(n, size=max(1, n // 3), replace=False)] = 1.0
    table.obs["sparse_score"] = binary * rng.random(n).astype("float32")
    table.obs["flag"] = (binary > 0).astype(int)
    wsi.write_element("tiles_table", overwrite=True)
    wsi.close()
    return store


@pytest.fixture(scope="session")
def cohort(tmp_path_factory):
    """Three synthetic slides of differing size, written as .zarr stores."""
    tmpdir = str(tmp_path_factory.mktemp("cohort"))
    shapes = [SLIDE_SHAPE, (896, 1408), (1152, 1280)]
    stores = []
    for i, shape in enumerate(shapes):
        store = _build_store(tmpdir, f"SYNTH{i:02d}", seed=i, shape=shape)
        stores.append(_add_features(store, seed=100 + i))
    return stores


@pytest.fixture(scope="session")
def manifest(cohort):
    """A cohort manifest, the DataFrame shape iter_slides/agg_wsi both accept."""
    return pd.DataFrame({"store": cohort})


@pytest.fixture
def one_slide(cohort):
    """A single opened WSIData with image data attached."""
    import ezslide

    wsi = ezslide.read_slide(cohort[0], attach_images=True)
    yield wsi
    wsi.close()


@pytest.fixture
def one_table(one_slide):
    """A single slide's tile table."""
    return one_slide.tables["tiles_table"]


@pytest.fixture
def open_cohort(manifest):
    """{slide_id: WSIData} with pixels, as patch extraction and galleries need."""
    import mesoslide as ms

    slides = ms.open_slides(manifest)
    yield slides
    for wsi in slides.values():
        wsi.close()


@pytest.fixture
def stub_encoder():
    """A plain, non-ViT stub encoder -- no grid_size/encode_image_dense."""
    return StubEncoder()


@pytest.fixture
def vit_stub_encoder():
    """A dense-capable stub encoder for feature_extraction(dense=True) tests."""
    return StubViTEncoder()


@pytest.fixture
def real_transform_stub_encoder():
    """A stub encoder with a real ToImage/Normalize transform chain."""
    return RealTransformStubEncoder()


# ---------------------------------------------------------------------------
# Optional: the project's real cohort
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def real_manifest():
    """Manifest over MESOSLIDE_TEST_STORES, or skip.

    Stores that spatialdata itself cannot read (checked cheaply via zarr attrs,
    without opening the whole slide) are dropped with a printed note rather
    than failing the whole run -- that is a data/environment problem in one
    store, not something these tests are checking for.
    """
    pattern = os.environ.get("MESOSLIDE_TEST_STORES")
    if not pattern:
        pytest.skip("set MESOSLIDE_TEST_STORES to a glob of .zarr stores")
    stores = sorted(glob.glob(pattern))
    if not stores:
        pytest.skip(f"MESOSLIDE_TEST_STORES matched nothing: {pattern}")

    import zarr

    usable = []
    for store in stores:
        try:
            attrs = dict(zarr.open(f"{store}/tables/tiles_table", mode="r").attrs)
            if not attrs:
                raise ValueError("no attrs on tiles_table group")
            usable.append(store)
        except Exception as e:
            print(f"real_manifest: skipping unreadable store {store}: {e}")
    if not usable:
        pytest.skip("no usable stores after filtering")
    return pd.DataFrame({"store": usable})
