"""Token ablation: encode_kept_tokens, token_ablation, feature_scorer, ordering='ablation'."""

import numpy as np
import pytest
import torch
from timm.models import VisionTransformer

import mesoslide as ms
from mesoslide.tools import summarize_token_ablation, token_ablation
from mesoslide.tools._model_stage import ImageModelStage
from mesoslide.tools.segmenters import TokenClusterer, fit_token_clusterer
from mesoslide.tools.sparse_coding import (
    LocalityConstrainedCoding,
    MiniBatchDictionaryCoding,
    SparseAutoencoder,
)
from mesoslide.tools.sparse_coding._feature_scorer import feature_column_index, feature_scorer
from tests.conftest import TILE_PX


class TinyTimmViT:
    """A real (tiny, random-weight) timm ViT with the lazyslide ViT model surface.

    `mean_patch=True` pools like Virchow/H-optimus: CLS concatenated with the
    mean of the patch tokens.
    """

    name = "tiny-vit"

    def __init__(self, mean_patch=False):
        torch.manual_seed(0)
        self.model = VisionTransformer(
            img_size=TILE_PX, patch_size=TILE_PX // 2, embed_dim=8, depth=2,
            num_heads=2, num_classes=0,
        ).eval()
        self.grid_size = tuple(self.model.patch_embed.grid_size)
        self.patch_size = tuple(self.model.patch_embed.patch_size)
        self.num_prefix_tokens = self.model.num_prefix_tokens
        self.mean_patch = mean_patch

    def to(self, device):
        self.model.to(device)
        return self

    def try_compile(self, **kwargs):
        pass

    def get_transform(self):
        return lambda x: x.float() / 255

    def encode_image_dense(self, x):
        from lazyslide_models.base import DenseTokens

        out = self.model.forward_features(x)
        return DenseTokens(cls_token=out[:, 0], patch_tokens=out[:, self.num_prefix_tokens:])

    def encode_image(self, x):
        if self.mean_patch:
            dense = self.encode_image_dense(x)
            return torch.cat([dense.cls_token, dense.patch_tokens.mean(1)], dim=-1)
        return self.model(x)


def _images(n=4, seed=0):
    return torch.as_tensor(np.random.default_rng(seed).integers(0, 256, (n, 3, TILE_PX, TILE_PX), dtype=np.uint8))


@pytest.mark.parametrize("mean_patch", [False, True])
def test_keeping_all_tokens_matches_plain_encoding(mean_patch):
    stage = ImageModelStage(TinyTimmViT(mean_patch), device="cpu")
    images = _images()
    keep_all = np.tile(np.arange(4), (len(images), 1))
    torch.testing.assert_close(stage.encode_kept_tokens(images, keep_all), stage(images))


def test_token_dropping_equals_attention_masking_for_cls_pooling():
    model = TinyTimmViT()
    stage = ImageModelStage(model, device="cpu")
    images = _images(2)
    keep = np.array([[0, 2, 3], [1, 2, 3]])
    dropped = stage.encode_kept_tokens(images, keep)

    key_mask = torch.ones(2, 5, dtype=torch.bool)
    key_mask[0, 1 + 1] = False   # patch token 1 removed from image 0 (after the CLS token)
    key_mask[1, 1 + 0] = False   # patch token 0 removed from image 1
    with torch.inference_mode():
        feats = model.model.forward_features(images.float() / 255, attn_mask=key_mask[:, None, None, :])
    torch.testing.assert_close(dropped, model.model.forward_head(feats), atol=1e-5, rtol=1e-5)


def test_mean_patch_pooling_averages_only_kept_tokens():
    model = TinyTimmViT(mean_patch=True)
    stage = ImageModelStage(model, device="cpu")
    images = _images(1)
    out = stage.encode_kept_tokens(images, np.array([[0, 3]]))
    dense = ImageModelStage(model, dense=True, device="cpu")
    # Mean part equals the mean of the two patch tokens produced when only they are kept.
    vit = model.model
    original = vit.patch_drop
    from mesoslide.tools._model_stage import _KeepPatchTokens
    vit.patch_drop = _KeepPatchTokens(torch.tensor([[0, 3]]), vit.num_prefix_tokens)
    try:
        kept_tokens = dense(images)
    finally:
        vit.patch_drop = original
    torch.testing.assert_close(out[0, 8:], kept_tokens[0].mean(0))


def test_token_ablation_rows_and_controls():
    images = _images(3)
    labels = np.array([[0, 0, 1, 2], [1, 1, 1, 1], [2, 0, 2, 0]])  # patch 1: one label covers all tokens
    results = token_ablation(images, labels, TinyTimmViT(), lambda E: E[:, 0],
                             full_scores=[1.0, 2.0, 3.0], device="cpu", progress_bar=False)
    assert sorted(zip(results["patch"], results["label"])) == [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]
    np.testing.assert_allclose(results["excess_drop"],
                               results["score_random_removed"] - results["score_label_removed"])
    assert results.loc[results["patch"] == 2, "full_score"].eq(3.0).all()
    assert results.loc[results["patch"] == 0, "n_tokens"].tolist() == [2, 1, 1]

    summary = summarize_token_ablation(results)
    assert set(summary.index) == {0, 1, 2}
    assert summary.loc[0, "n_patches"] == 2
    assert {"excess_drop", "wilcoxon_p", "full_score"} <= set(summary.columns)


def test_feature_column_index():
    assert feature_column_index("UNI_SAE_41985") == 41985
    assert feature_column_index("UNI_embedding_llc_7") == 7
    with pytest.raises(ValueError, match="column index"):
        feature_column_index("sparse_score")


def _fitted_sparse_models(dim=8):
    X = np.random.default_rng(0).standard_normal((64, dim)).astype(np.float32)
    sae = SparseAutoencoder(expansion_factor=2, batch_size=16, num_steps=20, random_state=0)
    sae.fit(X, device="cpu")
    llc = LocalityConstrainedCoding(n_codewords=16, batch_size=16, random_state=0).fit(X, device="cpu")
    dl = MiniBatchDictionaryCoding(n_components=16, alpha=0.1, batch_size=16, max_iter=2, n_jobs=1,
                                   random_state=0).fit(X)
    return X, {"sae": sae, "llc": llc, "dl": dl}


@pytest.mark.parametrize("kind", ["sae", "llc", "dl"])
def test_feature_scorer_matches_transform_column(kind):
    X, models = _fitted_sparse_models()
    model = models[kind]
    full = model.transform(X, device="cpu", progress_bar=False).toarray()
    score = feature_scorer(model, "prefix_3", device="cpu")
    np.testing.assert_allclose(score(X), full[:, 3], rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError, match="features"):
        feature_scorer(model, f"prefix_{model.embed_dim_}")


def test_sae_transform_uses_encoder_only_with_identical_codes():
    X, models = _fitted_sparse_models()
    sae = models["sae"]
    with torch.no_grad():
        _, h = sae.model_.forward(torch.tensor(X) * sae.scale_factor_)
    np.testing.assert_allclose(sae.transform(X, device="cpu", progress_bar=False).toarray(), h.numpy(),
                               rtol=1e-6, atol=1e-7)


def test_ablation_ordering_requires_pixels_not_y():
    c = TokenClusterer(ordering="ablation")
    X = np.random.default_rng(0).random((6, 4, 8))
    c.fit(X)   # KMeans only, identity order
    assert c.cluster_order_.tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="fit_order_ablation"):
        TokenClusterer(ordering="ablation").fit(X, np.arange(6.0))
    with pytest.raises(ValueError, match="fit_order_ablation"):
        c.fit_order(X, np.arange(6.0))


def test_fit_token_clusterer_ablation_with_callable_scorer(manifest, open_cohort):
    model = TinyTimmViT()
    c = fit_token_clusterer(
        manifest, "sparse_score", model,
        clusterer=TokenClusterer(n_clusters=2, ordering="ablation", name="label"),
        scorer=lambda E: E[:, 0], n_patches=6, top_fraction=0.5,
        image_slides=open_cohort, device="cpu", progress_bar=False,
    )
    assert c.name == "label" and c.feature_name_ == "sparse_score"
    assert sorted(c.cluster_order_.tolist()) == [0, 1]
    assert set(c.ablation_summary_.index) <= {0, 1}
    assert "cluster" in c.ablation_results_ and "full_score" in c.ablation_summary_
    # Canonical order: higher cluster id <-> larger mean excess drop.
    present = c.ablation_summary_["excess_drop"].sort_index()
    assert present.is_monotonic_increasing

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = c.plot_cluster_ablation()
    assert len(axes[0].collections) == len(c.ablation_summary_)   # one series per cluster present
    plt.close(fig)
    with pytest.raises(RuntimeError, match="ordering='ablation'"):
        TokenClusterer().fit(np.random.default_rng(0).random((6, 4, 8))).plot_cluster_ablation()

    # Random controls are seeded from the feature name: a refit reproduces the results.
    import zlib
    assert c.ablation_seed_ == zlib.crc32(b"sparse_score")
    again = fit_token_clusterer(
        manifest, "sparse_score", model,
        clusterer=TokenClusterer(n_clusters=2, ordering="ablation", name="label"),
        scorer=lambda E: E[:, 0], n_patches=6, top_fraction=0.5,
        image_slides=open_cohort, device="cpu", progress_bar=False,
    )
    np.testing.assert_allclose(again.ablation_results_["score_random_removed"],
                               c.ablation_results_["score_random_removed"])


@pytest.mark.parametrize("kind", ["sae", "llc", "dl"])
def test_fit_token_clusterer_ablation_routes_sparse_models_through_feature_scorer(manifest, open_cohort, kind):
    """A sparse-coding model is wrapped with feature_scorer on feature_name, so a
    feature name without a column index is rejected before any work is done."""
    _, models = _fitted_sparse_models()
    with pytest.raises(ValueError, match="column index"):
        fit_token_clusterer(
            manifest, "sparse_score", TinyTimmViT(),
            clusterer=TokenClusterer(ordering="ablation"), scorer=models[kind],
            n_patches=6, top_fraction=0.5, image_slides=open_cohort, device="cpu",
        )
