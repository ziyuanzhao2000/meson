"""The memory properties the streaming design exists for.

If these fail, something started concatenating a cohort that should have been
streamed -- the design is still "correct" but has lost the point.

The fixture slides are intentionally tiny (~400 tiles each), so fixed overhead
dominates any single measurement here; these tests check *growth rate* against
cohort size, not absolute streamed-vs-concatenated numbers, since the latter is
only unambiguous at realistic slide sizes (verified separately against the
project's ~48k-tile stores: streamed selection grew 1.34x from 1 to 8 slides
where concatenation would have grown ~8x).
"""

import gc
import tracemalloc

import numpy as np
import pandas as pd
import pytest

import mesoslide as ms


def peak_mb(fn):
    gc.collect()
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1] / 1e6
    finally:
        tracemalloc.stop()
        gc.collect()


@pytest.fixture(scope="module")
def wide_manifest(cohort):
    """The same slides repeated under distinct ids, to grow cohort size cheaply."""
    def build(n):
        stores = (cohort * ((n // len(cohort)) + 1))[:n]
        return pd.DataFrame({"store": stores,
                             "slide_id": [f"s{i:02d}" for i in range(n)]})
    return build


def test_selection_peak_grows_far_slower_than_the_data(wide_manifest):
    """Peak must be sublinear in cohort size, not proportional to it."""
    peak_mb(lambda: ms.select_top_patches(wide_manifest(1), "score", n=10))  # warm up
    one = peak_mb(lambda: ms.select_top_patches(wide_manifest(1), "score", n=10))
    nine = peak_mb(lambda: ms.select_top_patches(wide_manifest(9), "score", n=10))
    growth = nine / one
    assert growth < 9, (
        f"selection peak grew {growth:.1f}x while the data grew 9x; "
        "that is proportional, so the cohort is being held rather than streamed"
    )


def test_concat_with_obsm_does_grow(wide_manifest):
    """The contrast that justifies iter_slides being the default."""
    peak_mb(lambda: ms.concat_slides(wide_manifest(1), obsm_keys=["stub_embedding"]))
    one = peak_mb(lambda: ms.concat_slides(wide_manifest(1), obsm_keys=["stub_embedding"]))
    many = peak_mb(lambda: ms.concat_slides(wide_manifest(9), obsm_keys=["stub_embedding"]))
    assert many > one, "concat_slides should scale with the cohort; it materialises it"


def test_selection_output_obsm_scales_with_the_selection(manifest):
    """Embeddings ride along for selected rows only -- never the whole cohort."""
    small = ms.select_top_patches(manifest, "score", n=5)
    large = ms.select_top_patches(manifest, "score", n=25)
    assert small.obsm["stub_embedding"].shape[0] == 5
    assert large.obsm["stub_embedding"].shape[0] == 25


def test_sae_streaming_peak_is_flat_in_cohort_size():
    """Accumulators are (n_features, n_features); slide count must not enter."""
    import anndata as ad
    import scipy.sparse as sp

    from mesoslide.tools.sparse_coding import FeatureClusterer

    n_feat = 30

    def parts(n_slides):
        out = []
        for i in range(n_slides):
            X = sp.random(400, n_feat, density=0.05, format="csr", random_state=i)
            a = ad.AnnData(X=X)
            a.var_names = [f"UNI_SAE_{j}" for j in range(n_feat)]
            out.append(a)
        return out

    idx = np.arange(n_feat)
    fit = lambda n: FeatureClusterer().compute_iou(parts(n), "UNI_SAE", idx, progress=False)
    peak_mb(lambda: fit(1))  # warm up numba
    one, many = peak_mb(lambda: fit(1)), peak_mb(lambda: fit(8))
    assert many < one * 3, f"streamed IoU peak grew {many / one:.1f}x from 1 to 8 slides"
