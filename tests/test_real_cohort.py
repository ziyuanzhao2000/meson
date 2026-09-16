"""Optional: exercise the same paths against the project's real TB cohort.

Skipped unless MESOSLIDE_TEST_STORES names a glob of real .zarr stores, e.g.:

    MESOSLIDE_TEST_STORES='/n/scratch/.../data/*.zarr' pytest tests/test_real_cohort.py

The synthetic-fixture suite is what runs by default and in CI; this file exists
so the design's memory claims can be re-checked against realistic slide sizes
(the project's stores run ~48k tiles / 1024-d embeddings, vs. ~400 tiles in the
synthetic fixtures) without making that dependency part of the default run.
"""

import gc
import tracemalloc

import numpy as np
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


def test_selection_matches_a_full_sort(real_manifest):
    import pandas as pd

    every = []
    for sid, wsi in ms.iter_slides(real_manifest):
        t = wsi.tables["tiles_table"]
        every.append(t.obs[["y"]].assign(slide_id=sid))
    every = pd.concat(every)
    expected = every.sort_values("y", ascending=False).head(10)["y"].to_numpy()

    got = ms.select_top_patches(real_manifest, "y", n=10).obs["_feature_score"].to_numpy()
    assert np.allclose(got, expected)


def test_selection_peak_stays_within_a_small_multiple_of_one_slide(real_manifest):
    """The claim this whole design rests on, at realistic slide sizes.

    A single slide's tile table plus its selected-row copy is the honest lower
    bound; anything growing near-linearly with cohort size means something
    concatenated a whole cohort's embeddings instead of streaming them.
    """
    n = len(real_manifest)
    if n < 2:
        pytest.skip("need at least 2 slides to compare growth")

    peak_mb(lambda: ms.select_top_patches(real_manifest.head(1), "y", n=50))  # warm up
    one = peak_mb(lambda: ms.select_top_patches(real_manifest.head(1), "y", n=50))
    all_ = peak_mb(lambda: ms.select_top_patches(real_manifest, "y", n=50))

    growth = all_ / one
    assert growth < n / 2, (
        f"selection peak grew {growth:.2f}x over {n} slides; expected roughly "
        "flat growth, not proportional to cohort size"
    )


def test_extraction_matches_read_region(real_manifest):
    sel = ms.select_random_patches(real_manifest, 5, random_state=0)
    slides = ms.open_slides(real_manifest)
    try:
        imgs = ms.pp.extract_patch_images(sel, slides, channel_first=False, progress_bar=False)
        row = sel.obs.iloc[0]
        wsi = slides[row["slide_id"]]
        spec = wsi.tile_spec("tiles")
        ref = wsi.read_region(int(row.x), int(row.y), spec.width, spec.height)
        assert np.array_equal(imgs[0], ref)
    finally:
        for w in slides.values():
            w.close()
