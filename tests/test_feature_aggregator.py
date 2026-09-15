"""Tests for grouping sparse-coding features into per-group .obs scores
(mesoslide.tools.sparse_coding._feature_aggregator).
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from mesoslide.tools.sparse_coding._feature_aggregator import (
    FeatureGroupAggregator,
    aggregate_feature_groups,
)

PREFIX = "UNI_SAE"


def _make_table(X, n_obs=None):
    """A minimal patch table: sparse .X with var_names 'UNI_SAE_{i}'."""
    X = sp.csr_matrix(np.asarray(X, dtype=np.float32))
    n_obs = n_obs or X.shape[0]
    var_names = [f"{PREFIX}_{i}" for i in range(X.shape[1])]
    var = pd.DataFrame(index=var_names)
    obs = pd.DataFrame({"tile_id": np.arange(n_obs)})
    return ad.AnnData(X=X, var=var, obs=obs)


@pytest.fixture
def two_slides():
    # Feature 0 and 1 -> group 1, feature 2 -> group 2.
    slide_a = _make_table([[1.0, 4.0, 2.0],
                            [2.0, 0.0, 0.0]])
    slide_b = _make_table([[3.0, 1.0, 6.0],
                            [0.0, 2.0, 3.0],
                            [4.0, 3.0, 0.0]])
    return {"a": slide_a, "b": slide_b}


@pytest.fixture
def feature_to_group():
    return {0: 1, 1: 1, 2: 2}


class TestFit:
    def test_rejects_empty_mapping(self, two_slides):
        with pytest.raises(ValueError, match="non-empty"):
            FeatureGroupAggregator().fit(two_slides, PREFIX, {})

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="method must be"):
            FeatureGroupAggregator(method="mean")

    def test_builds_inverse_index(self, two_slides, feature_to_group):
        agg = FeatureGroupAggregator().fit(two_slides, PREFIX, feature_to_group)
        assert agg.group_to_features_ == {1: [0, 1], 2: [2]}

    def test_global_max_is_cohort_wide(self, two_slides, feature_to_group):
        agg = FeatureGroupAggregator(method="max").fit(two_slides, PREFIX, feature_to_group)
        # feature 0 max is 4 (slide_b row 2), feature 1 max is 4 (slide_a row 0),
        # feature 2 max is 6 (slide_b row 0).
        assert agg.global_max_ == {0: 4.0, 1: 4.0, 2: 6.0}

    def test_sum_does_not_normalize_by_default(self, two_slides, feature_to_group):
        agg = FeatureGroupAggregator(method="sum").fit(two_slides, PREFIX, feature_to_group)
        assert agg.normalize is False
        assert agg.global_max_ is None

    def test_transform_before_fit_raises(self, two_slides):
        with pytest.raises(RuntimeError, match="Call fit"):
            FeatureGroupAggregator().transform(two_slides)


class TestTransform:
    def test_max_aggregate_matches_manual_computation(self, two_slides, feature_to_group):
        aggregate_feature_groups(two_slides, PREFIX, feature_to_group,
                                  method="max", group_prefix="group", progress=False)

        # Global max per feature: f0=4, f1=4, f2=6 (see TestFit.test_global_max_is_cohort_wide).
        # normalized scores per row, per feature {0, 1, 2}
        table_a = two_slides["a"]
        norm_a = np.array([[1.0 / 4, 4.0 / 4, 2.0 / 6],
                            [2.0 / 4, 0.0 / 4, 0.0 / 6]])
        assert np.allclose(table_a.obs["group_1"], norm_a[:, [0, 1]].max(axis=1))
        assert np.allclose(table_a.obs["group_2"], norm_a[:, 2])

        table_b = two_slides["b"]
        norm_b = np.array([[3.0 / 4, 1.0 / 4, 6.0 / 6],
                            [0.0 / 4, 2.0 / 4, 3.0 / 6],
                            [4.0 / 4, 3.0 / 4, 0.0 / 6]])
        assert np.allclose(table_b.obs["group_1"], norm_b[:, [0, 1]].max(axis=1))
        assert np.allclose(table_b.obs["group_2"], norm_b[:, 2])

    def test_sum_without_normalization(self, two_slides, feature_to_group):
        aggregate_feature_groups(two_slides, PREFIX, feature_to_group,
                                  method="sum", group_prefix="group", progress=False)

        table_a = two_slides["a"]
        assert np.allclose(table_a.obs["group_1"], [1.0 + 4.0, 2.0 + 0.0])
        assert np.allclose(table_a.obs["group_2"], [2.0, 0.0])

    def test_single_table_input_has_no_slide_id(self, feature_to_group):
        table = _make_table([[1.0, 2.0, 3.0]])
        results = aggregate_feature_groups(table, PREFIX, feature_to_group,
                                            method="max", progress=False)
        assert list(results.keys()) == [None]
        assert "group_1" in table.obs and "group_2" in table.obs

    def test_missing_feature_raises_key_error(self, two_slides):
        with pytest.raises(KeyError):
            aggregate_feature_groups(two_slides, PREFIX, {99: 1}, progress=False)

    def test_write_requires_wsidata(self, two_slides):
        with pytest.raises(TypeError, match="WSIData"):
            FeatureGroupAggregator().fit_transform(
                two_slides, PREFIX, {0: 1}, write=True, progress=False,
            )
