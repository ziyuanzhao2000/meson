"""embed_patch's table contract: what the plotting and selection paths rely on."""

import numpy as np
import pytest

import mesoslide as ms


def test_table_has_the_columns_downstream_expects(one_table):
    for col in ("tile_id", "tissue_id", "x", "y", "library_id"):
        assert col in one_table.obs.columns


def test_tile_id_dtype_matches_the_tiles_element(one_slide):
    """SpatialData matches instance_key values against the element index.

    A str/int mismatch makes the table look unrelated to its shapes, and
    spatialdata_plot then refuses to render the overlay.
    """
    table = one_slide.tables["tiles_table"]
    tiles = one_slide.shapes["tiles"]
    assert table.obs["tile_id"].dtype == tiles["tile_id"].dtype


def test_obs_index_is_str(one_table):
    """AnnData requires string obs names."""
    assert all(isinstance(i, str) for i in one_table.obs_names)


def test_obs_index_carries_no_colliding_name(one_table):
    """Regression: an index named after a column whose values differ fails on write.

    The column is int and the index is str, so if the index inherits the name
    'tile_id' anndata rejects the whole table at write time -- which only shows
    up with save=True.
    """
    assert one_table.obs.index.name != "tile_id"


def test_table_round_trips_through_a_written_store(cohort):
    """The fixtures write with save=True, so reaching this at all exercises it."""
    import ezslide

    wsi = ezslide.read_wsi(cohort[0])
    table = wsi.tables["tiles_table"]
    assert table.n_obs == len(wsi.shapes["tiles"])
    assert "stub_embedding" in table.obsm
    assert table.obsm["stub_embedding"].shape == (table.n_obs, 8)


def test_x_y_are_tile_origins(one_slide):
    """extract_patches reads from these, so they must be the level-0 top-left."""
    table = one_slide.tables["tiles_table"]
    bounds = one_slide.shapes["tiles"].bounds
    assert np.array_equal(table.obs["x"].to_numpy(), bounds["minx"].to_numpy())
    assert np.array_equal(table.obs["y"].to_numpy(), bounds["miny"].to_numpy())
