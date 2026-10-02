"""read_annotations, tile_coverage and label_tiles."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

import mesoslide as ms


def _points(poly):
    return " ".join(f"{x},{y}" for x, y in poly.exterior.coords[:-1])


@pytest.fixture
def point_csv(tmp_path):
    """OMERO-style export: names in 'Text', an unhelpful 'Name', vertices in 'all_points'."""
    rows = [("LA1", box(0, 0, 10, 20)), ("GC1", box(5, 5, 6, 6)), ("LA2", box(30, 30, 40, 40))]
    df = pd.DataFrame({"Name": ["undefined"] * 3, "Text": [r[0] for r in rows],
                       "all_points": [_points(r[1]) for r in rows]})
    path = tmp_path / "ann.csv"
    df.to_csv(path, index=False)
    return path


def test_point_csv_names_filter_and_scale(point_csv):
    gdf = ms.read_annotations(point_csv, name_filter="LA", scale=(2, 4))
    assert list(gdf["name"]) == ["LA1", "LA2"]
    assert gdf.geometry.iloc[0].bounds == (0, 0, 20, 80)


def test_swap_xy(point_csv):
    gdf = ms.read_annotations(point_csv, name_filter="LA1", swap_xy=True)
    assert gdf.geometry.iloc[0].bounds == (0, 0, 20, 10)


def test_callable_name_filter(point_csv):
    gdf = ms.read_annotations(point_csv, name_filter=lambda names: names.str.endswith("1"))
    assert list(gdf["name"]) == ["LA1", "GC1"]


def test_wkt_csv_and_explicit_name_col(tmp_path):
    path = tmp_path / "wkt.csv"
    pd.DataFrame({"label": ["a", "b"], "geometry": [box(0, 0, 1, 1).wkt, box(2, 2, 3, 3).wkt]}).to_csv(path, index=False)
    gdf = ms.read_annotations(path, name_col="label")
    assert list(gdf["name"]) == ["a", "b"] and len(gdf) == 2


def test_geojson_repair_and_largest_component(tmp_path):
    bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])  # self-intersecting
    multi = MultiPolygon([box(0, 0, 1, 1), box(5, 5, 9, 9)])
    path = tmp_path / "ann.geojson"
    gpd.GeoDataFrame({"name": ["GC1", None]}, geometry=[bowtie, multi]).to_file(path, driver="GeoJSON")

    gdf = ms.read_annotations(path)
    assert gdf.geometry.is_valid.all()
    assert list(gdf["name"]) == ["GC1", ""]
    assert gdf.geometry.iloc[0].area == pytest.approx(50)
    largest = ms.read_annotations(path, largest_component=True)
    assert largest.geometry.iloc[1].equals(box(5, 5, 9, 9))


def test_unknown_layout_is_reported(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="CSV layout"):
        ms.read_annotations(path)


class TestLabelTiles:
    @staticmethod
    def _tiles(one_slide):
        return ms.table_tile_geometries(one_slide)

    def test_coverage_of_known_boxes(self, one_slide):
        tiles = self._tiles(one_slide)
        x0, y0, x1, y1 = tiles.iloc[0].bounds
        w, h = x1 - x0, y1 - y0
        polys = [
            box(x0, y0, x0 + 0.3 * w, y1),            # 30% of tile 0
            box(x0 + 0.2 * w, y0, x0 + 0.5 * w, y1),  # overlaps the first: union is 50%
        ]
        cov = ms.tl.tile_coverage(one_slide, polys)
        assert cov[0] == pytest.approx(0.5)
        assert np.all((cov >= 0) & (cov <= 1))
        assert ms.tl.label_tiles(one_slide, polys, coverage_threshold=0.1)[0]
        assert not ms.tl.label_tiles(one_slide, polys, coverage_threshold=0.5)[0]  # strictly greater

    def test_whole_tile_cover_and_obs_column(self, one_slide):
        tiles = self._tiles(one_slide)
        gdf = gpd.GeoDataFrame(geometry=[tiles.iloc[3].buffer(1)])
        labels = ms.tl.label_tiles(one_slide, gdf, coverage_threshold=0.5, key_added="pos")
        assert labels[3]
        assert np.array_equal(one_slide.tables["tiles_table"].obs["pos"].to_numpy(), labels)

    def test_no_polygons_gives_no_positives(self, one_slide):
        assert not ms.tl.label_tiles(one_slide, [], coverage_threshold=0.1).any()

    def test_follows_instance_key_not_row_order(self, one_slide):
        """Table rows reordered relative to the tiles must still get their own tile's label."""
        table = one_slide.tables["tiles_table"]
        tiles = self._tiles(one_slide)
        target = tiles.iloc[5]
        expected_id = table.obs["tile_id"].iloc[5]
        perm = np.random.default_rng(0).permutation(table.n_obs)
        one_slide.tables["tiles_table"] = table[perm].copy()
        labels = ms.tl.label_tiles(one_slide, [target], coverage_threshold=0.9)
        reordered_ids = one_slide.tables["tiles_table"].obs["tile_id"].to_numpy()
        assert labels.sum() == 1
        assert reordered_ids[labels][0] == expected_id
