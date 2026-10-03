"""overwrite_element never leaves the element missing on disk."""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from mesoslide._utils import overwrite_element


@pytest.fixture
def store(cohort, tmp_path):
    import ezslide

    path = tmp_path / "slide.zarr"
    shutil.copytree(cohort[0], path)
    wsi = ezslide.read_slide(str(path))
    yield wsi, path


def _table_dir(path):
    return Path(path) / "tables" / "tiles_table"


def _reread_marker(path):
    import ezslide

    return ezslide.read_slide(str(path)).tables["tiles_table"].obs.get("marker")


def test_replaces_the_table_and_leaves_nothing_behind(store):
    wsi, path = store
    wsi.tables["tiles_table"].obs["marker"] = 7
    overwrite_element(wsi, "tiles_table")
    assert (_reread_marker(path) == 7).all()
    assert sorted(p.name for p in _table_dir(path).parent.iterdir() if p.is_dir()) == ["tiles_table"]


def test_failed_write_keeps_the_original(store, monkeypatch):
    wsi, path = store
    wsi.tables["tiles_table"].obs["marker"] = 7
    original = type(wsi).write_element

    def failing(self, name, *a, **k):
        if name.endswith("__tmp_overwrite"):
            raise OSError("disk full")
        return original(self, name, *a, **k)

    monkeypatch.setattr(type(wsi), "write_element", failing)
    with pytest.raises(OSError, match="disk full"):
        overwrite_element(wsi, "tiles_table")
    assert _reread_marker(path) is None  # the original table, untouched
    assert "tiles_table__tmp_overwrite" not in wsi.tables


def test_failure_between_the_renames_restores_the_original(store, monkeypatch):
    wsi, path = store
    wsi.tables["tiles_table"].obs["marker"] = 7
    real_replace, calls = os.replace, []

    def flaky(src, dst):
        calls.append(src)
        if len(calls) == 2:
            raise OSError("rename failed")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    with pytest.raises(OSError, match="rename failed"):
        overwrite_element(wsi, "tiles_table")
    monkeypatch.setattr(os, "replace", real_replace)
    assert _table_dir(path).exists()
    assert _reread_marker(path) is None


def test_recovers_a_copy_stranded_by_an_earlier_interruption(store):
    wsi, path = store
    os.replace(_table_dir(path), _table_dir(path).with_name("tiles_table__old_overwrite"))
    wsi.tables["tiles_table"].obs["marker"] = 3
    overwrite_element(wsi, "tiles_table")
    assert (_reread_marker(path) == 3).all()
    assert not _table_dir(path).with_name("tiles_table__old_overwrite").exists()


def test_undeletable_old_copy_warns_but_the_new_table_is_in_place(store, monkeypatch):
    wsi, path = store
    wsi.tables["tiles_table"].obs["marker"] = 5
    real_rmtree = shutil.rmtree

    def rmtree(p, *a, **k):
        if str(p).endswith("__old_overwrite"):
            raise OSError(39, "Directory not empty")
        return real_rmtree(p, *a, **k)

    monkeypatch.setattr(shutil, "rmtree", rmtree)
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.warns(UserWarning, match="could not delete"):
        overwrite_element(wsi, "tiles_table")
    monkeypatch.setattr(shutil, "rmtree", real_rmtree)
    assert (np.asarray(_reread_marker(path)) == 5).all()
