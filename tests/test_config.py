"""Tests for the package-wide config module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from connects_common_connectivity.config import (
    CONFIG_FILENAME,
    Settings,
    find_config_file,
    get_settings,
    output_root,
    table_path,
)


@pytest.fixture(autouse=True)
def _reset_cache_and_env(monkeypatch, tmp_path):
    """Each test runs in an isolated tmp cwd with a cleared cache and no env override."""
    monkeypatch.delenv("CCC_OUTPUT_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _write_config(dir_: Path, **values) -> Path:
    import yaml

    path = dir_ / CONFIG_FILENAME
    path.write_text(yaml.safe_dump(values))
    return path


def test_get_settings_raises_actionable_error_when_missing(tmp_path):
    # tmp_path has no ccc_config.yaml anywhere up the tree (we chdir'd into it).
    with pytest.raises(RuntimeError, match=CONFIG_FILENAME):
        get_settings()


def test_find_and_load_from_nested_cwd(tmp_path, monkeypatch):
    _write_config(tmp_path, output_root=str(tmp_path / "out"), dry_run=True)
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    get_settings.cache_clear()

    found = find_config_file()
    assert found == (tmp_path / CONFIG_FILENAME).resolve()

    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.output_root == Path(str(tmp_path / "out"))
    assert settings.dry_run is True


def test_env_overrides_only_output_root(tmp_path, monkeypatch):
    _write_config(tmp_path, output_root=str(tmp_path / "from_file"), dry_run=True)
    monkeypatch.setenv("CCC_OUTPUT_ROOT", str(tmp_path / "from_env"))
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.output_root == Path(str(tmp_path / "from_env"))
    # dry_run still comes from the file; env cannot express it.
    assert settings.dry_run is True


def test_explicit_settings_wins_over_env_and_file(tmp_path, monkeypatch):
    _write_config(tmp_path, output_root=str(tmp_path / "from_file"), dry_run=True)
    monkeypatch.setenv("CCC_OUTPUT_ROOT", str(tmp_path / "from_env"))
    get_settings.cache_clear()

    explicit = Settings(output_root=tmp_path / "explicit", dry_run=False)

    # Simulate the caller-side precedence pattern documented for writers/readers.
    def writer(settings=None):
        return settings or get_settings()

    resolved = writer(settings=explicit)
    assert resolved is explicit
    assert resolved.output_root == tmp_path / "explicit"
    assert resolved.dry_run is False


def test_table_path_joins_and_returns_path(tmp_path):
    settings = Settings(output_root=tmp_path / "root")
    p = table_path(settings, "dataset")
    assert isinstance(p, Path)
    assert p == tmp_path / "root" / "dataset"
    # A few of the canonical subdir names used by the notebooks.
    for name in (
        "dataitem",
        "dataitem_dataset_association",
        "cellfeatureset",
        "cellfeaturematrix",
        "cluster",
        "clustermembership",
        "projectionmeasurementmatrix",
    ):
        assert table_path(settings, name) == tmp_path / "root" / name


def test_output_root_is_required(tmp_path):
    _write_config(tmp_path, dry_run=False)  # missing output_root
    get_settings.cache_clear()
    with pytest.raises(Exception):
        get_settings()


def test_unknown_keys_rejected(tmp_path):
    _write_config(tmp_path, output_root=str(tmp_path), nonsense_key=1)
    get_settings.cache_clear()
    with pytest.raises(Exception):
        get_settings()


def test_io_reexports_settings_helpers():
    from connects_common_connectivity.io import (
        Settings as IOSettings,
        get_settings as io_get_settings,
        table_path as io_table_path,
    )

    assert IOSettings is Settings
    assert io_get_settings is get_settings
    assert io_table_path is table_path


def test_get_settings_is_cached(tmp_path, monkeypatch):
    _write_config(tmp_path, output_root=str(tmp_path / "out"))
    get_settings.cache_clear()
    first = get_settings()
    # Mutating the file should not change the cached result.
    _write_config(tmp_path, output_root=str(tmp_path / "changed"))
    second = get_settings()
    assert first is second
    # After clearing, discovery re-runs.
    get_settings.cache_clear()
    third = get_settings()
    assert third.output_root == Path(str(tmp_path / "changed"))


def test_describe_includes_resolved_values(tmp_path):
    settings = Settings(output_root=tmp_path / "root", dry_run=True)
    text = settings.describe()
    assert "root" in text
    assert "dry_run=True" in text


def test_output_root_helper_appends_trailing_slash(tmp_path, monkeypatch):
    _write_config(tmp_path, output_root=str(tmp_path / "out"))
    get_settings.cache_clear()
    # cwd is tmp_path (autouse fixture), so relpath of tmp_path/out is "out".
    root = output_root()
    assert isinstance(root, str)
    assert root.endswith("/")
    assert root == "out/"


def test_output_root_helper_absolute_flag(tmp_path):
    settings = Settings(output_root=tmp_path / "explicit")
    assert output_root(settings, absolute=True) == str(tmp_path / "explicit") + "/"


def test_output_root_helper_accepts_explicit_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    explicit = Settings(output_root=tmp_path / "explicit")
    # Default returns path relative to cwd (tmp_path).
    assert output_root(explicit) == "explicit/"


def test_relative_output_root_in_config_is_anchored_at_config_dir(tmp_path, monkeypatch):
    # Config sits at tmp_path; output_root is relative ("scratch/x/").
    _write_config(tmp_path, output_root="scratch/x/")
    nested = tmp_path / "code"
    nested.mkdir()
    monkeypatch.chdir(nested)
    get_settings.cache_clear()

    settings = get_settings()
    # Settings.output_root is absolute, anchored at the config file's dir
    # (abspath, not resolve — symlinks must not be followed).
    assert settings.output_root == Path(os.path.abspath(tmp_path / "scratch" / "x"))

    # output_root() returns the path relative to cwd → "../scratch/x/".
    assert output_root() == "../scratch/x/"

    # table_path joins to an absolute path that works regardless of cwd.
    tp = table_path(settings, "dataset")
    assert tp.is_absolute()
    assert tp == Path(os.path.abspath(tmp_path / "scratch" / "x" / "dataset"))
