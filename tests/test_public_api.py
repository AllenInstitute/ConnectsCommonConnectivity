"""Lock the curated public surface of ``connects_common_connectivity.io``.

The public API is whatever ``__all__`` says — nothing more, nothing less.
"""

from __future__ import annotations

import importlib

import connects_common_connectivity.io as io_mod

EXPECTED = {
    "get_settings",
    "Settings",
    "write_models",
    "write_projection_matrix",
    "WrittenResult",
    "WRITABLE_CLASSES",
    "DatasetReader",
    "read_synapse_table",
}


def test_all_exact_set():
    """The IO public export list must match the curated API exactly."""
    assert set(io_mod.__all__) == EXPECTED


def test_all_resolves_to_non_none_objects():
    """Every declared IO export must resolve to an object."""
    for name in io_mod.__all__:
        obj = getattr(io_mod, name)
        assert obj is not None, f"io.{name} resolved to None"


def test_no_private_names_in_all():
    """The IO public export list must not contain private names."""
    for name in io_mod.__all__:
        assert not name.startswith("_"), f"private name {name!r} in __all__"


def test_each_name_imports_cleanly():
    """Every curated IO export must remain available after module reload."""
    mod = importlib.reload(io_mod)
    for name in EXPECTED:
        assert hasattr(mod, name), f"io.{name} missing"


def test_internal_modules_not_re_exported():
    """Internal IO modules must not leak into the public export list."""
    # arrow_utils / write_utils / write_spec / writers are accessible as
    # submodules (they're real modules) but their names must not leak into
    # io.__all__.
    forbidden = {"arrow_utils", "write_utils", "write_spec", "writers"}
    assert forbidden.isdisjoint(set(io_mod.__all__))
