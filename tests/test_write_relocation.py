"""No-shim regression test.

The deprecated shim modules at the package root
(``connects_common_connectivity.arrow_utils`` and ``connects_common_connectivity.write_utils``)
were removed after the W6 notebook migration. This test pins that contract:

1. The shim modules no longer exist on disk or as importable modules.
2. No source file (package, tests, notebooks, scripts) imports from the old paths.
3. The canonical IO modules still expose the public names.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from connects_common_connectivity.io.arrow_utils import (
    attach_linkml_metadata,
    build_arrow_schema,
    build_cell_feature_matrix_schema,
    models_to_table,
)
from connects_common_connectivity.io.write_utils import (
    append_new_dataitems,
    populate_region_coverage,
    walk_ancestors,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = ["src", "tests", "code", "scripts", "planning"]

EXCLUDED_DIRS = {".venv", ".git", ".pytest_cache", ".ruff_cache",
                 ".ipynb_checkpoints", ".Trash-0", "node_modules"}

SHIM_IMPORT_PATTERN = re.compile(
    r"connects_common_connectivity\.(?:arrow_utils|write_utils)\b"
)


def test_shim_modules_deleted():
    pkg = REPO_ROOT / "src" / "connects_common_connectivity"
    assert not (pkg / "arrow_utils.py").exists(), "shim arrow_utils.py must be deleted"
    assert not (pkg / "write_utils.py").exists(), "shim write_utils.py must be deleted"


def test_shim_modules_not_importable():
    with pytest.raises(ModuleNotFoundError):
        import connects_common_connectivity.arrow_utils  # noqa: F401
    with pytest.raises(ModuleNotFoundError):
        import connects_common_connectivity.write_utils  # noqa: F401


def _iter_source_files():
    for root in SEARCH_ROOTS:
        base = REPO_ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".ipynb"}:
                if not any(part in EXCLUDED_DIRS for part in path.parts):
                    yield path


def test_no_source_references_shim_paths():
    offenders: list[tuple[Path, list[str]]] = []
    for path in _iter_source_files():
        # Skip this test file itself (it intentionally mentions the names).
        if path.resolve() == Path(__file__).resolve():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if path.suffix == ".ipynb":
            # Search only code-cell source to avoid false positives in markdown prose.
            try:
                nb = json.loads(text)
            except json.JSONDecodeError:
                continue
            lines: list[str] = []
            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "code":
                    src = cell.get("source", "")
                    if isinstance(src, list):
                        src = "".join(src)
                    lines.append(src)
            haystack = "\n".join(lines)
        else:
            haystack = text
        hits = [m.group(0) for m in SHIM_IMPORT_PATTERN.finditer(haystack)]
        if hits:
            offenders.append((path.relative_to(REPO_ROOT), hits))
    assert not offenders, (
        "Old shim paths still referenced:\n"
        + "\n".join(f"  {p}: {set(hs)}" for p, hs in offenders)
    )


def test_public_names_from_io_paths():
    # Imports are performed at module top; assert the names resolve to callables
    # so the test fails loudly if the public surface ever regresses.
    for obj in (
        attach_linkml_metadata,
        build_arrow_schema,
        build_cell_feature_matrix_schema,
        models_to_table,
        append_new_dataitems,
        populate_region_coverage,
        walk_ancestors,
    ):
        assert callable(obj)
