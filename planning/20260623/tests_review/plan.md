# Tests Review — Implementation Plan

Five sequential work packages, implemented in order on the same execution track (no PR slicing).

---

## Work Package 1 — `conftest.py` foundation (enables everything else)

**Goal:** kill duplication and enforce stable test isolation (fresh cwd/env + cleared `get_settings` cache) in one shot.

```python
# tests/conftest.py
from __future__ import annotations
import pytest
from pathlib import Path
import polars as pl

import connects_common_connectivity as ccc
from connects_common_connectivity.config import Settings, get_settings
from connects_common_connectivity import models as _models_mod


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch, tmp_path):
    """Every test gets a clean cwd, no CCC_OUTPUT_ROOT, and a cleared cache."""
    monkeypatch.delenv("CCC_OUTPUT_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def models() -> dict:
    """Generate pydantic models once per session (expensive)."""
    return ccc.generate_pydantic_models()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(output_root=tmp_path)


@pytest.fixture
def read_delta():
    def _read(path) -> pl.DataFrame:
        return pl.read_delta(str(path))
    return _read
```

Then:
- delete the duplicated `_models()` from 4 schema files; switch tests to `def test_x(models):`
- delete the duplicated `settings` / `_read` from `test_writers.py`
- delete `_reset_cache_and_env` from `test_config.py` (now autouse globally)

**Decision:** keep autouse `chdir(tmp_path)` globally. In this package it is a feature, not a risk: config discovery is cwd-based and cached, so per-test cwd isolation prevents cross-test bleed. `test_write_relocation.py` is safe because `REPO_ROOT` is anchored from `__file__`, not cwd.

---

## Work Package 2 — Tighten exception assertions

**Pattern:** prefer the narrowest exception + a `match=` that names the *field or condition*, not the generic word.

`pytest.raises` signature reminder:
```python
with pytest.raises(ExpectedException, match=r"regex against str(exc)"):
    ...
```

**Concrete replacements:**

| File:line | Before | After |
|---|---|---|
| `test_basic.py:22` | `pytest.raises(Exception)` | `pytest.raises(ValidationError, match=r"project_id.*[Ff]ield required")` |
| `test_basic.py:37` | `pytest.raises(Exception)` | `pytest.raises(ValidationError, match=r"modality.*Input should be")` |
| `test_basic.py:69` | `pytest.raises(Exception)` | `pytest.raises(ValidationError, match=r"probability.*less than or equal to 1")` |
| `test_config.py:108` | `pytest.raises(Exception)` | `pytest.raises(ValidationError, match=r"output_root.*[Ff]ield required")` |
| `test_config.py:115` | `pytest.raises(Exception)` | `pytest.raises(ValidationError, match=r"[Ee]xtra inputs are not permitted")` *(verify what Settings raises first)* |
| `test_writers.py:328` | `pytest.raises(TypeError)` | `pytest.raises(TypeError, match=r"pydantic model or iterable")` |

Also add a **new** test for true registry rejection (different code path):

```python
from pydantic import BaseModel

class UnregisteredModel(BaseModel):
    id: str

with pytest.raises(KeyError, match=r"UnregisteredModel"):
    write_models(UnregisteredModel(id="u1"), settings=settings)
```

Note: this test needs `from pydantic import BaseModel` at the top of `test_writers.py`. Match on the class name rather than the exact error string — it's a more durable contract than the message text.

**Rule of thumb to leave in the package notes:**
> Never `pytest.raises(Exception)`. Always pick the narrowest class the production code raises, and always include `match=` naming the field or condition. If you don't know which exception the code raises, that's the first thing to find out — that's the contract.

For the dynamically-generated pydantic models in `test_basic.py`, import `from pydantic import ValidationError` at module top — it's the same class instance the dynamic models will raise.

---

## Work Package 3 — Failure messages on regression-critical asserts

Only add custom messages where the failure mode is non-obvious. Don't litter every assert.

**Targets:**

```python
# test_writers.py — patchseq regression
ids = sorted(rows["id"].to_list())
assert ids == ["visp_exc_patchseq", "visp_inh_patchseq"], (
    f"patchseq regression: second write wiped first. "
    f"Expected both datasets, got {ids}"
)
```

```python
# test_writers.py — idempotency, also strengthen content equality
rows = _read(settings.output_root / "dataset")
assert rows.shape[0] == 1, f"idempotent rewrite produced {rows.shape[0]} rows"
assert rows["id"].to_list() == ["d1"], "row identity changed across rewrites"
assert rows["name"].to_list() == ["example"], "row content drifted across rewrites"
```

```python
# test_write_utils.py — partial rerun
assert n == 1, f"expected only 'c' to be new; appended {n} rows"
```

```python
# test_write_validation.py — IO-never-happened check
assert not (tmp_path / "cluster").exists(), (
    "validation failure should short-circuit before any IO; "
    "cluster/ directory was created anyway"
)
```

Skip messages on simple positive assertions like `assert cfd.range_max is None` — pytest's introspection already shows the value.

---

## Work Package 4 — Coverage drift guards & list-failure tests

### 4a. `WRITABLE_CLASSES` ↔ fixture drift

Replace the if/elif tower in `_make_instance` with a registry dict + drift test:

```python
# tests/_fixtures.py  (or in conftest)
INSTANCE_FACTORIES = {
    DataSet: lambda: DataSet(id="ds1", name="ds", project_id="p1"),
    DataItem: lambda: DataItem(id="di1", name="di1", project_id="p1"),
    # ...
}

def make_instance(cls):
    try:
        return INSTANCE_FACTORIES[cls]()
    except KeyError:
        pytest.fail(
            f"No fixture for {cls.__name__}. Add an entry to "
            f"INSTANCE_FACTORIES in tests/_fixtures.py."
        )
```

```python
def test_every_writable_class_has_a_fixture():
    missing = set(WRITABLE_CLASSES) - set(INSTANCE_FACTORIES)
    assert not missing, (
        f"WRITABLE_CLASSES added entries without fixtures: "
        f"{sorted(c.__name__ for c in missing)}"
    )
    stale = set(INSTANCE_FACTORIES) - set(WRITABLE_CLASSES)
    assert not stale, (
        f"INSTANCE_FACTORIES has stale entries not in WRITABLE_CLASSES: "
        f"{sorted(c.__name__ for c in stale)}"
    )
```

This makes the drift visible as a dedicated test failure instead of a parametrized round-trip error.

### 4b. Negative-path coverage for `validate_for_write` with a list

```python
def test_validate_for_write_list_reports_failing_row():
    spec = REGISTRY["Cluster"]
    items = [
        Cluster(id="c1", hierarchy_id="h1"),
        Cluster(id="c2"),  # missing hierarchy_id
    ]
    with pytest.raises(ValueError, match=r"hierarchy_id") as ei:
        validate_for_write(items, spec)
    # row identity should appear in the error to make debugging tractable
    assert "c2" in str(ei.value), (
        f"error should name failing row; got: {ei.value}"
    )
```

(If the production code doesn't currently name the row, that's a real finding to file — the test documents the desired contract.)

---

## Work Package 5 — Plug the real coverage gaps

This is the only work package that may touch behavior beyond test infra. Split per module to keep diffs small.

### 5a. `dry_run` semantics
```python
def test_dry_run_does_not_write(tmp_path):
    settings = Settings(output_root=tmp_path, dry_run=True)
    ds = DataSet(id="d1", name="d", project_id="p1")
    result = write_models(ds, settings=settings)
    assert result.rows_written == 0, "dry_run must report 0 rows written"
    assert not (tmp_path / "dataset").exists(), "dry_run must not create tables"
```
If this fails, you've found a bug — `dry_run` exists in `Settings` but nothing checks it's honored.

### 5b. `cli.py`
This CLI is `argparse`, not Click. Use `subprocess.run([sys.executable, "-m", "connects_common_connectivity.cli", ...])`.
Cover: top-level `--help`, `info` (assert package version text appears), one happy-path command (`bundle`), one error path (bad subcommand/args → nonzero exit).

Skip `cmd_validate` and `etl-brain-regions` — both are marked `# pragma: no cover` in `cli.py` as runtime smoke commands; respect the existing exclusion.

### 5c. `parquet_loader.py`
Test the public contract of `load_parquet_to_models(...)`: write a tiny parquet, load into a concrete class (e.g. `DataItem`), assert instance count + key field values + report counts/mapping. Add one negative test where required data is missing and assert the failure is surfaced in `report["errors"]`.

### 5d. Extra escapes in `_build_predicate` (1-line additions to existing test)
```python
@pytest.mark.parametrize("value,expected_literal", [
    ("O'Hara", "'O''Hara'"),
    ("",       "''"),
    ("a\\b",   "'a\\b'"),       # backslash is not special in SQL string literals
    ("café",   "'café'"),
])
def test_build_predicate_escapes(value, expected_literal):
    assert _build_predicate(["name"], [value]) == f"name = {expected_literal}"
```

### 5e. Repo-walk hardening in `test_write_relocation.py`
```python
SEARCH_ROOTS = ["src", "tests", "code", "scripts", "planning"]

def _iter_source_files():
    for root in SEARCH_ROOTS:
        base = REPO_ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".ipynb"}:
                if not any(p in EXCLUDED_DIRS for p in path.parts):
                    yield path
```
Drops `data/`, `results/`, `scratch/`, `metadata/`, `environment/` from the walk.

---

## Sequential execution guardrails (hard stops between packages)

Do **not** start the next work package until the current package meets its guardrail.

1. **WP1 → WP2:** `conftest.py` is in place, duplicated helper fixtures are removed from target files, and settings/cache isolation behavior remains intact.
2. **WP2 → WP3:** broad `pytest.raises(Exception)` uses targeted in this plan are replaced with narrow exception types and meaningful `match=` checks.
3. **WP3 → WP4:** custom assertion messages were added only to regression-critical/non-obvious assertions (no blanket message churn).
4. **WP4 → WP5:** fixture drift guard(s) are in place and list-failure validation coverage is added; registry/fixture mismatch now fails with explicit guidance.
5. **WP5 completion:** coverage-gap tests are in place; if `dry_run` exposes a real bug, fix behavior in the same package before declaring completion.

---

## Sequencing & rollout

| Work package | Effort | Risk | Blocks |
|---|---|---|---|
| 1. conftest | 1h | low | 2, 3 |
| 2. exceptions | 30m | low | — |
| 3. messages | 30m | none | — |
| 4. drift guards | 1h | low | — |
| 5. coverage gaps | 2–4h | medium (may surface real bugs) | — |

Work packages 1–4 are pure test refactors. Work package 5 is where you'll likely find a `dry_run` bug; budget time for an immediate behavior fix in the same execution sequence.
