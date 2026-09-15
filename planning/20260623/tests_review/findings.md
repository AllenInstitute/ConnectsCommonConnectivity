# Tests Review — Findings

Review of `tests/` on branch `ingestion-v2` (12 files, ~1,540 LOC).

## 🔴 High priority

1. **No `conftest.py`.** Shared helpers are duplicated across files: `_models()` is redefined in 4 schema files; `_make_table`, `_read`, the `settings` / `tmp_path` fixtures appear ad-hoc. Promote `settings`, `_read`, `_models`, `_make_instance` to `tests/conftest.py` as fixtures. Will shrink the suite and stop drift.

2. **`pytest.raises(Exception)` is too broad** in `test_basic.py` (lines 22, 37, 69) and `test_config.py` (108, 115). It will pass on completely unrelated failures (ImportError, TypeError from a refactor). Use `ValidationError` / `RuntimeError` with `match=` like the other schema tests already do.

3. **Cross-test cache pollution risk.** `get_settings` is `lru_cache`d but only `test_config.py` clears it (via autouse fixture). If any other test imports `get_settings` first, later config tests can flake. Move the `_reset_cache_and_env` autouse fixture into `conftest.py` so it runs for every test.

4. **`test_no_source_references_shim_paths` walks `REPO_ROOT.rglob("*")`** including `data/`, `results/`, `scratch/`, `metadata/`, `.venv` siblings, etc. It's slow and brittle. Either restrict to `{src, tests, code, scripts, planning}` or add those large dirs to `EXCLUDED_DIRS`. Also worth caching the file list.

5. **`test_round_trip_each_writable_class` silently skips coverage drift.** If someone adds a class to `WRITABLE_CLASSES` and forgets to extend `_make_instance`, the test raises `AssertionError("no fixture for …")` — which *looks* like a test failure but doesn't tell you the spec is missing. Convert the `raise AssertionError` to `pytest.fail("…add a fixture in _make_instance")`, or better, register per-class fixtures in a dict and assert `set(fixtures) == set(WRITABLE_CLASSES)` as its own test.

## 🟡 Medium priority

6. **Assertion failure messages are mostly bare.** Examples:
   - `test_patchseq_regression_two_datasets_same_project` → `assert ids == [...]` with no `, f"…"` message. When this fails in CI, you'll get `AssertionError: assert ['x'] == ['visp_exc_patchseq','visp_inh_patchseq']` and nothing about which write was lost. Add messages like `f"second write wiped first; remaining ids={ids}"`.
   - `test_first_write_appends_all`, `test_idempotent_partial_rerun`: same — a custom message naming the scenario would speed debugging by months over time.

7. **Lots of inline `import` statements** (`test_basic.py` imports `pytest` and `ccc` inside every test, `test_write_validation.test_write_models_calls_validation_before_io` imports `Settings` and `write_models` inside the function). Lift to module top for consistency with the rest of the suite.

8. **`test_enum_validation` and `test_projection_measurement_matrix_laterality` use overly permissive assertions:** `assert str(ds.modality) in {Modality.TRACER.value, Modality.TRACER.name, str(Modality.TRACER)}`. That comment says "depending on dynamic generation" — pin it. If the schema can return three things, the schema isn't deterministic and *that* is the bug; if it's deterministic, assert exactly one.

9. **No negative test for `validate_for_write` with a `list` containing a bad row.** `test_validate_for_write_accepts_a_list` covers the happy path; add a counterpart that passes `[good, bad]` and asserts the error names *which row* failed.

10. **`test_write_models_rejects_unregistered_class`** uses `pytest.raises(TypeError)` without `match=`. Add `match="WRITABLE_CLASSES"` or similar so a misleading TypeError from elsewhere doesn't false-positive.

11. **`test_describe_includes_resolved_values`** asserts substring `"root"` which trivially matches the path. Strengthen: assert `str(settings.output_root)` is in the output verbatim.

12. **Idempotency assertion in `test_overwrite_scoped_is_idempotent`** checks only row count, not row equality. If the writer silently overwrites with wrong content, the test passes. Read back and assert the row matches `ds`.

## 🟢 Low priority / polish

13. **Naming consistency.** Some files use `def _models()` factory, others import directly from `connects_common_connectivity.models`. Pick one — preferably the direct import, since `generate_pydantic_models()` is re-invoked on every test and is presumably expensive.

14. **`test_basic.py` is a grab bag** (imports, model generation, enum, required field, multivalued, bounds). Split into `test_import.py` + fold the rest into the topical schema files that already exist.

15. **No markers / no test plan.** Consider `pytest.mark.slow` for the full per-class round-trip and the repo-walk shim test. Speeds local TDD.

16. **`test_write_relocation.py` test name is misleading** — it's about shim removal, not relocation. Rename to `test_no_shim_imports.py`.

17. **Missing coverage:**
    - No tests for `cli.py` (the `ccc` entry point).
    - No tests for `parquet_loader.py`.
    - No tests for `dry_run=True` actually being honored by `write_models` (config has the flag; writer behavior under it is untested).
    - No concurrent-write / locking behavior for delta tables, even a basic sanity test.
    - `_build_predicate_escapes_single_quotes` covers `'` — also test backslash, empty string, and unicode.

18. **`test_io_reexports_settings_helpers`** asserts identity (`is`) which is fine, but the same pattern in `test_public_api` uses `hasattr`. Pick one approach for re-export tests.

## ✅ What's working well

- **Excellent module-level docstrings** stating *why* the test exists (`test_writers.py`, `test_write_validation.py`, `test_write_relocation.py`, `test_public_api.py`). Keep doing this.
- **Headline regression test** (`test_patchseq_regression_two_datasets_same_project`) is exactly right — named for the bug, documents the prior failure mode in its docstring.
- **Parametrization over the registry** in `test_write_spec.py` is the right shape — it auto-grows with new entries.
- **`extra="forbid"` enforcement test** (`test_cluster_rejects_project_id`) prevents silent schema breakage. Good.
- Strong **regex `match=` usage** in schema tests catches the right error *and* the right field.
