# Test Suite Analysis

**Analysis date:** 2026-08-15  
**Reference:** [Allen Institute Confluence, "Testing"](https://alleninstitute.atlassian.net/wiki/spaces/IT/pages/1667334178/Testing)

## Classification

The suite most closely fits the document's **Testing Pyramid** strategy:

- A large base of fast unit and schema-contract tests.
- A smaller but meaningful integration layer using real Delta tables, Parquet files,
  filesystem configuration, and subprocesses.
- Only a thin command-level end-to-end layer.

It is not a testing diamond or honeycomb because integration tests do not dominate and the
project is not a microservice architecture. It is not a testing trophy because static analysis
is neither green nor automatically enforced. It also avoids the testing ice cone and testing
crab anti-patterns because it does not rely primarily on manual, visual, or end-to-end testing.

## What Is Present

- 169 tests across 14 test modules.
- Clean pytest collection with no import errors or deselected tests.
- All 169 tests passed in 6.64 seconds during the analysis.
- No skipped or xfailed tests, and no broad `pytest.raises(Exception)` assertions.
- Extensive generated-model and schema validation covering required fields, enums, field types,
  mappings, and public API contracts.
- Strong writer coverage for idempotency, project isolation, batch validation, dry runs, output
  overrides, and known regressions.
- A parameterized round-trip smoke test for every registered writable model class.
- Integration tests that write and reread real Delta tables.
- Parquet integration tests that create and load real Parquet files.
- CLI subprocess tests for help, package information, bundle creation, and invalid commands.
- Shared fixtures that isolate the working directory, environment, settings cache, and output
  paths between tests.
- Registry and fixture drift tests that fail when writable models and their test factories fall
  out of sync.

## What Is Missing

- No CI workflow was present to run tests automatically.
- No coverage dependency, report, or enforced threshold was configured, so the document's
  suggested 70-90% range could not be assessed.
- `io/arrow_utils.py` was checked only for importability and callability. Its conversion,
  normalization, schema construction, metadata, and NumPy-to-Arrow type-mapping behavior lacked
  direct tests.
- `walk_ancestors` had no behavioral tests.
- The Parquet loader had only a basic happy path and missing-required-field case. Alias mapping,
  list coercion, object-reference resolution, unresolved references, non-strict mode, and
  `max_errors` behavior were untested.
- The CLI `validate` and `etl-brain-regions` commands had no behavioral tests.
- No complete user workflow test covered source data through model conversion and persisted
  output.
- No performance, load, security, or other non-functional tests were present.
- Ruff and mypy were configured, but no automated enforcement was present. Ruff reported 237
  findings when run against `src` and `tests`; many came from generated models, but the results
  also included import-order and unused-variable findings. Therefore, static analysis did not
  constitute a reliable testing-trophy layer.

## Stale-Test Assessment

No test was demonstrably stale: every test collected and passed, and repository history showed
that the write-path tests had been updated recently. The following tests nevertheless deserved
maintenance review:

- `test_write_relocation.py` permanently guards against shim modules removed during an earlier
  migration. The checks remained valid, but could be retired once compatibility with those
  historical import paths is no longer relevant.
- `test_config.py::test_explicit_settings_wins_over_env_and_file` tested a locally defined fake
  `writer()` precedence pattern rather than production writer behavior. It could remain green if
  the real precedence behavior regressed.
- `test_projection_schema.py::test_region_coverage_on_pmm` showed that a manually selected
  coverage list was a subset of a manually selected region list, but did not prove that the model
  enforces that relationship.

## Conclusion

The repository had a healthy, fast **Testing Pyramid** with especially strong schema and writer
coverage and a useful real-storage integration layer. The highest-value missing pieces were CI
enforcement, measurable coverage, direct Arrow utility tests, deeper Parquet loader cases, and
behavioral coverage for the remaining CLI commands. The suite contained no confirmed stale or
disabled tests, although three low-value or transitional checks warranted periodic review.