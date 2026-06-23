# PR Message

Implemented the full `planning/tests_review/plan.md` sequence (WP1 to WP5) end-to-end, with package-by-package verification gates in order.

- Added shared test foundations in `tests/conftest.py` (settings/cache/cwd isolation + shared fixtures) and removed duplicated helpers across tests.
- Tightened exception assertions to specific exception classes with meaningful `match=` checks.
- Added high-signal regression assertion messages where failures are otherwise hard to diagnose.
- Added fixture/registry drift guards for writable model coverage and improved list-validation failure reporting to include row context.
- Closed remaining coverage gaps by adding tests for CLI behavior, parquet loader contract, predicate escaping edge cases, relocation scan roots, and dry-run semantics.
- Fixed writer behavior so `write_models(..., settings=Settings(..., dry_run=True))` does not write any data and returns `rows_written=0`.
