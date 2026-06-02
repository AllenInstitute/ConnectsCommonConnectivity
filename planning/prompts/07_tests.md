# Agent prompt — Test suite

> Prepend `00_shared_context.md`. Run after writers/readers exist (can be built alongside).

## Goal
Pull the suite together and fill the gaps. Several cases are already specified in their
owning prompts — do NOT re-specify them here, just ensure they exist and run as one suite:
- Registry↔schema drift → `02_write_spec.md` (`tests/test_write_spec.py`).
- Patchseq shared-partition regression, idempotency, append-new-by-id, predicate
  construction → `04_writers.md` (`tests/test_writers.py`).
- Round-trip + cross-dataset reads → `05_readers.md` (`tests/test_readers.py`).
- Strict-validation failures → `03_validation.md` (`tests/test_write_validation.py`).
- Public-API surface → `09_public_api.md` (`tests/test_public_api.py`).

Use small synthetic models written to a `tmp_path` Delta root (set `CCC_OUTPUT_ROOT` to
`tmp_path`) so tests never touch real data.

## Gaps this prompt owns (not covered elsewhere)
1. **No-shim regression (TODO 5.4):** after migration, assert no module imports the old
   paths `arrow_utils`, `write_utils`, `parquet_loader` (grep the repo or import-scan); the
   shims must be gone, not lingering.
2. **End-to-end smoke:** a single test exercising write → read → analysis on a tiny fixture,
   proving the modules compose.
3. Confirm the whole suite is collected and green together (no per-prompt drift).

## Reporting
Run `pytest -q` and paste the summary. Do not mark complete with failures.
