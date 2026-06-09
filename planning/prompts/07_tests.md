# Agent prompt — Write-side test suite

> Prepend `00_shared_context.md`. Run after the write path + validation exist. (Reader/
> analysis tests are deferred with that work.)

## Goal
Pull the write-side suite together and fill the gaps. Several cases are already specified in
their owning prompts — do NOT re-specify them here, just ensure they exist and run as one
suite:
- Registry↔schema drift → `02_write_spec.md` (`tests/test_write_spec.py`).
- Patchseq shared-partition regression, idempotency, append-new-by-id, predicate
  construction → `03_writers.md` (`tests/test_writers.py`).
- Strict-validation failures → `05_validation.md` (`tests/test_write_validation.py`).
- Public-API surface → `04_public_api.md` (`tests/test_public_api.py`).

Use small synthetic models written to a `tmp_path` Delta root (point `CCC_OUTPUT_ROOT` at
`tmp_path`, or a tmp `ccc_config.yaml`) so tests never touch real data.

## Gaps this prompt owns (not covered elsewhere)
1. **Per-class write-example smoke:** every writable class in the registry has a tiny write
   that round-trips through `write_models` without error (the prototyping evidence as a test).
2. **No-shim regression (TODO 3.4):** after migration, assert no module imports the old
   write-side paths `arrow_utils`, `write_utils` (grep / import-scan); the shims must be gone.
3. Confirm the suite is collected and green together (no per-prompt drift).

Round-trip and cross-dataset read tests are deferred to the read-side work.

## Reporting
Run `pytest -q` and paste the summary. Do not mark complete with failures.
