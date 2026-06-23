# Agent prompt — Write-side test suite

> Prepend `00_shared_context.md`. Run after the write path + validation exist. (Reader/
> analysis tests are deferred with that work.)

## Goal
This is the LAST write-side prompt that will run — prompts 02–05 will not be re-executed.
That means this prompt is responsible for both the gaps below AND any cleanup left over
from earlier prompts. Several cases are already specified in their owning prompts:
- Registry↔schema drift → `02_write_spec.md` (`tests/test_write_spec.py`).
- Patchseq shared-partition regression, idempotency, append-new-by-id, predicate
  construction → `03_writers.md` (`tests/test_writers.py`).
- Strict-validation failures → `05_validation.md` (`tests/test_write_validation.py`).
- Public-API surface → `04_public_api.md` (`tests/test_public_api.py`).

If any of those tests are missing, red, or do not actually assert what their prompt
claimed, **fix them here** — there is no second pass. When you patch a test owned by an
earlier prompt, list which prompt and which test in the report so the spec docs can be
updated later.

Use small synthetic models written to a `tmp_path` Delta root (point `CCC_OUTPUT_ROOT` at
`tmp_path`, or a tmp `ccc_config.yaml`) so tests never touch real data.

## Gaps this prompt owns (not covered elsewhere)
1. **Per-class write-example smoke:** every writable class in the registry has a tiny write
   that round-trips through `write_models` without error (the prototyping evidence as a test).
2. **No-shim regression (TODO 3.4):** after migration, assert no module imports the old
   write-side paths `arrow_utils`, `write_utils` (grep / import-scan); the shims must be gone.
3. Confirm the suite is collected and green together (no per-prompt drift).
4. Patch any 02–05 test gaps discovered while running the suite (see goal above).

Round-trip and cross-dataset read tests are deferred to the read-side work.

## Reporting
Run `uv run pytest -q` (this repo uses `uv` — plain `pytest` will not pick up the
project venv) and paste the summary. Do not mark complete with failures. Also list any
tests you patched on behalf of an earlier prompt and a one-line reason for each.
