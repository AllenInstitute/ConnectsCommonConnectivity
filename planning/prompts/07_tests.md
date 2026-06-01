# Agent prompt — Test suite

> Prepend `00_shared_context.md`. Run after writers/readers exist (can be built alongside).

## Goal
A focused pytest suite under `tests/` covering the safe-writing guarantees. Use small
synthetic models written to a `tmp_path` Delta root (set `CCC_OUTPUT_ROOT` to `tmp_path`)
so tests never touch real data.

## Required cases
1. **Shared-partition safety (patchseq regression):** write `DataSet(id="A")` and
   `DataSet(id="B")` both with `project_id="P"`; assert both rows survive. This is the
   core regression for the bug.
2. **Idempotency:** writing the same models twice → no duplicates, no row loss, for both
   `overwrite_scoped` and `append_new_by_id`.
3. **Append-new-by-id:** second write with one new + one existing id appends exactly one.
4. **Strict validation:** a model missing a `required_for_write` slot, or violating a
   cross-field rule, raises before any file is written (assert the Delta dir is unchanged).
5. **Registry↔schema drift:** (from `02_write_spec.md`) every registry entry's class and
   columns exist in `models.py`.
6. **Round-trip:** write → read back via readers → equality on scope columns.
7. **Predicate construction:** the predicate is derived from `scope_columns` (assert the
   DataSet predicate includes both `project_id` and `id`).

## Reporting
Run `pytest -q` and paste the summary. Do not mark complete with failures.
