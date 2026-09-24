Replaces scoped-overwrite writes with Delta MERGE upserts so ETL notebooks that
contribute disjoint rows to a shared scope stop deleting each other's data. All
15 live registry entries move to `merge_scoped`. Based on `wp1-schema-scope`;
review against that branch.

## What changed

1. **`merge_scoped` write mode**, with per-class identity keys declared in the
   registry — `write_spec.py::REGISTRY`. `WriteSpec.validate_merge_on` refuses to
   register a key that is neither schema-required-non-nullable nor listed in
   `required_for_write`; the writer raises on a null key value. This is how
   `ClusterMembership.item`, `.cluster`, and `.hierarchy_id` become mandatory at
   the IO boundary while staying optional in the LinkML schema.
2. **Pure MERGE upsert dispatch** — `writers.py::_dispatch_merge_scoped`. Matched
   rows update, new rows insert, rows absent from the batch are retained.
3. **Partition-pruned merge predicates** — `writers.py::_build_partition_prune_predicate`.
4. **Last-row-wins batch deduplication** via Arrow group-by — `writers.py::_deduplicate_on_keys`.
5. **`rows_written` counts inserted + changed rows only** — `writers.py::WrittenResult`,
   `_build_merge_update_predicate`. An unchanged rerun now reports `0`.
6. **Quoted column identifiers** in every generated predicate — `writers.py::_quote_identifier`.
7. **Creation-race handling**: a writer that loses the initial-create race reopens
   the table and merges — `writers.py::_dispatch_merge_scoped`.
8. **`dry_run` honored when `output_root` overrides only the destination** —
   `config.py`, `writers.py::_resolve_output_root`.
9. **Notebook migration**: the VISp patch-seq notebooks drop their
   read-union-rewrite workarounds and gain sibling-preservation guards;
   `etl_wnm_exc_04` moves to `write_models`; the unused
   `write_utils.append_new_dataitems` helper is deleted.

`_dispatch_overwrite_scoped` is retained with direct test coverage for the
deferred bulk `SynapseConnectivityLong` registration, where MERGE would be
wasteful at 10^7 rows. It has no live registry entry today.

Also in the diff, unrelated to the milestone: two `.github/skills/` entries and a
`.gitignore` line for `.vscode`.

| Issue | Closed by |
|---|---|
| Closes #13 — merge_scoped mode with declared merge_on keys | 1, 2 |
| Closes #14 — ClusterMembership merge keys in io, not schema | 1 |
| Closes #15 — multi-writer data loss in shared scopes | 2, 9 |
| Closes #16 — dry_run ignored when output_root is passed | 8 |

## Why

**Data loss (#15).** `overwrite_scoped` replaced every row in a declared scope.
Several patch-seq notebooks write disjoint rows into the same
`(project_id, dataset_id)` and `(project_id, hierarchy_id)` scopes, so later
notebooks silently deleted earlier contributions — the inhibitory association
count fell 2,759 → 520 → 495, and excitatory/inhibitory MET memberships could
overwrite one another. Change 2 makes each contribution an atomic upsert; change
9 removes the workarounds the notebooks had grown to compensate.

**Stale metadata.** The old `append_new_by_id` path *skipped* an existing
`DataItem` rather than applying revised fields. Change 1 gives these classes real
update semantics while leaving deletion explicit and separate (tracked in #21).

**Merge cost (change 3).** A predicate built only from `target.c = source.c`
equalities gives the Delta planner no literal to compare against file statistics,
so it scans every target file. Restating the batch's distinct partition values as
literals restores file skipping: `test_merge_scoped_predicate_prunes_untouched_partitions`
asserts that a single-partition merge into a four-file table scans 1 file and
skips 3. Only columns that are both partition columns *and* merge keys are
constrained — a target row outside the batch's value set for such a column can
never satisfy the equality join — so the narrowing removes no candidate match.

**Scope limit.** The guarantee here is sequential, not concurrent. Each MERGE
commit is atomic and change 7 covers the initial-create race, but two writers
committing to the same table simultaneously can still hit an
optimistic-concurrency conflict. No retries or distributed concurrency tests are
added, so closing #15 does not claim otherwise.

## How to test

```bash
uv run pytest -q          # 215 passed
uv run ruff check $(git diff --name-only origin/wp1-schema-scope...HEAD -- '*.py')
```

The change-3 file-skipping numbers are asserted from delta-rs merge metrics in
`tests/test_writers.py`. The modified notebooks parse as valid JSON and every
code cell compiles; `git diff --check` passes.

A Code Ocean acceptance run (2026-09-22, fresh `scratch/wp2_acceptance_20260922/`
output root) executed the Tasic and VISp MET-type taxonomies plus six patch-seq
notebooks, then reran the incremental ones without clearing output: 2,879
inhibitory associations, 2,637 MET memberships (1,152 exc + 1,485 inh) across 879
distinct items, zero duplicate association or membership merge keys, all notebook
assertions passing. The capsule id, tested SHA, and data-asset versions were not
recorded in the handover report.

> **Caveat — the committed notebook outputs are older than the code.** The
> sibling-preservation guards and the `etl_wnm_exc_04` migration (change 9) were
> added after that run and have never been executed: those cells carry
> `execution_count: null` beside outputs from the earlier run. Changes 3–7 are
> covered by the test suite but are likewise absent from the committed outputs.
> Read the numbers above as evidence for the first-pass implementation only; a
> rerun is needed before merge.

## Reviewer focus

- Composite `merge_on` choices for associations and memberships
  (`write_spec.py::REGISTRY`) — a wrong key silently merges distinct rows.
- `validate_merge_on` is the only barrier against adopting a nullable merge key.
- `_build_partition_prune_predicate`: the 100-literal ceiling and the null skip.
- Pure-upsert semantics: MERGE deliberately does not delete target rows missing
  from a batch; explicit removal stays separate work.
- Change 5 alters what an unchanged rerun reports — confirm no caller depends on
  the old count.
- Whether to rerun acceptance testing on the reviewed code before merge.
- Deferred decisions in `planning/20260820/wp2_out_of_scope_findings.md`:
  concurrency retries, richer MERGE metrics, repository-wide Ruff backlog.