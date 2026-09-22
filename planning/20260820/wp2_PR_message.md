## What changed

- Added `merge_scoped` writes with registry-declared `merge_on` keys for the 15 currently writable identity-bearing metadata and association classes.
- All 15 live registry entries now use `merge_scoped`; there are currently zero live `overwrite_scoped` entries. The overwrite dispatcher is intentionally retained for the deferred bulk `SynapseConnectivityLong` registration, where MERGE would be wasteful at 10^7 rows, and remains directly tested until that registration is enabled.
- Implemented pure Delta MERGE upserts: matching rows are updated, new rows are inserted, and rows absent from the incoming batch are retained.
- Deduplicated incoming batches on their complete merge key with deterministic last-row-wins behavior.
- Replaced `DataItem`'s append-only dispatch with merge-scoped upserts, so existing metadata can be updated instead of silently ignored.
- Enforced nullable `ClusterMembership.item`, `.cluster`, and `.hierarchy_id` fields at the IO boundary through `required_for_write`; the LinkML schema remains optional.
- Preserved discovered `dry_run` and other settings when `output_root` overrides only the write destination, including calls that omit an explicit `settings` argument.
- Removed the VISp patch-seq notebooks' manual read-union-rewrite workarounds. The notebooks now submit only their own association and membership rows and verify persisted merge-key uniqueness.
- Refreshed outputs for the Tasic, VISp MET-type, and six VISp patch-seq notebooks from the Code Ocean acceptance run.
- Kept the standalone `append_new_dataitems` helper for compatibility; `write_models(DataItem)` no longer uses it.

This PR is based on `wp1-schema-scope` and should be reviewed against that branch.

## Why

`overwrite_scoped` replaced every row in a declared scope. Multiple patch-seq notebooks contribute disjoint rows to the same `(project_id, dataset_id)` association scope and `(project_id, hierarchy_id)` membership scope, so later notebooks silently deleted earlier contributions. The observed inhibitory association count shrank from 2,759 to 520 to 495, and excitatory/inhibitory MET memberships could overwrite one another.

Identity-bearing metadata also needed true update behavior: the previous `append_new_by_id` path skipped an existing `DataItem` instead of applying revised metadata. Delta MERGE makes these incremental workflows transactional at the table-operation level while leaving deletion explicit and separate.

Closes #13
Closes #14
Closes #15
Closes #16

## How to test

Local automated validation:

```bash
uv run pytest -q
# 207 passed

uv run ruff check \
  src/connects_common_connectivity/io/write_spec.py \
  src/connects_common_connectivity/io/writers.py \
  tests/test_write_spec.py \
  tests/test_write_validation.py \
  tests/test_writers.py
# passed
```

The three modified patch-seq notebooks parse as valid JSON and every code cell compiles. `git diff --check` also passes.

The Code Ocean acceptance run used a fresh `scratch/wp2_acceptance_20260922/` output root and executed the Tasic taxonomy, VISp MET-type taxonomy, and six excitatory/inhibitory patch-seq notebooks. The incremental notebooks were rerun without clearing the output. Persisted results remained stable:

- 2,879 inhibitory dataset associations;
- 2,637 shared MET hierarchy memberships: 1,152 excitatory plus 1,485 inhibitory;
- 879 distinct membership items: 384 excitatory plus 495 inhibitory;
- zero duplicate `(project_id, dataset_id, dataitem_id)` association keys;
- zero duplicate `(project_id, hierarchy_id, item, cluster)` membership keys;
- all notebook assertions passed on the initial run and rerun.

The refreshed notebook outputs capture this acceptance run. The capsule identifier, exact tested commit SHA, and attached data-asset versions were not recorded in the handover report.

## Reviewer focus (optional)

- The per-class `merge_on` assignments in `io/write_spec.py`, especially composite identities for associations and memberships.
- Pure-upsert semantics: MERGE intentionally does not delete target rows absent from an incoming batch; explicit removal remains separate work.
- Last-row-wins deduplication for duplicate merge keys within one incoming batch.
- Keeping `ClusterMembership` fields optional in the shared schema while requiring merge keys only in this IO wrapper.
- The changed configuration path (`scratch/wp2_acceptance_20260922/`) and committed notebook outputs from Code Ocean acceptance testing.
- Deferred decisions documented in `planning/20260820/wp2_out_of_scope_findings.md`: optimistic-concurrency retries, the lifecycle of `append_new_dataitems`, and richer MERGE metrics in `WrittenResult`.