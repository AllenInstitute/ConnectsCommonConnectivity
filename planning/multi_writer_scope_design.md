# Multi-writer Delta scopes: bug, contrast with minnie, design options

Captured 2026-06-23 from a debugging session that started with an
`AssertionError` in `etl_visp_inh_patchseq_03_cluster_membership_and_mapping.ipynb`.

## The problem

`write_models` dispatches `overwrite_scoped` writes that **replace every row** in
the scope defined by a spec's `scope_columns`. For example
(`src/connects_common_connectivity/io/write_spec.py`):

| Class | `scope_columns` |
|---|---|
| `DataItemDataSetAssociation` | `(project_id, dataset_id)` |
| `ClusterMembership` | `(project_id, hierarchy_id)` |
| `CellToClusterMapping` | `(project_id, mapping_set)` |

When **multiple notebooks contribute disjoint row subsets to the same scope**,
any one of them issuing `write_models([...own_rows...])` deletes the other
notebooks' rows. The latest writer wins, silently.

This bit us concretely in the patch-seq pipeline. After running
`etl_visp_inh_patchseq_01/02/03`:

```
visp_patchseq / visp_inh_patchseq associations: 495   (expected ≥ 2759)
```

`_03` Section 1's assertion surfaced it:

```
AssertionError: 2367 T-type CSV cells are not associated with visp_inh_patchseq
```

Root cause: `_01` writes 2759 association rows from the ttype CSV → `_02`
overwrites with 520 rows from the wide CSV → `_03` overwrites with 495 rows from
the MET CSV. Every step is a valid `overwrite_scoped` call; together they shrink
the scope monotonically. The same bug existed (silently) for `ClusterMembership`
under `(visp_patchseq, visp_met_types_taxonomy)`, where `etl_visp_exc_patchseq_03`'s
1152 rows were being wiped by `etl_visp_inh_patchseq_03`'s 1485-row overwrite.

### Numbers from patch-seq

| Source CSV (input) | Notebook | Rows in CSV | Scope written |
|---|---|---:|---|
| `patchseq_tx_cell_ttype_labels.csv` | `inh_01` | 2759 | `(visp_patchseq, visp_inh_patchseq)` |
| `inh_ivscc_features_wide_unnormalized.csv` | `inh_02` | 520 | `(visp_patchseq, visp_inh_patchseq)` |
| `visp_met_cell_assignments_text_names.csv` | `inh_03` § 0 | 495 | `(visp_patchseq, visp_inh_patchseq)` |
| `visp_met_cell_assignments_text_names.csv` | `inh_03` § 2 | 495 cells × 3 ancestors = 1485 | `(visp_patchseq, visp_met_types_taxonomy)` |
| `inferred_met_types.csv` | `exc_03` § 1 | 384 cells × 3 ancestors = 1152 | `(visp_patchseq, visp_met_types_taxonomy)` |

After the fix (read-existing → union → re-write the full scope):

```
visp_patchseq / visp_inh_patchseq associations:                    2879
visp_patchseq / visp_met_types_taxonomy clustermembership rows:    2637   (=1152 exc + 1485 inh)
visp_patchseq / visp_met_types_taxonomy clustermembership items:    879   (=384 exc + 495 inh)
```

### Origin: migration regression

This is a migration regression, not an original design flaw. The pre-migration
notebooks did the merge manually with raw
`write_deltalake(..., mode="overwrite", predicate=...)`:

```python
existing_cm = pl.read_delta(...).filter(predicate)
other_cm = existing_cm.filter(~pl.col("item").is_in(our_cell_ids))
all_memberships = [ClusterMembership(**r) for r in other_cm.to_dicts()] + new
write_deltalake(..., mode="overwrite", predicate=..., partition_by=...)
```

When that pattern was migrated to `write_models([...])`, the read-and-union step
was dropped (replaced with a stub `other_cm = pl.DataFrame({"item": []})`) and
the assertion `others_present.shape[0] == other_cm.shape[0]` continued to
"pass" because both sides became 0 — the verification was no longer
load-bearing.

## How minnie avoids the problem entirely

Minnie uses a **sub-dataset (cohort) pattern**: each notebook writes into its
own unique `(project_id, dataset_id)` scope, so `overwrite_scoped` calls never
collide.

| Notebook | `DATASET_ID` |
|---|---|
| `etl_minnie_01_dataset_dataitem` | `minnie65_v1300_nuclei` (the universe) |
| `etl_minnie_02_cell_features` | `minnie65_v1300_csm_cluster` (CSM cohort) |
| `etl_minnie_03_cluster_and_cluster_membership` | reuses `minnie65_v1300_csm_cluster`, but writes `ClusterMembership` under `hierarchy_id="minnie65_csm_cell_types"` — a hierarchy no other minnie notebook writes to |
| `etl_minnie_04_cell_cell` | proofread cohorts (`minnie65_v1300_proofread*`) |

For every `overwrite_scoped` write minnie issues, the **scope owner is exactly
one notebook**. No merge, no surprises.

Patch-seq took the opposite philosophy: one `DataSet`
(`visp_inh_patchseq`) is treated as a single coherent cohort, and multiple
notebooks add rows of different kinds to the **same** `(project, dataset)` and
`(project, hierarchy)` scopes. That's what creates the multi-writer hazard.

There's a meta-question buried here: should patch-seq follow minnie's cohort
pattern? E.g. `visp_inh_patchseq_ttype`, `visp_inh_patchseq_morph`,
`visp_inh_patchseq_met` as sibling sub-datasets. It would remove the merge
problem entirely but would also fragment what is currently a clean
"inh-cohort" abstraction. Not obvious which is better.

## Considered solutions

### Option A — Add a merging write mode to `write_models`

Add a new `WriteSpec.write_mode` value, e.g. `"merge_by_id"` or
`"overwrite_scoped_by_id"`, that:

1. Requires the spec to declare an **identity column** within the scope
   (`dataitem_id` for `DataItemDataSetAssociation`, `item` for
   `ClusterMembership`, …).
2. On write: reads existing rows in scope, replaces rows whose identity is in
   the incoming batch, keeps the rest.

**Pros**
- Eliminates the boilerplate currently duplicated in every patch-seq notebook.
- Makes the multi-writer contract explicit in the spec (it's *declared* that
  this scope is multi-writer and merged on column X).
- Closes the regression class that bit us — a future migration cannot
  accidentally strip the merge logic because the merge lives in the library.

**Cons**
- Silently merging vs. overwriting is a semantically distinct contract; a
  caller who actually wanted to *clear* sibling rows would have to opt out.
- Requires a read per write (negligible at current data sizes).
- The library implicitly trusts that the caller's batch is the authoritative
  subset for the ids it contains.

### Option B — Keep `write_models` overwrite-only, add a sibling helper

```python
write_models_merging_on(items, id_column="item", output_root=...)
```

**Pros**
- No change to existing call sites or `write_models` semantics.
- Explicit at the call site: a reader sees "this notebook merges into a shared
  scope" without having to look up the spec.
- Matches how minnie sidesteps the issue (use distinct scopes whenever
  possible; reach for the merging helper only when you can't).

**Cons**
- Still requires every shared-scope notebook to remember to use the merge
  variant; the next migration can still regress this.

### Option C — Status quo (don't change the library)

Document the convention; every notebook touching a shared scope does its own
read-and-union before `write_models`.

**Pros**
- Library stays minimal and explicit.

**Cons**
- This is exactly the trap the recent migration walked into. There is no
  structural mechanism preventing a recurrence.

### Option D — Forbid shared scopes (push patch-seq toward minnie's pattern)

Refactor patch-seq notebooks so each `(project, dataset)` and
`(project, hierarchy)` scope has a single owner — possibly by introducing
sub-datasets (`visp_inh_patchseq_ttype`, `_morph`, `_met`).

**Pros**
- Removes the multi-writer hazard at the data-model level rather than papering
  over it in the library.
- Brings patch-seq into stylistic alignment with minnie / V1DD.

**Cons**
- Larger change. Downstream queries that group rows by "the inh cohort" now
  need to union sub-datasets. May lose a useful natural grouping.
- Doesn't solve the `ClusterMembership` case (different MET-types
  contributors *do* share a hierarchy — that's the taxonomy's whole point).
  So a merge mechanism is probably still needed somewhere.

## Suggested next step (for discussion, not yet decided)

Lean toward **A + a scope-ownership audit**:

1. For every `overwrite_scoped` spec, decide whether the scope is
   single-writer (minnie-style) or multi-writer (patch-seq-style).
2. Single-writer specs stay as-is.
3. Multi-writer specs declare a merge key (Option A).
4. Bonus: `write_models` could detect "a write that would shrink the scope it
   targets" (i.e. incoming rows form a strict subset of the existing scope by
   the merge key) and warn/error when the spec is not marked multi-writer.
   That would have caught the regression at runtime.

Open questions:

- Should `WriteSpec` gain a `merge_on: list[str] | None` field?
- Is the implicit "incoming batch is the truth for these ids" contract
  acceptable for every multi-writer class, or do we need a more general
  "upsert by composite key" mode?
- Do we want to keep patch-seq as multi-writer at all, or migrate to
  sub-datasets and reserve the merge mechanism only for `ClusterMembership`
  (where taxonomy-sharing makes single-ownership impossible)?
