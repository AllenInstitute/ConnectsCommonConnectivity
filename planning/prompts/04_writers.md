# Agent prompt — Writers (dispatch core + typed wrappers)

> Prepend `00_shared_context.md`. Depends on `config.py`, `write_spec.py`, `validation.py`.

## Goal
Create `src/connects_common_connectivity/io/writers.py`: a single write dispatch that uses
the registry so notebooks never hand-write `mode` / `predicate` / `partition_by` again.

## Core
`write_models(models, *, settings=None) -> WriteResult`:
1. Accept a single model or an iterable; infer the class; require homogeneous type.
2. `settings = settings or Settings.load()`.
3. Look up the `WriteSpec` via `get_spec`.
4. Validate every model with `validate_for_write` (strict submodel) BEFORE any IO.
5. Convert via `arrow_utils.models_to_table` + `build_arrow_schema`; attach metadata with
   `attach_linkml_metadata(linkml_class=<class name>)`.
6. Resolve path with `table_path(settings, spec.subdir)`.
7. Dispatch on `spec.write_mode`:
   - `overwrite_scoped`: build the predicate from `spec.scope_columns` and the row values
     (e.g. `project_id = '...' AND id = '...'`), then
     `write_deltalake(path, table, mode="overwrite", predicate=..., partition_by=spec.partition_by)`.
     If a batch contains multiple distinct scope tuples, write per scope group (one
     predicate each) — never widen a predicate to cover rows it shouldn't.
   - `append_new_by_id`: delegate to `write_utils.append_new_dataitems` (the backend),
     passing `project_id` and id column.
8. Return a small result object: rows written/appended, path, mode, predicate used.

## Typed wrappers (one-liners over `write_models`)
`write_dataset`, `write_dataitem`, `write_association`, `write_features`, `write_cluster`,
`write_cluster_membership`, `write_cell_to_cluster_mapping`, `write_projection_matrix`.
Signatures should be ergonomic (accept the model(s) and optional `settings`).

## Wide feature matrices
`CellFeatureMatrix` is wide Parquet. Route it through a matrix-specific path using
`build_cell_feature_matrix_schema`; do not force it into the row-Delta path.

## Reconcile `write_utils.py`
Make `append_new_dataitems` the `append_new_by_id` backend. If you must generalize it
(e.g. parametrize the partition column), keep the existing signature working — its current
notebook callers must not break.

## Tests (`tests/test_writers.py`)
- Scoped overwrite writes only matching rows; a second dataset sharing `project_id` is
  preserved (patchseq regression: write DataSet A, write DataSet B, both rows exist).
- Re-writing identical models is idempotent (no dupes, no loss).
- `append_new_by_id` appends only new ids.
- Predicate is built from `scope_columns`, verified by string/inspection.

## Do not
- Hardcode any predicate. Touch `models.py` or schemas.
