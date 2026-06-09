# Agent prompt — Write spec registry

> Prepend `00_shared_context.md`. Depends on nothing (reads generated models).

## Goal
Create `src/connects_common_connectivity/io/write_spec.py`: an explicit registry, one
entry per writable class, that is the single source of truth for how each class is
written. Plus a test that the registry cannot drift from the schema.

## Approach: prototype, don't assume
Do NOT assume every class is scoped-overwrite-with-predicate. That pattern fits
DataSet/Association, but `append_new_by_id` already exists for DataItem because append was
right there, and other classes may want append or modes not yet named. **For each class,
build a small real write example in a notebook first** (paired with `03_writers.md`), see how
it actually wants to be written, and let that decide the entry. `write_mode` is an open
`Literal` you extend when an example doesn't fit the existing modes — not a constraint to
force classes into. Seed the three correctness-critical classes below; add the rest as their
examples are built rather than all at once up front.

## Registry shape
Define a dataclass/pydantic model `WriteSpec` with fields:
- `model_cls` — the generated pydantic class (import from `..models`).
- `subdir: str` — Delta subdir under `output_root` (must match the notebook paths).
- `partition_by: list[str]` — Delta partition columns.
- `scope_columns: list[str]` — for scoped-overwrite classes, columns defining the predicate
  (identity within the shared table). May be empty for append-mode classes.
- `write_mode: Literal[...]` — start with `"overwrite_scoped"`, `"append_new_by_id"`; add new
  members when a class's example shows neither fits. Keep it easy to extend.
- `required_for_write: list[str]` — slots that must be non-null to write safely (may be
  stricter than the schema's `required`).
- `cross_field_rules: list[str]` — names of cross-field checks (implemented in
  `write_validation.py`); empty for now is fine.

Expose `REGISTRY: dict[str, WriteSpec]` keyed by class name, and a
`get_spec(model_or_cls) -> WriteSpec` lookup.

## Seed these first (correctness-critical)
- `DataSet`: subdir `"dataset"`, partition `["project_id"]`,
  **scope `["project_id", "id"]`** (THIS is the patchseq bug fix), mode
  `overwrite_scoped`.
- `DataItem`: subdir `"dataitem"`, partition `["project_id"]`, mode `append_new_by_id`,
  id column `"id"`.
- `DataItemDataSetAssociation`: subdir `"dataitem_dataset_association"`, partition
  `["project_id"]`, scope `["project_id", "dataset_id"]`, mode `overwrite_scoped`.

Then add entries for `Cluster`, `ClusterHierarchy`, `ClusterMembership`,
`CellFeatureSet`, `CellFeatureDefinition`, `CellToClusterMapping`, `MappingSet`,
`ProjectionMeasurementMatrix`, etc. **as each one's write example is prototyped** — read how
it's written today in `code/etl_*.ipynb` (grep `write_deltalake` and `predicate=`), try it
through the writer, and only then fix its entry. Where a notebook's current predicate looks
wrong (like the DataSet case), prefer the correct scope and note it in a comment.
`CellFeatureMatrix` is wide Parquet, not row Delta — mark it so the writer routes it to the
matrix path (`build_cell_feature_matrix_schema`).

## Drift test (`tests/test_write_spec.py`)
- Every `REGISTRY` key resolves to a real class in `models.py`.
- Every column in `scope_columns` + `partition_by` + `required_for_write` corresponds to
  a field on that model (check `model_fields`). Fail loudly otherwise.

## Report
A table of each class → subdir / partition_by / scope_columns / write_mode, and call out
any notebook predicate you believe is wrong (do not fix notebooks here).
