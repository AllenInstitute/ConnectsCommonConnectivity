# Agent prompt — Write spec registry

> Prepend `00_shared_context.md`. Depends on nothing (reads generated models).

## Goal
Create `src/connects_common_connectivity/io/write_spec.py`: an explicit registry, one
entry per writable class, that is the single source of truth for how each class is
written and validated. Plus a test that the registry cannot drift from the schema.

## Registry shape
Define a dataclass/pydantic model `WriteSpec` with fields:
- `model_cls` — the generated pydantic class (import from `..models`).
- `subdir: str` — Delta subdir under `output_root` (must match the notebook paths).
- `partition_by: list[str]` — Delta partition columns.
- `scope_columns: list[str]` — columns defining the overwrite predicate (identity within
  the shared table).
- `write_mode: Literal["overwrite_scoped", "append_new_by_id"]`.
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
`ProjectionMeasurementMatrix`, etc. — derive `subdir`/`scope_columns` by reading how each
is written in `code/etl_*.ipynb` (grep `write_deltalake` and `predicate=`). Where a
notebook's predicate looks wrong (like the DataSet case), prefer the correct scope and
note it in a comment. `CellFeatureMatrix` is wide Parquet, not row Delta — mark it so the
writer routes it to the matrix path (`build_cell_feature_matrix_schema`).

## Drift test (`tests/test_write_spec.py`)
- Every `REGISTRY` key resolves to a real class in `models.py`.
- Every column in `scope_columns` + `partition_by` + `required_for_write` corresponds to
  a field on that model (check `model_fields`). Fail loudly otherwise.

## Report
A table of each class → subdir / partition_by / scope_columns / write_mode, and call out
any notebook predicate you believe is wrong (do not fix notebooks here).
