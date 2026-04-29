# Add ETL notebooks for VISp Patch-seq, WNM, and Minnie65 datasets

## Summary

This PR adds a set of ETL Jupyter notebooks that register neuroscience datasets into the shared Delta Lake store, along with a schema fix required to make cell feature tables project-scoped.

---

## ETL notebooks added

All notebooks follow the same conventions:
- Inputs are loaded (CSV or CAVE query), printed with shape and `head(3)`.
- Outputs are written with `mode="overwrite"` and `predicate=f"project_id = '{PROJECT_ID}'"` so re-runs are idempotent and other projects' rows in shared Delta tables are never touched.
- Every write step is followed by a verification cell (`pl.read_delta`, shape check, `assert`).

### `_01` Dataset + DataItem notebooks

| Notebook | `project_id` | `dataset_id` | Input |
|---|---|---|---|
| `etl_visp_inh_patchseq_01_dataset_dataitem.ipynb` | `visp_patchseq` | `visp_inh_patchseq` | `patchseq_tx_cell_ttype_labels.csv` (2,759 cells) |
| `etl_visp_exc_patchseq_01_dataset_dataitem.ipynb` | `visp_patchseq` | `visp_exc_patchseq` | `inferred_met_types.csv` (1,528 cells) |
| `etl_wnm_exc_01_dataset_dataitem.ipynb` | `visp_wnm` | `visp_exc_wnm` | `FullMorphMetaData_Master.csv` (341 cells; `.swc` suffix stripped from index) |
| `etl_minnie_01_dataset_dataitem.ipynb` | `minnie65` | `minnie65_v1412_nuclei` | CAVE `nucleus_detection_lookup_v1` (version 1412, `pt_root_id != 0`) |

Each notebook writes exactly three tables: `DataSet`, `DataItem`, and `DataItemDataSetAssociation`.

### `etl_minnie_02_cell_features.ipynb`

Writes cell features for the Minnie65 dataset. Requires `etl_minnie_01` to have been run first.

Outputs:

| Path | Class | Rows |
|---|---|---|
| `dataset/` | `DataSet` | +1 (`minnie65_v1412_csm_cluster` cohort) |
| `dataitem_dataset_association/` | `DataItemDataSetAssociation` | one per cell in `minnie_features.parquet` |
| `cellfeaturedefinition/` | `CellFeatureDefinition` | 82 CSM features (from CSV) + 3 standard_transform coordinates |
| `cellfeatureset/` | `CellFeatureSet` | 2 (`csm_cluster_features`, `minnie65_std_transform_coordinates`) |
| `cellfeatures/csm_cluster_features/` | wide parquet | per-cell × 82 features |
| `cellfeatures/minnie65_std_transform_coordinates/` | wide parquet | per-cell x/y/z in microns |
| `cellfeaturematrix/` | `CellFeatureMatrix` | 2 pointer rows |

Soma coordinates are computed by applying `standard_transform.minnie_transform_vx()` to `pt_position` from the CAVE nucleus view, then dividing nm → µm and casting to `float32`.

---

## Schema fix: `CellFeatureDefinition` and `CellFeatureSet` were not `ProjectScoped`

### The problem

`CellFeatureDefinition` and `CellFeatureSet` did not inherit the `ProjectScoped` mixin, so they had no `project_id` field. This broke the uniform write pattern used everywhere else in the project:

```python
write_deltalake(..., mode="overwrite", predicate=f"project_id = '{PROJECT_ID}'", partition_by=["project_id"])
```

Without `project_id`, there was no way to scope a predicate overwrite to a single project. The only alternatives were:

1. **Plain `mode="overwrite"`** — wipes the entire shared table across all projects. Unsafe.
2. **`mode="append"`** — accumulates duplicate rows on every re-run. Unusable.
3. **A non-`project_id` predicate** (e.g. `feature_set_id IN (...)`) — fragile; requires enumerating every id belonging to the project, and still breaks for a new project that reuses a feature set name.

In the first draft of `etl_minnie_02`, the CSM feature definitions were written with predicate-overwrite, and the three standard_transform coordinate definitions were written with `mode="append"` in a second call. This meant re-running the notebook would accumulate duplicate coordinate definitions.

### The fix

Added `mixins: [ProjectScoped]` and `- project_id` to both `CellFeatureSet` and `CellFeatureDefinition` in `schemas/cell_features_schema.yaml`, then regenerated `models.py`:

```bash
uv run gen-pydantic schemas/connectivity_schema.yaml > src/connects_common_connectivity/models.py
```

`models.py` is never edited manually; `schemas/connectivity_schema.yaml` is the source of truth.

With `project_id` on both classes, `etl_minnie_02` can collect all feature definitions (82 CSM + 3 coords = 85) and write them in a single idempotent overwrite, and likewise for `CellFeatureSet` (2 rows) and `CellFeatureMatrix` (2 rows).

### Follow-up: `feature_set_id` on `CellFeatureDefinition`

After adding `project_id`, a second scoping problem emerged: two feature sets in the same notebook (`csm_cluster_features` and `minnie65_std_transform_coordinates`) both belong to `project_id="minnie65"`. Writing their definitions in separate cells was still impossible — a `mode="overwrite"` with `predicate="project_id='minnie65'"` in the second cell would wipe the first cell's rows.

The fix was to add `feature_set_id` as an optional field to `CellFeatureDefinition` in the schema. This enables a two-level predicate:

```python
predicate=f"project_id = '{PROJECT_ID}' AND feature_set_id = '{FSI}'"
```

Each feature-set section in `etl_minnie_02` now scopes its `cellfeaturedefinition/` write to its own `feature_set_id`. The same two-level predicate is used for `cellfeaturematrix/` (which already had `feature_set_id`). `cellfeatureset/` uses `project_id + id` since the row's own `id` is the feature set id.

The notebook is now two self-contained sections (A: CSM, B: STD coordinates), each independently re-runnable without affecting the other. `cellfeaturedefinition/` also gains `partition_by=["project_id", "feature_set_id"]` for query performance.

### Tests

Added `tests/test_cell_features_schema.py` with 10 tests covering:
- `project_id` is required on both `CellFeatureDefinition` and `CellFeatureSet`
- Valid construction with all required fields
- `data_type` pattern validation (must be numpy dtype string, e.g. `<f4`, `<i4`)
- Optional `range_min` / `range_max` fields
- `feature_set_id` is optional on `CellFeatureDefinition` and can be set when provided

---

## `write_utils.py`: idempotent DataItem registration across shared `project_id`

### The problem

Two `_01` notebooks can share the same `project_id`: both `etl_visp_inh_patchseq_01` and `etl_visp_exc_patchseq_01` use `project_id="visp_patchseq"`. Previously both wrote `dataitem/` with:

```python
write_deltalake(OUTPUT_ROOT + "dataitem/", table_di,
                mode="overwrite", predicate=f"project_id = '{PROJECT_ID}'", ...)
```

A predicate-scoped overwrite on `project_id='visp_patchseq'` wipes **all** rows for that partition — so whichever notebook ran second silently deleted the first's cells. `etl_visp_inh_patchseq_02` saw "Already in DataItem: 0" even after `_01` had registered 2,759 cells, because `exc_01` overwrote the partition.

The `dataitem_dataset_association/` predicate also only scoped to `project_id`, making it equally fragile when a second dataset shares the project.

### The fix

**`src/connects_common_connectivity/write_utils.py`** introduces `append_new_dataitems`:

```python
def append_new_dataitems(output_path, table, *, project_id, id_column="id") -> int:
    """Append only rows whose id is not already present for this project.
    Idempotent: re-running appends nothing. Handles missing table gracefully."""
```

It reads existing `(project_id, id)` pairs, filters the incoming table to only new rows, and appends with `mode="append"`. Re-running returns 0 and writes nothing. Two notebooks sharing `project_id` each only add their own cells without touching the other's.

All four `_01` notebooks now import and use `append_new_dataitems` for the `dataitem/` write, and their `dataitem_dataset_association/` predicates are narrowed to `project_id = '...' AND dataset_id = '...'` so each dataset's association rows are independently idempotent.

`etl_visp_inh_patchseq_02` is updated to use `append_new_dataitems` for the 120 new-cell registrations, replacing the previous read-union-write pattern.

### Tests

Added `tests/test_write_utils.py` with 6 tests covering:
- First write (table does not exist): all rows appended
- Empty table: 0 rows appended
- Idempotent re-run: 0 rows appended on second call
- Partial re-run: only new rows appended
- Different `project_id` values don't interfere
- Two sources sharing `project_id` accumulate without conflict

All 22 tests pass (`uv run pytest -q`).
