# ETL Notebook Prompt Guide

Use this file as context when asking an AI assistant to create a new ETL notebook in this repository. Paste it (or point to it) at the start of your prompt.

---

## 1. Read first — before writing any code

Ask the AI to read these files **before** generating any notebook cells.

### Schemas (source of truth)
```
schemas/base_schema.yaml          # HasId, ProjectScoped, and other base mixins
schemas/core_schema.yaml          # DataSet, DataItem, DataItemDataSetAssociation, Modality
schemas/cell_features_schema.yaml # CellFeatureDefinition, CellFeatureSet, CellFeatureMatrix
```
Read the relevant domain schema too if writing projection, clustering, or mapping data:
```
schemas/clustering_schema.yaml
schemas/mappings_schema.yaml
schemas/projection_schema.yaml
schemas/single_cell_schema.yaml
```

### Package utilities (read-only reference)
```
src/connects_common_connectivity/models.py      # Pydantic models — read to understand fields
src/connects_common_connectivity/io/arrow_utils.py # build_arrow_schema, models_to_table,
                                                    # attach_linkml_metadata,
                                                    # build_cell_feature_matrix_schema
src/connects_common_connectivity/io/write_utils.py # walk_ancestors
src/connects_common_connectivity/io/writers.py     # write_models, write_projection_matrix
```

### Example notebooks (read for patterns)
```
code/etl_visp_inh_patchseq_01_dataset_dataitem.ipynb            # canonical _01 pattern
code/etl_visp_inh_patchseq_02_cell_features.ipynb               # _02 with new-cell registration
code/etl_visp_exc_patchseq_02_cell_features.ipynb               # _02 without new-cell registration
code/etl_wnm_exc_02_cell_features.ipynb                         # _02 with three feature sets, shared defs
code/etl_minnie_02_cell_features.ipynb                          # _02 with CAVE query + two feature sets
code/etl_tasic_01_cluster.ipynb                                 # _01 that owns a global cluster taxonomy (no project_id)
code/etl_visp_exc_patchseq_03_cluster_membership_and_mapping.ipynb  # canonical _03 — both membership and mapping
code/etl_minnie_03_cluster_and_cluster_membership.ipynb         # _03 that owns its own taxonomy (one notebook, both ends)
code/etl_minnie_04_cell_cell.ipynb                              # _04 cell-cell connectivity, two-folder example pattern
code/etl_wnm_exc_04_projection_matrix.ipynb                     # _04 projection matrix + new-cell registration
```

Also read `code/etl_examples_readme.ipynb` for a plain-language summary of existing datasets and feature sets.

---

## 2. Hard rules — never break these

1. **Never edit `src/` or `models.py` directly.**
   `models.py` is auto-generated. If a schema change is needed, edit the relevant `schemas/*.yaml` file and regenerate:
   ```bash
   uv run gen-pydantic schemas/connectivity_schema.yaml > src/connects_common_connectivity/models.py
   ```

2. **Schemas are the contract.** Do not invent fields that aren't in the schema. If a field you need doesn't exist, ask whether the schema should be extended first.

3. **Never cast id values.** Cell ids come from source files as strings (or ints that should be stored as strings). Use them as-is. Do not zero-pad, strip, or reformat.

4. **Use enum `.value` for enum slots**, e.g. `Modality.MORPHOLOGY.value`, never the raw string.

5. **Every write must have a verification cell** immediately after: read back with `pl.read_delta`, print shape and `head(3)`, and assert at least one invariant (row count, unique ids, or correct column value).

6. **Markdown cells: 1–3 sentences.** No prose dumps. The title cell states what is written and lists identifiers. The summary cell lists every output path with its row count.

7. **Scoping rules differ by table family.**
   - **Project-scoped** (most tables): scoped by `project_id`, plus a discriminator second level (see §5b).
   - **Global cluster taxonomy** (`Cluster`, `ClusterHierarchy`, `AlgorithmRun`): no `project_id`. Scoped by `hierarchy_id` (or `id` for the hierarchy/run rows themselves) so multiple taxonomies share one table.
   - **Global category vocabulary** (`HierarchyCategory`): no `project_id`, no `hierarchy_id`. Category ids (`class`, `subclass`, `cluster`) are intentionally shared across taxonomies — see §11.
   - **Project-scoped *and* taxonomy-scoped**: `ClusterMembership` (project + `hierarchy_id`), `CellToClusterMapping` (project + `mapping_set`).

8. **Output root.** All notebooks write to `OUTPUT_ROOT = "../scratch/em_patchseq_wnm_v1/"`. Define this as a constant in cell 3.

---

## 3. Notebook naming convention

```
etl_<dataset>_<NN>_<schemas>.ipynb
```

- `<dataset>`: `minnie`, `tasic`, `visp_met_types`, `visp_inh_patchseq`, `visp_exc_patchseq`, `wnm_exc`
- `<NN>`: two-digit run-order within the dataset (`01`, `02`, …)
- `<schemas>`: snake_case names joined by `_and_`
  - Allowed values: `dataset_dataitem`, `cluster`, `cluster_membership`, `cell_features`, `projection_matrix`, `cell_cell`, `single_cell_recon`, `brain_region_assoc`, `cell_to_cluster_mapping`, `mapping`

Examples: `etl_visp_inh_patchseq_01_dataset_dataitem.ipynb`, `etl_wnm_exc_02_cell_features.ipynb`, `etl_visp_exc_patchseq_03_cluster_membership_and_mapping.ipynb`, `etl_minnie_04_cell_cell.ipynb`, `etl_wnm_exc_04_projection_matrix.ipynb`.

---

## 4. Canonical notebook structure

Every notebook follows this cell order. Do not skip sections.

| # | Cell type | Content |
|---|---|---|
| 1 | Markdown | **Title** — what is written, dataset/project identifiers, prerequisites |
| 2 | Code | **Imports** — stdlib, pandas, polars, pyarrow, deltalake, package imports |
| 3 | Code | **Constants** — all paths and identifiers as ALL_CAPS variables; `print` each |
| 4 | Code | **Prerequisite check** — assert prior notebooks' outputs exist; fail loudly with a message naming the missing notebook |
| 5+ | Code + Code | **Load input** → print shape and `head(3)` |
| … | Code + Code | **Write section** → build models → arrow schema → write → verify cell |
| Last | Markdown | **Summary** — table of every output path with row counts; note any intentionally omitted columns |

---

## 5. Write pattern reference

### 5a. Registry-backed tables — use `write_models` (do not hand-build predicates)

```python
from connects_common_connectivity.io import write_models

result = write_models(rows, output_root=OUTPUT_ROOT)
print(result.mode, result.rows_written, result.predicates)
```

- `write_models` infers class from `rows`, then applies the registered `subdir`, `partition_by`, scope predicate, and write mode from `io/write_spec.py`.
- For registry tables, notebook code should **not** call `write_deltalake` directly and should not build `predicate=` strings by hand.
- Re-running is idempotent when you pass the full intended scope slice (the standard pattern in current notebooks).

### 5b. `DataItem` writes are id-deduped append via `write_models`

```python
new_dataitems = [DataItem(id=cid, name=cid, project_id=PROJECT_ID) for cid in new_ids]
written = write_models(new_dataitems, output_root=OUTPUT_ROOT).rows_written
print(f"DataItems appended: {written}")
```

- `DataItem` dispatches to `append_new_by_id` (id dedupe within one `project_id` per call).
- Re-running with the same ids appends nothing.
- **Do not** use scoped overwrite for `dataitem/`.

### 5c. Wide-form feature parquet — `mode="overwrite"` with predicate on `project_id`

Each feature set lives in its own subdirectory (`cellfeatures/<feature_set_id>/`), so the directory already scopes to one feature set. Predicate only needs `project_id`:

```python
write_deltalake(
    OUTPUT_ROOT + f"cellfeatures/{FEATURE_SET_ID}/", arrow_table,
    mode="overwrite",
    predicate=f"project_id = '{PROJECT_ID}'",
    partition_by=["project_id", "feature_set_id"],
)
```

**Exception:** if two notebooks write to the same `cellfeatures/<fsi>/` directory with different `project_id`s (e.g. `exc_visp_morph_features` shared between patchseq and WNM), the `project_id` predicate correctly scopes each notebook's write without touching the other's rows.

### 5d. New-cell registration in `_02` notebooks

If a feature CSV contains cell ids not present in the `_01` DataItems, register them before writing features:

1. Read `dataitem_dataset_association/` filtered to `project_id AND dataset_id` → collect existing ids.
2. Identify new ids (`set(csv_ids) - set(existing_ids)`).
3. Call `write_models([...DataItem(...)...], output_root=OUTPUT_ROOT)` for any new cells.
4. Re-assert the full `(project_id, dataset_id)` association scope with `write_models([...DataItemDataSetAssociation(...)...])` (pass the full intended set, not append-only deltas).

### 5e. Cluster taxonomy tables (global)

`cluster/`, `clusterhierarchy/`, `algorithmrun/`, and `hierarchycategory/` have **no `project_id`**. Write through `write_models`; registry scopes are:

- `Cluster`: `hierarchy_id`
- `ClusterHierarchy`: `id`
- `AlgorithmRun`: `id`
- `HierarchyCategory`: `id`

```python
write_models(cluster_rows, output_root=OUTPUT_ROOT)
write_models([hierarchy_row], output_root=OUTPUT_ROOT)
write_models([run_row], output_root=OUTPUT_ROOT)
write_models(category_rows, output_root=OUTPUT_ROOT)
```

See `etl_tasic_01_cluster.ipynb` and `etl_visp_met_types_01_cluster.ipynb`.

### 5f. Membership and mapping (project-scoped, per-hierarchy)

- `ClusterMembership` is scoped by `project_id AND hierarchy_id`.
- `CellToClusterMapping` is scoped by `project_id AND mapping_set`.
- `MappingSet` is scoped by `project_id AND id`.

When two notebooks merge into the same scoped slice (for example, both patch-seq `_03` notebooks writing memberships for the same `(project_id, hierarchy_id)`), each notebook should write the full intended slice via `write_models(...)`. Re-running either notebook remains idempotent.

### 5g. Cell-cell connectivity (`cellcellconnectivitylong/`)

`CellCellConnectivityLong` rows have no per-example discriminator yet (no `connectome_id` slot). Two examples for the same project would overwrite each other if written into the same folder. **Workaround until the schema adds a discriminator:** write each example to its own subdirectory, e.g.

```
cellcellconnectivitylong_proofread_pre_to_csm_post/
cellcellconnectivitylong_proofread_to_proofread/
```

Predicate `project_id` only; the folder scopes the example. See `etl_minnie_04_cell_cell.ipynb`.

### 5h. Projection matrix (`projectionmeasurementmatrix/` + wide-form parquet)

Use `write_projection_matrix(pmm_row, dense_matrix, output_root=OUTPUT_ROOT)` for `ProjectionMeasurementMatrix` rows; it computes `region_coverage` from the dense matrix and delegates to the registry-backed writer. Keep direct `write_deltalake` only for the underlying wide-form `projectionmeasurementmatrix/<matrix_id>/` parquet folders. See `etl_wnm_exc_04_projection_matrix.ipynb`.

### 5i. Membership vs mapping

Same shape (cell → cluster), different meaning:

- **`ClusterMembership`** — the cell *belongs to* this cluster by definition. Use when the cell was part of the cohort that **defined** the taxonomy (e.g. inhibitory and excitatory Patch-seq cells get memberships in the VISp MET-types taxonomy they helped define).
- **`CellToClusterMapping`** + a `MappingSet` row — the cell was *assigned* to this cluster after the fact by some named classifier (e.g. WNM cells get mappings into VISp MET-types via random forest, with `probability` per call).

If the cells were not in the cohort that defined the taxonomy, write `CellToClusterMapping`, not `ClusterMembership`.

### 5j. Parent propagation (`walk_ancestors`)

Every membership and mapping is parent-propagated: one row per (cell × ancestor) all the way up to the root. Use `walk_ancestors` from `io.write_utils`:

```python
from connects_common_connectivity.io.write_utils import walk_ancestors

for ancestor_id, is_leaf in walk_ancestors(leaf_id, parent_by_child):
    ...  # build one row, set probability/membership_score on the leaf only
```

`probability` (mapping) and `membership_score`/`distance` (membership) are set on the leaf row only; null on parents.

---

## 6. Building arrow tables

```python
from connects_common_connectivity.io.arrow_utils import (
    build_arrow_schema,
    models_to_table,
    attach_linkml_metadata,
    build_cell_feature_matrix_schema,  # for wide-form feature parquets only
)

schema = build_arrow_schema(MyModelClass)
table  = attach_linkml_metadata(
    models_to_table(list_of_model_instances, schema=schema),
    linkml_class="MyModelClass",
)
```

For wide-form cell feature tables, use `build_cell_feature_matrix_schema` instead of `build_arrow_schema`:

```python
schema = build_cell_feature_matrix_schema(
    feature_set_obj,        # CellFeatureSet instance
    feature_def_objs,       # list of CellFeatureDefinition instances (must match column order)
    cell_index_column="id",
)
arrow_table = pa.Table.from_pandas(wide_df, schema=schema)
```

Column order in the wide DataFrame must match the order of `feature_def_objs`. Build defs and the wide table from the same source to guarantee alignment.

**Both `models_to_table` and `attach_linkml_metadata` are kwarg-only.** Positional calls (`models_to_table(rows, MyModelClass)`, `attach_linkml_metadata(table, "MyModelClass")`) fail with confusing schema-construction errors. Always pass `schema=` and `linkml_class=` explicitly.

---

## 7. `CellFeatureMatrix.parquet_path` format

Must match `^(s3://|gs://|https?://|file://).+`. Use:

```python
from pathlib import Path
parquet_path = f"file://{Path(OUTPUT_ROOT).resolve()}/cellfeatures/{FEATURE_SET_ID}/"
```

---

## 8. `data_type` format for `CellFeatureDefinition`

Must be a numpy dtype string matching `^([<>|=])[tbiufcmMOSUV]\d+$`. Examples:

| Python/numpy type | `data_type` value |
|---|---|
| float32 | `<f4` |
| float64 | `<f8` |
| int32 | `<i4` |
| int64 | `<i8` |

`"float32"` or `"float64"` will **fail validation**. Always use the dtype string form.

---

## 9. Shared feature sets across projects

When two projects (different `project_id`) share a feature set (same `feature_set_id`):

- **One notebook owns the defs and `CellFeatureSet`** — the one that writes it first (by convention, the patchseq notebook).
- **The second notebook reads defs back** from `cellfeaturedefinition/` filtered to `feature_set_id`, uses them to build the schema, and writes only its own rows.
- The second notebook **must not** write `cellfeaturedefinition/` or `cellfeatureset/` for the shared set.
- Column order in the second notebook's wide table must match the shared defs exactly. If any column is missing, either fail loudly or NaN-fill with an explicit warning — never silently drop or reorder.

---

## 10. Common mistakes and how to avoid them

| Mistake | What goes wrong | Correct approach |
|---|---|---|
| Calling `write_deltalake` directly for a registry-backed model table | Notebook-level predicate/partition drift from `io/write_spec.py` | Use `write_models(...)` |
| Hand-building `predicate=` / `partition_by=` for model writes | Scope bugs (row loss or accidental clobber) | Let `write_models` apply the registered scope |
| Writing `DataItem` with overwrite or plain append | Clobbers or duplicates within a project partition | Use `write_models(DataItem(...))` (append_new_by_id) |
| Appending only delta associations in `_02`/`_03` notebooks | Partial reruns can leave missing links | Re-write the full `(project_id, dataset_id)` scoped association slice with `write_models` |
| Raw string for enum slot (`modality="MORPHOLOGY"`) | Pydantic validation error | Use `Modality.MORPHOLOGY.value` |
| Casting or reformatting id values | Ids won't match across tables | Use ids as-is from the source file |
| Editing `models.py` directly | Changes lost on next schema regen | Edit the schema YAML, then regenerate |
| Inventing a field not in the schema | Pydantic validation error | Check the schema YAML first; extend if needed |
| Verifying with `project_id` filter only on a shared table | Asserts pass but row count is wrong (includes other dataset) | Always filter by both `project_id` and `dataset_id` (or `feature_set_id`) |
| Positional `models_to_table(rows, ModelClass)` or `attach_linkml_metadata(table, "Cluster")` | Silent schema-construction error, opaque message | Use `schema=` and `linkml_class=` kwargs |
| Setting `AlgorithmRun.produced_hierarchies = [hierarchy]` | Pydantic expects an inlined dict, not a list — validation error | Omit it; `ClusterHierarchy.run` carries the inverse link |
| Manual overwrite on `clustermembership/` scoped only by `project_id` | Wipes other hierarchies' rows for the same project | Use `write_models` (`ClusterMembership` scope is `project_id AND hierarchy_id`) |
| Writing `ClusterMembership` for cells not in the cohort that defined the taxonomy | Misrepresents provenance — they were classified, not members | Use `CellToClusterMapping` + a `MappingSet` row instead |

---

## 11. Known limitations

- **`HierarchyCategory` rows are id-scoped global vocabulary rows.** Because ids like `class`, `subclass`, and `cluster` are shared across taxonomies, only write canonical shared definitions (same ids/meaning) via `write_models`. Do not invent taxonomy-specific category ids without a schema-level discriminator.
- **`CellCellConnectivityLong` has no `connectome_id` discriminator.** Two example connectomes for the same project must live in separate folders (see §5g). Schema addition would let them share a folder.
