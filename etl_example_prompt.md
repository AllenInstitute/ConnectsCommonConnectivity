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
Read the relevant domain schema too if writing projection, clustering, mapping, or spatial data:
```
schemas/clustering_schema.yaml
schemas/mappings_schema.yaml
schemas/projection_schema.yaml
schemas/single_cell_schema.yaml
schemas/spatial_schema.yaml
schemas/cell_gene_schema.yaml
```

### Package utilities (read-only reference)
```
src/connects_common_connectivity/models.py      # Pydantic models — read to understand fields
src/connects_common_connectivity/io/arrow_utils.py # build_arrow_schema, models_to_table,
                                                    # attach_linkml_metadata,
                                                    # build_cell_feature_matrix_schema
src/connects_common_connectivity/io/write_utils.py # walk_ancestors
src/connects_common_connectivity/io/writers.py     # write_models, write_projection_matrix
src/connects_common_connectivity/io/read.py        # read_spatial_locations, read_reference_spaces
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
    bash scripts/generate_models.sh
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
    - **Spatial coordinates**: `SpatialLocation` merges on `(project_id, dataitem_id, reference_space, location_type)`. `ReferenceSpace` merges on globally unique `id`; its optional `project_id` identifies ownership, not identity. See §5k.

8. **Output root.** All notebooks obtain the write root with `OUTPUT_ROOT = get_settings().output_root` (imported from `connects_common_connectivity.config`) — an absolute `pathlib.Path` sourced from `ccc_config.yaml`. Define this in cell 3 and build subpaths with `/`, e.g. `OUTPUT_ROOT / "dataset"`.

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
    OUTPUT_ROOT / f"cellfeatures/{FEATURE_SET_ID}", arrow_table,
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

Every `CellCellConnectivityLong` row requires `connectome_id`, which independently identifies the measurement context (segmentation version, proofreading state, and measurement semantics). `synapse_table_id` is optional provenance for connectivity derived from a single-synapse table; cell-cell connectivity produced by other methods can omit it. DataSet IDs remain reserved for collections of DataItems.

Use `derive_cell_cell_connectivity(...)` to aggregate a Polars synapse frame by project and pre/post endpoints. It always emits `SYNAPSE_COUNT` with unit `COUNT`. Supplying both `size_column` and `size_unit` additionally emits `SUM_ANATOMICAL_SIZE`; the size column must be numeric and contain no null values. When all input rows share one non-null `synapse_table_id`, the transform preserves it as optional provenance without using it for grouping or row IDs.

Use `read_cell_cell_connectivity(project_id, connectome_id, ...)` to read canonical `cellcellconnectivitylong/` storage. It can optionally filter `synapse_table_id` provenance, explicit presynaptic IDs, postsynaptic IDs, and measurement types. `read_synapse_table` instead requires its logical `synapse_table_id` and adds feature-join controls. DataSet and cluster cohort resolution is not implemented here and remains issue #23.

All cell-cell ETLs write to the same canonical directory:

```
cellcellconnectivitylong/
```

Direct Delta writes must overwrite only `(project_id, connectome_id)` and use consistent partitions. Issue #19 still owns `write_models` registration; folder consolidation does not depend on it. Canonical model-table paths and dataframe-backed payload roots come from `connects_common_connectivity.io.path_spec`. See `etl_minnie_04_cell_cell.ipynb` and `etl_v1dd_03_synapses.ipynb`.

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

### 5k. Spatial coordinates and reference spaces

Use `SpatialLocation` for anatomical coordinates, not a generic cell feature set.
`SingleCellReconstruction.soma_location` and `CellMetadata.spatial_location` have
been removed. Choose an explicit `LocationType`: `SOMA`, `CENTROID`,
`INJECTION_SITE`, or `OTHER`. There is no default location type. One DataItem can
have several types in one space, and locations in several spaces, but only one
row per type/space within its project.

Before writing coordinates:

1. Register every referenced cell as a `DataItem` in the same project. Preserve
     its existing ID. For cell x gene data, `CellGeneData.cell_index` is the ordered
     list of DataItem IDs, not an unrelated Zarr row-number namespace. Preserve the
     mapping from matrix rows to those IDs. Writers do not enforce foreign keys;
     check these references in the ETL.
2. Establish a `ReferenceSpace` ID for each distinct coordinate frame/version.
     IDs must be globally unique even for project-owned spaces. Do not reuse an
    ID for different units, transforms, or frame versions. Record known coordinate
    units in `unit`. For voxel coordinates, use `unit=Unit.VOXELS` and specify
    physical scale with `voxel_size` (ordered x, y, z) and `voxel_size_unit`.
    Record provenance and axis meanings in `description`.
3. Give each space a single owning ETL. Global spaces use `project_id=None`;
     dataset-local spaces use that dataset's project. Consumers read existing
     space definitions rather than repeatedly overwriting shared metadata.
4. Set `default_2d_view` only when its directions are confirmed. A missing view
     means no recommendation, not an implicit x/y or y-up fallback. Both axes are
     required when a view is supplied, and they must refer to different underlying
     axes: `PLUS_X` / `MINUS_X` is invalid.

Write `ReferenceSpace` and `SpatialLocation` through `write_models`, never a
hand-built Delta predicate. Storage is `referencespace/` (unpartitioned) and
`spatiallocation/` (partitioned by project). These are upserts: reruns update the
same identity and preserve other rows; they do not delete stale coordinates.

Example write and verification cells, using a previously registered `CELL_ID`
and a space owned by this ETL:

```python
from connects_common_connectivity.io import (
        read_reference_spaces, read_spatial_locations, write_models,
)
from connects_common_connectivity.models import (
    Default2DView, LocationType, ReferenceSpace, SignedAxis, SpatialLocation, Unit,
)

SPACE_ID = "v1dd_streamline"
space = ReferenceSpace(
        id=SPACE_ID,
        project_id=PROJECT_ID,
    unit=Unit.MICRONS_LENGTH,
        description="V1DD streamline coordinates in micrometers; y increases toward white matter.",
        default_2d_view=Default2DView(
                left_to_right=SignedAxis.PLUS_X.value,
                bottom_to_top=SignedAxis.MINUS_Y.value,
        ),
)
write_models(space, output_root=OUTPUT_ROOT)
```

```python
spaces = read_reference_spaces(
        project_id=PROJECT_ID, reference_space_ids=SPACE_ID, output_root=OUTPUT_ROOT,
)
print(spaces.shape, spaces.head(3))
assert spaces.height == 1
assert spaces["default_2d_view"].struct.field("bottom_to_top").to_list() == ["MINUS_Y"]
```

```python
location = SpatialLocation(
        project_id=PROJECT_ID, dataitem_id=CELL_ID, reference_space=SPACE_ID,
        location_type=LocationType.SOMA.value,
        x=soma_x, y=soma_y, z=soma_z,
)
write_models(location, output_root=OUTPUT_ROOT)
```

```python
locations = read_spatial_locations(
        PROJECT_ID, reference_spaces=SPACE_ID, dataitem_ids=CELL_ID,
        location_types=LocationType.SOMA.value, output_root=OUTPUT_ROOT,
)
print(locations.shape, locations.head(3))
assert locations.height == 1
assert locations.select("x", "y", "z").row(0) == (soma_x, soma_y, soma_z)
```

For larger batches, verify complete identity uniqueness, reference membership,
and source-coordinate equality, not just row count. The writer resolves duplicate
incoming keys using the final row, so detect unintended duplicates before writing.
Reject or explicitly report missing coordinate triplets; do not silently invent
zero coordinates. Read helpers preserve numerical values and do not transform
coordinates or join reference-space metadata automatically.

To display a confirmed view, select the axis named by each signed value and
negate it for `MINUS_`. For the streamline example, screen horizontal is x and
screen vertical is -y, putting pia above white matter. The view remains metadata;
do not negate stored y values just to orient a plot.

#### V1DD pilot handoff (future Code Ocean work)

The current V1DD notebooks still contain the legacy `v1dd_soma_spatial` feature
output. This guide defines the new contract; no notebook execution, production
backfill, or scientific transform verification has been performed locally.

- Coordinate production is in `code/etl_v1dd_02_cave.ipynb`; coordinate read
    examples are in `code/etl_v1dd_04_read.ipynb`. Restrict the future change to
    their relevant spatial write/read cells rather than reworking the full ETL.
- Preserve registered DataItem IDs and produce two `SOMA` rows per eligible cell:
    one in a separately identified original EM space, one in `v1dd_streamline`.
- Preserve original EM source values. The query requests CAVE resolution
    `[1, 1, 1]` despite the old `soma_voxel_*` names. Verify the source unit contract
    in Code Ocean before choosing the space ID/description; never infer units from
    those names or label unknown values as voxels. Leave this space's view unset.
- Reuse the current transformed values, including the existing conversion to
    micrometers. Do not apply that conversion twice or recalculate the transform.
    `v1dd_streamline` uses the confirmed `PLUS_X` / `MINUS_Y` default.
- For global `CCF_v3`, the agreed reference view is `PLUS_Z` / `MINUS_Y` with
    `project_id=None`. Its owner seeds it once; do not relabel V1DD-local coordinates
    as CCF without an actual registration transform.
- Keep nucleus volume and other non-coordinate measurements as cell features.
    Split the mixed legacy output deliberately, updating feature definitions,
    feature-set metadata, and matrix pointers together. Do not drop volume while
    removing the six coordinate feature columns.
- Verify source units, coordinate equality, expected rows per space, complete
    identity uniqueness, DataItem membership, and pia-up display on real data.
    Rerun the targeted write to confirm idempotency and preservation of other scopes.
- Do not delete or backfill existing published feature matrices as part of this
    change. Their cleanup and any compatibility period need a separate ETL scope.

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
parquet_path = f"file://{OUTPUT_ROOT.resolve()}/cellfeatures/{FEATURE_SET_ID}/"
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
- **Canonical cell-cell persistence is not registered yet.** ETLs write directly to canonical `cellcellconnectivitylong/`, scoped by `(project_id, connectome_id)`. Issue #19 still owns registration with the generic model writer (see §5g).
