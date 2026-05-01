# Add ETL notebooks for VISp Patch-seq, WNM, and Minnie65 datasets

## Summary

- Adds ETL Jupyter notebooks across four projects (`visp_patchseq`, `visp_wnm`, `minnie65`, plus the global Tasic / VISp MET-types reference taxonomies), covering phases `_01` (DataSet + DataItem), `_02` (cell features), `_03` (cluster membership and cell-to-cluster mapping), and `_04` (cell-cell connectivity, projection matrices).
- Three schema fixes required to support the above: `CellFeatureDefinition` / `CellFeatureSet` made `ProjectScoped` (+ `feature_set_id`), `Cluster` made global with `hierarchy_id` (+ `ClusterMembership.hierarchy_id`), `MappingSet` endpoints generalized to `DataSet` or `ClusterHierarchy`.
- New `src/connects_common_connectivity/write_utils.py` helpers: `append_new_dataitems` (idempotent registration across notebooks sharing `project_id`) and `walk_ancestors` (parent propagation for `_03` notebooks).
- Documentation: `code/etl_examples_readme.ipynb`, `etl_example_prompt.md`. All 51 tests pass.

---

## Schema and source-code changes

### Schema fix: `CellFeatureDefinition` and `CellFeatureSet` were not `ProjectScoped`

#### The problem

`CellFeatureDefinition` and `CellFeatureSet` did not inherit the `ProjectScoped` mixin, so they had no `project_id` field. This broke the uniform write pattern used everywhere else in the project:

```python
write_deltalake(..., mode="overwrite", predicate=f"project_id = '{PROJECT_ID}'", partition_by=["project_id"])
```

Without `project_id`, there was no way to scope a predicate overwrite to a single project. The only alternatives were:

1. **Plain `mode="overwrite"`** — wipes the entire shared table across all projects. Unsafe.
2. **`mode="append"`** — accumulates duplicate rows on every re-run. Unusable.
3. **A non-`project_id` predicate** (e.g. `feature_set_id IN (...)`) — fragile; requires enumerating every id belonging to the project, and still breaks for a new project that reuses a feature set name.

In the first draft of `etl_minnie_02`, the CSM feature definitions were written with predicate-overwrite, and the three standard_transform coordinate definitions were written with `mode="append"` in a second call. This meant re-running the notebook would accumulate duplicate coordinate definitions.

#### The fix

Added `mixins: [ProjectScoped]` and `- project_id` to both `CellFeatureSet` and `CellFeatureDefinition` in `schemas/cell_features_schema.yaml`, then regenerated `models.py`:

```bash
uv run gen-pydantic schemas/connectivity_schema.yaml > src/connects_common_connectivity/models.py
```

`models.py` is never edited manually; `schemas/connectivity_schema.yaml` is the source of truth.

With `project_id` on both classes, `etl_minnie_02` can collect all feature definitions (82 CSM + 3 coords = 85) and write them in a single idempotent overwrite, and likewise for `CellFeatureSet` (2 rows) and `CellFeatureMatrix` (2 rows).

#### Follow-up: `feature_set_id` on `CellFeatureDefinition`

After adding `project_id`, a second scoping problem emerged: two feature sets in the same notebook (`csm_cluster_features` and `minnie65_std_transform_coordinates`) both belong to `project_id="minnie65"`. Writing their definitions in separate cells was still impossible — a `mode="overwrite"` with `predicate="project_id='minnie65'"` in the second cell would wipe the first cell's rows.

The fix was to add `feature_set_id` as an optional field to `CellFeatureDefinition` in the schema. This enables a two-level predicate:

```python
predicate=f"project_id = '{PROJECT_ID}' AND feature_set_id = '{FSI}'"
```

Each feature-set section in `etl_minnie_02` now scopes its `cellfeaturedefinition/` write to its own `feature_set_id`. The same two-level predicate is used for `cellfeaturematrix/` (which already had `feature_set_id`). `cellfeatureset/` uses `project_id + id` since the row's own `id` is the feature set id.

The notebook is now two self-contained sections (A: CSM, B: STD coordinates), each independently re-runnable without affecting the other. `cellfeaturedefinition/` also gains `partition_by=["project_id", "feature_set_id"]` for query performance.

### Schema fix: cluster classes — global taxonomies, project-scoped memberships

#### The problem

Cluster ETL (next set of notebooks) needs taxonomies (`ClusterHierarchy`, `Cluster`, `AlgorithmRun`) to be **global reference artifacts** — owned by their authoring ETL but consumable across projects (e.g. VISp MET types written by `etl_visp_met_types_01` and consumed by both `etl_visp_exc_patchseq_03` and `etl_wnm_exc_03`). Per-cell tables (`ClusterMembership`, `CellToClusterMapping`) remain project-scoped because they belong to the project whose cells are being labelled.

Two inconsistencies broke this model:

1. `Cluster` had `mixins: [ProjectScoped]`. A `Cluster` belongs to a `ClusterHierarchy`, not a project — and forcing a `project_id` makes cross-project consumption awkward (WNM cells mapping to VISp MET clusters would need to query a different `project_id` than their own). Compounding this, `Cluster` had **no taxonomy discriminator at all**, so multiple taxonomies coexisting in the same `cluster/` Delta table couldn't be overwritten independently.
2. `ClusterMembership` had no taxonomy discriminator. A single project can label its cells against multiple taxonomies (e.g. `visp_patchseq` mapping cells to both Tasic and VISp MET); without a `hierarchy_id` slot, the standard `predicate=project_id AND <discriminator>` write pattern can't isolate one taxonomy's memberships from another's.

`CellToClusterMapping` already has `mapping_set` (required) + `ProjectScoped`, so its scoping was already correct — no change needed there.

#### The fix

In `schemas/clustering_schema.yaml`:
- Removed `mixins: [ProjectScoped]` from `Cluster`. Taxonomies are now global; `Cluster` no longer carries a `project_id` field.
- Added an optional top-level `hierarchy_id` slot (range: `string`, references `ClusterHierarchy.id`, not inlined) and added it to both `ClusterMembership.slots` and `Cluster.slots`. Same slot is reused on both classes; mirrors the `feature_set_id` pattern on `CellFeatureDefinition`. Not added to `HierarchyCategory` (rows like `class`, `subclass`, `cluster` are reused across taxonomies) or `AlgorithmRun` (its own `id` is the discriminator and one run produces one or more hierarchies).

Models regenerated via `bash scripts/generate_models.sh`. After regeneration, `class Cluster(ConfiguredBaseModel)` (no longer `ProjectScoped`) gains `hierarchy_id: Optional[str]`, and `class ClusterMembership(ProjectScoped)` gains `hierarchy_id: Optional[str]`.

### Schema fix: `MappingSet` endpoints — DataSet or ClusterHierarchy

#### The problem

`MappingSet` required both `source_dataset: DataSet` and `target_dataset: DataSet`. Under the just-completed cluster-schema change, `Cluster` and `ClusterHierarchy` are global reference artifacts — not linked to any `DataSet`. So for `CellToClusterMapping` the target is a hierarchy (no honest `target_dataset`), and `ClusterToClusterMapping` has the same problem on both ends. Only `CellToCellMapping` fits the original schema.

#### The fix

In `schemas/mappings_schema.yaml`:
- Added top-level slots `source_hierarchy` and `target_hierarchy` (range `ClusterHierarchy`, `inlined: false`).
- Added both to `MappingSet.slots`.
- Dropped `required: true` from `MappingSet.source_dataset` and `MappingSet.target_dataset`; both are now optional. LinkML can't enforce "exactly one of {dataset, hierarchy}" per side, so the convention is documented in `MappingSet.description` and the per-mapping class descriptions:
  - `CellToCellMapping`     → `MappingSet` populates `source_dataset`    + `target_dataset`.
  - `CellToClusterMapping`  → `MappingSet` populates `source_dataset`    + `target_hierarchy`.
  - `ClusterToClusterMapping` → `MappingSet` populates `source_hierarchy` + `target_hierarchy`.

Models regenerated via `bash scripts/generate_models.sh`. `MappingSet` now exposes all four endpoint slots, all optional.

### `write_utils.py`: idempotent DataItem registration + parent-propagation helper

#### The problem

Two `_01` notebooks can share the same `project_id`: both `etl_visp_inh_patchseq_01` and `etl_visp_exc_patchseq_01` use `project_id="visp_patchseq"`. Previously both wrote `dataitem/` with:

```python
write_deltalake(OUTPUT_ROOT + "dataitem/", table_di,
                mode="overwrite", predicate=f"project_id = '{PROJECT_ID}'", ...)
```

A predicate-scoped overwrite on `project_id='visp_patchseq'` wipes **all** rows for that partition — so whichever notebook ran second silently deleted the first's cells. `etl_visp_inh_patchseq_02` saw "Already in DataItem: 0" even after `_01` had registered 2,759 cells, because `exc_01` overwrote the partition.

The `dataitem_dataset_association/` predicate also only scoped to `project_id`, making it equally fragile when a second dataset shares the project.

#### The fix

**`src/connects_common_connectivity/write_utils.py`** introduces `append_new_dataitems`:

```python
def append_new_dataitems(output_path, table, *, project_id, id_column="id") -> int:
    """Append only rows whose id is not already present for this project.
    Idempotent: re-running appends nothing. Handles missing table gracefully."""
```

It reads existing `(project_id, id)` pairs, filters the incoming table to only new rows, and appends with `mode="append"`. Re-running returns 0 and writes nothing. Two notebooks sharing `project_id` each only add their own cells without touching the other's.

All four `_01` notebooks now import and use `append_new_dataitems` for the `dataitem/` write, and their `dataitem_dataset_association/` predicates are narrowed to `project_id = '...' AND dataset_id = '...'` so each dataset's association rows are independently idempotent.

`etl_visp_inh_patchseq_02` is updated to use `append_new_dataitems` for the 120 new-cell registrations, replacing the previous read-union-write pattern.

#### Follow-up: `walk_ancestors`

The `_03` cluster-membership/mapping notebooks share a parent-propagation step (one row per (cell × ancestor) up to the root). `walk_ancestors(leaf_id, parent_of)` was added to `write_utils.py` as a small generator, given the `parent_of` map built from a `Cluster` table filtered to a single `hierarchy_id`. All four `_03` notebooks consume it.

---

## ETL notebooks

All notebooks follow the same conventions:
- Inputs are loaded (CSV or CAVE query), printed with shape and `head(3)`.
- Outputs use `mode="overwrite"` with two-level predicates (`project_id AND <discriminator>`) so re-runs are idempotent and other projects' rows in shared Delta tables are never touched.
- `dataitem/` uses `append_new_dataitems` (see below) instead of predicate-overwrite.
- Every write step is followed by a verification cell (`pl.read_delta`, shape check, `assert`).

### `_01` Dataset + DataItem notebooks

| Notebook | `project_id` | `dataset_id` | Input |
|---|---|---|---|
| `etl_visp_inh_patchseq_01_dataset_dataitem.ipynb` | `visp_patchseq` | `visp_inh_patchseq` | `patchseq_tx_cell_ttype_labels.csv` (2,759 cells) |
| `etl_visp_exc_patchseq_01_dataset_dataitem.ipynb` | `visp_patchseq` | `visp_exc_patchseq` | `inferred_met_types.csv` (1,528 cells) |
| `etl_wnm_exc_01_dataset_dataitem.ipynb` | `visp_wnm` | `visp_exc_wnm` | `FullMorphMetaData_Master.csv` (341 cells; `.swc` suffix stripped from index) |
| `etl_minnie_01_dataset_dataitem.ipynb` | `minnie65` | `minnie65_v1412_nuclei` | CAVE `nucleus_detection_lookup_v1` (version 1412, `pt_root_id != 0`) |

Each notebook writes exactly three tables: `DataSet`, `DataItem`, and `DataItemDataSetAssociation`.

#### Cluster-taxonomy `_01` notebooks (no DataItem)

Two `_01`s register **global** cluster reference taxonomies — they own `algorithmrun/`, `clusterhierarchy/`, `cluster/`, `hierarchycategory/` rows and intentionally register no `DataItem`s (the cells that defined or were assigned to these taxonomies are registered by their own project's `_01`/`_02`).

| Notebook | `hierarchy_id` | Source |
|---|---|---|
| `etl_tasic_01_cluster.ipynb` | `tasic_2018_visp_taxonomy` | `anno.feather` (class → subclass → cluster, synthetic `cell` root) |
| `etl_visp_met_types_01_cluster.ipynb` | `visp_met_types_taxonomy` | `met_type_colors.json` (45 MET-types, class → cluster, synthetic `cell` root; class colors borrowed from Tasic for visual consistency) |

### `_02` Cell feature notebooks

#### `etl_minnie_02_cell_features.ipynb`

Writes cell features for the Minnie65 dataset. Requires `etl_minnie_01`.

| Path | Class | Rows |
|---|---|---|
| `dataset/` | `DataSet` | +1 (`minnie65_v1412_csm_cluster` cohort) |
| `dataitem_dataset_association/` | `DataItemDataSetAssociation` | one per cell in `minnie_features.parquet` |
| `cellfeaturedefinition/` | `CellFeatureDefinition` | 82 CSM features + 3 std-transform coordinates |
| `cellfeatureset/` | `CellFeatureSet` | 2 (`csm_cluster_features`, `minnie65_std_transform_coordinates`) |
| `cellfeatures/csm_cluster_features/` | wide parquet | per-cell × 82 features |
| `cellfeatures/minnie65_std_transform_coordinates/` | wide parquet | per-cell x/y/z in µm |
| `cellfeaturematrix/` | `CellFeatureMatrix` | 2 pointer rows |

Soma coordinates are computed by applying `standard_transform.minnie_transform_vx()` to `pt_position`, then dividing nm → µm and casting to `float32`.

#### `etl_visp_inh_patchseq_02_cell_features.ipynb`

Writes inhibitory VISp Patch-seq morphology features. Requires `etl_visp_inh_patchseq_01`. Also registers 120 cells present in the wide CSV but not in the `_01` source.

| Path | Class | Rows |
|---|---|---|
| `cellfeaturedefinition/` | `CellFeatureDefinition` | 46 (`inh_visp_morph_features`) |
| `cellfeatureset/` | `CellFeatureSet` | 1 |
| `cellfeatures/inh_visp_morph_features/` | wide parquet | 520 cells × 46 features |
| `cellfeaturematrix/` | `CellFeatureMatrix` | 1 pointer row |
| `dataitem/` | `DataItem` | +120 new cells (via `append_new_dataitems`) |
| `dataitem_dataset_association/` | `DataItemDataSetAssociation` | +120 new associations |

#### `etl_visp_exc_patchseq_02_cell_features.ipynb`

Writes excitatory VISp Patch-seq morphology features. Requires `etl_visp_exc_patchseq_01`. All 389 cells were already registered by `_01`.

| Path | Class | Rows |
|---|---|---|
| `cellfeaturedefinition/` | `CellFeatureDefinition` | 50 (`exc_visp_morph_features`) |
| `cellfeatureset/` | `CellFeatureSet` | 1 |
| `cellfeatures/exc_visp_morph_features/` | wide parquet | 389 cells × 50 features |
| `cellfeaturematrix/` | `CellFeatureMatrix` | 1 pointer row |

#### `etl_wnm_exc_02_cell_features.ipynb`

Writes WNM excitatory neuron features across three feature sets. Requires `etl_wnm_exc_01` and `etl_visp_exc_patchseq_02` (shared defs for Set 1). Also registers 4 cells present in the Set 1 CSV but not in `_01`.

| Path | Class | Rows | Notes |
|---|---|---|---|
| `cellfeatures/exc_visp_morph_features/` | wide parquet | 345 WNM rows | Set 1; defs/set owned by exc patchseq `_02` |
| `cellfeaturematrix/` | `CellFeatureMatrix` | 1 | Set 1 pointer |
| `cellfeaturedefinition/` | `CellFeatureDefinition` | 51 (`wnm_exc_local_axon_features`) | Set 2 |
| `cellfeatureset/` | `CellFeatureSet` | 1 | Set 2 |
| `cellfeatures/wnm_exc_local_axon_features/` | wide parquet | 345 cells × 51 features | Set 2 |
| `cellfeaturematrix/` | `CellFeatureMatrix` | 1 | Set 2 pointer |
| `cellfeaturedefinition/` | `CellFeatureDefinition` | 18 (`wnm_exc_complete_axon_features`) | Set 3 (fMOST) |
| `cellfeatureset/` | `CellFeatureSet` | 1 | Set 3 |
| `cellfeatures/wnm_exc_complete_axon_features/` | wide parquet | 341 cells × 18 features | Set 3 |
| `cellfeaturematrix/` | `CellFeatureMatrix` | 1 | Set 3 pointer |
| `dataitem/` | `DataItem` | +4 new cells | via `append_new_dataitems` |
| `dataitem_dataset_association/` | `DataItemDataSetAssociation` | +4 new associations | |

For Set 1, the shared feature definitions (owned by `etl_visp_exc_patchseq_02`) are read back from `cellfeaturedefinition/` to build the wide-form arrow schema. Six columns present in the exc patchseq defs but absent from `RawFeaturesWide_ChamferCorr.csv` are NaN-filled with an explicit warning. Set 2 and Set 3 build `CellFeatureDefinition` rows directly from column names with `data_type="<f8"`.

---

### `_03` Cluster membership and cell-to-cluster mapping

Each notebook consumes globally-registered taxonomies (Tasic, VISp MET-types, or the legacy CSM hierarchy) and writes per-cell `ClusterMembership` and/or `CellToClusterMapping` rows for its project. All four use the parent-propagated convention from the reference Patch-seq pipeline: one row per (cell × ancestor) up to the root, `probability` set on the leaf only. The walk is provided by a new `walk_ancestors(leaf_id, parent_of)` helper added to `src/connects_common_connectivity/write_utils.py`.

When two notebooks write `clustermembership/` rows under the same `(project_id, hierarchy_id)` predicate (e.g. exc and inh Patch-seq both populating `(visp_patchseq, visp_met_types_taxonomy)`), they use a **merge-then-overwrite** pattern: read existing rows under the predicate, drop those whose `item` belongs to this notebook's cohort, re-validate the rest through Pydantic, concat, and overwrite. This keeps both notebooks idempotent and order-independent.

#### `etl_visp_exc_patchseq_03_cluster_membership_and_mapping.ipynb`

Source: `inferred_met_types.csv` (`t_type`, `met_type` columns).

| Path | Class | Rows |
|---|---|---|
| `mappingset/` | `MappingSet` | +1 (`visp_exc_patchseq_ttype_mapping`, `source_dataset=visp_exc_patchseq`, `target_hierarchy=tasic_2018_visp_taxonomy`) |
| `celltoclustermapping/` | `CellToClusterMapping` | one per (cell × Tasic ancestor); `t_type` legacy `ET → PT` rename applied |
| `clustermembership/` | `ClusterMembership` | one per (cell × MET-type ancestor); merge-then-overwrite |

#### `etl_visp_inh_patchseq_03_cluster_membership_and_mapping.ipynb`

Sources: `patchseq_tx_cell_ttype_labels.csv` (t-types), `visp_met_cell_assignments_text_names.csv` (MET-types). The MET CSV introduces 103 cells absent from `dataitem/` for this project — registered up front via `append_new_dataitems`, with `dataitem_dataset_association/` extended (read-union-overwrite) to keep the `(project, dataset)` predicate idempotent.

| Path | Class | Rows |
|---|---|---|
| `dataitem/` | `DataItem` | +103 new cells |
| `dataitem_dataset_association/` | `DataItemDataSetAssociation` | +103 new associations |
| `mappingset/` | `MappingSet` | +1 (`visp_inh_patchseq_ttype_mapping`) |
| `celltoclustermapping/` | `CellToClusterMapping` | one per (cell × Tasic ancestor); no `ET → PT` rename |
| `clustermembership/` | `ClusterMembership` | one per (cell × MET-type ancestor); merge-then-overwrite shared with the exc `_03` |

#### `etl_wnm_exc_03_cell_to_cluster_mapping.ipynb`

WNM excitatory cells are *mapped* into the VISp MET-types taxonomy (they didn't define it), so this notebook writes only `CellToClusterMapping` — no `ClusterMembership`. Source: `predicted_met_type` and `probability` columns of `FullMorphMetaData_Master.csv`.

| Path | Class | Rows |
|---|---|---|
| `mappingset/` | `MappingSet` | +1 (`visp_exc_wnm_mettype_mapping`, routed-RF classifier trained on the Patch-seq mMET cohort) |
| `celltoclustermapping/` | `CellToClusterMapping` | one per (cell × MET-type ancestor); leaf-only `probability` |

#### `etl_minnie_03_cluster_and_cluster_membership.ipynb`

Self-contained CSM cell-type taxonomy + memberships for Minnie65 (`hierarchy_id="minnie65_csm_cell_types"`).

The reference notebook (`code/parse_minnie_clustering.ipynb`) builds this from CAVE table `cell_type_multifeature_v1`, but **that table is gone at materialization v1412** (`404 NOT FOUND`). The cell-type metadata in `minnie_features.parquet` uses a different vocabulary (`L2a/L2b/L3a/...`) than the reference's hardcoded palette (`L2IT/L3IT/L5ET/...`), with only `L5ET` and `L5NP` shared. Mapping one to the other would require fabrication.

**Temp solution:** read the already-built legacy delta lakes at `data/microns1412/{cluster,clustermembership}/` and translate to the current schema (drop legacy `Cluster.project_id`, drop legacy `heirachy_category` typo column, stamp `hierarchy_id` everywhere). This faithfully reproduces the reference's outputs without re-running the algorithm. Caveats and the `HierarchyCategory` TODO are documented in the notebook header.

| Path | Class | Rows |
|---|---|---|
| `algorithmrun/` | `AlgorithmRun` | +1 (`minnie65_csm_clustering`) |
| `clusterhierarchy/` | `ClusterHierarchy` | +1 (`minnie65_csm_cell_types`) |
| `cluster/` | `Cluster` | 16 (1 root + 2 classes + 13 leaves) |
| `clustermembership/` | `ClusterMembership` | 107,340 (35,780 cells × 3 ancestors) |

`HierarchyCategory` writes are skipped here: the class has no `project_id`/`hierarchy_id` discriminator and category ids like `class`/`subclass`/`cluster` are intentionally shared across taxonomies, so neither overwrite nor append is safe. Needs a global-dedup append helper before any taxonomy notebook should write to that table — flagged as a TODO at the top of the notebook.

---

### `_04` Connectivity and projection matrices

#### `etl_minnie_04_cell_cell.ipynb`

Writes `CellCellConnectivityLong` synapse data for Minnie65 v1412. Creates the `minnie65_v1412_proofread` cohort DataSet (proofread ∩ CSM cells) and demonstrates two filtering patterns on the same precomputed `minnie_soma_soma_connectivity.parquet`:

| Path | Class | Rows | Filter |
|---|---|---|---|
| `dataset/` | `DataSet` | +1 (`minnie65_v1412_proofread`) | — |
| `dataitem_dataset_association/` | `DataItemDataSetAssociation` | one per proofread ∩ CSM cell | — |
| `cellcellconnectivitylong_proofread_pre_to_csm_post/` | `CellCellConnectivityLong` | 2 per pair | (proofread ∩ CSM)-pre × CSM-post; `SYNAPSE_COUNT` + `SUM_ANATOMICAL_SIZE` |
| `cellcellconnectivitylong_proofread_to_proofread/` | `CellCellConnectivityLong` | 1 per pair | (proofread ∩ CSM) × (proofread ∩ CSM); `SYNAPSE_COUNT` only |

The proofread cohort is built by querying CAVE `proofreading_status_and_strategy` (filter `status_axon`), joining to `nucleus_detection_lookup_v1`, then intersecting with the CSM cohort.

##### Current design: two separate output folders

Each example writes to its own Delta Lake path (`cellcellconnectivitylong_proofread_pre_to_csm_post/` and `cellcellconnectivitylong_proofread_to_proofread/`). This avoids overwrite collisions — since both examples share `project_id` and the `SYNAPSE_COUNT` measurement type, a single-folder predicate overwrite scoped to `project_id` would wipe one example's rows when the other runs.

#### `etl_wnm_exc_04_projection_matrix.ipynb`

Writes the WNM excitatory projection matrices (one ipsilateral + one contralateral) for `project_id="visp_wnm"`, `dataset_id="visp_exc_wnm"`. Source: `ProjectionMatrix_tip_and_branch_roll_up.csv` (345 cells × 152 `ipsi_<ACRONYM>` + 68 `contra_<ACRONYM>`). Cell ids are the SWC filename with `.swc` stripped, matching `_01`.

| Path | Class / shape | Rows | Write |
|---|---|---|---|
| `dataitem/` | `DataItem` | +4 new cells | `append_new_dataitems` (`_01` registered 341 of 345) |
| `dataitem_dataset_association/` | `DataItemDataSetAssociation` | 345 | overwrite, predicate `project_id AND dataset_id` |
| `projectionmeasurementmatrix/wnm_exc_proj_ipsi/` | wide parquet | 345 × (3 id cols + 152 acronyms) | overwrite, predicate `project_id`, `partition_by=["project_id"]` |
| `projectionmeasurementmatrix/wnm_exc_proj_contra/` | wide parquet | 345 × (3 id cols + 68 acronyms) | overwrite, predicate `project_id`, `partition_by=["project_id"]` |
| `projectionmeasurementmatrix/` | `ProjectionMeasurementMatrix` | 2 | overwrite, predicate `id IN (...)` |

`region_coverage` = acronyms with any non-zero value across the 345 cells (152/152 ipsi, 66/68 contra). `values=` is a `file://` URI to the per-laterality wide-parquet directory. `data_item_index` is the cell-id list in wide-parquet row order.

---

## Tests

All 51 tests pass (`uv run pytest -q`).

### `tests/test_cell_features_schema.py` (10 tests)

- `project_id` is required on both `CellFeatureDefinition` and `CellFeatureSet`
- Valid construction with all required fields
- `data_type` pattern validation (must be numpy dtype string, e.g. `<f4`, `<i4`)
- Optional `range_min` / `range_max` fields
- `feature_set_id` is optional on `CellFeatureDefinition` and can be set when provided

### `tests/test_write_utils.py` (6 tests)

- First write (table does not exist): all rows appended
- Empty table: 0 rows appended
- Idempotent re-run: 0 rows appended on second call
- Partial re-run: only new rows appended
- Different `project_id` values don't interfere
- Two sources sharing `project_id` accumulate without conflict

### `tests/test_clustering_schema.py` (11 tests)

- `Cluster` has no `project_id` field; constructs without it; rejects `project_id` with `Extra inputs are not permitted` (pydantic config is `extra='forbid'`).
- `Cluster.hierarchy_id` is optional, round-trips when set, and is type-checked (rejects non-string). Regression guard: adding `hierarchy_id` did not re-introduce `ProjectScoped` on `Cluster`.
- `ClusterMembership` still requires `project_id`; `hierarchy_id` is optional, round-trips when set, and is type-checked (rejects non-string).

### `tests/test_mappings_schema.py` (9 tests)

- All three endpoint shapes round-trip (dataset→dataset, dataset→hierarchy, hierarchy→hierarchy).
- All four endpoint slots are optional at the schema level.
- `method_name` and `project_id` are still required.
- `target_hierarchy` is type-checked as a string (rejects non-string input).
- `CellToClusterMapping` round-trips with the cell-to-cluster `MappingSet`; `target_cluster` still required.

## Documentation added

### `code/etl_examples_readme.ipynb`

A markdown-only notebook (no code cells) providing a skimmable overview of all registered datasets and feature sets. Covers:
- What a DataItem is and how it links to datasets
- Minnie65: why there are two datasets (all nuclei vs. CSM-classified subset)
- VISp Patch-seq: shared `project_id`, why inh (2,759 T-type) ≠ exc (1,528 MET-type), the 120/4 extra cells registered by `_02` notebooks
- Feature set ownership, the shared `exc_visp_morph_features` split between patchseq and WNM, and WNM-only sets

### `etl_example_prompt.md`

A prompt guide for AI assistants creating new ETL notebooks. Includes:
- Which files to read first (schemas, src utils, example notebooks)
- Hard rules: never edit `src/` or `models.py`, schemas are the source of truth, no id casting
- Canonical notebook structure (cell order)
- Write pattern reference per table type with correct predicate shapes
- Common mistakes table (10 failure modes with correct remedies)

Both files are linked from `README.md`.

---

## Follow-ups / not in this PR

### `etl_wnm_exc_05_single_cell_recon_and_brain_region_assoc.ipynb` (not written)

- **`BrainRegionAssociation` and `SingleCellReconstruction` are not `ProjectScoped`** and have no natural unique key (assoc) / no project partition (recon). Should these classes get `ProjectScoped`, an explicit `id`, or both? Or is "global, deduplicated by `(region_id, dataitem_id)`" the intended semantics?
- **`brainregion/` is not populated** in the current Delta Lake. `examples/etl_brain_regions.py` exists but isn't in the run order. WNM ETL needs `BrainRegion.id` lookup by acronym; the bootstrap step needs to be wired in first.
- **`ccf_registered_file` URI convention undefined.** Schema example uses `swc://s3://...`. WNM swc file public location is not confirmed.
- **Cells with NaN `ccf_soma_*` coordinates** — write the `SingleCellReconstruction`/`BrainRegionAssociation` rows with NaN values, or skip those cells (still keeping them in `dataitem/`)?

### Open questions on `etl_wnm_exc_04_projection_matrix.ipynb`

- `measurement_type=MICRONS_OF_AXON` is inferred from value magnitudes (~10⁴, max ≈ 22,700). Filename suggests tip counts. Confirm with data owner.
- `ProjectionMeasurementMatrix` is not `ProjectScoped`; metadata predicate is `id IN (...)` only. Recommend adding `mixins: [ProjectScoped]` (same gap on `BrainRegionAssociation`, `SingleCellReconstruction` — see `_05` below).
- `region_index` stores raw acronyms instead of `BrainRegion.id`s; `brainregion/` not yet populated. Re-run after that bootstrap.
- `values` is typed `ZarrArray` but stored as a delta-path string (mirrors `CellFeatureMatrix.parquet_path`). Either add a `parquet_path` slot or commit to zarr.

### Connectome version discriminator on `CellCellConnectivityLong`

Currently `etl_minnie_04_cell_cell.ipynb` writes two separate output folders (`cellcellconnectivitylong_proofread_pre_to_csm_post/` and `cellcellconnectivitylong_proofread_to_proofread/`) to avoid overwrite collisions on a shared `project_id`. Long-term, all rows could live in a single `cellcellconnectivitylong/` table distinguished by an identifier field (e.g. `connectome_id`, `cohort_description`) — analogous to `feature_set_id` on `CellFeatureDefinition`. Schema addition required.

### `HierarchyCategory` global-dedup append helper

`HierarchyCategory` is global with no `project_id`/`hierarchy_id` discriminator, and category ids like `class`/`subclass`/`cluster` are intentionally shared across taxonomies. A scoped overwrite from any taxonomy notebook would clobber sibling taxonomies' rows; a plain append collides on `id`. `etl_minnie_03_…` skips the write entirely and flags this in its header. Needs a global-dedup append helper before any new taxonomy notebook should write here.
