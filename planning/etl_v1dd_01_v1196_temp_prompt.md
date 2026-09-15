# Handoff prompt — continue building `etl_v1dd_01_v1196.ipynb`

You are picking up an in-progress ETL notebook that ingests the V1DD release 1196 dataset into the Common-Connectivity (CCC) Delta-lake schemas. One previous agent built the skeleton + section 1 (DataSets). The user wants the remaining sections filled in **one at a time, together** — finish a section, show the result, wait for the user to review before moving on.

---

## Read first (in this order)

### Authoritative conventions
- `/root/capsule/etl_example_prompt.md` — full ETL conventions guide. **Read end-to-end before writing any code.** Pay special attention to:
  - §2 hard rules (never edit `src/` or `models.py`; never cast ids; use enum `.value`; every write has a verification cell).
  - §4 canonical notebook structure.
  - §5a–§5j write patterns per table family.
  - §10 common mistakes table.
- `/root/capsule/CHANGELOG.md` — only relevant if you end up changing schemas (don't unless the user asks).

### The notebook in progress (the one you'll be editing)
- `/root/capsule/code/etl_v1dd_01_v1196.ipynb`
  - Cells 0–6: title, imports, constants, prereq check, **master id decision**, §1 DataSets (DONE, written + verified).
  - Cells 7+: §2…§10 are skeletons with markdown plans + `# TODO` code stubs + per-section open questions.
  - **`OUTPUT_ROOT = "../scratch/v1dd_1196_v1/"`** — relative to `code/`. The §1 outputs are already there under `dataset/`.
  - Re-execute the whole notebook with `cd /root/capsule/code && uv run jupyter nbconvert --to notebook --execute --inplace etl_v1dd_01_v1196.ipynb` after every change.

### Exploration / scratch reference
- `/root/capsule/code/etl_v1dd_00_explore.ipynb` — initial exploration of every input file with schema-fit notes. Useful for sanity-checking shapes/columns.

### Example notebooks to mirror (same modality as V1DD = MICrONS Minnie)
- `/root/capsule/code/etl_minnie_01_dataset_dataitem.ipynb` — DataSet + DataItem + association pattern.
- `/root/capsule/code/etl_minnie_02_cell_features.ipynb` — CellFeatureSet/Definition/Matrix + wide parquet.
- `/root/capsule/code/etl_minnie_03_cluster_and_cluster_membership.ipynb` — Cluster taxonomy + parent-propagated memberships.
- `/root/capsule/code/etl_minnie_04_cell_cell.ipynb` — CellCellConnectivityLong with per-example folder convention.
- `/root/capsule/code/etl_wnm_exc_04_projection_matrix.ipynb` — for the `SingleCellReconstruction` + `SpatialLocation` pattern if needed.

### Schemas (source of truth — do not modify without explicit user request)
- `/root/capsule/schemas/base_schema.yaml`
- `/root/capsule/schemas/core_schema.yaml` — `DataSet`, `DataItem`, `DataItemDataSetAssociation`, `SpatialLocation`, `Modality`.
- `/root/capsule/schemas/cell_features_schema.yaml`
- `/root/capsule/schemas/clustering_schema.yaml`
- `/root/capsule/schemas/mappings_schema.yaml` — `MappingSet`, `CellToCellMapping`, `CellToClusterMapping`.
- `/root/capsule/schemas/cell_cell_schema.yaml` — `CellCellConnectivityLong`, `SynapticMeasurementType` enum.
- `/root/capsule/schemas/single_cell_schema.yaml` — `SingleCellReconstruction`.
- The user has already added `Modality.CALCIUM_IMAGING` and regenerated `src/connects_common_connectivity/models.py`. Trust this.

### Package utilities (read-only)
- `/root/capsule/src/connects_common_connectivity/models.py` — auto-generated pydantic models; read to confirm field names and enum values.
- `/root/capsule/src/connects_common_connectivity/io/writers.py` — `write_models(models, *, output_root=...)` dispatches by class. Use `output_root=OUTPUT_ROOT` because we are NOT writing to the shared `ccc_config.yaml` location.
- `/root/capsule/src/connects_common_connectivity/io/write_spec.py` — per-class WriteSpec (predicates, partition keys); consult before any write that isn't already in the writer registry.
- `/root/capsule/src/connects_common_connectivity/write_utils.py` — `append_new_dataitems`, `walk_ancestors`.
- `/root/capsule/src/connects_common_connectivity/arrow_utils.py` — `build_arrow_schema`, `models_to_table`, `attach_linkml_metadata`, `build_cell_feature_matrix_schema` (kwarg-only — see §10 of the prompt guide).

---

## Raw V1DD data — `/data/v1dd_1196/`

| File | Shape | Notes |
|---|---|---|
| `data_description.json`, `subject.json`, `metadata.nd.json` | aind-data-schema records | provenance; `name`, `project_name`, modalities, S3 location |
| `soma_and_cell_type_1196.feather` | (207 455, 11) | soma centroids + `cell_type_coarse` ∈ {E,I} + `cell_type` (12 leaves) |
| `proofread_axon_list_1196.npy` | (1 210,) int64 | `pt_root_id`s with proofread axons; 1164/1210 are in soma catalog |
| `proofread_dendrite_list_1196.npy` | (63 986,) int64 | `pt_root_id`s with proofread dendrites; all in soma catalog |
| `snr_by_cell.feather` | (4 458, 5) | functional ROI `(volume, column, plane, roi)` + `snr` |
| `coregistration_1196.feather` | (571, 5) | EM↔functional mapping; pre/post not unique on either side |
| `syn_df_all_to_proofread_to_all_1196.feather` | (8 204 497, 13) | per-synapse rows; `pre_pt_root_id`/`post_pt_root_id` + positions + `size` |
| `syn_label_df_all_to_proofread_to_all_1196.feather` | (6 706 286, 1) | per-synapse `tag` (`spine`, …), indexed by synapse `id` |
| `cell_cell_correlations_by_stimulus.feather` | (8 846 260, 13) | all-ROI functional Pearson corr per stimulus, ROI-tuple-keyed, tuples unique |
| `cell_cell_correlations_by_stimulus_coregistered.feather` | (148 728, 9) | same but EM-rootid-keyed; 142 410 unique pairs (≈4 % repeat), 12 self-pairs |

---

## Master decisions already made (do not relitigate)

- **One notebook for all of V1DD 1196**, no `_02`/`_03` follow-ups.
- **`OUTPUT_ROOT = "../scratch/v1dd_1196_v1/"`.**
- **`PROJECT_ID = "v1dd"`.**
- **Five DataSets** (`v1dd_1196_em`, `v1dd_1196_proofread_axons`, `v1dd_1196_proofread_dendrites`, `v1dd_1196_func`, `v1dd_1196_func_coregistered`). Already written, do not rewrite.
- **EM `DataItem.id = str(pt_root_id)`** — single source of truth for EM cells. See the `master-id-decision` markdown cell for the table + numbers + collapse policy (drop `pt_root_id==0`, keep largest-`volume` soma when multiple rows share a root).
- **Functional `DataItem.id = f"{volume}-{column}-{plane}-{roi}"`** (planned in §6 skeleton).
- **`publication = "https://github.com/AllenInstitute/v1dd_physiology"`** for every DataSet.

---

## Sections still to fill (in order)

| § | Title | Status |
|---|---|---|
| 1 | DataSet rows | ✅ DONE |
| 2 | EM DataItems + cohort associations | TODO — next up |
| 3 | EM soma `CellFeatureMatrix` (`v1dd_em_soma_geometry`) | TODO |
| 4 | `SingleCellReconstruction` + `SpatialLocation` (CCF) | TODO |
| 5 | V1DD cell-type taxonomy (Cluster + ClusterMembership) | TODO; includes a v1dd↔minnie taxonomy comparison table already verified |
| 6 | Functional DataItems + coregistered cohort | TODO |
| 7 | Functional feature sets (`v1dd_func_qc`, `v1dd_func_imaging_position`) | TODO |
| 8 | `CellToCellMapping` for EM↔functional coregistration | TODO |
| 9 | Synapse aggregation → `CellCellConnectivityLong` | TODO (per-synapse schema question is intentionally OPEN) |
| 10 | Functional correlations → `CellCellConnectivityLong` × 7 stimuli × 2 tables | TODO |

Each section has open questions in its markdown cell. **Ask the user before answering them yourself** — they want to review each section before you wire the writes.

---

## Working agreement (per user instruction)

1. **Build one section at a time.** Do not jump ahead. After each section: re-execute the full notebook, show the verification cell output to the user, then stop and wait.
2. **Update the markdown of each section as decisions are resolved** — remove answered open questions, keep unresolved ones, keep the section concise.
3. **Don't touch `src/` or `schemas/`** unless the user explicitly asks for a schema change.
4. **Don't relitigate the master id decision.** If a section's open question is rendered moot by it, just delete the question.
5. **Verification cell after every write** — read back with `pl.read_delta`, print shape + head(3), assert at least one invariant (row count, unique ids, expected categorical value).
6. **Use `write_models(..., output_root=OUTPUT_ROOT)`** for everything that has a WriteSpec. For things without one (wide-form feature parquets, cell-cell folders), fall back to `deltalake.write_deltalake` with the patterns in §5b–§5g of the prompt guide.
7. **Run the notebook headless to validate**: `cd /root/capsule/code && uv run jupyter nbconvert --to notebook --execute --inplace etl_v1dd_01_v1196.ipynb`.

---

## Next action when the user returns

Start with **§2 (EM DataItems + cohort associations)**. The skeleton is in place and the master id decision is documented. The two remaining open questions in that section are:
1. Proofread axon roots missing from the soma catalog (46/1210) — skip silently or log + skip?
2. `neuroglancer_link` — populate, or leave null?

Ask the user, then implement, write, verify, and stop.
