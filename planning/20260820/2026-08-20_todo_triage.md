# TODO triage — 2026-08-20

Basis: branch `v1dd-ingest` (= PR #6 head, base `ingestion-v2` = PR #5), `schemas/`, `src/connects_common_connectivity/io/`, `planning/` docs (`20260623/TODO.md`, `ARCHITECTURE.md`, `multi_writer_scope_design.md`), and both open PR descriptions.

Labels: **feature** · **bugfix** · **design decision** · **research/alignment** · **chore**
Status: not started · partially complete · mostly done — verify · answered (question resolved, follow-ups extracted)

---

## A. Schema changes

### S1. AIT compatibility for cluster schemas — research/alignment + feature · not started
Make `Cluster` / `ClusterHierarchy` / `ClusterMembership` / `AlgorithmRun` interoperable with [AllenInstituteTaxonomy](https://github.com/AllenInstitute/AllenInstituteTaxonomy) conventions (term sets, annotations, abbreviations), so CCM taxonomies can be imported from / exported to AIT.
Note: `origin/u/morriscb/abcAtlasAccessDraftSchema` already drafts `hmba_taxonomy_annotation_schema.yaml` (`ClusterAnnotationTermSet`, `ClusterAnnotationTerm`, `ClusterToClusterAnnotationMembership`) — coordinate before designing independently.

### S2. Alignment with BKP explorer ingest schema — research/alignment · not started
Compare CCM schemas against the [BkpPlotGenerator ingest template](https://github.com/AllenInstitute/BkpPlotGenerator/blob/main/docs/Ingest-Template/README.md#visualization-files) (visualization files). Deliverable: similarities/differences memo and whether a CCM→BKP export is mechanical or needs schema additions.

### S3. Alignment with aind-data-schema — research/alignment · not started
Assess whether CCM dataset/session metadata can live as a JSON sidecar conforming to [aind-data-schema](https://aind-data-schema.readthedocs.io) next to the CCM store. Deliverable: mapping memo + decision on adopting/embedding.

### S4. `connectome_id` discriminator for CellCellConnectivityLong — feature (schema) · not started, well-scoped
`CellCellConnectivityLong` has no `connectome_id`/`dataset_id`, so two connectomes from the same project (even same dataset, different cell subsets) can't share a table. Gap is already documented in `etl_example_prompt.md` §5g/§11, the `write_cellcellconnectivitylong` stub docstring, and `etl_examples_readme.ipynb`. Current workaround: one subdirectory per connectome (`cellcellconnectivitylong_proofread_to_proofread/` etc.). Adding the field unblocks a registry entry (see P5).

### S5. ProjectScoped gaps — bugfix (schema) · not started, anticipated in code
`ProjectionMeasurementMatrix`, `SingleCellReconstruction`, `BrainRegionAssociation` (and `BrainRegion`, whose mixin is commented out) are not `ProjectScoped`. Known consequence: `etl_wnm_exc_05` blocked; `ProjectionMeasurementMatrix` write spec scopes on `id` only (registry comment says to widen once schema gains ProjectScoped). `BrainRegionAssociation` additionally has no `id` at all — currently unwritable. Decide per class: add mixin vs deliberately global (like the cluster taxonomy).

### S6. Missing identity fields causing duplicate/overwrite bugs — bugfix (schema + specs) · answered → follow-ups
Question answered; concrete gaps found:
- `ClusterMembership`: no `id`, everything Optional; scope `(project_id, hierarchy_id)` → multi-writer clobber risk.
- `DataItemDataSetAssociation`: no `id`; within-batch duplicates never deduped.
- `CellFeatureMatrix`: identity is `id` but scope is `(project_id, feature_set_id)` → two pointer rows clobber each other.
- `Cluster`: scope is `hierarchy_id` alone → any partial write deletes the rest of the hierarchy.
- `HierarchyCategory`: global ids like `class`/`subclass` collide across taxonomies (known, `etl_example_prompt.md` §11).
- `CellFeatureMeasurement`: **LinkML bug** — `feature_set_id` and `unit` are declared under `slot_usage:` but not `slots:`, so the generated model lacks them (and `project_id`).
- `DataItem` re-runs with changed `name`/`neuroglancer_link` are silent no-ops (no update path).
- `HierarchyCategory.level` generates as `str` while `Cluster.level` is `int` (missing range).

### S7. PROOFREAD_AXON / PROOFREAD_DENDRITE / PROOFREAD_CELL QC flags — feature (schema) · not started
No QC enum/schema exists anywhere. Proofread status is currently encoded as dataset membership (`v1dd_1196_proofread_axons`, `minnie65_v1300_proofread*`), answered by a join through the association table. Design a per-cell QC flag schema (enum + association or DataItem slots) and survey what QC concepts other modalities need (patch-seq QC calls, recon completeness, mapping confidence). Closest existing thing: unused `CellMetadata.quality_score` float.

### S8. Spatial features schema (CCF / UMAP / voxel coordinates) — feature (schema) · not started, workaround in use
Coordinates are currently smuggled through generic `CellFeatureSet`s (e.g. `v1dd_soma_spatial` with `soma_voxel_x/y/z`, `soma_transformed_x/y/z`) — untyped floats, no reference-space semantics. `SpatialLocation` exists (`x,y,z,reference_space`) but is nested-only and **cannot round-trip through `write_models`** (arrow layer stringifies embedded models without an `id` — latent bug). For UMAP: `EmbeddingSet`/`EmbeddingMethod` draft exists on `origin/u/morriscb/abcAtlasAccessDraftSchema`. Design one typed spatial-feature story covering CCF, voxel, and embedding coordinates.

### S9. CCF-area coverage flag when projection exists — mostly done — verify
`ProjectionMeasurementMatrix.region_coverage` (list of region ids with ≥1 non-zero value) exists and is auto-populated by `write_utils.populate_region_coverage` via `write_projection_matrix` (from the ingestion-v2 work). If the intent was "binary flag per CCF area", the id-list is informationally equivalent. Remaining ideas if any: per-DataItem coverage, or the deferred `compare_region_coverage()` analysis (see `_deferred/09_analysis.md`). Likely closeable.

---

## B. Package changes (read/write library)

### P1. Multi-dataset reader — feature · partially complete
Single-dataset read is done (`DatasetReader`, PR #6). Missing: reading across datasets that share feature sets / cluster memberships / mappings — e.g. "all DataItems that have feature set X or belong to cluster Y". A design already exists: deferred `planning/20260623/prompts/_deferred/08_readers.md` Layer 2, flagship `read_dataitems_for_clusters(cluster_ids, via=("membership","mapping"))`. Also still pending from that plan: move `parquet_loader.py` → `io/parquet_loader.py`.

### P2. write_models / registry coverage gaps — answered → follow-ups
Question answered. Registered: 15 classes. **Not registered:** `SynapseConnectivityLong` (commented out — see P5), `CellCellConnectivityLong` + `CellCellMeasurementMatrix` (blocked on S4), `CellToCellMapping`, `ClusterToClusterMapping`, `CellFeatureMeasurement` (blocked on S6 bug), `BrainRegion`, `BrainRegionAssociation`, `SingleCellReconstruction` (blocked on S5), the whole `cell_gene_schema` set, `SpatialLocation`, `ZarrArray`/`ZarrDataset`/`ParquetDataset` (pointer-only, may be intentional). Decide which are intentional non-goals vs backlog.

### P3. Delete/remove functions — feature · not started
No delete/vacuum/partition-removal exists anywhere; the only way to remove rows is a predicated overwrite that omits them. Design explicit ops: remove a DataItem, retire a feature set/column, drop a hierarchy, delete a scope from delta tables — plus tombstone/vacuum policy.

### P4. Overwrite vs append strategy — design decision + research/alignment · design doc exists, undecided
Current modes: `overwrite_scoped` (per-scope predicated overwrite, not atomic across scope groups) and `append_new_by_id` (DataItem only). No upsert; delta-rs `MERGE` is the industry-standard third option and is unused. `planning/multi_writer_scope_design.md` lays out Options A–D (A: `merge_by_id` mode; B: `write_models_merging_on()`; C: convention; D: forbid shared scopes / minnie-style cohorts) — decision pending. This is the umbrella decision for P6 and parts of S6.

### P5. SynapseConnectivityLong write model + wide-form writes — design decision + feature · partially complete
`SynapseFeatureMatrix` is registered; `SynapseConnectivityLong` is deliberately commented out because `validate_for_write` re-validates every row with pydantic — unusable at 8M rows (commit `6c5b68d`: "later problem"). Answer to "why validate every row": that's the only validation mechanism; there is no vectorized/arrow-level path. Options: sampled validation, arrow-schema-only validation for long tables, or a `bulk` write mode.
Sub-items:
- **Wide-form write functions in specs** — the `wide_parquet` write mode was designed in `ARCHITECTURE.md` but explicitly not built (W3 deviation); all wide tables (`cellfeatures/`, `synapsefeatures/`, wide projection, `synapse/`) are raw `write_deltalake` calls in notebooks.
- **Write wide form + its pointer row atomically** — nothing enforces that `CellFeatureMatrix`/`SynapseFeatureMatrix` pointer rows and the parquet they point at are written together; a combined API would remove a whole class of dangling-pointer bugs.

### P6. DataItemDataSetAssociation union/merge on write (patch-seq problem) — bugfix + design decision · partially complete
The exact failure is documented in `multi_writer_scope_design.md`: three patch-seq notebooks sharing scope `(project_id, dataset_id)` shrank associations 2759 → 520 → 495 (last writer wins). Current workaround is inlined read-union-rewrite in the notebooks (`etl_visp_*` 02/03). Resolution = pick a P4 option; your own note's alternatives (gather-all-IDs-first, minnie-like sub-dataset cohorts) are Options C/D. Keep in mind incremental writers (QC/typing arriving over time) — which argues for A/B over D.

### P7. Edit existing outputs (schema-evolution writes) — feature · not started, conventions only
No library support for: adding feature columns to an existing feature set, adding DataItems to an existing dataset, or any delta schema evolution (`schema_mode="merge"` unused). Conventions live in `etl_example_prompt.md` §5d/§9 ("fail loudly or NaN-fill, never silently drop") but aren't code. Depends on P4's merge decision; overlaps P3 (removing columns).

---

## C. Other (ecosystem, repos, tooling)

### O1. Repo split — design decision · not started
Decide boundaries: schema pkg vs read/write pkg (currently one repo), viz toolbox together or separate (currently separate: `ConnectsCommonConnectivity-VizToolbox`), demo notebooks public vs private (currently `code/` in-repo + `ConnectsCommonConnectivity-demos`). Constraint to note: generated `models.py` couples schemas to the package; splitting means versioned schema releases.

### O2. LLM-automated ETL (new data → CCM) — feature, likely new repo · seed exists
Automate: explore a new dataset, semantically map files → schemas, leave open questions (units, descriptions, cell-type name matching) for the user. `etl_example_prompt.md` (17KB, in-repo) is already this in prompt form — the decision is packaging: skill.md (agent workflow) vs drag-and-drop webapp with a questions UI. Would import the schema + rw packages; natural consumer of O3 for post-ETL sanity plots.

### O3. VizToolbox — feature (separate repo) · in progress there
Goal: immediate sanity-check plots after an ETL (ETL repo imports viz repo). Sub-items:
- Design for ipywidgets wrapping: dropdown-able parameters typed as enums/lists.
- Move code from `ConnectsCommonConnectivity-demos` (viz-ic3-presentation branch) into the `ConnectsCommonConnectivity-VizToolbox` package (viz-plan branch, `src/ccc_viz`).
Main repo has zero viz code (only `neuroglancer_link`), so this stays out-of-repo. Small chore here: `seaborn` is a runtime dependency of the main package but unused in `src/` — demote to dev/extras.

---

## Dropped / merged

- Empty bullet (4th item) — dropped.
- "what's missing in write_models" and "what's missing in the registry" — merged into P2 (same registry answers both).

## Bonus findings (small chores, not from your list)

- Dead import: `SynapseConnectivityLong` imported in `write_spec.py` but its entry is commented out.
- `dry_run=True` is silently ignored when caller passes `output_root=` (writers resolve `settings=None`).
- `planning/20260623/tests_review/findings.md` has an ungraded backlog (broad `pytest.raises(Exception)`, no dry-run test, no concurrent-write test).

## Suggested discussion order (dependency-aware)

1. **P4** (write strategy) — umbrella for P6, S6, P7.
2. **S6 + S4 + S5** (identity/scope schema fixes) — mostly mechanical once P4 is decided.
3. **P5** (synapse/wide-form writes), **P3** (delete), **P7** (edit).
4. **P1** (multi-dataset read) — design already drafted.
5. **S7, S8** (QC flags, spatial schema) — new design work.
6. **S1–S3** (external alignment research) — parallelizable, delegable.
7. **O1–O3** (repo org, LLM ETL, viz) — strategic, discuss when 1–4 settle.
