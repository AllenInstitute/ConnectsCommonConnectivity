# Git issues draft — 2026-08-20 (rev 2)

Source: `2026-08-20_todo_triage.md` + `2026-08-20_work_packages.md` + review discussion with YY. Status: **draft — nothing opened yet.**

Labels: repo has GitHub defaults only; create `schema`, `io`, `research` before opening.

---

## A. Schema defects (→ WP1)

### I1 — CellFeatureMeasurement: slot_usage fields are never generated
Labels: `bug`, `schema` · WP1
> `cell_features_schema.yaml` declares `feature_set_id` and `unit` for `CellFeatureMeasurement` under `slot_usage:` but not under `slots:`, so LinkML generates neither — the documented "denormalized reference to the feature set" does not exist in `models.py`.
> **Fix:** add both to `slots:`, regenerate models.
> **Open question — project scoping:** the long-term goal is common features across modalities/projects, so feature sets/definitions should ideally not be hard project-scoped, even though most sets will in practice be project-specific. With merge-scoped writes (I11) the write-side cost of `project_id` drops to ~zero (it's just part of a merge key, no clobber risk); the remaining concerns are (a) `dataitem_id` is only unique per project, so an unscoped measurement row is ambiguous in joins, and (b) namespacing for genuinely shared sets — sentinel project (e.g. `project_id="common"`) vs optional `project_id` on `CellFeatureSet`/`CellFeatureDefinition`. Decide here; don't silently enforce the mixin.

### I2 — HierarchyCategory: per-taxonomy vocabulary needs a discriminator; level type mismatch
Labels: `bug`, `schema` · WP1
> Category ids like `class`/`subclass`/`cluster` are global today, but there is no common agreement of levels across taxonomies and we do not want to force a level structure — `Cluster` holds *hierarchical categorical features* generally, not necessarily cell types. Merge-scoped writes (I11) do **not** fix this: two taxonomies writing `class` upsert the same row, so the last writer's `level`/`description` silently wins.
> **Fix:** add a hierarchy/taxonomy discriminator slot so category vocabularies are per-taxonomy; declare `level: integer` (currently generates `Optional[str]` while `Cluster.level` is `int`).

### I4 — ProjectionMeasurementMatrix: add ProjectScoped
Labels: `enhancement`, `schema` · WP1
> Not ProjectScoped; write spec scopes on `id` only, with an explicit registry comment to widen once the schema gains project scoping. Same matrix id in two projects collides. Blocks `etl_wnm_exc_05`.
> **Fix:** add `ProjectScoped` mixin; widen `partition_by`/`scope_columns`; update `etl_wnm_exc_04`.

### I5 — SingleCellReconstruction: add ProjectScoped
Labels: `enhancement`, `schema` · WP1
> Not ProjectScoped; its `id` references a DataItem (project-scoped), so two projects using the same cell id collide once this class becomes writable. No documented rationale for the omission (unlike BrainRegion, deliberately global per the schema comment). Blocks `etl_wnm_exc_05`.

### I6 — BrainRegionAssociation: no identity, unwritable as designed
Labels: `bug`, `schema` · WP1
> Purpose (per schema): "An association between a DataItem and a BrainRegion" — records which region a cell/dataitem is in (region-to-region projection lives in `ProjectionMeasurementMatrix`). Currently: two Optional fields, no `id`, no `project_id` — cannot be validated, scoped, or written. Blocks `etl_wnm_exc_05`.
> **Fix:** make `brainregion_id` and `dataitem_id` required (or write-required, per I3's convention), add `ProjectScoped`; natural identity `(project_id, dataitem_id, brainregion_id)` — no surrogate id needed once merge keys (I11) exist.
> **Design note:** a cell may get region assignments from multiple methods (CCF registration vs manual) or term sets — consider a `method`/`source` slot.

## B. Schema features

### I7 — CellCellConnectivityLong: add connectome_id (measurement-context discriminator) (→ WP3)
Labels: `enhancement`, `schema`
> `connectome_id` identifies the **measurement context** — segmentation version + proofreading state + measurement semantics — **not** the cohort. Cohorts are read-side filters: write the superset connectome once; pull subsets by joining `dataitem_dataset_association` or cluster membership ("connectivity of dataitems with cell type xx"). Today, with no discriminator, rows from two measurement contexts are indistinguishable and their overwrite predicates clobber each other — hence the current one-subdirectory-per-connectome workaround (`cellcellconnectivitylong_*`). Note minnie's two folders (`proofread_pre_to_csm_post` vs `proofread_to_proofread`) are two measurement contexts → two `connectome_id`s in one table.
> **Fix:** add `connectome_id`; register WriteSpec (`overwrite_scoped` by `(project_id, connectome_id)`); **rerun the affected ETLs with the new io + schema** (no folder migration); add a subset-read helper (ties to I19). Own planning session before implementation.

### I8 — Spatial coordinates table: promote SpatialLocation (→ WP9)
Labels: `enhancement`, `schema`
> Per-cell coordinates are smuggled through generic CellFeatureSets (`v1dd_soma_spatial`: `soma_voxel_x/y/z`, ...) — untyped floats, no reference-space semantics. `SpatialLocation` (`x,y,z,reference_space`) exists in `core_schema.yaml` but is unreachable on disk: its only attachment points (`SingleCellReconstruction.soma_location`, `CellMetadata.spatial_location`) were never written by any ETL, it has no id/WriteSpec (designed as an embedded struct), and the write path has no struct support (I18).
> **Fix:** first-class table `(dataitem_id, reference_space, x, y, z [, project_id])`; `reference_space` distinguishes CCF-registered vs dataset-original vs dataset-corrected coordinates. Pilot: migrate `v1dd_soma_spatial`. Flattening to a table bypasses I18 (kept separate as the general bug).

### I9 — Embedding coordinates schema (UMAP/tSNE), distinct from spatial (→ WP9)
Labels: `question`, `schema`
> Computed embeddings must not be conflated with anatomical coordinates (the BKP ingest template doesn't distinguish them; we should). `origin/u/morriscb/abcAtlasAccessDraftSchema` drafts `EmbeddingSet` + `EmbeddingMethod` (UMAP/TSNE/PCA/MDS/OTHER). Question: align with/adopt that draft? Coordinate with @morriscb.

### I10 — QC flag schema: per-cell proofreading and modality QC (→ WP8)
Labels: `enhancement`, `schema`, `question`
> No QC concept exists; proofread status is encoded as DataSet membership (`v1dd_1196_proofread_axons` etc.). Need per-cell flags like PROOFREAD_AXON / PROOFREAD_DENDRITE / PROOFREAD_CELL and equivalents for other modalities.
> **Inclination (TBD):** per-modality QC scores + controlled vocabulary, **plus** a commonly-understandable cross-modality `acceptable` flag. Candidate shape: `QCFlag` association table `(dataitem_id, flag, value, source, timestamp, project_id)` with per-modality vocabularies (possibly via the `ValueSet` draft on the morriscb branch) — flags arrive incrementally, matching merge-scoped writes (I11).
> **First step:** survey QC vocabulary across modalities.

### I27 — Auto-derive CellCellConnectivityLong from a synapse table (→ WP3)
Labels: `enhancement`, `io`
> When a `SynapseConnectivityLong` table exists, cell-cell connectivity is derivable: group by `(presynaptic_cell, postsynaptic_cell)` and aggregate. Provide a function that populates `CellCellConnectivityLong` automatically.
> **Decide aggregation functions:** total synapse count always; total synapse size when a per-synapse size column is available (via the dataset's `SynapseFeatureMatrix`). Each aggregation → its own `measurement_type` row set (`SynapticMeasurementType`). Depends on I7 (`connectome_id` on the output) and pairs with I13 (registered synapse writes).

## C. Write layer

### I3 — ClusterMembership merge keys: enforce in io, not schema (→ WP2)
Labels: `enhancement`, `io` *(reclassified from schema bug)*
> `item` and `cluster` are Optional in the schema and there is no `id`; merge keys (I11) must be non-null at write time.
> **Convention decided:** the schema is the standard (eventually its own repo); the io package is one wrapper around it — others may build their own io on raw polars/pandas/deltalake. Constraints required only by *this* write method belong in the io layer (`required_for_write` / `strict_model_for`, as already done for `hierarchy_id`), not in the schema. So: add `item`, `cluster` to `required_for_write`; schema stays Optional. Apply the same convention wherever merge keys land (I6 etc.).

### I11 — merge_scoped write mode with declared merge_on keys (→ WP2)
Labels: `enhancement`, `io`
> Only `overwrite_scoped` and `append_new_by_id` exist; no upsert. Incremental workflows (adding dataitems, revising feature values, QC over time) need merge semantics for identity-bearing tables; bulk tables (synapse long, wide matrices) stay overwrite-scoped for memory reasons (delta MERGE joins incoming vs existing — fine at 10³–10⁶ rows, wasteful at 10⁷).
> **Fix:** `write_mode="merge_scoped"` + `merge_on: list[str]` in `WriteSpec`; delta-rs MERGE; in-batch dedupe; per-class assignment per `planning/20260820/2026-08-20_work_packages.md` §P4. Replaces `append_new_by_id` for DataItem (fixes silent no-op metadata updates). Interacts with I1's scoping question: with merge, project scoping of feature sets becomes a namespacing decision, not a write-safety need.

### I12 — Multi-writer shared-scope data loss (patch-seq associations)
Labels: `bug`, `io` · WP2
> Notebooks sharing scope `(project_id, dataset_id)` silently delete each other's rows: associations shrank 2759 → 520 → 495 across `etl_visp_inh_patchseq_01/02/03` (options analysis in `planning/multi_writer_scope_design.md`). Workaround is inlined read-union-rewrite in the notebooks.
> **Fix:** resolved by I11 (merge_on `(project_id, dataset_id, dataitem_id)`); remove the inlined workarounds in the same PR. Kept as the user-visible bug record.

### I13 — Register SynapseConnectivityLong with a bulk validation mode (→ WP4)
Labels: `enhancement`, `io`
> Registry entry commented out because `validate_for_write` re-validates every row with pydantic — unusable at 8M rows; `etl_v1dd_03` bypasses `write_models` with raw `write_deltalake`. Dead import remains in `write_spec.py`.
> **Fix:** bulk mode (arrow-schema-level validation, optional row sampling); register `overwrite_scoped` on `(project_id, dataset_id)`; route the notebook through `write_models`; remove the dead import.

### I14 — wide_parquet mode + combined wide-matrix/pointer write API (→ WP4)
Labels: `enhancement`, `io`
> The `wide_parquet` mode designed in `ARCHITECTURE.md` was never built (W3 deviation); all wide tables are raw `write_deltalake` calls in notebooks. Nothing enforces pointer rows (`CellFeatureMatrix`, `SynapseFeatureMatrix`) and their parquet being written together — dangling-pointer risk.
> **Fix:** wide_parquet mode; combined API writing payload (overwrite) then merging the pointer row, validating index columns/paths agree. Depends on I11's mode vocabulary.

### I15 — Delete/remove operations for parquet/deltalake outputs (→ WP5)
Labels: `enhancement`, `io`
> No delete, vacuum, or partition removal exists; the only row removal is a predicated overwrite that happens to exclude rows. No way to drop a DataItem, retire a feature set/column, or delete a hierarchy.
> **Fix:** explicit remove operations complementary to I11 + vacuum/tombstone policy. Test-first development (see I20 conventions).

### I16 — Post-write edit workflows: add columns / add dataitems (→ WP6)
Labels: `enhancement`, `documentation`, `io`
> No library support for adding feature columns to an existing feature set or dataitems to an existing dataset; conventions live only in `etl_example_prompt.md` §5d/§9. Mostly falls out of I11 + I14; remaining work is a thin API + documented recipes. Removals → I15.

### I17 — dry_run silently ignored when output_root is passed
Labels: `bug`, `io`, `good first issue`
> `write_models(..., output_root=...)` resolves `settings=None`, so `dry_run=True` is silently ignored and data is written. Add the missing dry_run test.

### I18 — Nested models (SpatialLocation) don't round-trip through the arrow layer
Labels: `bug`, `io` · WP9 (kept separate from I8)
> Write path: pydantic models → dicts → arrow table → parquet. `flatten_refs` collapses a nested model to its id string — works for references (DataItem has an id); `SpatialLocation` has no id, so it survives as a raw dict `{x, y, z, reference_space}`. `build_arrow_schema` maps embedded-model fields to the fallback type `string`. Result: a dict hits a string column — pyarrow errors or stringifies the dict into a text blob that is unqueryable, schema-less, and unparseable on read. Model → parquet → model loses structure.
> **Fix options:** (a) struct columns in `arrow_utils`; (b) explicit JSON-column convention; (c) don't nest — I8 flattens spatial to a table and bypasses this. Keep this issue as the general record for any future nested model.

## D. Read layer

### I19 — Cross-dataset reads: union DataItems by shared features/clusters (→ WP7)
Labels: `enhancement`, `io`
> `DatasetReader` is single-dataset; cohorts sharing feature sets or memberships need N calls + manual concat (`etl_v1dd_04_read`). Design exists: `planning/20260623/prompts/_deferred/08_readers.md` Layer 2, flagship `read_dataitems_for_clusters(cluster_ids, via=("membership","mapping"))`.
> Subtasks: move `parquet_loader.py` → `io/parquet_loader.py` (pure move); subset-connectivity filter helper (see I7, I27).

## E. Tests & docs

### I20 — Test-writing conventions + skill: consistent errors, warnings, docstrings (→ WP10)
Labels: `enhancement`, `documentation`
> Establish testing conventions and encode them as a reusable skill: consistent error raising and warning patterns, descriptive docstrings on tests, test-first scaffolding for new io features (first consumer: I15).
> References: `planning/test_suite_analysis_2026-08-15.md` (current state: healthy 169-test pyramid; top gaps = CI enforcement, coverage measurement, direct `arrow_utils` tests, deeper parquet-loader cases, CLI behavioral tests) and `planning/20260623/tests_review/findings.md` (older graded review). The gap items can become checkboxes here or follow-up issues.

### I28 — Update README
Labels: `documentation`
> README predates the v1dd-ingest work and the upcoming write-mode changes. Refresh after I11 lands: document read API (`DatasetReader`, `read_synapse_table`), write modes (merge/overwrite/bulk/wide), config, and current schema coverage.

## F. Research / discussion

### I22 — AIT schema comparison + adapter/migration strategy (→ WP11)
Labels: `question`, `research`, `schema`
> **Main task:** review the [AllenInstituteTaxonomy](https://github.com/AllenInstitute/AllenInstituteTaxonomy) schema and compare it to CCM's clustering schemas (`Cluster`, `ClusterHierarchy`, `ClusterMembership`, `AlgorithmRun`) — field-mapping memo.
> Context: AIT is a spec (anndata h5ad + markdown/CSV schema; R tooling), no releases or pip package, actively evolving. **Versioning approach:** pin a specific AIT commit as our de-facto version and maintain a migration script for switching pinned versions; a conformance test against one published taxonomy from their `taxonomies.md` breaks loudly when the mapping drifts. Coordinate with `hmba_taxonomy_annotation_schema.yaml` draft (@morriscb branch).

### I23 — BKP ingest schema comparison (→ WP11)
Labels: `question`, `research`
> Research: compare CCM schemas to the [BKP explorer ingest template](https://github.com/AllenInstitute/BkpPlotGenerator/blob/main/docs/Ingest-Template/README.md#visualization-files) — similarities, differences, and use-case contrast (fast vis-app serving vs cross-modality commonality). CCM→BKP export itself is owned by another team; this is for understanding. *(Wording to be revised by YY.)*

### I24 — aind-data-schema: explore incorporation (→ WP11)
Labels: `question`, `research`
> Explore how [aind-data-schema](https://aind-data-schema.readthedocs.io/en/latest/index.html) can be incorporated — a sidecar JSON next to CCM was requested by a scientist. Capture the concrete use case (which aind core files?), then assess a small exporter util (aind-data-schema is a versioned pip package, so an optional-extra exporter is cheap once the mapping is known).

### I25 — Repo split: options and partial migration path (→ WP13)
Labels: `question`
> Options A–D in `planning/20260820/2026-08-20_work_packages.md` §O1. Recommendation on record: defer until write-path schema churn settles; Option B (schemas+package / viztoolbox / demos) is the natural first cut; migration can be partial.
> Includes: **seaborn dependency** — runtime dep in `pyproject.toml` but imported only by `code/` notebooks; belongs in a `demos`/`viz` optional-extra (`pip install ccc[demos]`) or dev group, and moves out entirely when demos split into their own repo.

### I26 — Refresh etl_example_prompt.md to package-first (→ WP12)
Labels: `documentation`
> Current prompt predates the write registry and reader; the future ETL skill should teach `write_models` + merge modes + `DatasetReader` verification without notebook-copying overhead. Timing: after I11.

---

## Removed / folded

- ~~I21 seaborn~~ → folded into I25.
- O4 test skill was "not a repo issue" in rev 1 — now **is** the repo issue I20 (conventions + skill), per discussion.

## WP ↔ issue map

| WP | Issues |
|---|---|
| WP1 schema identity & scoping | I1 I2 I4 I5 I6 |
| WP2 merge-scoped writes | I3 I11 I12 (+I17 opportunistically) |
| WP3 cell-cell connectivity | I7 I27 |
| WP4 wide+pointer+bulk | I13 I14 |
| WP5 delete/remove | I15 |
| WP6 edit workflows | I16 |
| WP7 cross-dataset reader | I19 |
| WP8 QC schema | I10 |
| WP9 spatial + embeddings | I8 I9 I18 |
| WP10 test conventions/skill | I20 |
| WP11 research memos | I22 I23 I24 |
| WP12 ETL prompt/skill | I26 |
| WP13 repo split | I25 |
| docs (after WP2) | I28 |

## Decisions log

2026-08-20 (rev 1): individual issues over umbrella; research items go in public tracker; create `schema`/`io`/`research` labels; no per-WP epics; handoffs carry the WP↔issue map.

2026-08-20 (rev 2): I1 project scoping left as open question (common cross-modality features are the goal); I2 confirmed per-taxonomy vocabulary → discriminator; I3 reclassified io-layer (`required_for_write`), schema stays Optional — schema=standard, io=one wrapper; I7 = measurement-context semantics, rerun ETL not migrate; I18 kept separate from I8; I20 = test conventions + skill referencing `test_suite_analysis_2026-08-15.md`; I21 folded into I25; I22 pin-commit-as-version + migration script; links added to I22/I23/I24; new I27 (synapse→cell-cell aggregation) and I28 (README).
