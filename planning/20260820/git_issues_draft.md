# Git issues draft — 2026-08-20 (rev 3)

Source: `2026-08-20_todo_triage.md` + `2026-08-20_work_packages.md` + review discussions with YY. Status: **draft — nothing opened yet.**

Rev 3 changes: renumbered in WP order (= submission order), bodies rewritten to a uniform Problem/Fix/Context template, old I1 split into #1 (bug) + #28 (scoping question). Old→new mapping at the bottom.

Before opening: create labels `schema`, `io`, `research` (repo has GitHub defaults only). Optional: one milestone per WP for grouping (lighter than epics, consistent with the no-epics decision).

**Cross-references:** `#N` below are draft numbers — GitHub will assign its own (PRs #5/#6 already consumed numbers). Submit in order 1→28, record the real numbers, then do one edit pass replacing draft refs with real links.

---

## WP1 — schema identity & scoping

### 1. CellFeatureMeasurement: `feature_set_id` and `unit` are never generated
Labels: `bug`, `schema` *(old I1, bug half)*

> **Problem:** `cell_features_schema.yaml` declares `feature_set_id` and `unit` for `CellFeatureMeasurement` under `slot_usage:` but not under `slots:`. LinkML generates neither, so `models.py` lacks both fields — the documented "denormalized reference to the feature set" doesn't exist.
> **Fix:** declare both under `slots:`; regenerate models.
> **Related:** whether feature sets are project-scoped is #28 — don't silently add the ProjectScoped mixin here.

### 2. HierarchyCategory: per-taxonomy vocabulary; fix `level` type
Labels: `bug`, `schema` *(old I2)*

> **Problem:** category ids like `class`/`subclass`/`cluster` are global, but taxonomies don't share a level structure. Two taxonomies writing `class` collide — the last writer's `level`/`description` wins. Merge-scoped writes (#6) don't fix this. Also `level` generates as `Optional[str]` while `Cluster.level` is `int`.
> **Fix:** add a hierarchy/taxonomy discriminator slot so category vocabularies are per-taxonomy; declare `level: integer`.
> **Context:** `Cluster` holds hierarchical categorical features generally, not only cell types — we don't want to force a common level structure.

### 3. ProjectionMeasurementMatrix: add ProjectScoped
Labels: `enhancement`, `schema` *(old I4)*

> **Problem:** not ProjectScoped; the write spec scopes on `id` only (registry comment says to widen once the schema allows). Same matrix id in two projects collides. Blocks `etl_wnm_exc_05`.
> **Fix:** add the `ProjectScoped` mixin; widen `partition_by`/`scope_columns`; update `etl_wnm_exc_04`.

### 4. SingleCellReconstruction: add ProjectScoped
Labels: `enhancement`, `schema` *(old I5)*

> **Problem:** not ProjectScoped, but its `id` references a project-scoped DataItem — two projects reusing a cell id collide once this class becomes writable. Unlike `BrainRegion` (deliberately global, documented in the schema), there is no rationale for this omission. Blocks `etl_wnm_exc_05`.
> **Fix:** add the `ProjectScoped` mixin.

### 5. BrainRegionAssociation: no identity, unwritable as designed
Labels: `bug`, `schema` *(old I6)*

> **Problem:** this class records which brain region a DataItem is in (region-to-region projection lives in `ProjectionMeasurementMatrix`), but has only two Optional fields — no `id`, no `project_id`. It can't be validated, scoped, or written. Blocks `etl_wnm_exc_05`.
> **Fix:** add `ProjectScoped`; require `brainregion_id`/`dataitem_id` at write time via the io layer (`required_for_write`, per #7's convention). Natural identity is `(project_id, dataitem_id, brainregion_id)` — no surrogate id needed once merge keys exist (#6).
> **Design note:** a cell may get region assignments from multiple methods (CCF registration vs manual) or term sets — consider a `method`/`source` slot.

## WP2 — merge-scoped writes

### 6. Add `merge_scoped` write mode with declared `merge_on` keys
Labels: `enhancement`, `io` *(old I11)*

> **Problem:** only `overwrite_scoped` and `append_new_by_id` exist — no upsert. Incremental workflows (adding dataitems, revising feature values, QC arriving over time) silently lose data (#8) or no-op.
> **Fix:** add `write_mode="merge_scoped"` + `merge_on: list[str]` to `WriteSpec`; implement with delta-rs MERGE; dedupe within batch; assign modes per class per the table in `planning/20260820/2026-08-20_work_packages.md` §P4. Replaces `append_new_by_id` for DataItem (fixes silent no-op metadata updates).
> **Scope:** merge is for identity-bearing metadata/association tables (10³–10⁶ rows). Bulk long tables and wide matrices stay `overwrite_scoped` — MERGE joins incoming vs existing rows, wasteful at 10⁷.
> **Decide in the PR:** pure upsert (accumulate) vs sync-with-delete within a sub-scope. Pure upsert fits our workflows; deletion stays explicit (#14).

### 7. ClusterMembership: enforce merge keys in io, not schema
Labels: `enhancement`, `io` *(old I3)*

> **Problem:** `item` and `cluster` are Optional in the schema and there is no `id`; merge keys (#6) must be non-null at write time.
> **Fix:** add `item`, `cluster` to `required_for_write` (as already done for `hierarchy_id`); schema stays Optional.
> **Convention (decided):** the schema is the standard (eventually its own repo); this io package is one wrapper around it. Constraints required only by *this* write method belong in the io layer, not the schema. Apply wherever merge keys land (#5 etc.).

### 8. Multi-writer data loss in shared scopes (patch-seq associations)
Labels: `bug`, `io` *(old I12)*

> **Problem:** notebooks sharing scope `(project_id, dataset_id)` silently delete each other's rows — associations shrank 2759 → 520 → 495 across `etl_visp_inh_patchseq_01/02/03`. Current workaround: inlined read-union-rewrite in the notebooks. Options analysis: `planning/multi_writer_scope_design.md`.
> **Fix:** resolved by #6 with `merge_on (project_id, dataset_id, dataitem_id)`; remove the inlined workarounds in the same PR. This issue is the user-visible bug record.

### 9. dry_run silently ignored when output_root is passed
Labels: `bug`, `io`, `good first issue` *(old I17)*

> **Problem:** `write_models(..., output_root=...)` resolves `settings=None`, so `dry_run=True` is silently ignored and data is written.
> **Fix:** honor `dry_run`; add the missing test.

## WP3 — cell-cell connectivity

### 10. CellCellConnectivityLong: add `connectome_id`
Labels: `enhancement`, `schema` *(old I7)*

> **Problem:** no measurement-context discriminator — rows from two measurement contexts are indistinguishable and their overwrite predicates clobber each other. Hence today's one-subdirectory-per-connectome workaround (`cellcellconnectivitylong_*`).
> **Semantics:** `connectome_id` identifies the **measurement context** (segmentation version + proofreading state + measurement semantics), **not** the cohort. Cohorts are read-side filters: write the superset connectome once; pull subsets by joining `dataitem_dataset_association` or cluster membership. Minnie's two folders (`proofread_pre_to_csm_post`, `proofread_to_proofread`) are two measurement contexts → two `connectome_id`s in one table.
> **Fix:** add `connectome_id`; register a WriteSpec (`overwrite_scoped` on `(project_id, connectome_id)`); rerun affected ETLs with the new io + schema (no folder migration); add a subset-read helper (ties to #16). Needs its own planning session before implementation.

### 11. Derive CellCellConnectivityLong from a synapse table
Labels: `enhancement`, `io` *(old I27)*

> **Feature:** when a `SynapseConnectivityLong` table exists, cell-cell connectivity is derivable — group by `(presynaptic_cell, postsynaptic_cell)` and aggregate. Provide a function that populates `CellCellConnectivityLong` automatically.
> **Decide:** aggregations — total synapse count always; total synapse size when a per-synapse size column exists (via the dataset's `SynapseFeatureMatrix`). Each aggregation gets its own `measurement_type` (`SynapticMeasurementType`).
> **Depends on:** #10 (`connectome_id` on the output); pairs with #12.

## WP4 — bulk + wide writes

### 12. Register SynapseConnectivityLong with a bulk validation mode
Labels: `enhancement`, `io` *(old I13)*

> **Problem:** registry entry is commented out because `validate_for_write` re-validates every row with pydantic — unusable at 8M rows; `etl_v1dd_03` bypasses `write_models` with raw `write_deltalake`. A dead import remains in `write_spec.py`.
> **Fix:** add a bulk mode (arrow-schema-level validation, optional row sampling); register `overwrite_scoped` on `(project_id, dataset_id)`; route the notebook through `write_models`; remove the dead import.

### 13. wide_parquet mode + combined wide-matrix/pointer write API
Labels: `enhancement`, `io` *(old I14)*

> **Problem:** the `wide_parquet` mode designed in `ARCHITECTURE.md` was never built; all wide tables are raw `write_deltalake` calls in notebooks. Nothing enforces that pointer rows (`CellFeatureMatrix`, `SynapseFeatureMatrix`) and the parquet they point at are written together — dangling-pointer risk.
> **Fix:** build `wide_parquet`; add a combined API that writes the payload (overwrite) then merges the pointer row, validating that index columns/paths agree.
> **Depends on:** #6 (mode vocabulary).

## WP5 — delete/remove

### 14. Delete/remove operations for parquet/deltalake outputs
Labels: `enhancement`, `io` *(old I15)*

> **Problem:** no delete, vacuum, or partition removal exists; the only row removal is a predicated overwrite that happens to exclude rows. No way to drop a DataItem, retire a feature set/column, or delete a hierarchy.
> **Fix:** explicit remove operations complementary to #6, plus a vacuum/tombstone policy. Test-first development (use #21's conventions).

## WP6 — edit workflows

### 15. Post-write edit workflows: add columns / add dataitems
Labels: `enhancement`, `documentation`, `io` *(old I16)*

> **Problem:** no library support for adding feature columns to an existing feature set or dataitems to an existing dataset; conventions live only in `etl_example_prompt.md` §5d/§9.
> **Fix:** mostly falls out of #6 + #13; remaining work is a thin API plus documented recipes. Removals → #14.

## WP7 — cross-dataset reader

### 16. Cross-dataset reads: union DataItems by shared features/clusters
Labels: `enhancement`, `io` *(old I19)*

> **Problem:** `DatasetReader` is single-dataset; cohorts sharing feature sets or memberships need N calls + manual concat (`etl_v1dd_04_read`).
> **Fix:** implement `planning/20260623/prompts/_deferred/08_readers.md` Layer 2; flagship `read_dataitems_for_clusters(cluster_ids, via=("membership","mapping"))`.
> **Subtasks:** move `parquet_loader.py` → `io/parquet_loader.py` (pure move); subset-connectivity filter helper (see #10, #11).

## WP8 — QC schema

### 17. QC flag schema: per-cell proofreading and modality QC
Labels: `enhancement`, `schema`, `question` *(old I10)*

> **Problem:** no QC concept exists; proofread status is encoded as DataSet membership (`v1dd_1196_proofread_axons` etc.). Need per-cell flags like PROOFREAD_AXON / PROOFREAD_DENDRITE / PROOFREAD_CELL and equivalents for other modalities.
> **Inclination (TBD):** per-modality QC vocabularies **plus** a cross-modality `acceptable` flag. Candidate shape: `QCFlag` association table `(dataitem_id, flag, value, source, timestamp, project_id)`, vocabularies possibly via the `ValueSet` draft (morriscb branch). Flags arrive incrementally — matches merge-scoped writes (#6).
> **First step:** survey QC vocabulary across modalities.

## WP9 — spatial + embeddings

### 18. Promote SpatialLocation to a first-class coordinates table
Labels: `enhancement`, `schema` *(old I8)*

> **Problem:** per-cell coordinates are smuggled through generic CellFeatureSets (`v1dd_soma_spatial`: `soma_voxel_x/y/z`, …) — untyped floats, no reference-space semantics. `SpatialLocation` (`x, y, z, reference_space`) exists in `core_schema.yaml` but is unreachable on disk: its only attachment points were never written by any ETL, it has no id/WriteSpec (designed as an embedded struct), and the write path has no struct support (#20).
> **Fix:** first-class table `(dataitem_id, reference_space, x, y, z [, project_id])`; `reference_space` distinguishes CCF-registered vs dataset-original vs dataset-corrected coordinates. Pilot: migrate `v1dd_soma_spatial`. Flattening to a table bypasses #20 (kept separate as the general bug).

### 19. Embedding coordinates schema (UMAP/tSNE), distinct from spatial
Labels: `question`, `schema` *(old I9)*

> **Question:** computed embeddings must not be conflated with anatomical coordinates (the BKP ingest template doesn't distinguish them; we should). `origin/u/morriscb/abcAtlasAccessDraftSchema` drafts `EmbeddingSet` + `EmbeddingMethod` (UMAP/TSNE/PCA/MDS/OTHER). Align with / adopt that draft? Coordinate with @morriscb.

### 20. Nested models (SpatialLocation) don't round-trip through the arrow layer
Labels: `bug`, `io` *(old I18)*

> **Problem:** the write path is pydantic → dicts → arrow → parquet. `flatten_refs` collapses a nested model to its id string — fine for references, but `SpatialLocation` has no id, so it survives as a raw dict while `build_arrow_schema` maps embedded-model fields to `string`. A dict then hits a string column: pyarrow errors, or stringifies into an unqueryable text blob. Model → parquet → model loses structure.
> **Fix options:** (a) struct columns in `arrow_utils`; (b) explicit JSON-column convention; (c) don't nest — #18 flattens spatial and bypasses this. Keep this issue as the general record for any future nested model.

## WP10 — tests

### 21. Test-writing conventions + skill
Labels: `enhancement`, `documentation` *(old I20)*

> Establish testing conventions and encode them as a reusable skill: consistent error raising and warning patterns, descriptive docstrings, test-first scaffolding for new io features (first consumer: #14).
> **References:** `planning/test_suite_analysis_2026-08-15.md` (healthy 169-test pyramid; top gaps: CI enforcement, coverage measurement, direct `arrow_utils` tests, deeper parquet-loader cases, CLI behavioral tests) and `planning/20260623/tests_review/findings.md`. Gap items can become checkboxes here or follow-up issues.

## WP11 — research memos

### 22. AIT schema comparison + adapter/migration strategy
Labels: `question`, `research`, `schema` *(old I22)*

> **Task:** review the [AllenInstituteTaxonomy](https://github.com/AllenInstitute/AllenInstituteTaxonomy) schema and compare it to CCM's clustering schemas (`Cluster`, `ClusterHierarchy`, `ClusterMembership`, `AlgorithmRun`) — field-mapping memo.
> **Context:** AIT is a spec (anndata h5ad + markdown/CSV schema; R tooling) — no releases, no pip package, actively evolving.
> **Versioning approach:** pin a specific AIT commit as our de-facto version; maintain a migration script for switching pins; conformance test against one published taxonomy from their `taxonomies.md` breaks loudly on drift. Coordinate with the `hmba_taxonomy_annotation_schema.yaml` draft (@morriscb branch).

### 23. BKP ingest schema comparison
Labels: `question`, `research` *(old I23)*

> **Task:** compare CCM schemas to the [BKP explorer ingest template](https://github.com/AllenInstitute/BkpPlotGenerator/blob/main/docs/Ingest-Template/README.md#visualization-files) — similarities, differences, and use-case contrast (fast vis-app serving vs cross-modality commonality). CCM→BKP export is owned by another team; this is for understanding.
> ⚠️ **Wording still to be revised by YY before submission.**

### 24. aind-data-schema: explore incorporation
Labels: `question`, `research` *(old I24)*

> **Task:** explore how [aind-data-schema](https://aind-data-schema.readthedocs.io/en/latest/index.html) can be incorporated — a sidecar JSON next to CCM was requested by a scientist. Capture the concrete use case (which aind core files?), then assess a small exporter util (aind-data-schema is a versioned pip package, so an optional-extra exporter is cheap once the mapping is known).

## WP12 — ETL prompt

### 25. Refresh etl_example_prompt.md to package-first
Labels: `documentation` *(old I26)*

> **Problem:** the current prompt predates the write registry and reader.
> **Fix:** teach `write_models` + merge modes + `DatasetReader` verification, without notebook-copying overhead; the future ETL skill builds on this.
> **Blocked by:** #6.

## WP13 — repo split

### 26. Repo split: options and partial migration path
Labels: `question` *(old I25)*

> **Question:** Options A–D in `planning/20260820/2026-08-20_work_packages.md` §O1. On record: defer until write-path schema churn settles; Option B (schemas+package / viztoolbox / demos) is the natural first cut; migration can be partial.
> **Checkbox:** `seaborn` is a runtime dep in `pyproject.toml` but imported only by `code/` notebooks — move to a demos/viz optional-extra (`pip install ccc[demos]`) or dev group; moves out entirely when demos split off.

## Docs

### 27. Update README
Labels: `documentation` *(old I28)*

> **Problem:** README predates the v1dd-ingest work and the write-mode changes.
> **Fix:** document read API (`DatasetReader`, `read_synapse_table`), write modes (merge/overwrite/bulk/wide), config, and current schema coverage.
> **Blocked by:** #6.

## Design questions

### 28. Should feature sets/definitions be project-scoped?
Labels: `question`, `schema` *(old I1, question half)*

> **Question:** the long-term goal is common features across modalities/projects, so feature sets/definitions should ideally not be hard project-scoped — though most sets will in practice be project-specific. With merge-scoped writes (#6) the write-side cost of `project_id` is ~zero (just part of a merge key, no clobber risk). Remaining concerns: (a) `dataitem_id` is only unique per project, so an unscoped measurement row is ambiguous in joins; (b) namespacing for genuinely shared sets — sentinel project (e.g. `project_id="common"`) vs optional `project_id` on `CellFeatureSet`/`CellFeatureDefinition`.
> **Informs:** #1, #6.

---

## Old → new mapping

| Old | New | | Old | New | | Old | New |
|---|---|---|---|---|---|---|---|
| I1 | 1 + 28 | | I11 | 6 | | I19 | 16 |
| I2 | 2 | | I12 | 8 | | I20 | 21 |
| I3 | 7 | | I13 | 12 | | I22 | 22 |
| I4 | 3 | | I14 | 13 | | I23 | 23 |
| I5 | 4 | | I15 | 14 | | I24 | 24 |
| I6 | 5 | | I16 | 15 | | I25 | 26 |
| I7 | 10 | | I17 | 9 | | I26 | 25 |
| I8 | 18 | | I18 | 20 | | I27 | 11 |
| I9 | 19 | | I21 | folded into 26 | | I28 | 27 |
| I10 | 17 | | | | | | |

## WP ↔ issue map

| WP | Issues |
|---|---|
| WP1 schema identity & scoping | 1–5 |
| WP2 merge-scoped writes | 6–8 (+9 opportunistically) |
| WP3 cell-cell connectivity | 10, 11 |
| WP4 wide+pointer+bulk | 12, 13 |
| WP5 delete/remove | 14 |
| WP6 edit workflows | 15 |
| WP7 cross-dataset reader | 16 |
| WP8 QC schema | 17 |
| WP9 spatial + embeddings | 18–20 |
| WP10 test conventions/skill | 21 |
| WP11 research memos | 22–24 |
| WP12 ETL prompt/skill | 25 |
| WP13 repo split | 26 |
| docs (after WP2) | 27 |
| design questions | 28 |

## Decisions log

2026-08-20 (rev 1): individual issues over umbrella; research items go in public tracker; create `schema`/`io`/`research` labels; no per-WP epics; handoffs carry the WP↔issue map.

2026-08-20 (rev 2): I1 project scoping left as open question (common cross-modality features are the goal); I2 confirmed per-taxonomy vocabulary → discriminator; I3 reclassified io-layer (`required_for_write`), schema stays Optional — schema=standard, io=one wrapper; I7 = measurement-context semantics, rerun ETL not migrate; I18 kept separate from I8; I20 = test conventions + skill referencing `test_suite_analysis_2026-08-15.md`; I21 folded into I25; I22 pin-commit-as-version + migration script; links added to I22/I23/I24; new I27 (synapse→cell-cell aggregation) and I28 (README).

2026-08-27 (rev 3): renumbered in WP order = submission order; uniform Problem/Fix/Context template; draft-history notes moved out of bodies; old I1 split into #1 (bug) + #28 (scoping question); cross-ref plan = submit 1→28 then one edit pass replacing draft numbers with real GitHub links; explicit "Blocked by #6" on #25/#27; #23 wording still pending YY.
