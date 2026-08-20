# Git issues draft — 2026-08-20

Source: `2026-08-20_todo_triage.md` + `2026-08-20_work_packages.md`. Status: **draft for discussion — nothing opened yet.**

Repo labels available today are only the GitHub defaults (`bug`, `enhancement`, `documentation`, `question`, `good first issue`, ...). Proposed new labels (to create if agreed): `schema`, `io`, `research`. Labels below assume they exist; falls back to defaults otherwise.

Issues are atomic and independently closable; the WP column maps to work packages (a PR/handoff may close several).

---

## A. Schema defects (→ WP1)

### I1 — CellFeatureMeasurement: slot_usage fields are never generated
Labels: `bug`, `schema` · WP1
> `cell_features_schema.yaml` declares `feature_set_id` and `unit` for `CellFeatureMeasurement` under `slot_usage:` but not under `slots:`, so LinkML generates neither. The documented "denormalized reference to the feature set (helps partitioning and joins)" does not exist in `models.py`. The class also lacks `project_id`, so long-form measurements cannot be scoped or partitioned.
> **Fix:** add both to `slots:`, add `ProjectScoped` mixin, regenerate models.
> **Done when:** generated model has `feature_set_id`, `unit`, `project_id`; schema test covers it.

### I2 — HierarchyCategory: cross-taxonomy id collisions and level type mismatch
Labels: `bug`, `schema` · WP1
> Ids like `class`/`subclass`/`cluster` are shared across taxonomies while the write scope is `["id"]` — writes from one taxonomy overwrite another's rows (known limitation, `etl_example_prompt.md` §11). Separately, `HierarchyCategory.level` has no declared range and generates as `Optional[str]`, while `Cluster.level` is `int`.
> **Fix:** add a taxonomy/hierarchy discriminator slot; declare `level: integer`; widen write scope accordingly.
> **Done when:** two taxonomies can write categories without collision; `level` types agree.

### I3 — ClusterMembership: identity fields are Optional
Labels: `bug`, `schema` · WP1
> `item` and `cluster` are Optional (everything except `project_id` is), and the class has no `id`. A membership row without an item or cluster is meaningless, and these fields must be non-null to serve as merge keys (see I11).
> **Fix:** make `item` and `cluster` required.
> **Done when:** model rejects membership rows missing item/cluster; write validation covers it.

### I4 — ProjectionMeasurementMatrix: add ProjectScoped
Labels: `enhancement`, `schema` · WP1
> Not ProjectScoped; write spec scopes on `id` only, with an explicit registry comment to widen once the schema gains project scoping. Same matrix id in two projects collides. Blocks `etl_wnm_exc_05`.
> **Fix:** add `ProjectScoped` mixin; widen `partition_by`/`scope_columns` in the registry; update `etl_wnm_exc_04`.

### I5 — SingleCellReconstruction: add ProjectScoped
Labels: `enhancement`, `schema` · WP1
> Not ProjectScoped; its `id` references a DataItem (which is project-scoped), so two projects using the same cell id would collide once this class becomes writable. Git history shows no documented rationale for the omission (unlike BrainRegion, which is deliberately global per the comment in `brain_region_schema.yaml`).
> **Fix:** add mixin; add a WriteSpec when the class is first written. Blocks `etl_wnm_exc_05`.

### I6 — BrainRegionAssociation: no identity, unwritable as designed
Labels: `bug`, `schema` · WP1
> Two Optional fields (`brainregion_id`, `dataitem_id`), no `id`, no `project_id`. Cannot be validated, scoped, or written. Blocks `etl_wnm_exc_05`.
> **Fix:** make both fields required, add `ProjectScoped`; natural composite identity is `(project_id, dataitem_id, brainregion_id)` — no surrogate id needed if merge keys (I11) land.

## B. Schema features

### I7 — CellCellConnectivityLong: add connectome_id discriminator (→ WP3)
Labels: `enhancement`, `schema`
> No discriminator, so two connectomes from the same project (even the same dataset with different cell subsets) cannot share a table. Documented in `etl_example_prompt.md` §5g/§11, the `write_cellcellconnectivitylong` stub, and `etl_examples_readme.ipynb`. Current workaround: one subdirectory per connectome.
> **Fix:** add `connectome_id` slot; register a WriteSpec (`overwrite_scoped` by `(project_id, connectome_id)`); migrate the `cellcellconnectivitylong_*` folders; update `etl_minnie_04` / `etl_v1dd_03`.
> Own planning session before implementation.

### I8 — Spatial coordinates table: promote SpatialLocation (→ WP9)
Labels: `enhancement`, `schema`
> Per-cell coordinates are currently smuggled through generic CellFeatureSets (e.g. `v1dd_soma_spatial` with `soma_voxel_x/y/z`) — untyped floats with no reference-space semantics. `SpatialLocation` (`x,y,z,reference_space`) exists in `core_schema.yaml` but is nested-only and never written.
> **Fix:** first-class table `(dataitem_id, reference_space, x, y, z, project_id?)`; `reference_space` distinguishes CCF-registered vs dataset-original vs dataset-corrected coordinates. Migrate `v1dd_soma_spatial` as the pilot.

### I9 — Embedding coordinates schema (UMAP/tSNE), distinct from spatial (→ WP9)
Labels: `enhancement`, `schema`
> Computed embeddings must not be conflated with anatomical coordinates (the BKP ingest template doesn't distinguish them; we should). `origin/u/morriscb/abcAtlasAccessDraftSchema` drafts `EmbeddingSet` + `EmbeddingMethod` (UMAP/TSNE/PCA/MDS/OTHER).
> **Fix:** align with/adopt that draft; embedding coordinates as a set-scoped table alongside method + params. Coordinate with @morriscb.

### I10 — QC flag schema: per-cell proofreading and modality QC (→ WP8)
Labels: `enhancement`, `schema`, `question`
> No QC concept exists; proofread status is encoded as DataSet membership (`v1dd_1196_proofread_axons` etc.). Need per-cell flags like PROOFREAD_AXON / PROOFREAD_DENDRITE / PROOFREAD_CELL, plus whatever other modalities need.
> **Design options:** (a) per-schema quality slots; (b) base-schema QC enum slot; (c) a `QCFlag` association table `(dataitem_id, flag, value, source, timestamp, project_id)` with per-modality controlled vocabularies (possibly via the `ValueSet` draft on the morriscb branch). Leaning (c) — flags arrive incrementally, which matches merge-scoped writes (I11).
> **First step:** survey QC vocabulary across modalities (research), then decide.

## C. Write layer

### I11 — merge_scoped write mode with declared merge_on keys (→ WP2)
Labels: `enhancement`, `io`
> Only `overwrite_scoped` and `append_new_by_id` exist; no upsert. Incremental workflows (adding dataitems to datasets, revising feature values, QC arriving over time) need merge semantics for identity-bearing tables, while bulk tables (synapse long, wide matrices) should stay overwrite-scoped for memory reasons.
> **Fix:** add `write_mode="merge_scoped"` + `merge_on: list[str]` to `WriteSpec`; implement via delta-rs MERGE; dedupe within batch; per-class assignment table in `planning/20260820/2026-08-20_work_packages.md` (§P4). Replaces `append_new_by_id` for DataItem (also fixes silent no-op metadata updates). Prereqs: I3 (and I4/I6 for later classes).

### I12 — Multi-writer shared-scope data loss (patch-seq associations) (→ WP2)
Labels: `bug`, `io`
> Notebooks sharing scope `(project_id, dataset_id)` silently delete each other's rows: patch-seq associations shrank 2759 → 520 → 495 across `inh_01/02/03` (documented with options A–D in `planning/multi_writer_scope_design.md`). Current workaround is inlined read-union-rewrite in `etl_visp_*` notebooks.
> **Fix:** resolved by I11 (merge_on `(project_id, dataset_id, dataitem_id)`); remove the inlined workarounds in the same PR. Kept separate from I11 as the user-visible bug record.

### I13 — Register SynapseConnectivityLong with a bulk validation mode (→ WP4)
Labels: `enhancement`, `io`
> Registry entry is commented out because `validate_for_write` re-validates every row with pydantic — unusable at 8M rows; the notebook bypasses `write_models` entirely with raw `write_deltalake`. Dead `SynapseConnectivityLong` import remains in `write_spec.py`.
> **Fix:** add a bulk mode (arrow-schema-level validation, optional row sampling) to the WriteSpec vocabulary; register with `overwrite_scoped` on `(project_id, dataset_id)`; route `etl_v1dd_03` through `write_models`; remove the dead import.

### I14 — wide_parquet mode + combined wide-matrix/pointer write API (→ WP4)
Labels: `enhancement`, `io`
> The `wide_parquet` write mode designed in `ARCHITECTURE.md` was never built (W3 deviation); all wide tables (`cellfeatures/`, `synapsefeatures/`, wide projection) are raw `write_deltalake` calls in notebooks. Nothing enforces that pointer rows (`CellFeatureMatrix`, `SynapseFeatureMatrix`) and the parquet they point at are written together — dangling-pointer risk.
> **Fix:** wide_parquet mode; a combined API that writes the payload (overwrite) then merges the pointer row, validating index columns/paths agree. Depends on I11's mode vocabulary.

### I15 — Delete/remove operations for parquet/deltalake outputs (→ WP5)
Labels: `enhancement`, `io`
> No delete, vacuum, or partition-removal exists anywhere; the only row removal is a predicated overwrite that happens to exclude rows. No way to drop a DataItem, retire a feature set/column, or delete a hierarchy.
> **Fix:** explicit remove operations complementary to I11's merge semantics + a vacuum/tombstone policy. Test-first development.

### I16 — Post-write edit workflows: add columns / add dataitems (→ WP6)
Labels: `enhancement`, `documentation`, `io`
> No library support for adding feature columns to an existing feature set or dataitems to an existing dataset; conventions live only in `etl_example_prompt.md` §5d/§9. Mostly falls out of I11 (associations merge) + I14 (defs/pointer merge, wide rewrite); remaining work is a thin API + documented recipes. Removals belong to I15.

### I17 — dry_run silently ignored when output_root is passed
Labels: `bug`, `io`, `good first issue`
> `write_models(..., output_root=...)` resolves `settings=None`, so `dry_run=True` from config is silently ignored and data is written. Also no test asserts dry_run is honored (tests_review finding).
> **Fix:** honor dry_run on all resolution paths; add the missing test.

### I18 — SpatialLocation (nested models) don't round-trip through the arrow layer
Labels: `bug`, `io`
> `arrow_utils._arrow_field_for` falls through to `pa.string()` for embedded BaseModels, and `flatten_refs` only collapses dicts with an `id` key — `SpatialLocation` has none. So `SingleCellReconstruction.soma_location` cannot survive `write_models`. Latent today (class unwritten); blocking for I8/WP9.
> **Fix:** either support nested struct fields in arrow_utils or flatten to a table (I8's direction).

## D. Read layer

### I19 — Cross-dataset reads: union DataItems by shared features/clusters (→ WP7)
Labels: `enhancement`, `io`
> `DatasetReader` is single-dataset; reading cohorts that share feature sets or cluster memberships means N calls + manual concat (`etl_v1dd_04_read`). Design exists: `planning/20260623/prompts/_deferred/08_readers.md` Layer 2, flagship `read_dataitems_for_clusters(cluster_ids, via=("membership","mapping"))`.
> **Fix:** implement Layer 2 readers. Include as a small subtask: move `parquet_loader.py` → `io/parquet_loader.py` (pure move, per the deferred plan).

## E. Tests & chores

### I20 — Test-suite improvements from tests_review findings
Labels: `enhancement` · (tests)
> `planning/20260623/tests_review/findings.md` grades 5 high / 13 medium items: over-broad `pytest.raises(Exception)`, `test_round_trip_each_writable_class` silently skipping coverage drift, no dry_run test (see I17), no concurrent-write/locking test, etc.
> **Fix:** work the high items; adopt consistent error-raising and docstring conventions.

### I21 — Demote seaborn from runtime dependency to dev/extras
Labels: `good first issue`
> `seaborn>=0.13.2` is a runtime dep in `pyproject.toml` but imported nowhere in `src/` (only `code/` notebooks use it).

## F. Research / discussion (→ WP11, WP13, WP12)

### I22 — AIT interoperability: adapter design (→ WP11)
Labels: `question`, `research`, `schema`
> AIT is a spec (h5ad + markdown/CSV schema, R tooling) — no releases, no pip package, actively evolving (last activity 2026-07-28). A package dependency is impossible; divergence is guaranteed.
> **Proposal:** optional `ait_adapter` module mapping AIT h5ad fields ↔ CCM cluster tables, pinned to an AIT commit, with a conformance test against one published taxonomy from their `taxonomies.md`. Coordinate with the `hmba_taxonomy_annotation_schema.yaml` draft (@morriscb branch).
> **First deliverable:** field-mapping memo.

### I23 — BKP ingest schema comparison memo (→ WP11)
Labels: `question`, `research`
> Another team owns CCM→BKP; our deliverable is understanding: entity mapping CCM ↔ BKP ingest visualization files, and a use-case contrast (vis-serving speed vs cross-modality commonality). AIT's README identifies BKP as the ABC Atlas data model, so this pairs naturally with I22.

### I24 — aind-data-schema sidecar: requirements + exporter feasibility (→ WP11)
Labels: `question`, `research`
> Requested by a scientist; use case must be captured first (which aind core files?). aind-data-schema is a versioned pip package, so a small optional exporter emitting a sidecar JSON per DataSet is cheap once the mapping is known. No CCM schema changes expected.

### I25 — Repo split: options and partial migration path (→ WP13)
Labels: `question`
> Options A–D captured in `planning/20260820/2026-08-20_work_packages.md` §O1. Recommendation on record: defer until the write-path schema churn (I1–I14) settles; Option B is the natural first cut. Issue exists to hold the discussion.

### I26 — Refresh etl_example_prompt.md to package-first (→ WP12)
Labels: `documentation`
> Current prompt predates the write registry and reader; the future ETL skill should teach `write_models` + merge modes + `DatasetReader` verification, without notebook-copying overhead.
> **Timing:** after I11 lands, so it documents final semantics.

---

## Not repo issues (on purpose)

- **O4 test-writing skill** — our workflow tooling (lives with Claude skills, not the repo); the repo-side test fixes are I20.
- **VizToolbox work (O3)** — belongs to the VizToolbox repo's tracker.
- **ETL skill packaging (O2 step 2)** — belongs to the demos repo once created; only the prompt refresh (I26) is in this repo.

## WP ↔ issue map

| WP | Issues |
|---|---|
| WP1 schema identity & scoping | I1 I2 I3 I4 I5 I6 |
| WP2 merge-scoped writes | I11 I12 (+I17 opportunistically) |
| WP3 connectome_id | I7 |
| WP4 wide+pointer+bulk | I13 I14 |
| WP5 delete/remove | I15 |
| WP6 edit workflows | I16 |
| WP7 cross-dataset reader | I19 |
| WP8 QC schema | I10 |
| WP9 spatial + embeddings | I8 I9 I18 |
| WP10 test skill | (I20 repo-side) |
| WP11 research memos | I22 I23 I24 |
| WP12 ETL prompt/skill | I26 |
| WP13 repo split | I25 |

## Decisions (2026-08-20, YY)

1. **Granularity:** individual issues I1–I6 (independently closable; WP1 PR closes them together).
2. **Research as issues:** yes — I22–I24 go in the public tracker.
3. **Labels:** create `schema`, `io`, `research` before opening.
4. **No epics:** handoff docs + PR descriptions carry the WP↔issue mapping.
5. Open: I12 kept separate from I11 (bug record vs feature) — merge if fewer issues preferred.
