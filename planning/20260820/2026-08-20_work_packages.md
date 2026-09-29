# Answers to triage notes + work packages — 2026-08-20

Companion to `2026-08-20_todo_triage.md`. Answers follow the discussion order below (revised from the triage doc: your notes resolved P6→P4, spun S4/P1/P3/P5 off as their own PRs, and added S10 + O4).

**Revised discussion order:** P4 (+ which S5/S6 survive it) → S5 → S6 → spun-off PRs (S4, P1, P3, P5, P7) → S7 → S8/S10 → S1–S3 → O1–O4.

---

## P4. Merge-scoped vs overwrite-scoped writes

**Your memory concern, checked:** delta-rs `MERGE` runs in the Rust engine with partition pruning, not Python row-by-row — but it still joins incoming rows against the existing scope. Fine at metadata/association scale (10³–10⁶ small rows), wasteful at synapse scale (10⁷). Your split is the right call: **merge for identity-bearing metadata/association tables, overwrite for bulk long tables and wide matrices.** This is purely a registry vocabulary change: add `write_mode="merge_scoped"` + a `merge_on: list[str]` field to `WriteSpec`.

### Proposed per-class assignment

**merge_scoped** (upsert on `merge_on`; incremental writers safe):

| Class | merge_on | prerequisite |
|---|---|---|
| DataSet | project_id, id | — |
| DataItem | project_id, id | — (replaces `append_new_by_id`; fixes the silent no-op update) |
| DataItemDataSetAssociation | project_id, dataset_id, dataitem_id | none — composite key suffices, **no surrogate id needed**; writer should dedupe within batch |
| ClusterMembership | project_id, hierarchy_id, item, cluster | S6: make `item`, `cluster` required |
| Cluster | hierarchy_id, id | — (fixes "partial write deletes hierarchy") |
| ClusterHierarchy | id | — |
| AlgorithmRun | id | — |
| HierarchyCategory | hierarchy/taxonomy discriminator + id | S6: add discriminator field (merge can't fix colliding `class`/`subclass` ids) |
| MappingSet | project_id, id | — |
| CellToClusterMapping | project_id, id | — |
| CellFeatureSet | project_id, id | — |
| CellFeatureDefinition | project_id, feature_set_id, id | — (enables adding feature columns → P7) |
| CellFeatureMatrix | project_id, id | — (fixes pointer-row clobber; registry fix only) |
| SynapseFeatureMatrix | project_id, id | — |
| ProjectionMeasurementMatrix | project_id, id | S5: add ProjectScoped |
| Later: SingleCellReconstruction | project_id, id | S5 |
| Later: BrainRegionAssociation | project_id, dataitem_id, brainregion_id | S5/S6: add identity + scoping |
| Later: BrainRegion | id | stays global (see S5) |

**overwrite_scoped** (bulk republish; keep current semantics):

| Class | scope | note |
|---|---|---|
| SynapseConnectivityLong | project_id, dataset_id | 8M rows; register with a bulk/arrow-only validation mode (P5 PR) |
| CellCellConnectivityLong | project_id, connectome_id | after S4 PR |
| Wide tables (cellfeatures/, synapsefeatures/, wide projection) | per feature_set / matrix id | via P5 `wide_parquet` mode |
| Later: CellFeatureMeasurement | project_id, feature_set_id | long-form bulk; after its schema bug fix |

Design detail worth deciding in the PR: whether merge also deletes rows absent from the batch within a declared sub-scope (true sync) or only upserts (accumulate). For "workflows keep adding dataitems and revising feature values," pure upsert is right; deletion stays explicit (P3).

### Which S5/S6 schema changes survive P4

Still required (merge keys need them):
1. **S5:** ProjectScoped on `ProjectionMeasurementMatrix`, `SingleCellReconstruction`; identity + scoping on `BrainRegionAssociation`. `BrainRegion` stays global — intentional, see S5 below.
2. **S6:** `CellFeatureMeasurement` slot_usage bug (`feature_set_id`, `unit` never generated) + missing `project_id` — independent of P4 entirely.
3. **S6:** `HierarchyCategory` needs a taxonomy discriminator; also `level` typed `str` vs `Cluster.level` `int`.
4. **S6:** `ClusterMembership.item` / `.cluster` must become required (merge keys can't be nullable).

Made unnecessary by P4 (registry/writer fixes only, no schema change):
- Surrogate `id` on `DataItemDataSetAssociation` (composite merge key suffices).
- `Cluster` partial-write deletion (merge_on fixes it).
- `CellFeatureMatrix` pointer clobber (merge_on `id`).
- `DataItem` update path (merge = upsert).
- P6 patch-seq association shrink — solved by merge_on; delete the inlined read-union-rewrite blocks from the `etl_visp_*` notebooks in the same PR.

---

## S5. Was the missing scoping intentional?

Git history answer: the schemas were prototyped by **Forrest Collman, Oct–Nov 2025**. For `BrainRegion` it **was intentional and documented** — commit `b80cb3c` (2025-10-22) added, in `brain_region_schema.yaml`:
> "BrainRegion may often be global; include ProjectScoped only if some regions are project-specific. Commented out by default to keep ontology global."

For `SingleCellReconstruction`, `ProjectionMeasurementMatrix`, `BrainRegionAssociation`: **no comment, no commit message rationale — almost certainly just not gotten to** (they were never wired to any writer either). If it had been intentional, the justification would presumably be the same "global reference data, project reachable by join through DataItem" logic — the same reasoning you yourself applied when dropping ProjectScoped from `Cluster` (`7d33d76`). That logic holds for BrainRegion (an atlas ontology) but not for projection matrices or reconstructions, which are project outputs — and `etl_wnm_exc_05` being blocked proves the need. Recommendation: scope all three, keep BrainRegion global.

## S6. Writers' fault, or truly missing?

Three layers:
1. **Truly missing (schema):** identity/key fields listed above. The prototype era had no scoped write path — writes were ad hoc — so nothing ever demanded row identity. Your colleague couldn't have hit it: the collisions only occur when **multiple datasets/notebooks share a scope**, exactly your patch-seq case.
2. **Writer semantics expose it:** `overwrite_scoped` turns "missing identity" into silent data loss (last writer wins). Not a writer bug per se — a missing third mode, which P4 adds.
3. **Registry + write_validation do NOT solve it:** `validate_for_write` is presence/type per row only. No uniqueness check, no key declaration, no referential check (that's the deferred `check_refs`). After P4, `merge_on` becomes the registry's declared identity — which is the actual fix.

## S9 / P2 — closed / answered. S4, P1, P3, P5, P7 — spun off as PRs

- **S4 (connectome_id):** own PR + planning session. Soft dependency: its registry entry should use the P4 mode vocabulary, so land after (or rebase on) the P4 PR.
- **P1 (multi-dataset reader):** own PR; includes the `parquet_loader.py` → `io/` move as a small task. Independent of the write-side critical path — can run in parallel. Start from `_deferred/08_readers.md` Layer 2.
- **P3 (delete/remove):** own PR, test-first as you suggested — pairs with O4 (write the tests skill first, use it here). Depends on P4 (deletion semantics should complement merge semantics, not fight them).
- **P5 (wide-form + pointer combined API):** own PR. **How P4 affects it:** pointer rows (`CellFeatureMatrix`, `SynapseFeatureMatrix`, `ProjectionMeasurementMatrix`) become merge_scoped → re-registering a matrix is idempotent; the wide payload stays overwrite_scoped. The combined API is then a sequenced pair: write payload (overwrite) → merge pointer, with validation that index columns/paths agree — killing the dangling-pointer class of bugs. The third piece, a `bulk` mode for `SynapseConnectivityLong` (arrow-schema validation instead of per-row pydantic), belongs in this same PR since it reuses the mode vocabulary. So: P4 defines the vocabulary, P5 consumes it — strict ordering.
- **P7 (edit workflows):** mostly falls out of P4 (add dataitems = merge associations; revise feature values = overwrite feature-set scope) + P5 (add columns = rewrite wide + merge defs/pointer). Removal of rows/columns follows in P3's package, as you said.

## S7. QC flags — the design choice

Three options for the discussion:
- **(a) Per-schema quality slots** (like `CellMetadata.quality_score`): matches each modality's vocabulary, but N ad-hoc slots, unqueryable across modalities, and schema churn per new QC concept.
- **(b) Base-schema QC slot with permissible values:** one enum in `base_schema.yaml`; but a single enum mixing `PROOFREAD_AXON` with patch-seq QC states gets ugly, and a slot on DataItem forces one value per cell.
- **(c) QC as an association table** (my lean): `QCFlag(dataitem_id, flag, value/bool, source, timestamp, project_id)` — multiple flags per cell, flags arrive incrementally as QC happens (which is exactly the merge_scoped write pattern from P4), and permissible values per modality live in a controlled vocabulary. Notably, morriscb's draft branch already has a `value_sets_schema.yaml` (`ValueSet`, `ValueSetHierarchy`) that could host per-modality QC vocabularies instead of hardcoded enums.
- Current state for reference: proofread status is encoded as DataSet membership (`v1dd_1196_proofread_axons` etc.) — option (c) would supersede that pattern without breaking it.
- Open subtask: survey QC vocabulary in other modalities (patch-seq QC calls, recon completeness, mapping confidence) — delegable research.

## S8 + new S10. Spatial coordinates

- Branch link: <https://github.com/AllenInstitute/ConnectsCommonConnectivity/tree/u/morriscb/abcAtlasAccessDraftSchema> — embedding schema file: <https://github.com/AllenInstitute/ConnectsCommonConnectivity/blob/u/morriscb/abcAtlasAccessDraftSchema/schemas/hmba_embedding_schema.yaml> (`EmbeddingSet`, `EmbeddingMethod` enum: UMAP/TSNE/PCA/MDS/OTHER).
- **S10 answer — yes, an unused schema exists:** `SpatialLocation` in `core_schema.yaml` (`x`, `y`, `z` required, `reference_space` string e.g. `'CCF_v3'`). It's nested-only today (inside `SingleCellReconstruction.soma_location`) and **cannot round-trip through `write_models`** (the arrow layer stringifies embedded models without an `id`). Adoption path: promote it to a first-class per-cell table — `(dataitem_id, reference_space, x, y, z [, project_id])` — where `reference_space` distinguishes exactly your three cases: `CCF_v3`-registered, `dataset_original`, `dataset_corrected`. That replaces the `v1dd_soma_spatial` feature-set workaround with typed semantics.
- **S8 distinction you wanted (cell coords vs UMAP coords), which the BKP template lacks:** keep anatomical coordinates in the SpatialLocation-derived table (reference spaces are physical), and computed embeddings under `EmbeddingSet` + its own coordinates (method + params matter, dimensionality varies). Two schemas, deliberately not one.

## S1. AIT — divergence, cadence, packaging

- **Can it be imported as a package? No.** AIT is a *spec*, not a library: an anndata `.h5ad` file convention with a markdown+CSV schema (`schema/`), companion tooling is the **scrattch R libraries**. Nothing pip-installable exists.
- **How often does it change?** No releases, no tags, ever — 111 commits total; latest activity 2026-07-28 (schema-doc alignment against their CSV source of truth). So: actively evolving, in bursts, with no versioned artifacts to pin against.
- **Consequence — your helper-util instinct is correct:** a dependency is impossible and undesirable; divergence is guaranteed. Build a small **adapter module** (e.g. `ait_adapter`, optional extra, not inside `io/`) that maps AIT h5ad fields ↔ CCM cluster tables (`Cluster`/`ClusterHierarchy`/`ClusterMembership`/`AlgorithmRun`), pinned to an AIT commit hash recorded in the adapter, with a conformance test against one small published taxonomy from their `taxonomies.md`. When AIT moves, the test breaks loudly and you update the mapping — divergence becomes a maintained seam instead of silent drift. Zero impact on the io package.
- Coordinate with morriscb's `hmba_taxonomy_annotation_schema.yaml` draft (annotation-term classes) before designing the CCM side of the mapping.

## S2. BKP explorer — scoped to a comparison memo

Since another team owns CCM→BKP, our deliverable is understanding: an entity-mapping memo (CCM classes ↔ BKP ingest visualization files), plus a use-case contrast — BKP optimizes for fast vis-app serving (denormalized, per-visualization files); CCM optimizes for capturing cross-modality commonality (normalized, long-form + pointers). Useful side-fact: AIT's own README identifies BKP as the ABC Atlas / MapMyCells data model, so the S1 and S2 research overlap — one subagent can do both memos in a single pass.

## S3. aind-data-schema sidecar

Requested by a scientist → treat as requirements-gathering first: get the concrete use case from them (which aind core files — `data_description`, `subject`, `session`, `procedures`?). Unlike AIT, **aind-data-schema IS a versioned pip package**, so a small exporter util (optional extra) emitting a sidecar JSON per DataSet is cheap once the mapping is known. Deliverable: memo + (if justified) a tiny exporter, no schema changes to CCM.

## O1–O4

- **O1 (repo split):** agree — partial and deferred. Practical note: keep schemas+package together until the WP1–WP4 schema churn settles; splitting now means cross-repo PRs for every identity fix. The only near-term forcing function is O2 (where the skill lives). Revisit after the write-path PRs land; Option B (schemas+package / viztoolbox / demos) is the natural first cut since it just formalizes what already exists.
- **O2 (ETL skill):** agreed sequence — step 1 is revising `etl_example_prompt.md` into a package-first prompt (drop notebook-copying overhead, teach `write_models` + the new merge modes + `DatasetReader` verification). Do this *after* the P4 PR so the skill teaches the final semantics, not soon-to-change ones. Step 2: package it as a skill living in the demos(/+viztoolbox) repo, which imports the ccc package — dependency direction stays clean, and it can call viztoolbox for post-ETL sanity plots (O3).
- **O3:** stays in the viz repo; its only coupling here is O2's skill importing it and P1's reader being what it plots. No main-repo work.
- **O4 (test-writing skill):** quick win, independent, do early. Inputs already exist: `planning/20260623/tests_review/findings.md` (graded backlog: over-broad `pytest.raises(Exception)`, silent coverage-drift skip, missing dry-run and concurrent-write tests). The skill should encode: error-raising conventions, docstring style, coverage-drift guards, test-first scaffolding. Use it first on P3's test-first delete package.

---

## Work packages and dependencies

| WP | Content | Depends on |
|---|---|---|
| **WP1** | Schema identity & scoping PR: S5 (ProjectScoped ×3, BRA identity) + surviving S6 (CellFeatureMeasurement bug, HierarchyCategory discriminator + level type, ClusterMembership required fields) | — |
| **WP2** | Merge-scoped writes PR (P4/P6): `merge_on` in WriteSpec, delta MERGE writer, batch dedupe, per-class assignment table above, de-inline notebook workarounds | WP1 |
| **WP3** | connectome_id PR (S4), own planning session | soft: WP2 (mode vocab) |
| **WP4** | Wide-form + pointer combined API + bulk long-table mode PR (P5) | WP2 |
| **WP5** | Delete/remove PR (P3), test-first | WP2, WP10 |
| **WP6** | Edit workflows (P7) — thin layer / docs over WP2+WP4; removals live in WP5 | WP2, WP4 |
| **WP7** | Multi-dataset reader PR (P1) + parquet_loader move | — (parallel) |
| **WP8** | QC schema (S7): decide option a/b/c, then PR | — (design first) |
| **WP9** | Spatial coordinates schema (S10): promote SpatialLocation to table, fix/bypass arrow nested bug | — (design first) |
| **WP10** | Test-writing skill (O4) | — (quick, do early) |
| **WP11** | Research memos (S1 AIT + S2 BKP together; S3 aind; S7 QC vocab survey) — subagent-delegable, parallel | — |
| **WP12** | ETL prompt refresh → skill (O2) | WP2 (+ WP4 ideally) |
| **WP13** | Repo split (O1), partial | after WP1–WP6 settle |

**Critical path:** WP1 → WP2 → {WP3, WP4, WP5, WP6} → WP12.
**Anytime/parallel:** WP7, WP8, WP9, WP10, WP11.
**Deferred:** WP13.

Suggested first moves: WP10 + WP11 kick off immediately (cheap, delegable); WP1 as a small focused PR; WP2 planning session next.
