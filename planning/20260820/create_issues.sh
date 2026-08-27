#!/usr/bin/env bash
# Create GitHub labels, milestones, and issues from planning/20260820/git_issues_draft.md (rev 3).
#
# Requires: gh CLI, authenticated with issues:write on the repo (`gh auth login`).
#
# Usage (run phases in order):
#   ./create_issues.sh setup     # create labels + milestones (idempotent)
#   ./create_issues.sh create    # create issues 1-28 in order; records real numbers in issue_number_map.txt
#   ./create_issues.sh fixrefs   # replace {{N}} draft cross-refs in issue bodies with real #numbers
#
#   DRY_RUN=1 ./create_issues.sh create   # print what would be created, submit nothing
#
# Nothing is submitted unless you run it.

set -euo pipefail

REPO="AllenInstitute/ConnectsCommonConnectivity"
DIR="$(cd "$(dirname "$0")" && pwd)"
BODIES="$DIR/issue_bodies"
MAP="$DIR/issue_number_map.txt"
DRY_RUN="${DRY_RUN:-0}"

# ---------------------------------------------------------------- milestones
MILESTONES=(
  "WP1 schema identity & scoping"
  "WP2 merge-scoped writes"
  "WP3 cell-cell connectivity"
  "WP4 wide+pointer+bulk writes"
  "WP5 delete/remove"
  "WP6 edit workflows"
  "WP7 cross-dataset reader"
  "WP8 QC schema"
  "WP9 spatial + embeddings"
  "WP10 test conventions & skill"
  "WP11 research memos"
  "WP12 ETL prompt & skill"
  "WP13 repo split"
)

# ------------------------------------------------- manifest: num|milestone|labels|title
# milestone "-" = none
MANIFEST=(
  "01|WP1 schema identity & scoping|bug,schema|CellFeatureMeasurement: feature_set_id and unit are never generated"
  "02|WP1 schema identity & scoping|bug,schema|HierarchyCategory: per-taxonomy vocabulary; fix level type"
  "03|WP1 schema identity & scoping|enhancement,schema|ProjectionMeasurementMatrix: add ProjectScoped"
  "04|WP1 schema identity & scoping|enhancement,schema|SingleCellReconstruction: add ProjectScoped"
  "05|WP1 schema identity & scoping|bug,schema|BrainRegionAssociation: no identity, unwritable as designed"
  "06|WP2 merge-scoped writes|enhancement,io|Add merge_scoped write mode with declared merge_on keys"
  "07|WP2 merge-scoped writes|enhancement,io|ClusterMembership: enforce merge keys in io, not schema"
  "08|WP2 merge-scoped writes|bug,io|Multi-writer data loss in shared scopes (patch-seq associations)"
  "09|WP2 merge-scoped writes|bug,io,good first issue|dry_run silently ignored when output_root is passed"
  "10|WP3 cell-cell connectivity|enhancement,schema|CellCellConnectivityLong: add connectome_id"
  "11|WP3 cell-cell connectivity|enhancement,io|Derive CellCellConnectivityLong from a synapse table"
  "12|WP4 wide+pointer+bulk writes|enhancement,io|Register SynapseConnectivityLong with a bulk validation mode"
  "13|WP4 wide+pointer+bulk writes|enhancement,io|wide_parquet mode + combined wide-matrix/pointer write API"
  "14|WP5 delete/remove|enhancement,io|Delete/remove operations for parquet/deltalake outputs"
  "15|WP6 edit workflows|enhancement,documentation,io|Post-write edit workflows: add columns / add dataitems"
  "16|WP7 cross-dataset reader|enhancement,io|Cross-dataset reads: union DataItems by shared features/clusters"
  "17|WP8 QC schema|enhancement,schema,question|QC flag schema: per-cell proofreading and modality QC"
  "18|WP9 spatial + embeddings|enhancement,schema|Promote SpatialLocation to a first-class coordinates table"
  "19|WP9 spatial + embeddings|question,schema|Embedding coordinates schema (UMAP/tSNE), distinct from spatial"
  "20|WP9 spatial + embeddings|bug,io|Nested models (SpatialLocation) don't round-trip through the arrow layer"
  "21|WP10 test conventions & skill|enhancement,documentation|Test-writing conventions + skill"
  "22|WP11 research memos|question,research,schema|AIT schema comparison + adapter/migration strategy"
  "23|WP11 research memos|question,research|BKP ingest schema comparison"
  "24|WP11 research memos|question,research|aind-data-schema: explore incorporation"
  "25|WP12 ETL prompt & skill|documentation|Refresh etl_example_prompt.md to package-first"
  "26|WP13 repo split|question|Repo split: options and partial migration path"
  "27|-|documentation|Update README"
  "28|-|question,schema|Should feature sets/definitions be project-scoped?"
)

# ---------------------------------------------------------------- issue bodies
# {{N}} = draft cross-reference, replaced with the real #number by `fixrefs`.
write_bodies() {
  mkdir -p "$BODIES"

cat > "$BODIES/01.md" <<'EOF'
**Problem:** `cell_features_schema.yaml` declares `feature_set_id` and `unit` for `CellFeatureMeasurement` under `slot_usage:` but not under `slots:`. LinkML generates neither, so `models.py` lacks both fields — the documented "denormalized reference to the feature set" doesn't exist.

**Fix:** declare both under `slots:`; regenerate models.

**Related:** whether feature sets are project-scoped is {{28}} — don't silently add the ProjectScoped mixin here.
EOF

cat > "$BODIES/02.md" <<'EOF'
**Problem:** category ids like `class`/`subclass`/`cluster` are global, but taxonomies don't share a level structure. Two taxonomies writing `class` collide — the last writer's `level`/`description` wins. Merge-scoped writes ({{6}}) don't fix this. Also `level` generates as `Optional[str]` while `Cluster.level` is `int`.

**Fix:** add a hierarchy/taxonomy discriminator slot so category vocabularies are per-taxonomy; declare `level: integer`.

**Context:** `Cluster` holds hierarchical categorical features generally, not only cell types — we don't want to force a common level structure.
EOF

cat > "$BODIES/03.md" <<'EOF'
**Problem:** not ProjectScoped; the write spec scopes on `id` only (registry comment says to widen once the schema allows). Same matrix id in two projects collides. Blocks `etl_wnm_exc_05`.

**Fix:** add the `ProjectScoped` mixin; widen `partition_by`/`scope_columns`; update `etl_wnm_exc_04`.
EOF

cat > "$BODIES/04.md" <<'EOF'
**Problem:** not ProjectScoped, but its `id` references a project-scoped DataItem — two projects reusing a cell id collide once this class becomes writable. Unlike `BrainRegion` (deliberately global, documented in the schema), there is no rationale for this omission. Blocks `etl_wnm_exc_05`.

**Fix:** add the `ProjectScoped` mixin.
EOF

cat > "$BODIES/05.md" <<'EOF'
**Problem:** this class records which brain region a DataItem is in (region-to-region projection lives in `ProjectionMeasurementMatrix`), but has only two Optional fields — no `id`, no `project_id`. It can't be validated, scoped, or written. Blocks `etl_wnm_exc_05`.

**Fix:** add `ProjectScoped`; require `brainregion_id`/`dataitem_id` at write time via the io layer (`required_for_write`, per the convention in {{7}}). Natural identity is `(project_id, dataitem_id, brainregion_id)` — no surrogate id needed once merge keys exist ({{6}}).

**Design note:** a cell may get region assignments from multiple methods (CCF registration vs manual) or term sets — consider a `method`/`source` slot.
EOF

cat > "$BODIES/06.md" <<'EOF'
**Problem:** only `overwrite_scoped` and `append_new_by_id` exist — no upsert. Incremental workflows (adding dataitems, revising feature values, QC arriving over time) silently lose data ({{8}}) or no-op.

**Fix:** add `write_mode="merge_scoped"` + `merge_on: list[str]` to `WriteSpec`; implement with delta-rs MERGE; dedupe within batch; assign modes per class per the table in `planning/20260820/2026-08-20_work_packages.md` §P4. Replaces `append_new_by_id` for DataItem (fixes silent no-op metadata updates).

**Scope:** merge is for identity-bearing metadata/association tables (10³–10⁶ rows). Bulk long tables and wide matrices stay `overwrite_scoped` — MERGE joins incoming vs existing rows, wasteful at 10⁷.

**Decide in the PR:** pure upsert (accumulate) vs sync-with-delete within a sub-scope. Pure upsert fits our workflows; deletion stays explicit ({{14}}).
EOF

cat > "$BODIES/07.md" <<'EOF'
**Problem:** `item` and `cluster` are Optional in the schema and there is no `id`; merge keys ({{6}}) must be non-null at write time.

**Fix:** add `item`, `cluster` to `required_for_write` (as already done for `hierarchy_id`); schema stays Optional.

**Convention (decided):** the schema is the standard (eventually its own repo); this io package is one wrapper around it. Constraints required only by *this* write method belong in the io layer, not the schema. Apply wherever merge keys land ({{5}} etc.).
EOF

cat > "$BODIES/08.md" <<'EOF'
**Problem:** notebooks sharing scope `(project_id, dataset_id)` silently delete each other's rows — associations shrank 2759 → 520 → 495 across `etl_visp_inh_patchseq_01/02/03`. Current workaround: inlined read-union-rewrite in the notebooks. Options analysis: `planning/multi_writer_scope_design.md`.

**Fix:** resolved by {{6}} with `merge_on (project_id, dataset_id, dataitem_id)`; remove the inlined workarounds in the same PR. This issue is the user-visible bug record.
EOF

cat > "$BODIES/09.md" <<'EOF'
**Problem:** `write_models(..., output_root=...)` resolves `settings=None`, so `dry_run=True` is silently ignored and data is written.

**Fix:** honor `dry_run`; add the missing test.
EOF

cat > "$BODIES/10.md" <<'EOF'
**Problem:** no measurement-context discriminator — rows from two measurement contexts are indistinguishable and their overwrite predicates clobber each other. Hence today's one-subdirectory-per-connectome workaround (`cellcellconnectivitylong_*`).

**Semantics:** `connectome_id` identifies the **measurement context** (segmentation version + proofreading state + measurement semantics), **not** the cohort. Cohorts are read-side filters: write the superset connectome once; pull subsets by joining `dataitem_dataset_association` or cluster membership. Minnie's two folders (`proofread_pre_to_csm_post`, `proofread_to_proofread`) are two measurement contexts → two `connectome_id`s in one table.

**Fix:** add `connectome_id`; register a WriteSpec (`overwrite_scoped` on `(project_id, connectome_id)`); rerun affected ETLs with the new io + schema (no folder migration); add a subset-read helper (ties to {{16}}). Needs its own planning session before implementation.
EOF

cat > "$BODIES/11.md" <<'EOF'
**Feature:** when a `SynapseConnectivityLong` table exists, cell-cell connectivity is derivable — group by `(presynaptic_cell, postsynaptic_cell)` and aggregate. Provide a function that populates `CellCellConnectivityLong` automatically.

**Decide:** aggregations — total synapse count always; total synapse size when a per-synapse size column exists (via the dataset's `SynapseFeatureMatrix`). Each aggregation gets its own `measurement_type` (`SynapticMeasurementType`).

**Depends on:** {{10}} (`connectome_id` on the output); pairs with {{12}}.
EOF

cat > "$BODIES/12.md" <<'EOF'
**Problem:** registry entry is commented out because `validate_for_write` re-validates every row with pydantic — unusable at 8M rows; `etl_v1dd_03` bypasses `write_models` with raw `write_deltalake`. A dead import remains in `write_spec.py`.

**Fix:** add a bulk mode (arrow-schema-level validation, optional row sampling); register `overwrite_scoped` on `(project_id, dataset_id)`; route the notebook through `write_models`; remove the dead import.
EOF

cat > "$BODIES/13.md" <<'EOF'
**Problem:** the `wide_parquet` mode designed in `ARCHITECTURE.md` was never built; all wide tables are raw `write_deltalake` calls in notebooks. Nothing enforces that pointer rows (`CellFeatureMatrix`, `SynapseFeatureMatrix`) and the parquet they point at are written together — dangling-pointer risk.

**Fix:** build `wide_parquet`; add a combined API that writes the payload (overwrite) then merges the pointer row, validating that index columns/paths agree.

**Depends on:** {{6}} (mode vocabulary).
EOF

cat > "$BODIES/14.md" <<'EOF'
**Problem:** no delete, vacuum, or partition removal exists; the only row removal is a predicated overwrite that happens to exclude rows. No way to drop a DataItem, retire a feature set/column, or delete a hierarchy.

**Fix:** explicit remove operations complementary to {{6}}, plus a vacuum/tombstone policy. Test-first development (use the conventions from {{21}}).
EOF

cat > "$BODIES/15.md" <<'EOF'
**Problem:** no library support for adding feature columns to an existing feature set or dataitems to an existing dataset; conventions live only in `etl_example_prompt.md` §5d/§9.

**Fix:** mostly falls out of {{6}} + {{13}}; remaining work is a thin API plus documented recipes. Removals → {{14}}.
EOF

cat > "$BODIES/16.md" <<'EOF'
**Problem:** `DatasetReader` is single-dataset; cohorts sharing feature sets or memberships need N calls + manual concat (`etl_v1dd_04_read`).

**Fix:** implement `planning/20260623/prompts/_deferred/08_readers.md` Layer 2; flagship `read_dataitems_for_clusters(cluster_ids, via=("membership","mapping"))`.

**Subtasks:** move `parquet_loader.py` → `io/parquet_loader.py` (pure move); subset-connectivity filter helper (see {{10}}, {{11}}).
EOF

cat > "$BODIES/17.md" <<'EOF'
**Problem:** no QC concept exists; proofread status is encoded as DataSet membership (`v1dd_1196_proofread_axons` etc.). Need per-cell flags like PROOFREAD_AXON / PROOFREAD_DENDRITE / PROOFREAD_CELL and equivalents for other modalities.

**Inclination (TBD):** per-modality QC vocabularies **plus** a cross-modality `acceptable` flag. Candidate shape: `QCFlag` association table `(dataitem_id, flag, value, source, timestamp, project_id)`, vocabularies possibly via the `ValueSet` draft (morriscb branch). Flags arrive incrementally — matches merge-scoped writes ({{6}}).

**First step:** survey QC vocabulary across modalities.
EOF

cat > "$BODIES/18.md" <<'EOF'
**Problem:** per-cell coordinates are smuggled through generic CellFeatureSets (`v1dd_soma_spatial`: `soma_voxel_x/y/z`, …) — untyped floats, no reference-space semantics. `SpatialLocation` (`x, y, z, reference_space`) exists in `core_schema.yaml` but is unreachable on disk: its only attachment points were never written by any ETL, it has no id/WriteSpec (designed as an embedded struct), and the write path has no struct support ({{20}}).

**Fix:** first-class table `(dataitem_id, reference_space, x, y, z [, project_id])`; `reference_space` distinguishes CCF-registered vs dataset-original vs dataset-corrected coordinates. Pilot: migrate `v1dd_soma_spatial`. Flattening to a table bypasses {{20}} (kept separate as the general bug).
EOF

cat > "$BODIES/19.md" <<'EOF'
**Question:** computed embeddings must not be conflated with anatomical coordinates (the BKP ingest template doesn't distinguish them; we should). `origin/u/morriscb/abcAtlasAccessDraftSchema` drafts `EmbeddingSet` + `EmbeddingMethod` (UMAP/TSNE/PCA/MDS/OTHER). Align with / adopt that draft? Coordinate with @morriscb.
EOF

cat > "$BODIES/20.md" <<'EOF'
**Problem:** the write path is pydantic → dicts → arrow → parquet. `flatten_refs` collapses a nested model to its id string — fine for references, but `SpatialLocation` has no id, so it survives as a raw dict while `build_arrow_schema` maps embedded-model fields to `string`. A dict then hits a string column: pyarrow errors, or stringifies into an unqueryable text blob. Model → parquet → model loses structure.

**Fix options:** (a) struct columns in `arrow_utils`; (b) explicit JSON-column convention; (c) don't nest — {{18}} flattens spatial and bypasses this. Keep this issue as the general record for any future nested model.
EOF

cat > "$BODIES/21.md" <<'EOF'
Establish testing conventions and encode them as a reusable skill: consistent error raising and warning patterns, descriptive docstrings, test-first scaffolding for new io features (first consumer: {{14}}).

**References:** `planning/test_suite_analysis_2026-08-15.md` (healthy 169-test pyramid; top gaps: CI enforcement, coverage measurement, direct `arrow_utils` tests, deeper parquet-loader cases, CLI behavioral tests) and `planning/20260623/tests_review/findings.md`. Gap items can become checkboxes here or follow-up issues.
EOF

cat > "$BODIES/22.md" <<'EOF'
**Task:** review the [AllenInstituteTaxonomy](https://github.com/AllenInstitute/AllenInstituteTaxonomy) schema and compare it to CCM's clustering schemas (`Cluster`, `ClusterHierarchy`, `ClusterMembership`, `AlgorithmRun`) — field-mapping memo.

**Context:** AIT is a spec (anndata h5ad + markdown/CSV schema; R tooling) — no releases, no pip package, actively evolving.

**Versioning approach:** pin a specific AIT commit as our de-facto version; maintain a migration script for switching pins; conformance test against one published taxonomy from their `taxonomies.md` breaks loudly on drift. Coordinate with the `hmba_taxonomy_annotation_schema.yaml` draft (@morriscb branch).
EOF

cat > "$BODIES/23.md" <<'EOF'
**Task:** compare CCM schemas to the [BKP explorer ingest template](https://github.com/AllenInstitute/BkpPlotGenerator/blob/main/docs/Ingest-Template/README.md#visualization-files) — similarities, differences, and use-case contrast (fast vis-app serving vs cross-modality commonality). CCM→BKP export is owned by another team; this is for understanding.
EOF

cat > "$BODIES/24.md" <<'EOF'
**Task:** explore how [aind-data-schema](https://aind-data-schema.readthedocs.io/en/latest/index.html) can be incorporated — a sidecar JSON next to CCM was requested by a scientist. Capture the concrete use case (which aind core files?), then assess a small exporter util (aind-data-schema is a versioned pip package, so an optional-extra exporter is cheap once the mapping is known).
EOF

cat > "$BODIES/25.md" <<'EOF'
**Problem:** the current prompt predates the write registry and reader.

**Fix:** teach `write_models` + merge modes + `DatasetReader` verification, without notebook-copying overhead; the future ETL skill builds on this.

**Blocked by:** {{6}}.
EOF

cat > "$BODIES/26.md" <<'EOF'
**Question:** Options A–D in `planning/20260820/2026-08-20_work_packages.md` §O1. On record: defer until write-path schema churn settles; Option B (schemas+package / viztoolbox / demos) is the natural first cut; migration can be partial.

- [ ] `seaborn` is a runtime dep in `pyproject.toml` but imported only by `code/` notebooks — move to a demos/viz optional-extra (`pip install ccc[demos]`) or dev group; moves out entirely when demos split off.
EOF

cat > "$BODIES/27.md" <<'EOF'
**Problem:** README predates the v1dd-ingest work and the write-mode changes.

**Fix:** document read API (`DatasetReader`, `read_synapse_table`), write modes (merge/overwrite/bulk/wide), config, and current schema coverage.

**Blocked by:** {{6}}.
EOF

cat > "$BODIES/28.md" <<'EOF'
**Question:** the long-term goal is common features across modalities/projects, so feature sets/definitions should ideally not be hard project-scoped — though most sets will in practice be project-specific. With merge-scoped writes ({{6}}) the write-side cost of `project_id` is ~zero (just part of a merge key, no clobber risk). Remaining concerns: (a) `dataitem_id` is only unique per project, so an unscoped measurement row is ambiguous in joins; (b) namespacing for genuinely shared sets — sentinel project (e.g. `project_id="common"`) vs optional `project_id` on `CellFeatureSet`/`CellFeatureDefinition`.

**Informs:** {{1}}, {{6}}.
EOF
}

# ---------------------------------------------------------------- phases
setup() {
  echo "== labels =="
  run gh label create schema   --repo "$REPO" --color 1D76DB --description "Schema (LinkML) changes" --force
  run gh label create io       --repo "$REPO" --color 0E8A16 --description "io package (read/write layer)" --force
  run gh label create research --repo "$REPO" --color 5319E7 --description "External alignment / research" --force

  echo "== milestones =="
  for t in "${MILESTONES[@]}"; do
    if [ "$DRY_RUN" = "1" ]; then
      echo "DRY: gh api repos/$REPO/milestones -f title=\"$t\""
    else
      gh api "repos/$REPO/milestones" -f title="$t" --silent 2>/dev/null \
        && echo "created: $t" || echo "exists (or error): $t"
    fi
  done
}

create() {
  if [ -f "$MAP" ] && [ "$DRY_RUN" != "1" ]; then
    echo "ERROR: $MAP already exists — issues were already created. Delete it only if you're sure." >&2
    exit 1
  fi
  write_bodies
  [ "$DRY_RUN" = "1" ] || : > "$MAP"

  for row in "${MANIFEST[@]}"; do
    IFS='|' read -r num milestone labels title <<< "$row"
    body="$BODIES/$num.md"
    args=(--repo "$REPO" --title "$title" --body-file "$body" --label "$labels")
    [ "$milestone" != "-" ] && args+=(--milestone "$milestone")

    if [ "$DRY_RUN" = "1" ]; then
      echo "DRY: gh issue create [$num] '$title' labels=[$labels] milestone=[$milestone]"
      continue
    fi
    url=$(gh issue create "${args[@]}")
    real="${url##*/}"
    echo "$((10#$num))=$real" >> "$MAP"
    echo "draft $num -> #$real  $title"
  done
  [ "$DRY_RUN" = "1" ] || echo "Mapping written to $MAP — now run: ./create_issues.sh fixrefs"
}

fixrefs() {
  [ -f "$MAP" ] || { echo "ERROR: $MAP not found — run 'create' first." >&2; exit 1; }
  tmp="$(mktemp -d)"
  for row in "${MANIFEST[@]}"; do
    IFS='|' read -r num _ _ title <<< "$row"
    body="$BODIES/$num.md"
    grep -q '{{' "$body" || continue
    fixed="$tmp/$num.md"
    cp "$body" "$fixed"
    while IFS='=' read -r draft real; do
      sed -i.bak "s/{{$draft}}/#$real/g" "$fixed" && rm -f "$fixed.bak"
    done < "$MAP"
    if grep -q '{{' "$fixed"; then
      echo "WARNING: unresolved refs remain in $num — skipping edit"; continue
    fi
    real_num=$(grep "^$((10#$num))=" "$MAP" | cut -d= -f2)
    if [ "$DRY_RUN" = "1" ]; then
      echo "DRY: gh issue edit $real_num --body-file $fixed  ($title)"
    else
      gh issue edit "$real_num" --repo "$REPO" --body-file "$fixed" >/dev/null
      echo "fixed refs in #$real_num  $title"
    fi
  done
  echo "Done. Draft->real mapping is in $MAP."
}

run() { if [ "$DRY_RUN" = "1" ]; then echo "DRY: $*"; else "$@"; fi; }

case "${1:-}" in
  setup)   setup ;;
  create)  create ;;
  fixrefs) fixrefs ;;
  *) sed -n '2,15p' "$0"; exit 1 ;;
esac
