# IO Layer — TODO

Ordered, dependency-aware. See `ARCHITECTURE.md` for design and `prompts/` for the
agent prompt that implements each item. Hard rules: never edit `models.py`; never edit
`schemas/*.yaml` without explicit permission from YY.

## Phase 0 — groundwork
- [ ] **0.1 Config module** (`io/config.py`) — plain pydantic `Settings` with
  `output_root` default + `CCC_OUTPUT_ROOT` env override + `table_path()` helper.
  No new dependency. Prompt: `prompts/01_config.md`. Blocks everything that writes.

## Phase 1 — registry (the hub)
- [ ] **1.1 Write spec registry** (`io/write_spec.py`) — one entry per writable class:
  `subdir`, `partition_by`, `scope_columns`, `write_mode`, `required_for_write`,
  `cross_field_rules`. Seed DataSet/DataItem/Association first, then the rest.
  Prompt: `prompts/02_write_spec.md`. Blocked by: none (reads generated models).
- [ ] **1.2 Registry↔schema drift test** — assert every entry's class + scope/identifier
  slots exist in `models.py`. Part of `prompts/02_write_spec.md`.

## Phase 2 — validation
- [ ] **2.1 Strict submodel derivation** (`io/validation.py`) — `strict_model_for(cls)`
  flips `required_for_write` to required + attaches `cross_field_rules`. Auto-derived
  from generated models + registry. Prompt: `prompts/03_validation.md`. Blocked by 1.1.

## Phase 3 — writers
- [ ] **3.1 Write dispatch core** (`io/writers.py`) — `write_models(models, settings=...)`:
  infer class → registry lookup → strict-validate → arrow convert → metadata → write per
  `write_mode`. Reuses `arrow_utils` + `write_utils`. Prompt: `prompts/04_writers.md`.
  Blocked by 0.1, 1.1, 2.1.
- [ ] **3.2 Typed wrappers** — `write_dataset`, `write_dataitem`, `write_association`,
  `write_features`, `write_cluster`, `write_cluster_membership`,
  `write_cell_to_cluster_mapping`, `write_projection_matrix`. Part of `prompts/04_writers.md`.
- [ ] **3.3 Reconcile `write_utils.py`** — make `append_new_dataitems` the
  `append_new_by_id` backend without breaking current callers. Part of `prompts/04_writers.md`.

## Phase 4 — readers
- [ ] **4.1 Predicate-based readers** (`io/readers.py`) — `read_dataset`, `read_dataitem`,
  `read_features` scoped by project/dataset. Prompt: `prompts/05_readers.md`. Blocked by 1.1.
- [ ] **4.2 Cross-dataset reads** — flagship: DataItems with ClusterMembership OR
  CellToClusterMapping to a given cluster set. Part of `prompts/05_readers.md`.
- [ ] **4.3 Fold in analysis utils** — `populate_region_coverage`,
  `compare_region_coverage` from `io/io_plans.md`. Part of `prompts/05_readers.md`.

## Phase 5 — notebook migration
- [ ] **5.1 Migrate `_01_dataset_dataitem` notebooks** — Settings + typed writers; fixes
  the patchseq DataSet overwrite. Prompt: `prompts/06_notebook_migration.md`. Blocked by 3.x.
- [ ] **5.2 Migrate feature / cluster / mapping / projection notebooks.** Same prompt.
- [ ] **5.3 Patchseq regression check** — re-run exc then inh; assert both DataSet rows
  coexist. Acceptance test for the migration.

## Phase 6 — tests & docs
- [ ] **6.1 Test suite** — idempotency, shared-partition safety (patchseq regression),
  strict-validator failures, round-trip. Prompt: `prompts/07_tests.md`.
- [ ] **6.2 Update README / usage docs** for the new IO API. (Ask before large edits.)

## Decisions locked (2026-06-01)
- Config: plain pydantic, version-controlled default + env override, no new dep.
- Write spec: explicit registry, source of truth, schema-checked for drift.
- Validation: auto-derived strict submodels from generated models + registry.
- Scope: this session produced planning docs + prompts only.
