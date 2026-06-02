# IO Layer — TODO

Ordered, dependency-aware. Design lives in `ARCHITECTURE.md`; the implementing prompt for
each item is in `prompts/`. Hard rules: see `prompts/00_shared_context.md`.

## Phase 0 — groundwork
- [ ] **0.1 Config module** (`config.py`, package root — not `io/`) — pydantic `Settings` loaded from a discovered
  `ccc_config.yaml` (walk-up like `pyproject.toml`), cached `get_settings()`, `table_path()`
  helper. Precedence: explicit arg > `CCC_OUTPUT_ROOT` env (developer escape hatch, path
  only) > `ccc_config.yaml` > **error, no default path**. No `configure()` global, no `%run`.
  Deps: pydantic + PyYAML (already present). Prompt: `01_config.md`. Blocks everything that writes.

## Phase 1 — registry (the hub)
- [ ] **1.1 Write spec registry** (`io/write_spec.py`) — one entry per writable class:
  `subdir`, `partition_by`, `scope_columns`, `write_mode`, `required_for_write`,
  `cross_field_rules`. Seed DataSet/DataItem/Association first, then the rest.
  Prompt: `02_write_spec.md`. Blocked by: none (reads generated models).
- [ ] **1.2 Registry↔schema drift test** — assert every entry's class + scope/identifier
  slots exist in `models.py`. Part of `02_write_spec.md`.

## Phase 2 — validation (structural only)
- [ ] **2.1 Strict submodel derivation** (`io/write_validation.py`) — `strict_model_for(cls)`
  flips `required_for_write` to required + attaches *pure* `cross_field_rules`. No I/O, no
  reading other tables. Auto-derived from generated models + registry. Prompt:
  `03_validation.md`. Blocked by 1.1. (Referential checks are 4b.2, not here.)

## Phase 3 — writers
- [ ] **3.0 Relocate backends into `io/`** — move `arrow_utils.py`→`io/arrow_utils.py`,
  `write_utils.py`→`io/write_utils.py`, with re-export shims at old paths. Part of
  `04_writers.md`.
- [ ] **3.1 Write dispatch core** (`io/writers.py`) — `write_models(models, settings=...)`:
  infer class → registry lookup → strict-validate → arrow convert → metadata → write per
  `write_mode`. Reuses `io/arrow_utils.py` + `io/write_utils.py`. Prompt: `04_writers.md`.
  Blocked by 0.1, 1.1, 2.1.
- [ ] **3.2 Typed wrappers** — generated from the registry (not hand-maintained); hand-write
  only non-uniform signatures (e.g. `write_projection_matrix`). Part of `04_writers.md`.
- [ ] **3.3 Reconcile `write_utils.py`** — make `append_new_dataitems` the
  `append_new_by_id` backend without breaking current callers. Part of `04_writers.md`.
- [ ] **3.4 Write-side transform** — `populate_region_coverage` as a section in
  `writers.py` (pre-write enrichment of ProjectionMeasurementMatrix). Part of `04_writers.md`.

## Phase 3b — public API
- [ ] **3b.1 `io/__init__.py`** — curated exports, module docstring, `__all__`. Defines what
  users type after `from connects_common_connectivity.io import …`. Prompt: `09_public_api.md`.
  Blocked by 3.1.

## Phase 4 — readers
- [ ] **4.0 Fold `parquet_loader.py` into `io/readers.py`** (re-export shim at old path).
  Part of `05_readers.md`.
- [ ] **4.1 Predicate-based readers** (`io/readers.py`) — `read_dataset`, `read_dataitem`,
  `read_features` scoped by project/dataset. Prompt: `05_readers.md`. Blocked by 1.1.
- [ ] **4.2 Cross-dataset reads** — flagship: DataItems with ClusterMembership OR
  CellToClusterMapping to a given cluster set. Part of `05_readers.md`.

## Phase 4b — analysis & referential checks (need readers)
- [ ] **4b.1 Read-side analysis** — `compare_region_coverage` as a section in `readers.py`
  (read-side overlap summary). Prompt: `08_analysis.md`. Blocked by 4.1.
- [ ] **4b.2 Opt-in referential check** — `write_models(..., check_refs=True)` verifies an
  association's `dataset_id` exists among written DataSets. Uses readers; off the hot path.
  Part of `08_analysis.md`. Blocked by 4.1.

## Phase 5 — notebook migration
- [ ] **5.0 Create `ccc_config.yaml`** at the repo root — the single, version-controlled
  source of truth for `output_root` (+ `dry_run`). Part of `06_notebook_migration.md`.
- [ ] **5.1 Migrate `_01_dataset_dataitem` notebooks** — delete hardcoded `OUTPUT_ROOT`
  (no config cell; library discovers `ccc_config.yaml`) + typed writers; fixes the patchseq
  DataSet overwrite. Prompt: `06_notebook_migration.md`. Blocked by 3.x.
- [ ] **5.2 Migrate feature / cluster / mapping / projection notebooks.** Same prompt.
- [ ] **5.3 Patchseq regression check** — re-run exc then inh; assert both DataSet rows
  coexist. Acceptance test for the migration.
- [ ] **5.4 Remove re-export shims** — delete shims at `arrow_utils.py`, `write_utils.py`,
  `parquet_loader.py` once no notebook/test imports them. Add a test asserting no old import
  path is referenced anywhere. Blocked by 5.1–5.2.

## Phase 6 — tests & docs
- [ ] **6.1 Test suite** — see `07_tests.md`. Pulls together the cases already specified in
  `02` (drift), `04`/`06` (regression) rather than re-specifying them.
- [ ] **6.2 Update README / usage docs** for the new IO API. (Ask before large edits.)

## Decisions locked (2026-06-01)
- Config: declarative `ccc_config.yaml` at repo root, discovered by walk-up and validated by
  pydantic; no per-notebook setup, no `%run`, no global. Precedence: explicit arg > env
  (escape hatch) > `ccc_config.yaml` > **error (no default path)**. Deps already present.
- Write spec: explicit registry, source of truth, schema-checked for drift.
- Validation: auto-derived strict submodels; hot path is structural-only, no I/O.
- Packaging: seed transforms/analysis inside writers/readers; split out on second function.
  Public surface is `io/__init__.py`. Scope of this session: planning docs + prompts only.
