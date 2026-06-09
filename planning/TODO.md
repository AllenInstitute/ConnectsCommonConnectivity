# IO Layer — TODO

Ordered, dependency-aware. Design lives in `ARCHITECTURE.md`; the implementing prompt for
each item is in `prompts/`. Hard rules: see `prompts/00_shared_context.md`.

**Priority for this round: config → write IO → validation → notebook migration.**
Readers and analysis are deferred — see "Later — elaborations" (not actionable yet).

## Phase 0 — config
- [ ] **0.1 Config module** (`config.py`, package root — not `io/`) — pydantic `Settings`
  loaded from a discovered `ccc_config.yaml` (walk-up like `pyproject.toml`), cached
  `get_settings()`, `table_path()` helper. Precedence: explicit arg > `CCC_OUTPUT_ROOT` env
  (developer escape hatch, path only) > `ccc_config.yaml` > **error, no default path**. No
  `configure()` global, no `%run`. Deps already present. Prompt: `01_config.md`. Blocks writes.

## Phase 1 — write IO (prototype per class)
Approach this like prototyping. **Do not assume every class is scoped-overwrite-with-
predicate.** For each writable class, add a small real write example to a notebook, see how
it actually wants to be written, and let that set its registry entry. `append_new_by_id`
already exists for DataItem; other classes may want append or modes not yet named.

- [ ] **1.1 Write spec registry** (`io/write_spec.py`) — one entry per class: `subdir`,
  `partition_by`, `scope_columns`, `write_mode` (open `Literal`, extend as prototyping
  surfaces new modes), `required_for_write`, `cross_field_rules`. Seed
  DataSet/DataItem/Association; add others as their examples are built. Prompt: `02_write_spec.md`.
- [ ] **1.2 Registry↔schema drift test** — class + scope/identifier slots exist in `models.py`.
- [ ] **1.3 Relocate write backends into `io/`** — `arrow_utils.py`→`io/arrow_utils.py`,
  `write_utils.py`→`io/write_utils.py` (re-export shims). `populate_region_coverage` lands in
  `write_utils.py` (the projection writer calls it), NOT a transforms module. Part of `03_writers.md`.
- [ ] **1.4 Write dispatch core** (`io/writers.py`) — `write_models` + registry-generated
  typed wrappers; dispatch on `write_mode`. Include a **pass-through validation hook** so
  Phase 2 can slot the real validator in without restructuring. Prompt: `03_writers.md`.
  Blocked by 0.1, 1.1.
- [ ] **1.5 Per-class write examples in notebooks** — the prototyping evidence that informs
  1.1; one small example per writable class. Part of `02_write_spec.md` / `03_writers.md`.

## Phase 2 — validation (after write works)
- [ ] **2.1 Strict submodel derivation** (`io/write_validation.py`) — `strict_model_for(cls)`
  flips `required_for_write` to required + attaches *pure* `cross_field_rules` (no I/O); wire
  `validate_for_write` into `write_models`, replacing the pass-through hook. Prompt:
  `05_validation.md`. Blocked by 1.1, 1.4. (Referential checks deferred with readers.)

## Phase 3 — notebook migration
- [ ] **3.0 Create `ccc_config.yaml`** at repo root — single source of truth for
  `output_root` (+ `dry_run`). Part of `06_notebook_migration.md`.
- [ ] **3.1 Migrate `_01_dataset_dataitem` notebooks** — delete hardcoded `OUTPUT_ROOT` (no
  config cell; library discovers `ccc_config.yaml`) + typed writers; fixes the patchseq
  DataSet overwrite. Prompt: `06_notebook_migration.md`. Blocked by 1.x (2.x preferred).
- [ ] **3.2 Migrate feature / cluster / mapping / projection notebooks.** Same prompt.
- [ ] **3.3 Patchseq regression check** — re-run exc then inh; assert both DataSet rows coexist.
- [ ] **3.4 Remove write-side re-export shims** — delete shims at `arrow_utils.py`,
  `write_utils.py` once no notebook/test imports them; test asserts no old path is referenced.
  Blocked by 3.1–3.2.

## Phase 4 — write-side tests & docs
- [ ] **4.1 Write-side test suite** — drift, patchseq shared-partition regression, idempotency,
  append-new-by-id, predicate construction, per-class example smoke. Prompt: `07_tests.md`.
- [ ] **4.2 Update README / usage docs** for the write API. (Ask before large edits.)

## Later — elaborations (NOT actionable yet)
Deferred until the write path is done and notebooks migrated. Designs kept in `ARCHITECTURE.md`
and prompts `08_readers.md` / `09_analysis.md` for reference; do not start these now.
- **Readers** (`io/readers.py`) — predicate-based + cross-dataset reads. `parquet_loader.py`
  is **moved** to `io/parquet_loader.py` (pure move, NOT folded) when this starts.
- **Read-side analysis** — `compare_region_coverage`.
- **Opt-in referential check** — `write_models(..., check_refs=True)`; needs readers.

## Decisions locked (2026-06-01)
- Config: declarative `ccc_config.yaml` at repo root, discovered by walk-up, validated by
  pydantic; no per-notebook setup, no `%run`, no global. Precedence: explicit arg > env
  (escape hatch) > file > **error (no default path)**. `config.py` at package root, not `io/`.
- Write spec: explicit registry, prototyped per class via notebook examples; `write_mode` is
  an open vocabulary, not a forced overwrite assumption.
- `populate_region_coverage` lives in `write_utils.py` (write plumbing), not a transforms module.
- Validation: built after the write path; auto-derived strict submodels; structural-only, no I/O.
  Named `io/write_validation.py` (cli owns the generic LinkML conformance check).
- Readers, analysis, referential check: deferred. `parquet_loader.py` is a pure move, not a fold.
- Public surface is `io/__init__.py`. Scope of this session: planning docs + prompts only.
