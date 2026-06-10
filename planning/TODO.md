# IO Layer — TODO

Flat, ordered list. One row per prompt; sub-tasks live in the prompts. Design lives in
`ARCHITECTURE.md`. Hard rules: see `prompts/00_shared_context.md`.

**Priority for this round: W1 → W7. Readers and analysis are deferred.**

## This round (write path → migration → tests)

- [x] **W1 — Config** (`prompts/01_config.md`) — `config.py` at the package root, pydantic
  `Settings` loaded from a discovered `ccc_config.yaml` (walk-up like `pyproject.toml`),
  cached `get_settings()`, `table_path()` helper, plus `output_root()` convenience that
  returns the path relative to cwd (notebooks in `code/` see `../scratch/...`). Relative
  values in the file are anchored at the config file's directory using `os.path.abspath`
  (not `Path.resolve`, so Code Ocean's `scratch -> /scratch` symlink isn't followed).
  Precedence: explicit arg > `CCC_OUTPUT_ROOT` env > `ccc_config.yaml` > error. No
  `configure()` global, no `%run`. Re-exported from `io/__init__.py`.
  `ccc_config.yaml` seeded at repo root with `output_root: scratch/em_patchseq_wnm_v1/`.
  Tests: `tests/test_config.py` (14 tests, all passing).
- [ ] **W2 — Write spec registry (seed only)** (`prompts/02_write_spec.md`) —
  `io/write_spec.py`: `WriteSpec` pydantic model, `REGISTRY` seeded with **exactly three**
  entries (`DataSet`, `DataItem`, `DataItemDataSetAssociation`), `get_spec()` lookup, and
  the drift test (`tests/test_write_spec.py`). `required_for_write` and
  `cross_field_rules` left empty — W5 owns those. The remaining classes are W3's job.
- [ ] **W3 — Writers + relocation + registry expansion** (`prompts/03_writers.md`) —
  Move `arrow_utils.py`/`write_utils.py` into `io/` (re-export shims at old paths, removed
  in W6). Build `io/writers.py`: `write_models()` dispatch, `WriteResult` frozen dataclass,
  `WRITABLE_CLASSES` discovery tuple, pass-through `_validation_hook` for W5 to swap, plus
  `write_projection_matrix()` (the one non-`write_models` public writer, justified by its
  non-uniform signature). **No per-class wrappers** — `write_models` infers the class.
  Land `populate_region_coverage` in `io/write_utils.py`. **Expand the registry** by
  prototyping each remaining writable class one at a time (notebook write → registry entry
  → smoke test). `CellFeatureMatrix` stays in the registry under `wide_parquet` mode.
  Blocked by W1, W2.
- [ ] **W4 — Public API** (`prompts/04_public_api.md`) — `io/__init__.py`: curated
  re-exports + `__all__`. The user-facing surface; defines what autocomplete shows. Blocked
  by W3.
- [ ] **W5 — Write validation** (`prompts/05_validation.md`) — `io/write_validation.py`:
  `strict_model_for(cls)` flips `required_for_write` to required + attaches pure
  `cross_field_rules` (no I/O). Swap `validate_for_write` into the W3 hook. Blocked by W2, W3.
- [ ] **W6 — Notebook migration** (`prompts/06_notebook_migration.md`) — Migrate every
  ETL notebook to typed writers; delete hardcoded `OUTPUT_ROOT` and per-cell
  `mode`/`predicate`/`partition_by` (`ccc_config.yaml` already exists from W1). Run the
  patchseq regression (exc then inh, both DataSet rows must coexist). Remove the W3
  re-export shims and confirm nothing imports the old paths. Blocked by W3 (W5 preferred).
- [ ] **W7 — Write-side test suite** (`prompts/07_tests.md`) — Drift, patchseq regression,
  idempotency, append-new-by-id, predicate construction, per-class example smoke, no-shim
  regression, public-API surface. Owns only the gaps not specified by W2/W3/W4/W5.
- [ ] **W8 — README / usage docs** — Update README for the write API. No prompt; small task.
  Ask before large edits.

## Deferred (do not start; design kept for reference)

Designs live in `ARCHITECTURE.md` and `prompts/_deferred/`. Pick up only after W1–W7 land.

- **L1 — Readers** (`prompts/_deferred/08_readers.md`) — `io/readers.py` (predicate-based +
  cross-dataset). `parquet_loader.py` is **moved** to `io/parquet_loader.py` (pure move,
  not folded) when this starts.
- **L2 — Read-side analysis + opt-in referential check**
  (`prompts/_deferred/09_analysis.md`) — `compare_region_coverage` and
  `write_models(..., check_refs=True)`.

## Decisions locked (2026-06-01)
- Config: declarative `ccc_config.yaml` at repo root, discovered by walk-up, validated by
  pydantic; no per-notebook setup, no `%run`, no global. Precedence: explicit arg > env
  (escape hatch) > file > error (no default path). `config.py` at package root, not `io/`.
- Write spec: explicit registry, prototyped per class via notebook examples; `write_mode` is
  an open vocabulary, not a forced overwrite assumption.
- `populate_region_coverage` lives in `write_utils.py` (write plumbing), not a transforms module.
- Validation: built after the write path; auto-derived strict submodels; structural-only, no I/O.
  Named `io/write_validation.py` (cli owns the generic LinkML conformance check).
- Readers, analysis, referential check: deferred. `parquet_loader.py` is a pure move, not a fold.
- Public surface is `io/__init__.py`. Scope of this session: planning docs + prompts only.
