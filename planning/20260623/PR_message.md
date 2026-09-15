# IO layer: write path + validation

Ships the curated `connects_common_connectivity.io` write path end-to-end: package-wide configuration, a registry-driven write API, write-time validation derived from that same registry, ETL notebook migration to the new API, and the test suite to back it.

## Design: WriteSpec as the single source of truth

The `WriteSpec` registered per writable class is one declaration that drives both Delta dispatch (subdir, partitioning, scope columns, write mode) and write-time validation (`required_for_write` slots are flipped non-optional in auto-derived strict submodels and re-validated before any IO). Generated `models.py` is never touched.

## Configuration

- New `connects_common_connectivity.config`: pydantic `Settings`, cached `get_settings()`, walk-up discovery of `ccc_config.yaml`, plus `output_root()` / `table_path()` helpers. Relative values anchor at the config file's directory via `os.path.abspath` (avoids Code Ocean's `scratch -> /scratch` symlink).
- Precedence: explicit arg > `CCC_OUTPUT_ROOT` env > `ccc_config.yaml` > error.
- Repo-root `ccc_config.yaml` seeded.

## Write registry and dispatch

- `io/write_spec.py`: `WriteSpec`, `REGISTRY` (14 entries), `get_spec()`.
- `io/writers.py`: `write_models()` single-dispatch over the registry (no per-class wrappers), frozen `WriteResult` dataclass, `WRITABLE_CLASSES` tuple. `write_projection_matrix()` is the only non-`write_models` writer, justified by its non-uniform signature (dense matrix + model).
- `populate_region_coverage()` added in `io/write_utils.py`; derives `region_coverage` from the dense values before write.
- `DataSet` scope widened to `(project_id, id)` so patchseq exc/inh `DataSet` rows coexist (today's predicate-only-on-`project_id` behavior would overwrite one with the other).

## Write-time validation

- `io/write_validation.py`: `strict_model_for(cls)` flips `WriteSpec.required_for_write` slots to non-optional and strips `Optional` from those annotations (cached per class, no mutation of generated `models.py`). `validate_for_write()` re-validates instances and raises `ValueError` naming the missing slots before any IO. Wired into `write_models`.
- `required_for_write` populated for `Cluster`, `ClusterMembership`, `CellFeatureDefinition`.

## Public API surface

- Curated `io/__init__.py` re-exports pinned by `__all__`: `get_settings`, `Settings`, `table_path`, `write_models`, `write_projection_matrix`, `WriteResult`, `WRITABLE_CLASSES`.
- Per-call `output_root=` keyword on `write_models()` / `write_projection_matrix()` (mutually exclusive with `settings=`) so a single notebook can redirect its writes without mutating process-global config.
- `Modality.CALCIUM_IMAGING` added (for functional correlations in microns or v1dd-like datasets with EM + CI experiments).
- Removed `connects_common_connectivity.arrow_utils` / `connects_common_connectivity.write_utils` re-export shims; `arrow_utils.py` and `write_utils.py` now live exclusively under `io/`.

## ETL notebook migration

- Every registry-backed class is now exclusively written through `write_models` / `write_projection_matrix` in the ETL notebooks. Hand-rolled `write_deltalake` migrated. Per-notebook imports trimmed.
- Hardcoded `OUTPUT_ROOT = "../scratch/..."` strings replaced with `output_root()`.
- Patchseq exc/inh regression covered (see `DataSet` scope fix above).

## Tests

- Shared `tests/conftest.py` foundations (settings/cache/cwd isolation + shared fixtures); duplicated helpers removed.
- Tightened exception assertions to specific classes with meaningful `match=` checks.
- High-signal regression assertion messages where failures are otherwise hard to diagnose; list-validation failures now include row context.
- Per-class smoke parametrized over `WRITABLE_CLASSES`; registry-drift guard; no-shim regression (`test_shim_modules_deleted`, `_not_importable`, `_no_source_references_shim_paths`).
- Closed coverage gaps: CLI behavior, parquet loader contract, predicate escaping edge cases, relocation scan roots, dry-run semantics.
- Patchseq regression, idempotency, append-new-by-id, predicate construction, `output_root=` override, strict-validation failures, public-API surface.

## Not in this PR

- Wide cell-feature / projection-matrix parquet writes (still use `write_deltalake` directly).
- `CellCellConnectivityLong` — no registry entry yet; the `write_cellcellconnectivitylong` stub in `io/writers.py` documents the migration plan.
- The `etl_v1dd_01` new dataset ingestion prototype ongoing in parallel.
- A `merge_by_id` (read-existing → union → overwrite) write mode for shared scopes like `(visp_patchseq, visp_inh_patchseq)` where multiple notebooks contribute disjoint subsets. The union is currently inlined in patch-seq / WNM notebooks; see `planning/multi_writer_scope_design.md` for the draft design discussion.

## Verification

`uv run pytest -q` → 160 passed.
