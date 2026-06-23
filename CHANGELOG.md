# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [0.2.0] - 2026-06-23

### Added

- Added `connects_common_connectivity.config` with `Settings`,
  `get_settings()`, `find_config_file()`, `output_root()`, and
  `table_path()`. Settings are discovered from a `ccc_config.yaml` at (or
  above) the cwd; `CCC_OUTPUT_ROOT` overrides `output_root`. Relative
  `output_root` values are anchored at the config file's directory so a
  notebook in `code/` and a script at the repo root resolve to the same
  place.
- Added curated public API at `connects_common_connectivity.io`:
  `write_models()` (single dispatch core for all generated pydantic
  models), `write_projection_matrix()`, `WriteResult`,
  `WRITABLE_CLASSES`, and re-exports of `get_settings`, `Settings`, and
  `table_path`. The surface is pinned by `__all__`.
- Added write-time validation: `write_models()` now re-validates each
  model through a runtime-derived strict subclass that flips
  `WriteSpec.required_for_write` slots to non-optional, raising
  `ValueError` before any IO if a write-required slot is missing or
  `None`. Public helpers `strict_model_for()` and `validate_for_write()`
  live in `connects_common_connectivity.io.write_validation`.
- Added `WriteSpec` registry entries for `AlgorithmRun` and
  `HierarchyCategory` (both project-agnostic, scope=`["id"]`,
  `overwrite_scoped`). These classes are now writable through
  `write_models(...)` and surface in `WRITABLE_CLASSES`.
- Added an `output_root=` keyword to `write_models()` and
  `write_projection_matrix()` for per-call overrides of the on-disk root.
  Accepts a `str` or `Path` and writes to `<output_root>/<spec.subdir>/`,
  bypassing `ccc_config.yaml` for that call. Mutually exclusive with
  `settings=` (passing both raises `TypeError`). Lets a single notebook
  redirect its writes (e.g. an isolated test dataset) without mutating
  process-global config or environment variables.
- Added `populate_region_coverage()` in
  `connects_common_connectivity.io.write_utils` for deriving
  `ProjectionMeasurementMatrix.region_coverage` from a dense matrix.
- Added `CALCIUM_IMAGING` value to the `Modality` enum for calcium
  imaging based functional correlations.

### Changed

- Migrated `code/etl_*.ipynb` notebooks to the curated IO API:
  hardcoded `OUTPUT_ROOT = "../scratch/..."` strings are replaced with
  `output_root()` from `connects_common_connectivity.config`, and
  hand-rolled `write_deltalake(..., mode=..., predicate=..., partition_by=...)`
  calls for every registry-backed model are replaced with `write_models(...)`
  (and `write_projection_matrix(...)` for projection-matrix metadata rows).
  Wide cell-feature / projection-matrix parquets and `CellCellConnectivityLong`
  writes remain on raw `write_deltalake` pending registry support.
- Moved `connects_common_connectivity.arrow_utils` and
  `connects_common_connectivity.write_utils` under
  `connects_common_connectivity.io.*`.

### Removed

- Removed the deprecated re-export shims
  `connects_common_connectivity.arrow_utils` and
  `connects_common_connectivity.write_utils`. Import from
  `connects_common_connectivity.io.arrow_utils` /
  `connects_common_connectivity.io.write_utils` instead.

### Fixed

- Fixed `DataSet` writes to scope on `(project_id, id)` instead of
  `project_id` alone, so sibling notebooks sharing a `project_id` (e.g.
  patchseq exc/inh) no longer overwrite each other's `DataSet` rows.
- Fixed `write_models()` to honor `Settings.dry_run=True`: writes are now
  skipped, `rows_written` is reported as `0`, and no Delta table
  directories are created.
