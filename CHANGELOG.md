# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `WriteSpec` registry entries for `AlgorithmRun` and
  `HierarchyCategory` (both project-agnostic, scope=`["id"]`,
  `overwrite_scoped`). These classes are now writable through
  `write_models(...)` and surface in `WRITABLE_CLASSES`.
- Added write-time validation: `write_models()` now re-validates each
  model through a runtime-derived strict subclass that flips
  `WriteSpec.required_for_write` slots to non-optional, raising
  `ValueError` before any IO if a write-required slot is missing or
  `None`. Public helpers `strict_model_for()` and `validate_for_write()`
  live in `connects_common_connectivity.io.write_validation`.
- Added curated public API at `connects_common_connectivity.io`: imports
  for `get_settings`, `Settings`, `table_path`, `write_models`,
  `write_projection_matrix`, `WriteResult`, and `WRITABLE_CLASSES` are
  now stable and pinned by `__all__`.
- Added `connects_common_connectivity.io.writers` with `write_models()` (the
  single dispatch core for all generated pydantic models),
  `write_projection_matrix()`, `WriteResult`, and `WRITABLE_CLASSES`.
- Added `populate_region_coverage()` in
  `connects_common_connectivity.io.write_utils` for deriving
  `ProjectionMeasurementMatrix.region_coverage` from a dense matrix.

### Changed

- Migrated `code/etl_*.ipynb` notebooks to the curated IO API:
  hardcoded `OUTPUT_ROOT = "../scratch/..."` strings are replaced with
  `output_root()` from `connects_common_connectivity.config`, and
  hand-rolled `write_deltalake(..., mode=..., predicate=..., partition_by=...)`
  calls for registry-backed models are replaced with `write_models(...)`
  (and `write_projection_matrix(...)` for projection matrices).
- Moved `arrow_utils` and `write_utils` under
  `connects_common_connectivity.io.*`.

### Deprecated

### Removed

- Removed the deprecated re-export shims
  `connects_common_connectivity.arrow_utils` and
  `connects_common_connectivity.write_utils`. Import from
  `connects_common_connectivity.io.arrow_utils` /
  `connects_common_connectivity.io.write_utils` instead.

### Fixed

### Security
