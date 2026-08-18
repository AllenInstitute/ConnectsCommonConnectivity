# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `connects_common_connectivity.config` with `Settings`,
  `get_settings()`, and `find_config_file()`. Settings are discovered from
  `ccc_config.yaml` at or above the current working directory;
  `CCC_OUTPUT_ROOT` overrides the configured output root, and relative paths
  are anchored to the config file's directory.
- Added the curated `connects_common_connectivity.io` API with
  `write_models()`, `write_projection_matrix()`, `WrittenResult`,
  `WRITABLE_CLASSES`, `Settings`, and `get_settings()`. `write_models()`
  accepts one Pydantic model or a non-empty homogeneous iterable, including a
  one-shot generator, and dispatches writes through the model's registered
  `WriteSpec`.
- Added `settings=` and `output_root=` controls to `write_models()` and
  `write_projection_matrix()`. An explicit output root bypasses config
  discovery, while `Settings.dry_run=True` validates a write without creating
  a Delta table and reports zero rows written.
- Added write-time validation helpers in
  `connects_common_connectivity.io.write_validation`: `strict_model_for(spec)`
  derives validation rules from the supplied `WriteSpec`, and
  `validate_for_write(models, spec)` validates every member of a non-empty,
  exact-type model sequence while returning the original instances in a new
  list.
- Added `WriteSpec` registry entries for `AlgorithmRun` and
  `HierarchyCategory`, making both classes writable through `write_models()`
  and discoverable through `WRITABLE_CLASSES`.
- Added `populate_region_coverage()` in
  `connects_common_connectivity.io.write_utils` to return a
  `ProjectionMeasurementMatrix` copy whose region coverage is derived from a
  two-dimensional NumPy-compatible array.
- Added `CALCIUM_IMAGING` to the `Modality` enum for calcium-imaging-based
  functional correlations.

### Changed

- Migrated registry-backed writes in the `code/etl_*.ipynb` notebooks to
  `write_models()` and `write_projection_matrix()`. The notebooks now obtain
  their shared output root from `get_settings().output_root`; wide matrix
  Parquet files and `CellCellConnectivityLong` writes remain on direct Delta
  APIs pending registry support.

### Deprecated

### Removed

- Removed the top-level `connects_common_connectivity.arrow_utils` and
  `connects_common_connectivity.write_utils` modules. Import them from
  `connects_common_connectivity.io.arrow_utils` and
  `connects_common_connectivity.io.write_utils`, respectively.

### Fixed

- Fixed `DataSet` writes to scope on `(project_id, id)` instead of
  `project_id` alone, so sibling notebooks sharing a project no longer
  overwrite one another's `DataSet` rows.

### Security
