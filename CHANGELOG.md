# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `connects_common_connectivity.io.writers` with `write_models()` (the
  single dispatch core for all generated pydantic models),
  `write_projection_matrix()`, `WriteResult`, and `WRITABLE_CLASSES`.
- Added `populate_region_coverage()` in
  `connects_common_connectivity.io.write_utils` for deriving
  `ProjectionMeasurementMatrix.region_coverage` from a dense matrix.

### Changed

- Moved `arrow_utils` and `write_utils` under
  `connects_common_connectivity.io.*`. The old import paths
  (`connects_common_connectivity.arrow_utils`,
  `connects_common_connectivity.write_utils`) keep working as deprecated
  re-export shims.

### Deprecated

- Importing from `connects_common_connectivity.arrow_utils` and
  `connects_common_connectivity.write_utils`; use
  `connects_common_connectivity.io.arrow_utils` /
  `connects_common_connectivity.io.write_utils` instead. The shims will be
  removed once notebook migration completes.

### Removed

### Fixed

### Security
