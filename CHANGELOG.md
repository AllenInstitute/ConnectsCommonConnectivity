# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added merge-scoped upserts for identity-bearing metadata and association rows written through `write_models`.
- Added project scoping to `ProjectionMeasurementMatrix`, `SingleCellReconstruction`, and `BrainRegionAssociation`.
- Added taxonomy-local `hierarchy_id` values to `HierarchyCategory`.

### Changed

### Deprecated

### Removed

### Fixed

- Fixed later writes deleting rows contributed by other ETL notebooks in shared dataset and hierarchy scopes.
- Fixed existing `DataItem` metadata updates being silently ignored.
- Fixed `dry_run` being ignored when `output_root` overrides the configured destination.
- Fixed generated `CellFeatureMeasurement` models to include `feature_set_id` and `unit`, and to accept valid NumPy dtype strings.
- Fixed `HierarchyCategory.level` to generate as an integer and prevented category writes from colliding across taxonomies.

### Security
