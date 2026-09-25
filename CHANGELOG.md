# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added required measurement-context `connectome_id` values to `CellCellConnectivityLong`.
- Added required `synapse_table_id` identity to single-synapse rows and feature pointers, plus optional source provenance on derived cell-cell measurements; DataSet IDs remain reserved for DataItem collections.
- Added `derive_cell_cell_connectivity` for deriving count and optional anatomical-size measurements from Polars synapse tables.
- Added aligned endpoint filters to `read_synapse_table` and `read_cell_cell_connectivity`; synapse reads require table identity, while cell-cell reads require connectome identity and can optionally filter source provenance.
- Added `io.path_spec` as the shared source of canonical model-table and wide-payload paths for readers, writers, and ETLs.
- Added merge-scoped upserts for identity-bearing metadata and association rows written through `write_models`, scanning only the partitions a batch touches.
- Added project scoping to `ProjectionMeasurementMatrix`, `SingleCellReconstruction`, and `BrainRegionAssociation`.
- Added taxonomy-local `hierarchy_id` values to `HierarchyCategory`.

### Changed

- Changed the Minnie and V1DD cell-cell ETLs to share canonical `cellcellconnectivitylong/` storage with connectome-scoped overwrites.
- Changed identity-bearing metadata and association writes through `write_models` from scoped replacement to pure upserts; rerunning with fewer `Cluster`, `CellFeatureDefinition`, or `ClusterMembership` rows no longer deletes omitted rows. Explicit deletion support is tracked in #21.

### Deprecated

### Removed

- Removed the unused `io.write_utils.append_new_dataitems` helper; use merge-scoped `write_models` calls with `DataItem` models instead.

### Fixed

- Fixed Delta writes for column names that overlap SQL keywords or contain special characters.
- Fixed later writes deleting rows contributed by other ETL notebooks in shared dataset and hierarchy scopes.
- Fixed mappings and feature-matrix pointers with the same local ID merging across different parent sets.
- Fixed existing `DataItem` metadata updates being silently ignored.
- Fixed `dry_run` being ignored when `output_root` overrides the configured destination.
- Fixed generated `CellFeatureMeasurement` models to include `feature_set_id` and `unit`, and to accept valid NumPy dtype strings.
- Fixed `HierarchyCategory.level` to generate as an integer and prevented category writes from colliding across taxonomies.

### Security
