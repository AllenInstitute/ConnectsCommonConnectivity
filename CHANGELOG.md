# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added writable `ReferenceSpace` and `SpatialLocation` tables, explicit `LocationType` values, and optional `SignedAxis` enum `Default2DView` metadata for spatial coordinates.
- Added optional `SpatialLocation.description` details for anatomical points,
  particularly those with location type `OTHER`, preserved through writes and reads.
- Added `read_spatial_locations` and `read_reference_spaces` to the public IO API, with project and identity filters and preserved default-view structs.
- Added required measurement-context `connectome_id` values to
  `CellCellConnectivityLong` and `CellCellMeasurementMatrix`.
- Added required `synapse_table_id` identity to single-synapse rows and feature pointers, plus optional source provenance on derived cell-cell measurements; DataSet IDs remain reserved for DataItem collections.
- Added `derive_cell_cell_connectivity` for deriving count and optional
  anatomical-size measurements from Polars synapse tables; inputs must have
  unique, non-null synapse IDs and belong to the requested project.
- Added aligned endpoint filters to `read_synapse_table` and
  `read_cell_cell_connectivity`; synapse reads require table identity, while
  cell-cell reads require connectome identity and can optionally filter source
  provenance. Requesting a provenance filter for a table without provenance
  now raises a clear error.
- Added `io.path_spec` as the shared source of canonical model-table and wide-payload paths for readers, writers, and ETLs.
- Added merge-scoped upserts for identity-bearing metadata and association rows written through `write_models`, scanning only the partitions a batch touches.
- Added project scoping to `ProjectionMeasurementMatrix`, `SingleCellReconstruction`, and `BrainRegionAssociation`.
- Added taxonomy-local `hierarchy_id` values to `HierarchyCategory`.

### Changed

- Changed `read_reference_spaces` to select all scopes when `project_id` is omitted, only global spaces for explicit `None`, and only the named project's spaces for a string, without including global spaces automatically.
- Changed the Minnie and V1DD cell-cell ETLs to share canonical `cellcellconnectivitylong/` storage with connectome-scoped overwrites.
- Changed identity-bearing metadata and association writes through `write_models` from scoped replacement to pure upserts; rerunning with fewer `Cluster`, `CellFeatureDefinition`, or `ClusterMembership` rows no longer deletes omitted rows. Explicit deletion support is tracked in #21.
- `HierarchyCategory` now requires `level`, so every category has an unambiguous position in its taxonomy.

### Deprecated

### Removed

- Removed unused `SingleCellReconstruction.soma_location` and `CellMetadata.spatial_location` fields; write project-scoped `SpatialLocation` rows with an explicit reference space and location type instead.
- Removed the unused `io.write_utils.append_new_dataitems` helper; use merge-scoped `write_models` calls with `DataItem` models instead.

### Fixed

- Fixed `read_spatial_locations` filters mixing `LocationType` members and
  strings; matching rows are retained even when other requested values are unknown.
- Fixed `ReferenceSpace` writes accepting inconsistent voxel-scale metadata;
  supplied scale now requires paired size/unit fields, `VOXELS` coordinates,
  and three finite positive dimensions in supported physical length units.
  Omitting physical scale remains valid.
- Fixed `ReferenceSpace` writes accepting default-view directions that reuse
  the same data axis, regardless of sign.
- Fixed `write_models` skipping schema validation for models with no write-only
  constraints; every row is now re-validated before any IO, so rows built with
  `model_construct` can no longer reach a Delta table with missing or
  wrongly typed slots.
- Fixed `ReferenceSpace` writes with the same ID overwriting rows in other scopes; identity now combines `project_id` and `id`, with null `project_id` representing a separate global scope.
- Fixed embedded Pydantic models being stringified or mistaken for ID references during Arrow conversion; schema-declared structs, including lists of structs, now round-trip through Parquet.
- Fixed Delta writes for column names that overlap SQL keywords or contain special characters.
- Fixed later writes deleting rows contributed by other ETL notebooks in shared dataset and hierarchy scopes.
- Fixed mappings and feature-matrix pointers with the same local ID merging across different parent sets.
- Fixed existing `DataItem` metadata updates being silently ignored.
- Fixed `dry_run` being ignored when `output_root` overrides the configured destination.
- Fixed generated `CellFeatureMeasurement` models to include `feature_set_id` and `unit`, and to accept valid NumPy dtype strings.
- Fixed `HierarchyCategory.level` to generate as an integer and prevented category writes from colliding across taxonomies.

### Security
