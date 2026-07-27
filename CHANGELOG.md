# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added the `synapse_schema` module with the `SynapseConnectivityLong` and `SynapseFeatureMatrix` classes for per-synapse connectivity: long-form single-synapse rows plus a wide per-synapse feature matrix LEFT-joined on the synapse id.
- Added `read_synapse_table()` to `connects_common_connectivity.io` for reading the long single-synapse connectivity table, optionally LEFT-joining per-synapse feature columns from a wide feature table.
- Made `SynapseFeatureMatrix` writable through `write_models()`; it has registered `WriteSpec` entry and appear in `WRITABLE_CLASSES`.

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
