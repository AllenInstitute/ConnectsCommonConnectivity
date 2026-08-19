# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `DatasetReader` to `connects_common_connectivity.io` for listing datasets, discovering related cell feature sets and cluster hierarchies, and reading one wide Polars table per dataset with feature and cluster-level columns.
- Added the `synapse_schema` module with the `SynapseConnectivityLong` and `SynapseFeatureMatrix` classes for per-synapse connectivity: long-form single-synapse rows plus a wide per-synapse feature matrix LEFT-joined on the synapse id.
- Added `read_synapse_table()` to `connects_common_connectivity.io` for reading the long single-synapse connectivity table, optionally LEFT-joining per-synapse feature columns from a wide feature table.
- Made `SynapseFeatureMatrix` writable through `write_models()`; it has registered `WriteSpec` entry and appear in `WRITABLE_CLASSES`.

### Changed

### Deprecated

### Removed

### Fixed

### Security
