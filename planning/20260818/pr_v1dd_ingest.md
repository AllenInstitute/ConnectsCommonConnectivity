# v1dd ingestion: dataset reader, synapse schema, and soma-feature ETL

**Base:** `ingestion-v2` → **Head:** `v1dd-ingest`

Adds end-to-end v1dd ingestion plus read functions to complement the write API. New synapse schema.
Package bumped to `0.3.0`.

## Read API
- `DatasetReader` (`connects_common_connectivity.io`): lists datasets, discovers
  related cell-feature sets and cluster hierarchies, returns one wide Polars
  table per dataset.
- `read_synapse_table()`: reads the long single-synapse connectivity table, with
  an optional LEFT-join for per-synapse feature columns.

## Synapse schema
- `schemas/synapse_schema.yaml` and generated `SynapseConnectivityLong` /
  `SynapseFeatureMatrix` models. `SynapseConnectivityLong` is long-form (one row
  per synapse), so the pre/post `DataItem` pair is intentionally non-unique.
  `SynapseFeatureMatrix` points at a wide per-synapse feature Parquet keyed on
  the synapse id.
- `SynapseFeatureMatrix` is writable via `write_models()` (registered `WriteSpec`,
  listed in `WRITABLE_CLASSES`).

## Testing

```bash
uv run pytest
```

Adds `tests/test_read.py` and expands `test_writers.py`,
`test_write_validation.py`, `test_write_utils.py`, `test_write_spec.py`. The
`etl_v1dd_*` notebooks run top-to-bottom against v1dd source data.

## Reviewer focus
- Non-unique pre/post pair in `SynapseConnectivityLong` and LEFT-join semantics
  in `read_synapse_table()`.
- `DatasetReader` discovery of related feature sets / cluster hierarchies.
