Adds the WP3 cell-cell connectivity contract, reader, and derivation from synapse table API, with
canonical direct ETL writes that keep distinct connectomes isolated. Review
against `wp2-merge-write`; this branch is stacked on that completed work.

## What changed

1. **Defined connectivity identity and provenance** —
  `cell_cell_schema.yaml::CellCellConnectivityLong` and
  `cell_cell_schema.yaml::CellCellMeasurementMatrix` now require
  `connectome_id` as the measurement-context discriminator. The synapse
  schemas use `synapse_table_id` for logical table identity, while cell-cell
  rows may carry it as optional source provenance; generated models were
  refreshed from the LinkML sources.

2. **Added scoped connectivity reads** —
  `read.py::read_cell_cell_connectivity` requires project and connectome scope
  and supports optional source-provenance, pre- and post-synaptic cell, and
  measurement-type filters. Provenance-free tables remain readable, while
  requesting a `synapse_table_id` filter against one raises a clear error.
  Existing `read.py::read_synapse_table` uses the same endpoint filter shape
  while retaining its feature join.

3. **Added validated columnar derivation from synapse tables** —
  `write_utils.py::derive_cell_cell_connectivity` requires an explicit project
  and unique, non-null synapse IDs, rejecting cross-project rows and duplicate
  source identities before they can inflate totals. It always emits
  `SYNAPSE_COUNT` and optionally emits `SUM_ANATOMICAL_SIZE` with an explicit
  source column and unit. Native Polars expressions build readable IDs from
  project, connectome, endpoints, and measurement type; optional
  synapse-table provenance does not affect identity.
  `write_utils.py::cell_cell_connectivity_to_arrow` converts derived frames
  to the canonical model Arrow schema before persistence.

4. **Consolidated cell-cell ETL storage** —
   `etl_minnie_04_cell_cell.ipynb` writes both Minnie contexts to canonical
   `cellcellconnectivitylong/` with separate connectome-scoped overwrite
   predicates. `etl_v1dd_03_synapses.ipynb` names its logical synapse table,
  uses the derivation and canonical Arrow conversion APIs, and writes the
  derived context to the same canonical table; `etl_example_prompt.md` was
  updated accordingly.

5. **Centralized canonical storage paths** —
  `path_spec.py::MODEL_TABLE_PATHS` defines a path for every concrete
  project-scoped model, including models that are not registered for writes,
  while `path_spec.py::WIDE_PAYLOAD_PATHS` separately names dataframe-backed
  feature payloads. Readers, writer policies, and the edited ETLs consume
  those mappings instead of repeating directory literals.

6. **Locked contracts and delivery boundaries** —
   `tests/test_cell_cell.py`, `tests/test_read.py`,
   `tests/test_public_api.py`, and registry tests cover required scope,
  duplicate source rejection, aggregation, deterministic IDs, optional
  provenance, unavailable-provenance filters, empty results, exact Arrow
  schema conversion, and canonical-path drift.

| Issue | Closed by |
|---|---|
| Closes #17 — `CellCellConnectivityLong`: add `connectome_id` | 1, 2, 4, 5, 6 |
| Closes #18 — derive `CellCellConnectivityLong` from a synapse table | 1, 3, 4, 6 |

## Why

**Measurement-context collisions (#17).** Rows from different segmentation,
proofreading, or measurement contexts were indistinguishable. Changes 1, 2,
and 4 make `connectome_id` the required logical discriminator and let both
contexts coexist in one canonical Delta table without treating a cohort as a
connectome.

**Repeated manual aggregation (#18).** V1DD manually grouped synapses and built
one Pydantic object per derived row. Change 3 provides a schema-shaped Polars
transform with explicit count, size, unit, null, and readable deterministic-ID
behavior; change 4 adopts it in the ETL.

**Cross-project ambiguity (#18).** A scalar `connectome_id` cannot describe
rows from multiple projects. Change 3 requires the caller's `project_id`,
validates every input row against it, and includes project scope in generated
IDs so repeated endpoint names cannot collide across projects.

**Duplicate source inflation (#18).** Repeated synapse IDs could otherwise
increase both count and size totals. Change 3 validates source identity before
aggregation instead of silently deduplicating or overcounting.

**Incomplete provenance filters.** Cell-cell tables may legitimately omit
source-table provenance. Change 2 keeps those tables readable but rejects a
provenance filter when the stored column is unavailable.

**Arrow schema divergence.** Polars preserves column data types but not model
field nullability, and its Arrow conversion emits `large_string` rather than
the model's canonical `string`. Changes 3 through 5 give the V1DD write an
explicit canonical conversion and regression coverage, avoiding a table schema
that depends on producer write order.

**Storage-path drift.** Readers, writer policies, and notebooks previously
repeated directory literals. Change 5 separates canonical model-table paths
from wide dataframe payloads and verifies that every concrete project-scoped
model has a path without making read availability depend on writer
registration.

**Scope limit.** This closes the WP3 schema, reader, transform, and direct ETL
adoption work. The WP3/WP4 handoffs and changelog distinguish
this PR from generic writer registration and cohort resolution. `write_cellcellconnectivitylong`
and direct `write_deltalake` calls remain in
place intentionally. Issue #19 owns validated `write_table` registration and
replacement of those direct canonical writes, including enforcement that does
not depend on notebook callers invoking a conversion helper. The open ownership
decision is recorded in `planning/20260820/wp4_plan_handoff.md`.


## How to test

```bash
bash scripts/generate_models.sh
# passed; generated models hash:
# d75ffd682cf4ccc04d5db616e76c04ff73dcaf148c1cc26e8d3ce870038ef495
bash scripts/generate_models.sh
# passed with the same hash

pytest -q
# 268 passed, 1 warning

uv run ruff check \
  src/connects_common_connectivity/io/path_spec.py \
  src/connects_common_connectivity/io/read.py \
  src/connects_common_connectivity/io/write_utils.py \
  src/connects_common_connectivity/io/write_spec.py \
  src/connects_common_connectivity/io/writers.py \
  tests/test_cell_cell.py tests/test_read.py tests/test_write_spec.py \
  tests/test_public_api.py tests/test_writers.py
# all checks passed

git diff --check
# passed
```

A Python notebook validation pass parsed all four edited notebooks as JSON and
compiled all 57 code cells without executing external-data workflows. Synthetic
Delta tests verify project/connectome isolation, optional provenance filtering,
pre- and post-synaptic cell and measurement filters, feature joins,
missing-table errors, and schema-preserving empty results.

> The Minnie and V1DD ETLs were not rerun because their external source data
> was not available in this environment. Their stored outputs may predate the
> changed source cells and are not cited as verification evidence.

## Reviewer focus (optional)

- `cell_cell_schema.yaml::CellCellConnectivityLong`: `connectome_id` is required;
  `synapse_table_id` is optional provenance, not additional identity.
- `cell_cell_schema.yaml::CellCellMeasurementMatrix`: matrix-form measurements
  require the same `connectome_id` context as long-form measurements.
- `write_utils.py::_shape_cell_cell_measurements`: derived IDs use the readable
  `project_connectome_pre_post_measurement-type` form and native Polars string
  concatenation; source provenance is deliberately excluded. This helper does
  not standardize IDs produced by manual ETLs; that decision is deferred to
  WP4.
- `write_utils.py::derive_cell_cell_connectivity`: size aggregation rejects any
  null size rather than reporting a partial total. Its explicit `project_id`
  must match every input row, and source synapse IDs must be unique and
  non-null, so one invocation cannot span projects or double-count a source.
- `write_utils.py::cell_cell_connectivity_to_arrow`: the interim public helper
  applies canonical Arrow types and field nullability metadata for the V1DD
  write; it does not validate required values. Issue #19 must decide whether a
  CellCell writer or generic `write_table` owns conversion and exhaustive null
  validation for all producers.
- `read.py::read_cell_cell_connectivity`: project/connectome scope is mandatory;
  filtering by provenance requires that provenance to exist in storage;
  dataset and cluster cohort expansion remains #23.
- `path_spec.py::MODEL_TABLE_PATHS`: every concrete project-scoped model has a
  canonical path, including unregistered tables, without implying write
  eligibility; wide feature payloads remain separate in `WIDE_PAYLOAD_PATHS`.
- `write_spec.py::REGISTRY`: both long connectivity classes remain unregistered;
  registration and generic bulk writes remain #19.
- Minnie and V1DD notebook writes use canonical paths but still call Delta
  directly; V1DD canonicalizes through the helper first. Review their
  `(project_id, connectome_id)` overwrite predicates.
