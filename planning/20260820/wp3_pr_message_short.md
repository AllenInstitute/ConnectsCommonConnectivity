Adds the WP3 cell-cell connectivity contract, scoped reader, synapse-table
derivation, and canonical ETL writes. Review against `wp2-merge-write`; this
branch is stacked on that completed work.

## What changed

1. **Defined connectivity identity** —
   `cell_cell_schema.yaml::CellCellConnectivityLong` and
   `cell_cell_schema.yaml::CellCellMeasurementMatrix` require `connectome_id`.
   `synapse_table_id` remains optional source provenance on cell-cell rows.

2. **Added scoped reads** — `read.py::read_cell_cell_connectivity` requires
   project and connectome scope and supports provenance, endpoint-cell, and
   measurement-type filters. It reports when provenance filtering is requested
   for a table without provenance.

3. **Added validated derivation** —
   `write_utils.py::derive_cell_cell_connectivity` validates project scope and
   unique, non-null synapse IDs, then derives counts and optional anatomical-size
   sums with deterministic IDs. `cell_cell_connectivity_to_arrow` converts the
   result to the canonical model Arrow schema.

4. **Consolidated ETL storage** — The Minnie and V1DD notebooks write separate
   connectome contexts to canonical `cellcellconnectivitylong/` storage using
   scoped overwrite predicates; V1DD uses the new derivation and conversion APIs.

5. **Centralized storage paths** — `path_spec.py::MODEL_TABLE_PATHS` defines
   canonical project-scoped model paths, while `WIDE_PAYLOAD_PATHS` keeps
   dataframe-backed feature payloads separate.

6. **Added contract coverage** — Cell-cell, read, public API, and registry tests
   cover scope, duplicate rejection, aggregation, IDs, provenance, Arrow schema,
   empty results, and path drift.

| Issue | Closed by |
|---|---|
| Closes #17 — add `connectome_id` | 1, 2, 4, 5, 6 |
| Closes #18 — derive connectivity from a synapse table | 1, 3, 4, 6 |

## Why

**Context isolation (#17).** Changes 1, 2, and 4 make `connectome_id` the
required discriminator so multiple connectivity contexts can coexist without
collisions.

**Reliable derivation (#18).** Change 3 replaces manual row-by-row aggregation,
rejects cross-project input and duplicate source identities, and produces
schema-shaped count and size measurements.

**Consistent persistence.** Changes 3 through 5 prevent Arrow-schema and path
drift across producers, readers, and writer policies.

**Scope limit.** Issue #19 owns generic writer registration, replacement of
direct Delta writes, and exhaustive write-time validation. Cohort expansion
remains #23. The ownership decision is recorded in
`planning/20260820/wp4_plan_handoff.md`.

## How to test

```bash
bash scripts/generate_models.sh
# passed twice with identical generated-model hashes

pytest -q
# 268 passed, 1 warning

uv run ruff check src/connects_common_connectivity/io/ tests/
# all checks passed for the touched files

git diff --check
# passed
```

All four edited notebooks parsed as JSON, and all 57 code cells compiled.
Synthetic Delta tests verify scoped isolation, filters, joins, missing-table
errors, and schema-preserving empty results.

> Minnie and V1DD ETLs were not rerun because external source data was
> unavailable. Stored notebook outputs are not verification evidence.

## Reviewer focus (optional)

- `cell_cell_schema.yaml::CellCellConnectivityLong`: required `connectome_id`
  versus optional `synapse_table_id` provenance.
- `write_utils.py::derive_cell_cell_connectivity`: project and source-identity
  validation, null-size rejection, and provenance-free deterministic IDs.
- `write_utils.py::cell_cell_connectivity_to_arrow`: interim conversion scope;
  issue #19 owns final writer-side validation.
- `read.py::read_cell_cell_connectivity`: mandatory scope and unavailable-
  provenance behavior.
- `path_spec.py::MODEL_TABLE_PATHS`: canonical paths do not imply write
  eligibility.
- Notebook overwrite predicates isolate `(project_id, connectome_id)`; direct
  Delta writes remain until #19.