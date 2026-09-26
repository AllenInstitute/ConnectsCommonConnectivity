# WP4 bulk and wide writes handoff

## Objective

Implement GitHub issue
[#19](https://github.com/AllenInstitute/ConnectsCommonConnectivity/issues/19)
as the reusable fixed-schema bulk table API needed by both long connectivity
classes, and separately implement
[#20](https://github.com/AllenInstitute/ConnectsCommonConnectivity/issues/20)
for dynamic wide payloads and their pointer rows.

WP4 owns:

- generic validated `write_table` support for explicitly opted-in flat,
  fixed-schema `overwrite_scoped` models;
- `SynapseConnectivityLong` and `CellCellConnectivityLong` registration;
- validated registry-backed persistence for both classes;
- migration of affected ETLs from direct canonical Delta writes.
- a distinct wide-payload API for `CellFeatureMatrix` and
  `SynapseFeatureMatrix` workflows;
- validation and ordering between each dynamic payload and its pointer row;
- migration of ETLs that currently write those payloads directly.

The two APIs are not interchangeable. `write_table` validates one table against
a fixed LinkML row schema. The #20 API accepts dynamic feature columns and
coordinates a payload write with a separate pointer-model write.

## Dependency and delivery order

This handoff consumes the schema, reader, and derivation delivered by WP3
issues #17 and #18.

1. Land WP3 `connectome_id`, `read_cell_cell_connectivity`, and
  `derive_cell_cell_connectivity` first.
2. Base or rebase WP4 onto the completed WP3 state.
3. Complete #19: add `write_table`, register both long classes, and migrate
  their direct ETL writes to the generic writer.
4. Complete #20 as a separate workstream after #19, retaining its dependency on
  #13's mode vocabulary. It may reuse low-level settings and Delta helpers but
  must not route dynamic payloads through fixed-schema model validation.

Issue #19 should therefore depend on #17 and #18 and own their scalable
persistence and ETL adoption follow-through.

## Repository constraints

- Never edit generated `models.py` manually.
- Use `apply_patch` for existing text files and the notebook editor for existing
  notebooks.
- Do not revert or commit unrelated changes.
- Follow the changelog instruction and repository commit/PR skills.
- Preserve `write_models` as the Pydantic API for small model batches.

## Issue #19: fixed-schema bulk API boundary

Add public:

```python
write_table(
    table,
    *,
    model_cls,
    settings=None,
    output_root=None,
    validation_sample_size=...,
) -> WrittenResult
```

The first implementation supports only model classes whose `WriteSpec`
explicitly opts into table input and whose mode is `overwrite_scoped`.

Registration does not imply table-write eligibility. `merge_scoped` metadata
and association classes remain on `write_models`.

`write_table` does not accept arbitrary dynamic columns and does not write a
pointer model. Those behaviors belong exclusively to #20.

### Open decision: long-table ID creation

Before migrating the long-table ETLs, decide where canonical row IDs are
created. WP3 only gives `derive_cell_cell_connectivity` a private, readable
`connectome_pre_post_measurement-type` ID convention; it does not standardize
manual producers.

Choose and document one of these contracts for #19:

- `write_table` creates model-specific IDs through an explicit `WriteSpec`
  policy before validation; or
- producers create IDs through shared public helpers and `write_table`
  validates that the supplied IDs match those helpers.

Do not silently preserve multiple conventions when Minnie and V1DD move to the
same canonical table. Cover the selected contract with tests proving that a
manual cell-cell row and a derived row with the same identity receive the same
ID. Keep the IDs readable unless an explicit fixed-length opaque-ID requirement
is introduced.

## WriteSpec changes

Extend `src/connects_common_connectivity/io/write_spec.py::WriteSpec` with an
explicit table-write capability flag or equivalent narrow policy field.

Validate the policy itself:

- table-write eligibility is allowed only for `overwrite_scoped`;
- table-write eligibility must not be inferred from model shape;
- existing registry entries remain ineligible by default;
- incompatible combinations fail at spec construction.

Register `SynapseConnectivityLong`:

- subdirectory: `synapse`
- partition: `project_id`
- overwrite scope: `(project_id, synapse_table_id)`
- mode: `overwrite_scoped`
- table-write eligible: yes
- required-for-write fields: every nullable generated field needed for valid
  source identity and scope

Remove the commented registry block and stale/dead import state.

Register `CellCellConnectivityLong`:

- subdirectory: `cellcellconnectivitylong`
- partition and overwrite scope: `(project_id, connectome_id)`
- mode: `overwrite_scoped`
- table-write eligible: yes
- required-for-write fields: nullable generated endpoint and measurement fields
  that must be non-null in persisted connectivity rows

Remove the obsolete `write_cellcellconnectivitylong` placeholder from
`src/connects_common_connectivity/io/writers.py`. Both classes must use the same
generic path without class-name conditionals.

## Columnar validation

Add a clearly separated bulk-table validation section to
`src/connects_common_connectivity/io/write_validation.py`. Do not create a new
`table_validation.py` module.

Accept:

- `pyarrow.Table`
- `polars.DataFrame`

Normalize against `build_arrow_schema(model_cls)` and exhaustively validate:

- supported input container;
- non-empty input;
- exact allowed columns;
- required schema columns;
- optional missing columns filled with correctly typed null arrays;
- safe Arrow casts only;
- canonical schema column order;
- non-null schema-required fields;
- non-null `required_for_write` fields;
- non-null scope and partition fields;
- enum membership for all enum-backed columns.

Validation errors must identify the model, field, and representative invalid row
or value. Validation must finish before creating or mutating a Delta table.

A bounded Pydantic sample may be available as a diagnostic for generated
constraints not represented in Arrow. Sampling is not exhaustive and must not
be the basis for type, null, enum, partition, or scope guarantees.

Explicit non-guarantees:

- foreign-key existence;
- cross-table relationships;
- global uniqueness;
- arbitrary generated custom-validator equivalence;
- semantic validation of dynamic feature columns.

## Writer implementation

Add a clearly separated bulk-table section to
`src/connects_common_connectivity/io/writers.py`.

`write_table` must:

1. resolve the exact registered `model_cls` through `get_spec`;
2. reject specs without table-write eligibility;
3. reject modes other than `overwrite_scoped`;
4. normalize and validate before IO;
5. honor existing `settings`, `output_root`, and `dry_run` precedence;
6. attach LinkML class/version/schema metadata;
7. delegate to the existing `_dispatch_overwrite_scoped` implementation;
8. return the existing `WrittenResult` shape;
9. avoid Pydantic model construction and per-row `model_dump`.

Keep `write_models` behavior unchanged. Do not add checks such as
`if model_cls is SynapseConnectivityLong`; policy belongs in `WriteSpec`.

Export `write_table` through `src/connects_common_connectivity/io/__init__.py`
and update `tests/test_public_api.py`.

## Tests

Extend:

- `tests/test_write_validation.py`
- `tests/test_writers.py`
- `tests/test_write_spec.py`
- `tests/test_read.py` or a focused SynapseLong IO test
- `tests/test_public_api.py`

Cover the generic contract with a small test model/spec where useful, and cover
both long classes in integration tests.

Required cases:

- Arrow and Polars input;
- rejection of unsupported containers;
- empty input;
- exact extra/missing-column behavior;
- optional-column typed-null insertion;
- safe cast success and unsafe cast failure;
- canonical output schema order;
- required and write-required null failures;
- null project/dataset scope failure;
- enum membership where applicable;
- unsupported class and non-opted-in spec rejection;
- rejection of table-write eligibility on `merge_scoped` specs;
- dry-run performs validation but no IO;
- LinkML metadata attached to persisted data;
- two project/dataset scopes coexist;
- rewriting one scope preserves the other;
- two project/connectome scopes coexist;
- rewriting one connectome preserves the other;
- invalid input creates no Delta directory;
- round-trip compatibility with `read_synapse_table`;
- round-trip compatibility with `read_cell_cell_connectivity`;
- `write_models` regression coverage remains green.

Include a scale-oriented test or benchmark-sized synthetic check that proves the
path does not instantiate Pydantic rows. Do not put an 8-million-row fixture in
the unit suite.

## Documentation

Update:

- `etl_example_prompt.md`
- `CHANGELOG.md`
- public docstrings for `WriteSpec`, validation helpers, and `write_table`

Document:

- the distinction between `write_models` and `write_table`;
- fixed-schema table eligibility and `overwrite_scoped` restriction;
- exhaustive columnar guarantees and explicit non-guarantees;
- SynapseLong scope `(project_id, synapse_table_id)`;
- CellCellLong scope `(project_id, connectome_id)`;
- that WP3 supplies the CellCellLong schema, reader, and transform consumed here;
- that dynamic wide payload and pointer work remains #20.

## ETL migration

Edit only active notebooks whose filenames begin with `etl` and whose long-table
writes are affected by #19.

### `code/etl_v1dd_03_synapses.ipynb`

Replace the raw `write_deltalake` call for the roughly 8-million-row
`SynapseConnectivityLong` table with `write_table` using its existing columnar
Arrow or Polars representation.

Requirements:

- do not instantiate one Pydantic model per synapse;
- preserve project and synapse-table scope;
- preserve canonical `synapse/` path through the registry;
- preserve LinkML metadata through the public writer;
- verify with `read_synapse_table`;
- keep WP3's `derive_cell_cell_connectivity` call, explicit size column/unit,
  and stable connectome ID unchanged;
- replace only the derived cell-cell frame's direct canonical Delta persistence
  with `write_table`;
- verify it through `read_cell_cell_connectivity`.

### `code/etl_minnie_04_cell_cell.ipynb`

- preserve the stable connectome IDs added by WP3;
- construct schema-shaped columnar frames instead of Pydantic row lists;
- replace direct canonical `cellcellconnectivitylong/` writes with
  `write_table`;
- preserve separate `(project_id, connectome_id)` scopes;
- verify both scopes coexist through `read_cell_cell_connectivity`;
- verify rewriting one scope preserves the other.

### `code/etl_examples_readme.ipynb`

Update only paths and workflow descriptions made stale by these migrations.

Do not edit `code/example_microns_query.ipynb`,
`code/parse_minnie_clustering.ipynb`, non-ETL notebooks, or unrelated cells.

The wide-feature payload cell in `etl_v1dd_03_synapses.ipynb` remains unchanged
until the separate #20 migration.

## Issue #20: dynamic wide payload plus pointer API

Implement the `wide_parquet` behavior described by #20 separately from
`write_table`.

The public API name should follow repository conventions and may be settled
during implementation, but its contract must accept:

- a dynamic Arrow or Polars payload;
- a `CellFeatureMatrix` or `SynapseFeatureMatrix` pointer model;
- settings or output-root resolution consistent with existing writers.

It must:

1. resolve the pointer's canonical payload destination;
2. validate that the configured index column exists in the payload;
3. validate that pointer IDs, feature-set or dataset scope, index-column name,
   and stored path agree with the destination;
4. validate scope columns and required pointer fields before IO;
5. overwrite only the declared wide payload scope;
6. write the payload first;
7. merge the pointer row through `write_models` only after the payload succeeds;
8. honor dry-run without writing either destination;
9. return enough result information to identify both writes.

This sequencing prevents creation of a pointer to a payload write that already
failed. It is not a cross-table ACID transaction: if pointer persistence fails
after payload success, the payload may remain without a pointer. Document that
failure mode rather than calling the operation atomic.

Do not validate dynamic feature columns against
`build_arrow_schema(CellFeatureMatrix)` or
`build_arrow_schema(SynapseFeatureMatrix)`: those schemas describe the pointer
rows, not the payload columns.

### Issue #20 tests

Add focused coverage for:

- CellFeatureMatrix and SynapseFeatureMatrix payloads;
- dynamic columns preserved without fixed-model rejection;
- missing, null, duplicate, or incorrectly named index columns according to the
  agreed contract;
- pointer path and canonical destination mismatch;
- feature-set or dataset scope mismatch;
- payload failure prevents pointer creation;
- pointer write occurs only after payload success;
- dry-run suppresses both writes;
- rerunning one payload scope preserves sibling scopes;
- round-trip discovery through the existing pointer and reader paths.

### Issue #20 ETL migration

Audit active `etl*.ipynb` notebooks for raw writes of payloads represented by
`CellFeatureMatrix` or `SynapseFeatureMatrix`. Migrate those payload-plus-pointer
pairs to the combined API in the same #20 workstream.

Known candidates include:

- `code/etl_minnie_02_cell_features.ipynb`
- `code/etl_v1dd_01_v1196.ipynb`
- `code/etl_v1dd_02_cave.ipynb`
- the wide synapse-feature cell in `code/etl_v1dd_03_synapses.ipynb`
- `code/etl_v1dd_05_somafeatures.ipynb`
- `code/etl_visp_exc_patchseq_02_cell_features.ipynb`
- `code/etl_visp_inh_patchseq_02_cell_features.ipynb`
- `code/etl_wnm_exc_02_cell_features.ipynb`

Confirm each candidate actually writes a registered pointer row before editing
it. Do not mechanically migrate raw tables that have no
`CellFeatureMatrix`/`SynapseFeatureMatrix` pointer contract. Wide projection
payloads are excluded unless #20 is explicitly expanded to define their pointer
contract.

## Verification

Run at minimum:

```bash
pytest tests/test_write_validation.py tests/test_writers.py tests/test_write_spec.py tests/test_read.py tests/test_public_api.py -v
ruff check src/connects_common_connectivity/io tests/test_write_validation.py tests/test_writers.py tests/test_write_spec.py tests/test_read.py tests/test_public_api.py
pytest -v
```

Validate notebook JSON and imports after editing. Execute Minnie and V1DD only
when their external source data is available. If unavailable, use synthetic
Delta integration to prove both scope policies and reader compatibility, and
state which production notebooks were not rerun.

## Commit and PR boundaries

Recommended commits:

1. `WriteSpec` capability, columnar validation, `write_table`, both long-table
  registrations, tests, and non-notebook docs.
2. #19 Minnie and V1DD long-table ETL migration only.
3. #20 wide payload-plus-pointer API, tests, and non-notebook docs.
4. #20 relevant ETL notebook migrations only.
5. Final verified `planning/20260820/wp4_pr_message.md`.

Use `commit-and-push` after each boundary. The PR description must distinguish
#19's fixed-schema table API from #20's dynamic payload-plus-pointer API, state
that #19 consumes #17 and #18, and report verification for each workstream
separately.

## Out of scope

- Cell-cell schema, reader, or derivation implementation (owned by WP3)
- Wide projection payloads unless #20 gains an explicit pointer contract
- Zarr payload creation unrelated to the two supported pointer models
- `merge_scoped` table inputs
- Deletes
- Streaming/chunked writer design
- Non-ETL notebooks
