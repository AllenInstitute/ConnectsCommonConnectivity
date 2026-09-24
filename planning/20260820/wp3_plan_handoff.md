# WP3 cell-cell connectivity handoff

## Objective

Close GitHub issues
[#17](https://github.com/AllenInstitute/ConnectsCommonConnectivity/issues/17)
and
[#18](https://github.com/AllenInstitute/ConnectsCommonConnectivity/issues/18)
without taking ownership of the generic bulk-write implementation.

WP3 owns the `CellCellConnectivityLong` data contract, its scoped reader, and
the pure synapse-to-cell-cell derivation transform. It does not register either
long connectivity class for writing. It makes only the minimum ETL changes
needed to adopt the required field and transform while preserving the existing
raw Delta write paths.

## Relationship to WP4

WP3 is independently deliverable. It defines the schema and transformations
that WP4 later persists. WP4 issue #19 owns generic `write_table`, registration
of both `SynapseConnectivityLong` and `CellCellConnectivityLong`, canonical
overwrite policies, and migration from the temporary raw Delta writes to the
canonical table.

The reader in WP3 targets the future canonical `cellcellconnectivitylong/`
layout and is tested against synthetic Delta data. WP3 does not make the model
appear in `WRITABLE_CLASSES` or `REGISTRY`.

## Starting and repository constraints

- Repository root: `/workspaces/ConnectsCommonConnectivity`
- Current planning branch observed on 2026-09-24: `wp3-cell-conn`
- Never edit `src/connects_common_connectivity/models.py` manually.
- Make model changes under `schemas/`, then run
  `bash scripts/generate_models.sh`.
- Edit only the minimal cells described under "ETL compatibility updates";
  preserve valid notebook JSON and leave persistence migration to WP4.
- Follow `.github/instructions/changelog.instructions.md` for `CHANGELOG.md`.
- Use the repository `commit-and-push` and `write-pr-message` skills.
- Do not revert or commit unrelated user changes.

## Data semantics

`connectome_id` identifies a measurement context: segmentation version,
proofreading state, and measurement semantics. It is not a cohort identifier or
a replacement for `dataset_id`.

Cohorts are read-side filters. A connectome should be written once as the
appropriate superset; subsets are selected by endpoint IDs now and may later be
resolved from dataset associations or cluster membership by issue #23.

WP4 will persist `CellCellConnectivityLong` in canonical subdirectory
`cellcellconnectivitylong` with overwrite scope `(project_id, connectome_id)`.
WP3 defines and reads that layout but does not register its write policy.

## Work package 1: schema, reader, and derivation

### Schema

1. Add a required `connectome_id` string slot to
   `schemas/cell_cell_schema.yaml::CellCellConnectivityLong`.
2. Put the measurement-context semantics in the slot description.
3. Run `bash scripts/generate_models.sh`.
4. Run generation a second time and confirm no second diff.
5. Verify that generated `CellCellConnectivityLong` requires `connectome_id`.

### Reader

Add `read_cell_cell_connectivity` to
`src/connects_common_connectivity/io/read.py`.

Required behavior:

- require `project_id` and `connectome_id`;
- read canonical `cellcellconnectivitylong/` storage;
- optionally filter explicit presynaptic IDs;
- optionally filter explicit postsynaptic IDs;
- optionally filter measurement types;
- return an empty Polars frame for valid filters with no matches;
- raise a clear error when the canonical table is absent;
- do not resolve datasets, clusters, mappings, or memberships.

Export the reader from `src/connects_common_connectivity/io/__init__.py` and
update `tests/test_public_api.py`.

### Derivation transform

Add `derive_cell_cell_connectivity` to a clearly separated cell-cell section of
`src/connects_common_connectivity/io/write_utils.py`. Do not create another
Python module.

Required contract:

- input is a Polars DataFrame containing `project_id`, `presynaptic_cell`, and
  `postsynaptic_cell`;
- `connectome_id` and modality are explicit function arguments;
- output is a schema-shaped Polars DataFrame that WP4 can later pass to
  `write_table`;
- no Pydantic model is constructed per row;
- missing or null identity columns fail clearly;
- group by `(project_id, presynaptic_cell, postsynaptic_cell)`;
- always emit `SYNAPSE_COUNT` with unit `COUNT`;
- derive deterministic IDs from project, connectome, pre-cell, post-cell, and
  measurement type using a documented collision-resistant encoding or hash;
- enable `SUM_ANATOMICAL_SIZE` only when both `size_column` and `size_unit` are
  supplied;
- require the size column to exist and be numeric;
- define null-size behavior explicitly and cover it with tests;
- emit the caller-provided unit on size rows;
- reject either size argument when supplied alone.

Export the transform through `io/__init__.py` and update the public API test.

## Tests

Add or extend focused tests in:

- `tests/test_cell_cell.py`
- `tests/test_read.py`
- `tests/test_public_api.py`

Cover:

- generated model requires `connectome_id`;
- count aggregation for one and many synapses;
- optional size aggregation and explicit units;
- null size behavior;
- multiple projects and connectomes;
- deterministic IDs and context separation;
- missing, null, and non-numeric input columns;
- empty transform input;
- required reader scope and all filters;
- canonical table missing and empty-filter results;
- two connectome scopes can be selected independently from a synthetic
  canonical Delta table;
- `CellCellConnectivityLong` remains absent from the writable registry in WP3.

## Documentation

Update:

- `etl_example_prompt.md`
- `CHANGELOG.md`

Document:

- `connectome_id` semantics;
- the reader's expected canonical storage layout;
- the reader's explicit-ID filtering boundary;
- transform count and optional size behavior;
- that persistence and ETL adoption are deferred to WP4 #19.

Do not document dataset/cluster cohort resolution as implemented; that remains
#23.

## ETL compatibility updates

Edit only active notebooks whose filenames begin with `etl`. WP3 notebook
changes keep the existing direct Delta writes and temporary output directories;
they do not call `write_table` or depend on registry support.

### `code/etl_minnie_04_cell_cell.ipynb`

- define stable `connectome_id` values for the two existing measurement
  contexts;
- add the appropriate required `connectome_id` to every
  `CellCellConnectivityLong` row;
- retain the existing `cellcellconnectivitylong_proofread_pre_to_csm_post/` and
  `cellcellconnectivitylong_proofread_to_proofread/` destinations;
- retain the current raw `write_deltalake` persistence;
- update verification to assert each output contains the expected
  `connectome_id`.

Do not consolidate these folders in WP3. WP4 performs that migration after
`CellCellConnectivityLong` gains a validated table writer.

### `code/etl_v1dd_03_synapses.ipynb`

Change only the cell-cell derivation section:

- define a stable connectome ID for the existing measurement context;
- replace manual pair grouping and Pydantic row construction with
  `derive_cell_cell_connectivity`;
- pass the existing size column and its unit explicitly;
- retain the existing temporary cell-cell destination and raw
  `write_deltalake` call, converting the schema-shaped result to Arrow as
  needed;
- verify the derived output includes the expected connectome ID and measurement
  counts.

Do not change the source `SynapseConnectivityLong` write or wide synapse-feature
write in WP3. Those remain WP4 work.

### `code/etl_examples_readme.ipynb`

Update only descriptions made stale by the required `connectome_id` and
derivation helper. Keep the temporary output paths documented until WP4
consolidates them.

Do not edit `code/example_microns_query.ipynb`,
`code/parse_minnie_clustering.ipynb`, other non-ETL notebooks, or unrelated
cells.

## Verification

Run at minimum:

```bash
bash scripts/generate_models.sh
bash scripts/generate_models.sh
pytest tests/test_cell_cell.py tests/test_read.py tests/test_public_api.py -v
ruff check src/connects_common_connectivity/io tests/test_cell_cell.py tests/test_read.py tests/test_public_api.py
pytest -v
```

Use narrower test selectors when practical during implementation, then run the
listed final checks. Record exact outcomes; do not repeat remembered counts.

Validate notebook JSON, imports, required connectome fields, and transform API
usage. Execute the affected ETLs only when their external source data is
available. Otherwise report synthetic transform and Delta evidence separately
and do not cite stale stored outputs as a new run.

## Commit and PR boundaries

Recommended commits:

1. Schema, generated model, tests
2. reader, tests
3. transform, tests
4. minimal ETL compatibility updates
5. non-notebook docs and final verified `planning/20260820/wp3_pr_message.md`.

Use `commit-and-push` after each boundary. The PR message must map #17 and #18 to
numbered symbol-anchored changes, report only observed evidence, state that WP4
#19 owns canonical persistence migration, and identify #23 as deferred reader
work.

## Out of scope

- Implementing or changing generic `write_table` validation
- Registering `CellCellConnectivityLong` or migrating it to canonical storage
- Registering or migrating `SynapseConnectivityLong`
- Replacing raw Delta writes with `write_table`
- Dynamic wide Parquet/Zarr payloads and pointer atomicity (#20)
- Dataset/cluster cohort resolution (#23)
- Non-ETL notebooks
- Unrelated notebook cleanup
