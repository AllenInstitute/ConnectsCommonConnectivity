Adds project-scoped 2D embeddings for generic DataItems, separate from
anatomical coordinates, with metadata and coordinate tables writable through
`write_models`. Review against `wp9-spatial`; this is the second WP9 increment.

## What changed

1. **Defined embedding metadata and coordinates.**
   `embedding_schema.yaml::EmbeddingSpace`, `EmbeddingLocation`, and
   `EmbeddingMethod` describe one embedding run and its DataItem x/y positions.
   Both classes require project scope; metadata includes optional parameters,
   a CellFeatureSet ID, input-feature description, and creation date.
   `connectivity_schema.yaml::imports` includes the module, and the generated
   models expose these classes and the UMAP/TSNE/PCA/MDS/OTHER vocabulary.

2. **Enabled embedding writes.** `path_spec.py::MODEL_TABLE_PATHS` and
   `write_spec.py::REGISTRY` register `embeddingspace/` and `embeddinglocation/`,
   partitioned by project. Metadata merges on `(project_id, id)`; locations
   merge on `(project_id, dataitem_id, embedding_space)`.

3. **Covered the contracts and documented availability.**
   `test_embedding_schema.py` checks model requirements, optional metadata,
   dates, reference IDs, and the 2D-only shape. `test_write_spec.py` guards
   paths and identities; `test_writers.py` covers round trips, null transitions,
   idempotency, deduplication, and isolation. `test_write_validation.py` checks
   malformed constructed rows fail before IO. `CHANGELOG.md` records the new
   writable models; this planning document supplies the review handoff.

Deliberately unchanged: spatial models, the shared merge backend, readers, and
ETL notebooks. The generated-model diff includes class reordering and refreshed
LinkML metadata from generation, not manual model edits. No unrelated files
are included in this scope.

| Issue | Addressed by |
|---|---|
| Related to #26 - embedding coordinates distinct from spatial | 1, 2, 3 |

## Why

**Coordinate meaning (#26).** Computed embeddings must not be mistaken for
anatomical positions. Change 1 gives them their own coordinate space and
provenance, without restricting DataItems to cells. Change 2 makes repeated
ETL contributions update only matching identities.

**Scope limit.** This implements schema and write support, not embedding
computation or dedicated readers. Feature-set references are ID strings with a
same-project contract; existence and project consistency are not checked against
stored records. Arbitrary dimensionality is deferred to a separate follow-up;
its issue number is not recorded here. No automatic closure of #26 is requested.

## How to test

```bash
uv run pytest tests/test_embedding_schema.py tests/test_write_spec.py tests/test_writers.py tests/test_write_validation.py tests/test_spatial_schema.py tests/test_arrow_utils.py -q
# 290 passed, 14 skipped

uv run ruff check src/connects_common_connectivity/io/path_spec.py src/connects_common_connectivity/io/write_spec.py tests/test_embedding_schema.py tests/test_write_spec.py tests/test_writers.py tests/test_write_validation.py
# All checks passed; existing top-level lint-configuration deprecation warning.
```

Diff review used `wp9-spatial` as the base and included the current writer and
test changes. Assertions check date/null round trips, complete identity
isolation, unchanged reruns, last-row-wins duplicates, and pre-IO rejection.

> These are synthetic local Delta tests, not a production ETL run. The full
> repository suite was not run; foreign-key existence is not validated.

## Reviewer focus (optional)

- `EmbeddingLocation`: required x/y only; arbitrary dimensions remain deferred
  as described under Scope limit.
- `EmbeddingSpace`: required project scope, unlike global ReferenceSpace rows.
- `input_feature_set_id`: optional CellFeatureSet ID in the same project, not
  an embedded object or a verified foreign key.
- `input_features_description`: optional and usable with or without an ID.
- `REGISTRY`: complete merge keys and project partitions; omitted rows remain.
- `read.py`: dedicated embedding readers remain a later wiring step for #26.