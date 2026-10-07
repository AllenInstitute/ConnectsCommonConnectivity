Adds project-scoped 2D embeddings for generic DataItems, separate from
anatomical coordinates, with metadata and coordinate tables writable through
`write_models` and exposed through public readers. Review against `wp9-spatial`;
this is the second WP9 increment.

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

3. **Added project-scoped readers.** `read.py::read_embedding_locations` filters
   coordinates by embedding-space and DataItem IDs; `read_embedding_spaces`
   filters metadata by space IDs. Both require a project ID, preserve stored
   values and empty-result schemas, and are exported through `io/__init__.py`.

4. **Covered the contracts and documented availability.**
   `test_embedding_schema.py::test_embedding_metadata_json_round_trip` and its
   neighboring tests check model requirements, optional metadata,
   dates, reference IDs, and the 2D-only shape. `test_write_spec.py` guards
   paths and identities; `test_writers.py` covers round trips, null transitions,
   idempotency, deduplication, and isolation. `test_write_validation.py` checks
   malformed constructed rows fail before IO. `test_read.py` and
   `test_public_api.py` cover filters, project isolation, root selection,
   metadata round trips, missing storage, and exports. `CHANGELOG.md` records
   the writable models and readers; this document supplies the review handoff.

Deliberately unchanged: spatial behavior, the shared merge backend, and
ETL notebooks. The generated-model diff includes class reordering and refreshed
LinkML metadata from generation, not manual model edits. The branch also adds
a test-docstring requirement to `.github/copilot-instructions.md` and one-line
behavioral docstrings to the new tests; this does not change runtime behavior.

| Issue | Addressed by |
|---|---|
| Related to #26 - embedding coordinates distinct from spatial | 1, 2, 3, 4 |

## Why

**Coordinate meaning (#26).** Computed embeddings must not be mistaken for
anatomical positions. Change 1 gives them their own coordinate space and
provenance, without restricting DataItems to cells. Change 2 makes repeated
ETL contributions update only matching identities.

**Class-local coordinates.** In change 1,
`embedding_schema.yaml::EmbeddingLocation` defines x/y as `attributes` because
embedding positions and anatomical positions have different meanings despite
sharing field names. This keeps embedding descriptions and constraints local
to the class instead of reusing the spatial schema's shared x/y slots.

**Scope limit.** This implements schema, writes, and reads, not embedding
computation. Feature-set references are ID strings with a
same-project contract; existence and project consistency are not checked against
stored records. Arbitrary dimensionality is deferred to a separate follow-up;
its issue number is not recorded here. No automatic closure of #26 is requested.

## How to test

```bash
uv run pytest tests/test_embedding_schema.py tests/test_write_spec.py tests/test_writers.py tests/test_write_validation.py tests/test_spatial_schema.py tests/test_arrow_utils.py tests/test_read.py tests/test_public_api.py -q
# 359 passed, 14 skipped

uv run ruff check src/connects_common_connectivity/io/read.py src/connects_common_connectivity/io/__init__.py tests/test_read.py tests/test_public_api.py
# All checks passed; existing top-level lint-configuration deprecation warning.
```

Diff review used `wp9-spatial` as the base and included the current reader and
export changes. Assertions check date/null round trips, complete identity
isolation, unchanged reruns, last-row-wins duplicates, pre-IO rejection,
composable filters, typed empty results, and configured or explicit roots.

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
- `read_embedding_spaces` and `read_embedding_locations`: required project ID,
  no cross-project fallback, no reference joins, and no coordinate transforms.