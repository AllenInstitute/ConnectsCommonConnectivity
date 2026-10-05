Adds first-class spatial coordinates, reference-space metadata, and nested-model
Arrow persistence. Review against `wp3-cell-conn`; WP9 is stacked on that work.
The V1DD production pilot remains a separate Code Ocean task.

## What changed

1. **Defined coordinate identity.** `spatial_schema.yaml::SpatialLocation`
   replaces the embedded coordinate object with a project-scoped row keyed by
   `(project_id, dataitem_id, reference_space, location_type)`.
   `LocationType` distinguishes soma, centroid, injection-site, and other points.

2. **Added reference-space metadata.** `spatial_schema.yaml::ReferenceSpace`
   holds a globally unique frame/version ID, optional project ownership, units,
   an ordered three-element voxel size, its physical unit, and an optional
   `Default2DView`. `base_schema.yaml::Unit` adds nanometers, millimeters,
   centimeters, and voxels; `SignedAxis` defines screen directions.

3. **Preserved nested structures.** `arrow_utils.py::_arrow_field_for`,
   `model_to_row`, `flatten_refs`, and `models_to_table` preserve embedded models
   and lists as Arrow structs. Schema-declared references still collapse to IDs;
   an embedded object's own `id` no longer causes its contents to be discarded.

4. **Registered spatial persistence.** `path_spec.py::MODEL_TABLE_PATHS` and
   `write_spec.py::REGISTRY` register `referencespace/` by global ID and
   `spatiallocation/` by the complete project-scoped key.
   `write_validation.py::validate_for_write` rejects default views that reuse
   the same data axis, regardless of sign, before IO.

5. **Added filtered readers.** `read.py::read_spatial_locations` selects project,
   cell, space, and location type; `read_reference_spaces` includes global spaces
   when filtering by project. Both are exported through `io/__init__.py` and
   preserve stored values and types without applying transforms or view defaults.

6. **Removed redundant fields and documented the handoff.**
   `single_cell_schema.yaml::SingleCellReconstruction.soma_location` and
   `cell_gene_schema.yaml::CellMetadata.spatial_location` are removed, with the
   aggregator updated and models regenerated. Schema, Arrow, registry, writer,
   validation, reader, and public-API tests cover the new contracts. README,
   CHANGELOG, the spatial design document, and `etl_example_prompt.md` describe
   the replacement API and future V1DD migration.

Deliberately retained: existing ETL notebooks and published feature matrices,
the no-schema reference-flattening heuristic, and the existing merge backend.
No notebook or production-data migration is included. The README also closes an
existing code fence; the ETL guide has formatting and generation-command edits
beyond its new spatial section. No unrelated runtime feature is included.

| Issue | Closed by |
|---|---|
| #25 - first-class coordinates; pilot still pending | 1, 2, 4, 5, 6; do not auto-close |
| Closes #27 - nested models lose structure through Arrow | 3, 6 |
| Closes #42 - reference-space default display axes | 2, 3, 4, 5, 6 |
| Closes #43 - redundant spatial fields and missing location type | 1, 6 |

## Why

**Spatial meaning (#25, #43).** Coordinates hidden in generic feature matrices
cannot identify their frame or anatomical point. Changes 1, 2, and 4 give them
explicit identity and writable storage; change 6 removes the duplicate embedded
fields. `CellGeneData.cell_index` already declares DataItem references, and a
synthetic test uses those same IDs for coordinates.

**Orientation and persistence (#42, #27).** V1DD plots can invert the cortex
when they assume y-up. Change 2 records display directions without altering
stored coordinates. Its nested view requires change 3: flattening spatial rows
alone would leave the Arrow structure-loss bug unresolved.

**Scope limit.** #42's closure covers metadata, persistence, and validation, not
production seeding or plotting. #25 remains open for the V1DD pilot described in
`etl_example_prompt.md`, including source-unit verification and preservation of
nucleus-volume features. #12 and #26, foreign-key enforcement, transforms, and
published-data backfills are outside this change.

## How to test

```bash
uv run pytest -q --tb=short
# 346 passed

uv run ruff check \
  src/connects_common_connectivity/io/{arrow_utils,path_spec,write_spec,write_validation,read,__init__}.py \
  tests/test_{arrow_utils,spatial_schema,write_spec,write_validation,writers,read,public_api}.py \
  --output-format concise
# Failed: 9 findings in arrow_utils.py (1 I001, 8 E501).
# The same findings occur on wp3-cell-conn; no others in this selection.
```

Diff and assertion review covered coordinate-key isolation, unchanged reruns,
view updates and null transitions, struct/reference distinctions, typed empty
reads, and voxel-cardinality checks. The cell-gene fixture verifies identifier
compatibility, not referential integrity in a real Zarr dataset.

The agreed V1DD streamline view is `PLUS_X` / `MINUS_Y`. Original EM units and
orientation still require source verification; its default view stays unset.
No external dataset, production seeding, or notebook execution supplies evidence
for this draft.

> Results were obtained on 2026-10-03 from the current worktree, not solely the
> pushed HEAD (`85ae02d`). The Arrow implementation, its untracked test file,
> public-API expectations, and documentation edits still need committing.
> Rerun verification on the final committed revision before merge.

## Reviewer focus (optional)

- `SpatialLocation.location_identity`: one point per type/space/project, with
  no synthetic row ID; duplicate incoming keys follow last-row-wins merge rules.
- `ReferenceSpace.id`: globally unique even for project-owned spaces; project
  ownership is metadata, not a separate identity or access-control boundary.
- `ReferenceSpace.voxel_size`: cardinality is tested, but positivity, paired
  size/unit fields, and allowed unit combinations are not enforced by the
  current writer. Decide whether these need validation before merge.
- `arrow_utils.py::_flatten_typed_value`: preserve embedded IDs and recursive
  structures while retaining explicit-schema and legacy reference behavior.
- `validate_for_write`: reject repeated axes even when Pydantic construction
  was bypassed; do not mistake this check for general foreign-key validation.
- `read_reference_spaces`: project filtering includes global spaces; a null
  view must not become an implicit x/y or y-up fallback.
- `etl_example_prompt.md` V1DD handoff: #25 still owns real-data validation and
  the coordinate/volume feature split in Code Ocean; #12 and #26 remain deferred.