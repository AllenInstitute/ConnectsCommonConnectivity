Adds first-class spatial coordinates, reference-space metadata, and nested-model
Arrow persistence. Review against `wp3-cell-conn`; WP9 is stacked on that work.
The V1DD production pilot remains a separate Code Ocean task.

## What changed

1. **Defined coordinate identity.** `spatial_schema.yaml::SpatialLocation`
   replaces the embedded coordinate object with a project-scoped row keyed by
   `(project_id, dataitem_id, reference_space, location_type)`.
   `LocationType` distinguishes soma, centroid, injection-site, and other points.
  Optional `description` details, especially for `OTHER`, survive JSON and
  table round-trips; they are not part of the identity key.

2. **Added reference-space metadata.** `spatial_schema.yaml::ReferenceSpace`
  identifies a frame/version by `(project_id, id)`, with null project denoting
  an independent global scope. It carries optional units, an ordered
  three-element voxel size, its physical unit, and `Default2DView`;
  `base_schema.yaml::Unit` adds nanometers, millimeters, centimeters, and voxels.

3. **Preserved nested structures.** `arrow_utils.py::_arrow_field_for`,
   `model_to_row`, `flatten_refs`, and `models_to_table` preserve embedded models
   and lists as Arrow structs. Schema-declared references still collapse to IDs;
   an embedded object's own `id` no longer causes its contents to be discarded.

4. **Registered spatial persistence.** `path_spec.py::MODEL_TABLE_PATHS` and
  `write_spec.py::REGISTRY` register `referencespace/` and `spatiallocation/`
  by their complete scoped keys. `WriteSpec.nullable_merge_on` opts the
  reference-space project key into null-safe deduplication and merging;
  global and project-owned frames with the same ID remain separate.

5. **Added filtered readers.** `read.py::read_spatial_locations` selects project,
  cell, space, and location type; `read_reference_spaces` returns all scopes
  when project is omitted, only global rows for explicit `None`, and only the
  named project otherwise. Both are exported through `io/__init__.py` without
  applying transforms or view defaults.
  `read_spatial_locations` accepts mixed `LocationType` members and strings;
  unknown filter values are ignored, and empty or unmatched selections retain
  the table schema instead of raising `KeyError`.

6. **Removed redundant fields and documented the handoff.**
   `single_cell_schema.yaml::SingleCellReconstruction.soma_location` and
   `cell_gene_schema.yaml::CellMetadata.spatial_location` are removed, with the
   aggregator updated and models regenerated. Schema, Arrow, registry, writer,
   validation, reader, and public-API tests cover the new contracts. README,
   CHANGELOG, the spatial design document, and `etl_example_prompt.md` describe
   the replacement API and future V1DD migration.

7. **Expressed write-only constraints as model subclasses.**
   `write_spec.py::WriteSpec.write_cls` replaces the `required_for_write` and
  `cross_field_rules` name lists with generated-model subclasses housed in
  `write_validation.py`: `ClusterWrite`, `ClusterMembershipWrite`,
  `CellFeatureDefinitionWrite`, `HierarchyCategoryWrite`, and
  `ReferenceSpaceWrite`. `write_validation.py::validate_for_write` validates
  every row against `spec.validation_cls`, including `model_construct` rows;
  `WriteSpec` is imported there only under `TYPE_CHECKING` to avoid a cycle.

8. **Enforced reference-space coherence before IO.**
  `write_validation.py::ReferenceSpaceWrite` rejects repeated default-view
  data axes using `DATA_AXIS_BY_SIGNED_AXIS`. Supplied voxel scale requires
  paired size/unit fields, coordinate unit `VOXELS`, finite positive dimensions,
  and one of the four physical length units; unknown scale remains allowed.
  The schema descriptions and regenerated models identify these as write-time
  guarantees, not constructor-time validation.

Deliberately retained: existing ETL notebooks and published feature matrices,
the no-schema reference-flattening heuristic, and the merge backend, extended
for nullable identity keys.
No notebook or production-data migration is included. The README also closes an
existing code fence; the ETL guide has formatting and generation-command edits
beyond its new spatial section. No unrelated runtime feature is included.

| Issue | Closed by |
|---|---|
| #25 - first-class coordinates; pilot still pending | 1, 2, 4, 5, 6; do not auto-close |
| Closes #27 - nested models lose structure through Arrow | 3, 6 |
| Closes #42 - reference-space default display axes | 2, 3, 4, 5, 6, 7, 8 |
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

**Validation coverage (review follow-up).** Rows without write-required slots
could bypass validation, and documented voxel-scale constraints were not
enforced. Changes 7 and 8 validate every row before IO and keep row rules in
the validation module while the spec selects them. Voxel coordinates with
unknown physical scale remain valid; partially supplied or invalid scale does
not.

**Filter compatibility (review follow-up).** Mixed enum/string selections could
fail during Polars filter construction. Change 5 normalizes filter values to
strings so known values still match when unknown values are also requested.

## How to test

```bash
uv run pytest -q --tb=short
# 423 passed, 12 skipped

uv run ruff check \
  src/connects_common_connectivity/io/{arrow_utils,path_spec,write_spec,write_validation,read,__init__}.py \
  tests/test_{arrow_utils,spatial_schema,write_spec,write_validation,writers,read,public_api}.py \
  --output-format concise
# Failed: 9 findings in arrow_utils.py (1 I001, 8 E501).
# No other findings in this selection.

git show wp3-cell-conn:src/connects_common_connectivity/io/arrow_utils.py | \
  uv run ruff check \
    --stdin-filename src/connects_common_connectivity/io/arrow_utils.py \
    --output-format concise -
# Failed: the same 9 findings on the base branch.
```

Diff and assertion review covered coordinate-key isolation, unchanged reruns,
view updates and null transitions, struct/reference distinctions, typed empty
reads, mixed enum/string filters, optional location descriptions,
global/project scope isolation, and voxel-scale validation. The scale
tests include nonfinite and nonpositive dimensions, incomplete metadata,
invalid unit combinations, and constructed rows. The cell-gene fixture verifies
identifier compatibility, not referential integrity in a real Zarr dataset.

The agreed V1DD streamline view is `PLUS_X` / `MINUS_Y`. Original EM units and
orientation still require source verification; its default view stays unset.
No external dataset, production seeding, or notebook execution supplies evidence
for this draft.

> Results were obtained on 2026-10-06 from the worktree based on `356b9fe`,
> including uncommitted `SpatialLocation.description`, enum descriptions,
> regenerated models, round-trip tests, and an Arrow comment correction.
> These results do not describe pushed HEAD alone. Rerun on the final committed
> revision before merge; production-data validation remains pending.

## Reviewer focus (optional)

- `SpatialLocation.location_identity`: one point per type/space/project, with
  no synthetic row ID; duplicate incoming keys follow last-row-wins merge rules.
- `ReferenceSpace` identity: `(project_id, id)` with independent global scope;
  `WriteSpec.nullable_merge_on` permits null-safe matching only for opted-in
  keys. Check that global updates cannot overwrite same-ID project frames.
- `write_validation.py::ReferenceSpaceWrite`: voxel scale is optional, but
  supplied scale requires paired fields, physical units, and finite positive
  dimensions. Direct generated-model construction does not enforce these rules.
- `arrow_utils.py::_flatten_typed_value`: preserve embedded IDs and recursive
  structures while retaining explicit-schema and legacy reference behavior.
- `write_validation.py::validate_for_write`: every row is re-validated, adding one
  `model_dump` and one validation pass per row for classes that previously
  skipped it. Confirm the cost is acceptable for large batches; these checks
  do not enforce foreign keys.
- `write_validation.py::DATA_AXIS_BY_SIGNED_AXIS`: new `SignedAxis` members
  require a map entry; the tests assert full enum coverage.
- `read_reference_spaces`: distinguish omitted project from explicit `None`;
  project filtering has no global fallback. A null view supplies no default.
- `etl_example_prompt.md` V1DD handoff: #25 still owns real-data validation and
  the coordinate/volume feature split in Code Ocean; #12 and #26 remain deferred.