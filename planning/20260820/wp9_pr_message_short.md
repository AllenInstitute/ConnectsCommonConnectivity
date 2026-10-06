Adds first-class spatial coordinates, explicit default 2D views for reference
spaces, and nested-model Arrow persistence. Review against `wp3-cell-conn`;
this branch is stacked on that work.

## What changed

1. **`schemas/spatial_schema.yaml::SpatialLocation`** replaces embedded
   coordinate objects with a project-scoped row keyed by
   `(project_id, dataitem_id, reference_space, location_type)`. `LocationType`
   distinguishes soma, centroid, injection site, and other points; the optional
   `description` round-trips but is not part of the key.

2. **`schemas/spatial_schema.yaml::ReferenceSpace`** identifies a frame and
   version by `(project_id, id)`, where a null `project_id` denotes global
   scope. It carries optional units and an ordered three-element voxel size
   with its physical unit. `schemas/base_schema.yaml::Unit` gains nanometers,
   millimeters, centimeters, and voxels.

3. **`schemas/spatial_schema.yaml::Default2DView`**, embedded as
   `ReferenceSpace.default_2d_view`, records `left_to_right` and
   `bottom_to_top` as `SignedAxis` values. Consumers negate `MINUS_*` axes for
   display; the two axes must differ and the unused axis is depth. A null view
   leaves the choice to the consumer. Closes #42.

4. **`io/arrow_utils.py`** (`_arrow_field_for`, `model_to_row`, `flatten_refs`,
   `models_to_table`) preserves embedded models and lists as Arrow structs.
   Schema-declared references still collapse to IDs, but an embedded object's
   own `id` no longer causes its contents to be discarded. Closes #27.

5. **`io/path_spec.py::MODEL_TABLE_PATHS`** and **`io/write_spec.py::REGISTRY`**
   register `referencespace/` and `spatiallocation/` by their full scoped keys.
   `WriteSpec.nullable_merge_on` opts the reference-space project key into
   null-safe deduplication, so global and project-owned frames with the same
   `id` stay separate.

6. **`io/read.py`** adds `read_spatial_locations` (filters on project, cell,
   reference space, and location type) and `read_reference_spaces` (all scopes
   when `project` is omitted, global rows only for explicit `None`, the named
   project otherwise). Both are exported from `io/__init__.py`. Filter values
   accept `LocationType` members and strings; unknown values are ignored and
   empty results keep the table schema instead of raising `KeyError`.

7. **Removed the redundant fields.**
   `schemas/single_cell_schema.yaml::SingleCellReconstruction.soma_location`
   and `schemas/cell_gene_schema.yaml::CellMetadata.spatial_location` are gone,
   with the aggregator updated and `models.py` regenerated. Closes #43.

8. **`io/write_spec.py::WriteSpec.write_cls`** replaces the `required_for_write`
   and `cross_field_rules` name lists with generated-model subclasses in
   `io/write_validation.py`: `ClusterWrite`, `ClusterMembershipWrite`,
   `CellFeatureDefinitionWrite`, `HierarchyCategoryWrite`, and
   `ReferenceSpaceWrite`. `validate_for_write` now validates every row,
   including `model_construct` rows.

9. **`io/write_validation.py::ReferenceSpaceWrite`** rejects default views that
   repeat a data axis (via `DATA_AXIS_BY_SIGNED_AXIS`), and requires supplied
   voxel scale to pair size and unit, use coordinate unit `VOXELS`, be finite
   and positive, and name one of the four physical length units. Unknown scale
   remains allowed. These are write-time guarantees, not constructor-time
   validation.

| Issue | Closed by |
|---|---|
| Closes #25 — first-class coordinates (production pilot still pending) | 1, 2, 5, 6, 7 |
| Closes #27 — nested models lose structure through Arrow | 4, 7 |
| Closes #42 — explicit default 2D views | 2, 3, 4, 5, 6, 9 |
| Closes #43 — redundant spatial fields, missing location type | 1, 7 |

## Why

**#25, #43.** Coordinates stored inside generic feature matrices cannot say
which frame or anatomical point they describe. Items 1, 2, and 5 give them
explicit identity and writable storage; item 7 removes the duplicated embedded
fields.

**#42, #27.** Plots silently invert anatomy when they assume y-up. `Default2DView`
records the plotting plane and screen directions, and item 4 lets that nested
structure survive Arrow instead of working around the structure-loss bug with
flattened coordinate rows.

**Review follow-ups.** Rows missing write-required slots previously bypassed
validation, and the documented voxel-scale constraints were not enforced
(items 8 and 9). Mixed enum/string filter values could fail during Polars
filter construction, so `read.py` now normalizes them to strings (item 6).

**Out of scope.** No notebook or production-data migration, no plotting, and no
foreign-key enforcement or transforms (#12, #26 stay deferred). #25 remains
open for the production pilot described in `etl_example_prompt.md`.

## How to test

```bash
uv run pytest -q --tb=short
# 423 passed, 12 skipped

uv run ruff check \
  src/connects_common_connectivity/io/{arrow_utils,path_spec,write_spec,write_validation,read,__init__}.py \
  tests/test_{arrow_utils,spatial_schema,write_spec,write_validation,writers,read,public_api}.py \
  --output-format concise
# Failed: 9 findings in arrow_utils.py (1 I001, 8 E501); none elsewhere.

git show wp3-cell-conn:src/connects_common_connectivity/io/arrow_utils.py | \
  uv run ruff check \
    --stdin-filename src/connects_common_connectivity/io/arrow_utils.py \
    --output-format concise -
# Failed: the same 9 findings already exist on the base branch.
```

Tests cover coordinate-key isolation, idempotent reruns, default-view updates
and null transitions, struct/reference distinctions, typed empty reads, mixed
enum/string filters, optional `SpatialLocation.description`, global/project
scope isolation, and voxel-scale validation (nonfinite and nonpositive
dimensions, incomplete metadata, invalid unit combinations, constructed rows).

> Results are from the worktree based on `356b9fe` on 2026-10-06, including
> uncommitted changes. Rerun on the final committed revision before merge.
> No external dataset or production seeding supplies evidence here.

## Reviewer focus (optional)

- `SpatialLocation`: one row per `(project_id, dataitem_id, reference_space,
  location_type)`, no synthetic row ID; duplicate incoming keys follow
  last-row-wins merge rules.
- `ReferenceSpace` identity and `WriteSpec.nullable_merge_on`: confirm a global
  update cannot overwrite a project-owned frame with the same `id`.
- `write_validation.py::ReferenceSpaceWrite`: voxel scale is optional, but a
  supplied scale requires paired fields, physical units, and finite positive
  dimensions. Constructing the generated model directly bypasses these rules.
- `arrow_utils.py::_flatten_typed_value`: preserves embedded IDs and recursive
  structures while keeping explicit-schema and legacy reference behavior.
- `write_validation.py::validate_for_write`: adds one `model_dump` and one
  validation pass per row for classes that previously skipped it — confirm the
  cost is acceptable for large batches.
- `Default2DView`: screen directions must use distinct data axes, not just
  different signs. New `SignedAxis` members need an entry in
  `DATA_AXIS_BY_SIGNED_AXIS`; a test asserts full coverage.
- `read_reference_spaces`: omitted `project` and explicit `None` mean different
  things, and project filtering has no global fallback.
