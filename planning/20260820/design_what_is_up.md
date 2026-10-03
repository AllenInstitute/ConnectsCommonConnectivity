# Design: the "what is up" problem — default display axes for spatial coordinates

| | |
|---|---|
| Status | Draft for review (simplified 2026-10-01) |
| Issue | [#42](https://github.com/AllenInstitute/ConnectsCommonConnectivity/issues/42) |
| Milestone | [WP9 spatial + embeddings](https://github.com/AllenInstitute/ConnectsCommonConnectivity/milestone/9) |
| Depends on | [#27](https://github.com/AllenInstitute/ConnectsCommonConnectivity/issues/27) (nested models must round-trip — needed for `Default2DView`) |
| Soft blocks | [#25](https://github.com/AllenInstitute/ConnectsCommonConnectivity/issues/25) (SpatialLocation → coordinates table) |
| Owner | YY |

## 1. Summary

Implementation update (2026-10-02): this draft is implemented together with #25,
#27, and #43. The authoritative schema is `schemas/spatial_schema.yaml`.
`SpatialLocation` is now a project-scoped table keyed by
`(project_id, dataitem_id, reference_space, location_type)`; its former embedded
attachments are removed. `ReferenceSpace` locally overrides `project_id` and
`name` to optional, while its ID remains globally unique. The sketch below
records the original #42 design rather than the full combined schema.

V1DD streamline `PLUS_X` / `MINUS_Y` is confirmed. Original EM values are preserved,
with units awaiting verification in Code Ocean and the default view unset.
The existing ETL notebooks remain unchanged; `etl_example_prompt.md` carries
the future targeted migration and verification instructions. Full ETL execution,
published-data backfills, and typed unit/transform metadata remain out of scope.

Coordinates in the lake don't say how to display them, so downstream plots come out flipped. Example: V1DD soma y increases from pia to white matter (image convention, y down); a plot assuming y up put white matter at the top.

Fix: one new table, **`ReferenceSpace`**, in a new **`schemas/spatial_schema.yaml`**.

- A `ReferenceSpace` row is a coordinate frame *and version* (e.g. `CCF_v3`, `CCF_v4` are separate rows). Every `SpatialLocation` / #25 coordinate points at one via `reference_space`.
- It holds properties shared by all coordinates in that space, so they are stored once, not per DataItem. Today that is only an **optional `default_2d_view`**: a nested `Default2DView` object saying which signed data axis runs left → right and bottom → top on screen. Later it can hold more (e.g. which axis is dorsal/ventral).

## 2. Schema — `schemas/spatial_schema.yaml` (new)

Spatial definitions live in their own file, separate from core. `SpatialLocation` and slots `x`, `y`, `z`, `reference_space` move here from `core_schema.yaml`.

```yaml
id: https://brain-connects.org/ic3-spatial-schema
name: spatial_schema
description: Spatial reference spaces and coordinates.
imports:
  - base_schema
  - core_schema          # for dataitem_id in the #25 table; core does not import spatial
prefixes:
  linkml: https://w3id.org/linkml/
  cc: https://brain-connects.org/cc/
default_prefix: cc

enums:
  SignedAxis:
    description: A data axis and whether it increases (PLUS) or decreases (MINUS) along a screen direction.
    permissible_values: [PLUS_X, MINUS_X, PLUS_Y, MINUS_Y, PLUS_Z, MINUS_Z]

classes:
  ReferenceSpace:
    description: >-
      A coordinate frame (and version) that SpatialLocations refer to. Holds
      properties shared by all coordinates in the space, such as an optional
      default 2D view. Use description to say in words what the axes mean
      (e.g. "y increases toward white matter, pia at 0").
    mixins: [ProjectScoped]          # project_id null => global space (e.g. CCF_v3)
    slots: [id, name, description, default_2d_view]
    slot_usage:
      id: {required: true}

  Default2DView:
    description: >-
      Default screen layout for coordinates in a reference space. Embedded in
      ReferenceSpace (no id, not its own table). The unused data axis is depth.
    slots: [left_to_right, bottom_to_top]
    slot_usage:
      left_to_right: {required: true}
      bottom_to_top: {required: true}

  SpatialLocation:                    # moved from core_schema; only reference_space range changes
    description: 3D spatial coordinates in a reference space.
    slots: [x, y, z, reference_space]
    slot_usage:
      x: {required: true}
      y: {required: true}
      z: {required: true}
      reference_space: {required: true}

slots:
  x: {range: float, description: X coordinate in the reference space.}
  y: {range: float, description: Y coordinate in the reference space.}
  z: {range: float, description: Z coordinate in the reference space.}
  reference_space:
    range: ReferenceSpace             # was string; pydantic still sees a str id
    description: Id of the ReferenceSpace these coordinates are in (e.g. CCF_v3).
  default_2d_view:
    range: Default2DView
    inlined: true                     # stored inside the ReferenceSpace row
    required: false
    description: Optional default 2D view for this space.
  left_to_right: {range: SignedAxis, description: Data axis that increases left → right on screen.}
  bottom_to_top: {range: SignedAxis, description: Data axis that increases bottom → top on screen.}
```

Generated pydantic:

```python
class Default2DView(ConfiguredBaseModel):
    left_to_right: SignedAxis
    bottom_to_top: SignedAxis

class ReferenceSpace(ProjectScoped):
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    default_2d_view: Optional[Default2DView] = None
```

On disk (after #27): `default_2d_view` is a Parquet struct column `struct<left_to_right: string, bottom_to_top: string>`, null when no view is set. Read with `pl.col("default_2d_view").struct.field("left_to_right")` or `.unnest("default_2d_view")`.

To plot: use the two named axes, negating any `MINUS_` axis. If a space has no default view, the reader returns none and the consumer decides; there is no implicit fallback.

Rules:

- Both fields set or neither — enforced by the schema (object optional, its fields required).
- `left_to_right` and `bottom_to_top` use different data axes — write-time check, row-local.

### Seed rows

| id | left → right | bottom → top | result |
|---|---|---|---|
| `CCF_v3` | `PLUS_Z` | `MINUS_Y` | coronal, dorsal up |
| `v1dd_streamline` | `PLUS_X` | `MINUS_Y` | pia up |

## 3. Dependency on #27

Today the arrow layer (`io/arrow_utils.py`) can't store `Default2DView`:

1. `build_arrow_schema` maps any model-typed field to a `string` column.
2. `models_to_table` calls `str()` on the dict, writing a Python repr blob (`"{'left_to_right': 'PLUS_Z', ...}"`), not queryable and not read back as an object.
3. `flatten_refs` collapses any nested dict with an `id` key to that id, so a nested object must never gain an `id` field unless it is a reference.

#27 fix (option a, struct columns) needed for this design:

- `_arrow_field_for`: model-typed field → `pa.struct(...)` built recursively from the model's fields.
- `models_to_table`: pass dicts through for struct fields instead of `str()`.
- `flatten_refs`: don't collapse fields the schema declares as structs.
- Round-trip test: model → parquet → polars struct → model.

Combined with #43, `SingleCellReconstruction.soma_location` and
`CellMetadata.spatial_location` are removed instead of becoming struct columns.
No nested-field data migration is needed: per #25, no ETL has ever written them.

## 4. Changes

| File | Change |
|---|---|
| `io/arrow_utils.py` + tests | #27 struct-column support (§3). Lands first. |
| `schemas/spatial_schema.yaml` (new) | §2. |
| `schemas/core_schema.yaml` | Remove `SpatialLocation`, `x`, `y`, `z`, `reference_space`. |
| `schemas/single_cell_schema.yaml`, `cell_gene_schema.yaml` | Remove the nested spatial fields and unused slots (#43). |
| `schemas/connectivity_schema.yaml` | Import `spatial_schema`. |
| `src/connects_common_connectivity/models.py` | Regenerate via `scripts/generate_models.sh`. |
| `io/path_spec.py`, `io/write_spec.py` | Register `ReferenceSpace` (unpartitioned, merged by globally unique ID) and project-partitioned `SpatialLocation` (complete coordinate key). `Default2DView` is embedded. |
| `io/write_validation.py` | Default view uses two different data axes. |
| `io/read.py` | `read_reference_spaces()` and `read_spatial_locations()`. |
| `etl_example_prompt.md` | Reference-space ownership, spatial writes, and future V1DD Code Ocean pilot; no new seeding notebook or local ETL execution. |
| `tests/` | Round-trip with and without a default view; same-axis rejection. |
| `CHANGELOG.md`, `README.md` | Short entries. |

Order: #27 → this (#42) → #25 (the coordinates table also goes in `spatial_schema.yaml`).

## 5. Open questions

- **Q1 resolved (2026-10-02).** V1DD streamline y increases toward white matter; use `PLUS_X` / `MINUS_Y`.
- **Q2 deferred to Code Ocean.** Verify original EM units from the source contract. Preserve values and leave default axes unset until confirmed.

## 6. Deferred

Add only when a real need appears, as new optional `ReferenceSpace` fields where possible: per-axis anatomical direction fields such as `x_increases_toward` (dorsal/ventral etc., for axis labels / computing coronal, sagittal), units, multiple stored views (coronal, saggital, horizontal, laminar) or per-dataset overrides, handedness (right handed left handed coordinate system), Neuroglancer link generation, transforms between spaces.
