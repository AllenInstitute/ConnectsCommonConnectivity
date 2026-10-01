# Design: the "what is up" problem — default display axes for spatial coordinates

| | |
|---|---|
| Status | Draft for review (simplified 2026-10-01) |
| Milestone | [WP9 spatial + embeddings](https://github.com/AllenInstitute/ConnectsCommonConnectivity/milestone/9) |
| Related issues | #25 (SpatialLocation → coordinates table), #27 (nested models don't round-trip) |
| New issue | "what is up" — number TBD; should block #25 |
| Owner | YY |

## 1. Summary

Coordinates in the lake don't say how to display them, so downstream plots come out flipped. Example: V1DD soma y increases from pia to white matter (image convention, y down); a plot assuming y up put white matter at the top.

Fix: one new flat table, **`ReferenceSpace`**, in a new **`schemas/spatial_schema.yaml`**.

- A `ReferenceSpace` row is a coordinate frame *and version* (e.g. `CCF_v3`, `CCF_v4` are separate rows). Every `SpatialLocation` / #25 coordinate points at one via `reference_space`.
- It holds properties shared by all coordinates in that space, so they are stored once, not per DataItem. Today that is only an **optional default 2D view**: which signed data axis runs left → right and bottom → top on screen. Later it can hold more (e.g. which axis is dorsal/ventral).

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
    slots: [id, name, description, default_left_to_right, default_bottom_to_top]
    slot_usage:
      id: {required: true}
      # default_left_to_right / default_bottom_to_top: optional, set both or neither

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
  default_left_to_right: {range: SignedAxis, description: Default 2D view — data axis that increases left → right on screen.}
  default_bottom_to_top: {range: SignedAxis, description: Default 2D view — data axis that increases bottom → top on screen.}
```

The default 2D view is two flat columns with a shared `default_` prefix rather than a nested object, because nested objects don't round-trip yet (#27).

To plot: use the two named axes, negating any `MINUS_` axis. The unused axis is depth. If a space has no default view, the reader returns none and the consumer decides; there is no implicit fallback.

Write-time rules (row-local):

- `default_left_to_right` and `default_bottom_to_top` are both set or both null.
- If set, they use different data axes.

### Seed rows

| id | left → right | bottom → top | result |
|---|---|---|---|
| `CCF_v3` | `PLUS_Z` | `MINUS_Y` | coronal, dorsal up |
| `v1dd_streamline` | `PLUS_X` | `MINUS_Y` | pia up |
| `v1dd_em` | TBD | TBD | confirm with V1DD owners |

## 3. Changes

| File | Change |
|---|---|
| `schemas/spatial_schema.yaml` (new) | §2. |
| `schemas/core_schema.yaml` | Remove `SpatialLocation`, `x`, `y`, `z`, `reference_space`. |
| `schemas/single_cell_schema.yaml`, `cell_gene_schema.yaml`, `connectivity_schema.yaml` | Add `spatial_schema` to `imports:`. |
| `src/connects_common_connectivity/models.py` | Regenerate via `scripts/generate_models.sh`. |
| `io/path_spec.py`, `io/write_spec.py` | Register `ReferenceSpace` (`partition_by=[]`, `merge_on=["id"]`, project-scoped). |
| `io/write_validation.py` | Default view: both-or-neither, and different axes. |
| `io/read.py` | `read_reference_spaces()`. |
| `code/etl_reference_spaces.ipynb` (new), `code/etl_v1dd_02_cave.ipynb` | Seed rows above. |
| `tests/` | Round-trip with and without a default view; half-set and same-axis rejection. |
| `CHANGELOG.md`, `README.md` | Short entries. |

Blocks #25 (the coordinates table also goes in `spatial_schema.yaml`). Does not depend on #27.

## 4. Open questions

- **Q1.** V1DD streamline: confirm y increases toward white matter (pia at 0).
- **Q2.** V1DD EM coords: units (nm vs. voxel) and default axes.

## 5. Deferred

Add only when a real need appears, as new optional `ReferenceSpace` columns where possible: per-axis anatomical direction fields such as `x_increases_toward` (dorsal/ventral etc., for axis labels / computing coronal, sagittal), units, a nested `Default2DView` object (once #27 is fixed), multiple stored views or per-dataset overrides, handedness / Neuroglancer link generation, transforms between spaces.
