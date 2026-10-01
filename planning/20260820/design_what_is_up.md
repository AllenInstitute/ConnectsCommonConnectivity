# Design: the "what is up" problem — axis semantics and default views for spatial coordinates

| | |
|---|---|
| Status | Draft for review |
| Milestone | [WP9 spatial + embeddings](https://github.com/AllenInstitute/ConnectsCommonConnectivity/milestone/9) |
| Related issues | #25 (SpatialLocation → coordinates table), #27 (nested models don't round-trip), #12 (BrainRegionAssociation) |
| New issue | "what is up" — number TBD; should block #25 |
| Owner | YY |

## 1. Summary

Coordinates in the lake carry no statement of what direction each axis points or how they should be displayed. Downstream users guess, and guess differently. This design adds two small, flat, first-class tables to the schema:

1. **`ReferenceSpace`** — the coordinate frame. For each of x, y, z it states what the axis *increases toward* (e.g. CCF_v3: x → POSTERIOR, y → VENTRAL, z → RIGHT; V1DD streamline space: y → WHITE_MATTER).
2. **`SpatialView`** — a display convention. It states which signed data axis runs `left_to_right` and `bottom_to_top` on screen, either via a named preset (CORONAL, LAMINAR, …) or explicitly (CUSTOM). A reference space ships default views; a dataset may override the default.

`SpatialLocation.reference_space` (and the #25 coordinates table) then point at a `ReferenceSpace` id instead of a free string.

## 2. Background and motivation

Soma xy positions of cortical cells appeared upside down in a downstream plot. In the source data, y increases from pia to white matter (image convention, y down). The plotting code used Cartesian convention (y up), placing white matter at the top. Nothing in the data said which convention applied.

The same ambiguity exists for left/right and front/back in 3D coordinates, and for which plane to plot by default.

Current state in the repo:

- `schemas/core_schema.yaml` defines `SpatialLocation(x, y, z, reference_space: string)`. It is nested-only (`SingleCellReconstruction.soma_location`, `CellMetadata.spatial_location`) and cannot be written (#27).
- V1DD soma positions are written as generic floats in the `v1dd_soma_spatial` CellFeatureSet (`code/etl_v1dd_02_cave.ipynb`, cells 41–45): `soma_voxel_{x,y,z}` and `soma_transformed_{x,y,z}` (streamline-transformed cortical coordinates, µm). Neither carries axis semantics.
- #25 proposes a first-class table `(dataitem_id, reference_space, x, y, z [, project_id])`. That table needs `reference_space` to mean something; this design supplies it.

## 3. Goals and non-goals

**Goals**

- Every stored coordinate can be traced to a frame that states the anatomical or tissue direction of each axis.
- Any consumer can derive, without guessing, which axis goes on screen-horizontal and screen-vertical, and whether to flip each.
- Datasets can declare their own default view without being registered to CCF.
- Works for cortical slabs where anatomical names do not apply ("pia up").

**Non-goals**

- Registration or transforms between spaces (e.g. sample → CCF). Users own that. A registered dataset simply uses the CCF_v3 reference space.
- Storing affine transforms. Possible future extension (§11).
- Building a plotting or Neuroglancer library. §8 specifies the mapping; implementing a link generator is follow-up work.
- Per-row orientation metadata. Orientation belongs to the frame, not to each coordinate.

## 4. Terminology

| Term | Meaning |
|---|---|
| Reference space | A coordinate frame: axis directions, unit, optional handedness. |
| Axis direction | What a data axis increases toward (`AxisDirection` enum). Always "increases toward", never "from". |
| Signed axis | A data axis plus a sign: `PLUS_X`, `MINUS_Y`, … |
| View | Mapping of two signed data axes to screen `left_to_right` and `bottom_to_top`. The third (depth/slicing) axis is implied. |
| Preset | A named view defined in terms of axis directions, resolved per reference space. |
| Image convention | Screen y increases downward (images, SVG, canvas, Neuroglancer). Preset `IMAGE`. |
| Cartesian convention | Screen y increases upward (matplotlib, math). Preset `CARTESIAN`. |
| Neurological convention | Subject's left shown on screen left ("left is left"). Used by the anatomical presets. Radiology uses the opposite. |

## 5. Schema design

### 5.1 Enums — `schemas/base_schema.yaml`

```yaml
  AxisDirection:
    description: Direction a reference-space axis increases toward.
    permissible_values:
      ANTERIOR:      {description: Toward the nose (rostral). Opposite of POSTERIOR.}
      POSTERIOR:     {description: Toward the tail (caudal). Opposite of ANTERIOR.}
      DORSAL:        {description: Toward the top of the brain (superior). Opposite of VENTRAL.}
      VENTRAL:       {description: Toward the base of the brain (inferior). Opposite of DORSAL.}
      LEFT:          {description: Toward the subject's left. Opposite of RIGHT.}
      RIGHT:         {description: Toward the subject's right. Opposite of LEFT.}
      PIAL:          {description: Toward the pial surface along the cortical radial axis. Opposite of WHITE_MATTER.}
      WHITE_MATTER:  {description: Toward white matter along the cortical radial axis. Opposite of PIAL.}
      TANGENTIAL:    {description: Within the cortical sheet; no anatomical direction. Has no opposite.}
      UNDEFINED:     {description: Direction unknown or not meaningful. Has no opposite.}

  SignedAxis:
    description: A data axis and the sign along which it increases on screen.
    permissible_values:
      PLUS_X:  {description: Data x increases along the screen direction.}
      MINUS_X: {description: Data x decreases along the screen direction (axis flipped).}
      PLUS_Y:  {description: Data y increases along the screen direction.}
      MINUS_Y: {description: Data y decreases along the screen direction (axis flipped).}
      PLUS_Z:  {description: Data z increases along the screen direction.}
      MINUS_Z: {description: Data z decreases along the screen direction (axis flipped).}

  ViewPreset:
    permissible_values:
      CORONAL:    {description: "left_to_right → RIGHT, bottom_to_top → DORSAL. Neurological convention, viewed from behind."}
      HORIZONTAL: {description: "left_to_right → RIGHT, bottom_to_top → ANTERIOR. Viewed from above."}
      SAGITTAL:   {description: "left_to_right → POSTERIOR, bottom_to_top → DORSAL. Rostral on the left, viewed from the subject's left."}
      LAMINAR:    {description: "left_to_right → first TANGENTIAL axis, bottom_to_top → PIAL. Pia up."}
      EN_FACE:    {description: "left_to_right → first TANGENTIAL axis, bottom_to_top → second TANGENTIAL axis. Viewed from the pial surface."}
      IMAGE:      {description: "left_to_right → PLUS_X, bottom_to_top → MINUS_Y. Raw axes, y down."}
      CARTESIAN:  {description: "left_to_right → PLUS_X, bottom_to_top → PLUS_Y. Raw axes, y up."}
      CUSTOM:     {description: "left_to_right and bottom_to_top given explicitly."}

  Handedness:
    permissible_values: [RIGHT_HANDED, LEFT_HANDED]
```

`MEDIAL`/`LATERAL` are deliberately excluded: they flip at the midline and cannot describe a single fixed axis.

`Unit` gains `NANOMETERS_LENGTH` (EM coordinates are commonly in nm).

### 5.2 `ReferenceSpace` — `schemas/core_schema.yaml`

```yaml
  ReferenceSpace:
    description: >-
      A spatial coordinate frame. States what each axis increases toward, so
      coordinates in this space can be interpreted and displayed without guessing.
    mixins: [ProjectScoped]          # project_id null => global space (e.g. CCF_v3)
    slots: [id, name, description, x_increases_toward, y_increases_toward,
            z_increases_toward, unit, handedness]
    slot_usage:
      id:                 {required: true}
      x_increases_toward: {required: true}
      y_increases_toward: {required: true}
      z_increases_toward: {required: true}
      unit:               {required: true}
      handedness:
        required: false
        description: >-
          Anatomical handedness of (x, y, z). Derivable when all three axes are
          anatomical; must agree with the derived value if given. Required to
          interpret the depth axis when any axis is TANGENTIAL or UNDEFINED.

slots:
  x_increases_toward: {range: AxisDirection}
  y_increases_toward: {range: AxisDirection}
  z_increases_toward: {range: AxisDirection}
  handedness:         {range: Handedness}
```

The existing global slot `reference_space` changes range from `string` to `ReferenceSpace`. In generated pydantic this is still `str` (an id), so `SpatialLocation`, `SingleCellReconstruction` and `CellMetadata` keep their shape; the value must now be a registered `ReferenceSpace.id`.

### 5.3 `SpatialView` — `schemas/core_schema.yaml`

Views are a separate flat table, not a nested object on `ReferenceSpace` or `DataSet`. Reasons: nested models don't round-trip (#27); a space can have several named views; a dataset override must not widen the `DataSet` table.

```yaml
  SpatialView:
    description: >-
      A display convention for coordinates in one reference space: which signed
      data axis runs left-to-right and bottom-to-top on screen.
    mixins: [ProjectScoped]
    slots: [id, name, reference_space, dataset_id, preset, left_to_right,
            bottom_to_top, is_default]
    slot_usage:
      id:              {required: true}
      reference_space: {required: true}
      dataset_id:
        required: false            # overrides the global slot's required: true
        description: >-
          If set, this view belongs to one DataSet and can override the reference
          space's default for that dataset. If null, the view belongs to the
          reference space.
      preset:          {required: true}
      left_to_right:
        description: Data axis that increases left → right on screen. Required iff preset is CUSTOM; must be null otherwise.
      bottom_to_top:
        description: Data axis that increases bottom → top on screen. Required iff preset is CUSTOM; must be null otherwise.
      is_default:      {required: true}

slots:
  preset:        {range: ViewPreset}
  left_to_right: {range: SignedAxis}
  bottom_to_top: {range: SignedAxis}
  is_default:    {range: boolean}
```

ID convention: `{reference_space}:{name}` for space-level views (e.g. `CCF_v3:coronal`), `{dataset_id}:{name}` for dataset-level views.

## 6. Behavior

### 6.1 Resolving a view to signed axes

Pure function, no IO. For each of `left_to_right` then `bottom_to_top`:

1. If the preset entry is already a signed axis (`IMAGE`, `CARTESIAN`, `CUSTOM`), use it.
2. Otherwise find the first axis (x, y, z order), not yet used, whose direction equals the target → `PLUS_<axis>`.
3. Otherwise find the first unused axis whose direction is the target's opposite → `MINUS_<axis>`. (TANGENTIAL and UNDEFINED have no opposite.)
4. Otherwise raise `UnresolvableViewError`.

The two resolved axes must differ. The remaining axis is the depth (slicing) axis. Its toward-viewer sign is `left_to_right × bottom_to_top` computed in data coordinates.

Reference results (checked by script; these become test cases):

| Space (x, y, z increase toward) | Preset | left_to_right | bottom_to_top |
|---|---|---|---|
| CCF_v3 (POSTERIOR, VENTRAL, RIGHT) | CORONAL | PLUS_Z | MINUS_Y |
| CCF_v3 | HORIZONTAL | PLUS_Z | MINUS_X |
| CCF_v3 | SAGITTAL | PLUS_X | MINUS_Y |
| CCF_v3 | LAMINAR / EN_FACE | unresolvable | |
| V1DD streamline (TANGENTIAL, WHITE_MATTER, TANGENTIAL) | LAMINAR | PLUS_X | MINUS_Y |
| V1DD streamline | EN_FACE | PLUS_X | PLUS_Z |
| V1DD streamline | CORONAL / HORIZONTAL / SAGITTAL | unresolvable | |
| any | IMAGE | PLUS_X | MINUS_Y |
| any | CARTESIAN | PLUS_X | PLUS_Y |

Note: CCF_v3 `SAGITTAL` equals `IMAGE`. Neuroglancer's native xy panel on CCF data is already a sagittal view.

### 6.2 Choosing the default view for a dataset

`resolve_default_view(dataset_id, reference_space)`:

1. `SpatialView` with `dataset_id = D`, `reference_space = S`, `is_default = true`.
2. Else `SpatialView` with `dataset_id = null`, `reference_space = S`, `is_default = true`.
3. Else `IMAGE` (the most common acquisition convention), with a warning.

At most one default per `(reference_space, dataset_id)`.

### 6.3 Validation

| Rule | Where enforced |
|---|---|
| `preset = CUSTOM` ⇔ `left_to_right` and `bottom_to_top` both set | Write time (row-local) |
| `left_to_right` and `bottom_to_top` use different axes | Write time (row-local) |
| Preset resolves against its reference space | Write time when the `ReferenceSpace` is in the same batch or on disk; always in tests |
| `handedness`, if given, agrees with the value derived from three anatomical axes | Write time (row-local on `ReferenceSpace`) |
| Each `AxisDirection` pair (e.g. ANTERIOR/POSTERIOR) used by at most one axis | Write time (row-local on `ReferenceSpace`) |
| At most one `is_default` per `(reference_space, dataset_id)` | Within-batch at write time; across lake in reader / tests |
| `SpatialLocation.reference_space` and #25 table values exist as `ReferenceSpace.id` | Reader / ETL test (cross-table; not write time) |

A preset that does not resolve fails loudly. It never falls back silently.

## 7. Implementation locations

| File | Change |
|---|---|
| `schemas/base_schema.yaml` | Add enums `AxisDirection`, `SignedAxis`, `ViewPreset`, `Handedness`. Add `NANOMETERS_LENGTH` to `Unit`. |
| `schemas/core_schema.yaml` | Add classes `ReferenceSpace`, `SpatialView`. Add slots `x/y/z_increases_toward`, `handedness`, `preset`, `left_to_right`, `bottom_to_top`, `is_default`. Change `reference_space` slot range to `ReferenceSpace` and update its description. |
| `src/connects_common_connectivity/models.py` | Regenerate only, via `scripts/generate_models.sh`. Never hand-edit. |
| `src/connects_common_connectivity/io/path_spec.py` | Add `"ReferenceSpace": "referencespace"`, `"SpatialView": "spatialview"`. |
| `src/connects_common_connectivity/io/write_spec.py` | Register both. `ReferenceSpace`: `partition_by=[]`, `scope_columns=["id"]`, `merge_scoped`, `merge_on=["id"]`. `SpatialView`: same, `merge_on=["id"]`. Add `cross_field_rules=["spatial_view_axes"]` / `["reference_space_axes"]`. |
| `src/connects_common_connectivity/spatial.py` (new) | Pure helpers: `OPPOSITE`, `resolve_view(space, view) -> (SignedAxis, SignedAxis)`, `depth_axis(...)`, `derive_handedness(space)`, `check_reference_space(space)`, `check_spatial_view(view)`, `UnresolvableViewError`. No IO. |
| `src/connects_common_connectivity/io/write_validation.py` | First consumer of `WriteSpec.cross_field_rules`: map rule names to the `spatial.py` checks and run them in `validate_for_write`. (Decision D2.) |
| `src/connects_common_connectivity/io/read.py` | Add `read_reference_spaces()`, `read_spatial_views()`, and `DatasetReader.default_view(dataset_name, reference_space)` implementing §6.2. |
| `src/connects_common_connectivity/__init__.py` | Export `resolve_view` and the reader functions. |
| `code/etl_reference_spaces.ipynb` (new) | Seed global rows: `CCF_v3` (POSTERIOR, VENTRAL, RIGHT, µm, RIGHT_HANDED) and views `CCF_v3:coronal` (default), `CCF_v3:horizontal`, `CCF_v3:sagittal`. Written via `write_models`. |
| `code/etl_v1dd_02_cave.ipynb` | Add project-scoped `ReferenceSpace` rows for V1DD EM space and streamline space, and `SpatialView` `v1dd_streamline:laminar` (default). Coordinate data migration itself belongs to #25. |
| `tests/test_spatial.py` (new) | Resolution table in §6.1; unresolvable cases; CUSTOM validation; handedness derivation and mismatch; opposite-pair collision. |
| `tests/test_write_spec.py` | Covered by the existing parametrized registry tests once entries are added; add a round-trip write/read test for both classes in `tests/test_writers.py`. |
| `tests/test_read.py` | Default-view precedence (§6.2). |
| `CHANGELOG.md` | `### Added` entries under `[Unreleased]`. |
| `README.md` | Short "Spatial coordinates" section: reference spaces, views, how to pick axes for a plot. |

Existing seed and V1DD axis directions must be confirmed with the data owners before merging (Q1, Q2).

## 8. Neuroglancer mapping

The `neuroglancer_link` slot on `DataItem` can later be generated from a `SpatialView`. Relevant Neuroglancer state fields: `dimensions`, `position`, `layout` (`xy`, `xz`, `yz`, `3d`, `4panel`, …), `crossSectionOrientation` (quaternion `[x, y, z, w]`), `crossSectionScale`, `projectionOrientation`, `displayDimensions`.

Mapping, using only the `xy` panel to avoid depending on the exact orientation of the `xz`/`yz` panels:

- Neuroglancer's unrotated xy panel shows +x to the right and +y downward (= preset `IMAGE`).
- For a resolved view `(r, u)`: screen-down `d = −u`; normal `n = r × d`. The rotation matrix with columns `[r, d, n]` (in data coordinates) converts to the `crossSectionOrientation` quaternion.
- Because `n` is derived, `[r, d, n]` is always a proper rotation, so every view is reachable with a quaternion. An earlier note claimed some views would need a mirror; that is wrong for views defined by two signed axes.
- A `LEFT_HANDED` reference space shows anatomy mirrored in any viewer that treats xyz as right-handed. Flag it; correcting it is a data transform, out of scope.

The quaternion convention (active vs. passive rotation, element order) must be checked against a live Neuroglancer instance before a generator is written.

## 9. Rollout

1. **PR 1 — schema + helpers.** Enums, both classes, regenerated models, `spatial.py`, registry/path entries, write-time rules, tests. No data changes.
2. **PR 2 — seed data.** `etl_reference_spaces.ipynb` (CCF_v3) and V1DD reference spaces and views.
3. **PR 3 — #25.** Coordinates table with `reference_space` referencing `ReferenceSpace`; migrate `v1dd_soma_spatial`.
4. **PR 4 — reader.** `default_view` and readers; README section.
5. **Follow-up.** Neuroglancer link generator (§8); optional plotting helper.

This issue blocks #25. It does not depend on #27 (all new classes are flat).

## 10. Alternatives considered

| Alternative | Why not |
|---|---|
| A single "what is up" flag per dataset | Mixes frame semantics with display; can't express left/front or the default plane. |
| 3-letter orientation codes (`PIR`, `RAS`) as a string | Ambiguous "to" vs. "from" conventions across tools; can't express PIAL/TANGENTIAL. Codes can be derived from our fields for anatomical spaces. |
| Store `screen_out` as a third field | Determined by the other two; storing it invites contradictions. |
| `right` / `up` field names | "up" collides with anatomical up (dorsal/superior). `left_to_right` / `bottom_to_top` name screen directions only. |
| Nested `View` object on `ReferenceSpace` / `DataSet` | Blocked by #27; allows only one view; widens `DataSet`. |
| Per-row orientation columns on the coordinates table | Redundant; orientation is a property of the frame. |
| Require registration to CCF | Registration is the scientist's choice; cortical slabs need tissue-local frames. |

## 11. Open questions and decisions

| # | Item | Proposed answer |
|---|---|---|
| Q1 | V1DD streamline space: confirm y increases toward WHITE_MATTER with pia at 0, and that x/z are tangential. | Verify with `standard_transform.datasets.v1dd_streamline_nm` docs/owners. |
| Q2 | V1DD "voxel" coordinates: queried with `desired_resolution=[1, 1, 1]` (nm) but labelled `ARBITRARY_UNIT` "voxel" in `etl_v1dd_02_cave.ipynb`. Which is correct, and what are the EM axis directions? | Check with V1DD owners; fix in #25 migration. |
| Q3 | Are oblique samples in scope now? | No. Use the nearest direction or UNDEFINED; add an optional affine to a target space later. |
| D1 | `LAMINAR` with two TANGENTIAL axes: which goes left-to-right? | First in x, y, z order. Use CUSTOM to choose otherwise. |
| D2 | Wire the checks into `cross_field_rules` (first consumer) vs. call them only from ETLs/tests. | Wire into `write_validation.py`; row-local only. |
| D3 | Should `ReferenceSpace` be ProjectScoped? | Yes, with `project_id` null for global spaces; `merge_on=["id"]`, so ids must be globally unique. |
| D4 | Fallback when no default view exists. | `IMAGE`, with a warning. |

## 12. Acceptance criteria

- `ReferenceSpace` and `SpatialView` round-trip through `write_models` and the reader.
- `resolve_view` reproduces every row of the §6.1 table; unresolvable presets raise.
- Invalid CUSTOM views, duplicate axes, conflicting opposite pairs and handedness mismatches fail at write time.
- CCF_v3 and V1DD reference spaces and default views are seeded.
- `DatasetReader.default_view` follows the §6.2 precedence.
- CHANGELOG and README updated; `models.py` only regenerated.
