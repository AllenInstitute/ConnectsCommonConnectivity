# Agent prompt — Write spec registry (seed only)

> Prepend `00_shared_context.md`. Depends on nothing (reads generated models).

## Goal
Create `src/connects_common_connectivity/io/write_spec.py` with the `WriteSpec` shape, a
`REGISTRY` seeded with **exactly three** entries (DataSet, DataItem,
DataItemDataSetAssociation), a `get_spec()` lookup, and a drift test.

This prompt is the **minimum** needed to unblock W3. The remaining classes are added during
W3, where the writer exists to prototype against — see `03_writers.md` for that loop.

## `WriteSpec` shape
Pydantic v2 `BaseModel` (the rest of the codebase uses pydantic — match it):

```python
class WriteSpec(BaseModel):
    model_cls: type                       # the generated pydantic class
    subdir: str                           # Delta subdir under output_root
    partition_by: list[str]               # Delta partition columns
    scope_columns: list[str]              # columns defining the predicate
                                          # (or the id column for append_new_by_id)
    write_mode: Literal["overwrite_scoped", "append_new_by_id"]  # extend in W3 if needed
    required_for_write: list[str] = []    # leave empty here; W5 owns this
    cross_field_rules: list[str] = []     # leave empty here; W5 owns this
```

Notes:
- `scope_columns` does double duty: for `overwrite_scoped` it's the predicate; for
  `append_new_by_id` it's the id column(s) the backend dedupes on. One field, two
  interpretations dispatched on `write_mode`.
- `required_for_write` and `cross_field_rules` are owned by W5 (validation). Leave them as
  empty lists for the seed entries; do not guess.

Expose:
- `REGISTRY: dict[str, WriteSpec]` keyed by class name (`"DataSet"`, etc.).
- `get_spec(model_or_cls) -> WriteSpec` — accepts a class or an instance.

## Seed exactly these three

| class | subdir | partition_by | scope_columns | write_mode |
|---|---|---|---|---|
| `DataSet` | `dataset` | `["project_id"]` | `["project_id", "id"]` ← patchseq fix | `overwrite_scoped` |
| `DataItem` | `dataitem` | `["project_id"]` | `["id"]` | `append_new_by_id` |
| `DataItemDataSetAssociation` | `dataitem_dataset_association` | `["project_id"]` | `["project_id", "dataset_id"]` | `overwrite_scoped` |

The subdir names must match the existing notebook paths (grep
`code/etl_*_01_dataset_dataitem.ipynb` for `write_deltalake(` to confirm). The DataSet
scope is the patchseq fix — today's notebooks predicate only on `project_id`, which is why
`visp_inh_patchseq` overwrites `visp_exc_patchseq`.

**Do NOT add any other classes here.** `Cluster`, `ClusterMembership`, `MappingSet`,
`CellFeatureSet`, `CellToClusterMapping`, `ProjectionMeasurementMatrix`, `CellFeatureMatrix`,
etc. are W3's responsibility, added one at a time as their write examples are prototyped.

## Drift test (`tests/test_write_spec.py`)
- Every `REGISTRY` key resolves to a real class in `models.py` (importable, `model_cls`
  matches the key).
- For each entry, every name in `scope_columns + partition_by + required_for_write` is a
  field on `model_cls` (check `model_cls.model_fields`). Fail with the offending
  class/field name.
- `get_spec(SomeClass)` and `get_spec(SomeClass(...))` return the same entry.

## Report
- The three subdir names you wrote, and the matching paths grep'd from the notebooks.
- Confirmation that `tests/test_write_spec.py` passes (`pytest tests/test_write_spec.py -q`).

## Do not
- Add a fourth class. Edit `models.py` or schemas. Touch any notebook. Populate
  `required_for_write` or `cross_field_rules` (those are W5's job).
