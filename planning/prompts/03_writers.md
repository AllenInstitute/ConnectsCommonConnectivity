# Agent prompt — Writers (dispatch core + registry expansion)

> Prepend `00_shared_context.md`. Depends on `config.py` (W1), `write_spec.py` (W2).
> Validation (W5) slots into the pass-through hook below — not a dependency here.

## What W3 ships
1. The dispatch core `write_models()` and a `WriteResult` value object.
2. The remaining `WriteSpec` entries (everything except the three W2 seeded), each driven
   by a small write example so the entry reflects how the class actually wants to be
   written.
3. `write_projection_matrix()` — the **only** standalone writer function, because it
   needs a non-uniform signature (the dense matrix for `populate_region_coverage`).
4. The relocation of `arrow_utils.py` and `write_utils.py` into `io/`, plus the
   `populate_region_coverage` helper.

## No per-class wrapper functions
Decision: there are NO `write_dataset`, `write_dataitem`, `write_association`, etc.
wrappers. `write_models()` infers the class from its argument; renaming it eight times
adds no behavior, only drift surface. The single exception is `write_projection_matrix()`
because its signature is genuinely different (it accepts a dense matrix). Discoverability
is provided by `WRITABLE_CLASSES` (a tuple of `model_cls`) plus `write_models`'s docstring
listing them.

## Relocation first
Before writing new code, MOVE the existing backends into `io/` with one-line re-export
shims at the old paths (deleted in W6):
- `arrow_utils.py` → `io/arrow_utils.py`
- `write_utils.py` → `io/write_utils.py`

All new code imports from the `io/` locations. The shims look like:
```python
# src/connects_common_connectivity/arrow_utils.py
from .io.arrow_utils import *  # noqa: F401,F403  (deprecated; removed in W6)
```

Add a quick smoke test (`tests/test_write_relocation.py`) that asserts the public names
(`build_arrow_schema`, `models_to_table`, `attach_linkml_metadata`,
`build_cell_feature_matrix_schema`, `append_new_dataitems`, `walk_ancestors`) are
importable from BOTH the new and the shim path.

## Core: `write_models`
```python
def write_models(models, *, settings: Settings | None = None) -> WriteResult: ...
```

1. Accept a single model or an iterable; require homogeneous type; infer the class.
2. `settings = settings or get_settings()`. Explicit `settings=` always wins.
3. `spec = get_spec(cls)`.
4. **Validation hook** — call `_validation_hook(models, spec)` before any IO. In W3 this
   is a pass-through (identity) function defined at module top:
   ```python
   _validation_hook = lambda models, spec: models  # replaced in W5
   ```
   Wire the call site now; W5 monkey-patches the real validator in.
5. Convert via `arrow_utils.models_to_table` + `build_arrow_schema`; attach metadata with
   `attach_linkml_metadata(linkml_class=cls.__name__)`.
6. Resolve path with `table_path(settings, spec.subdir)`.
7. Dispatch on `spec.write_mode` (factor each branch into a private helper so the tests
   below can target each in isolation):
   - `_dispatch_overwrite_scoped`: group rows by their `scope_columns` tuple via
     `_group_by_scope`. **Write each group with its own predicate** — never widen a
     predicate to cover rows it shouldn't. Predicate built by `_build_predicate`, format
     `col1 = 'val1' AND col2 = 'val2'` (single quotes, AND-joined). One
     `write_deltalake(... mode="overwrite", predicate=..., partition_by=spec.partition_by)`
     call per group.
   - `_dispatch_append_new_by_id`: delegate to `write_utils.append_new_dataitems`. The
     existing signature (`output_path, table, *, project_id, id_column="id"`) already
     covers the seed entries; if a new `append_new_by_id` entry needs a different
     partition column, generalize then. Pull `id_column` from `spec.scope_columns[0]`
     and `project_id` from the row values.
8. Return a `WriteResult`.

`write_models` should know nothing class-specific; everything class-specific lives in the
registry. The only places that mention specific model classes are `write_spec.py` (the
registry) and `write_projection_matrix` (the one signature exception).

## `WriteResult`
A frozen dataclass — this is a return value, not validated data:

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class WriteResult:
    class_name: str
    path: Path
    mode: str
    predicates: tuple[str, ...]   # one per group; () for append_new_by_id / wide_parquet
    rows_written: int
```

Co-locate in `writers.py`.

## Discovery: `WRITABLE_CLASSES`
Replaces the per-class wrappers. One line in `writers.py`:

```python
WRITABLE_CLASSES: tuple[type, ...] = tuple(spec.model_cls for spec in REGISTRY.values())
```

`write_models`'s docstring should list `WRITABLE_CLASSES` (or instruct the reader to print
it) so users can see what's writable without reading the registry source.

## Registry expansion (the prototype loop — the main intellectual work of W3)
W2 only seeded `DataSet`, `DataItem`, `DataItemDataSetAssociation`. Add the rest now,
one at a time, each driven by a real write example. Do NOT batch them up front.

For each class below:
1. **Read the existing notebook write.** `grep -n 'write_deltalake' code/etl_*.ipynb` to
   find the call(s); note the current `mode`, `predicate`, and `partition_by`.
2. **Decide the mode.** If neither `overwrite_scoped` nor `append_new_by_id` fits, extend
   the `Literal` in `write_spec.py` with a new value, document it in one comment line,
   and add the dispatch branch in `write_models`. Don't force a class into a mode that
   doesn't fit.
3. **Add the entry to `REGISTRY`.** If a current notebook predicate looks wrong (like the
   DataSet case), use the correct scope and note it in a comment.
4. **Write a smoke test** in `tests/test_writers.py` (NOT a production notebook —
   notebooks are W6) that constructs one or two instances and round-trips them through
   `write_models`.
5. **Update the drift test** if the new entry exposes a column the test doesn't already
   check.

Classes to add this round, roughly grouped:
- Cluster side: `Cluster`, `ClusterHierarchy`, `ClusterMembership`.
- Mapping side: `MappingSet`, `CellToClusterMapping`. (`CellToCellMapping` and
  `ClusterToClusterMapping` only if a notebook actually writes them this round; otherwise
  defer.)
- Feature side: `CellFeatureSet`, `CellFeatureDefinition`. (`CellFeatureMatrix` is wide
  Parquet — see "Wide feature matrices" below.)
- Projection: `ProjectionMeasurementMatrix`. See "Projection pre-write helper" below.

If a class isn't written by any current notebook, skip it — adding an entry no caller
exercises violates "prototype, don't assume."

## Wide feature matrices
`CellFeatureMatrix` is wide Parquet, not row-Delta. It doesn't fit `overwrite_scoped` /
`append_new_by_id`. Keep it inside the registry by adding `write_mode = "wide_parquet"`
and routing it through `build_cell_feature_matrix_schema` + a Parquet write inside
`write_models`. Same registry, same dispatch, different branch — no separate wrapper.
(`write_models(cell_feature_matrix)` is the call.) If during prototyping the wide-Parquet
path turns out to need invariants that don't fit `WriteSpec` cleanly, stop and report
before adding a separate function.

## Projection pre-write helper + `write_projection_matrix`
Port `populate_region_coverage(pmm, matrix)` from `io/io_plans.md` into
`io/write_utils.py` (write plumbing — same shelf as `append_new_dataitems`, NOT a separate
`transforms` module). Pure function: derive `region_coverage` from the dense values array,
return a NEW `ProjectionMeasurementMatrix` instance (no mutation, no IO).

`write_projection_matrix` is the **one** non-`write_models` public writer:
```python
def write_projection_matrix(pmm, matrix, *, settings=None) -> WriteResult:
    enriched = populate_region_coverage(pmm, matrix)
    return write_models(enriched, settings=settings)
```
It exists because its signature is non-uniform (takes the dense matrix). Don't introduce
a second exception — if some other class needs pre-write enrichment, route it through
`write_models` with the enrichment done by the caller, not via a new wrapper.

Do NOT port `compare_region_coverage` — read-side, deferred (`_deferred/09_analysis.md`).

## Private helpers (factor these out for testability)
- `_build_predicate(scope_columns, row_values) -> str`
- `_group_by_scope(table, scope_columns) -> list[tuple[tuple, Table]]`
- `_dispatch_overwrite_scoped(table, spec, path) -> WriteResult`
- `_dispatch_append_new_by_id(table, spec, path) -> WriteResult`
- `_validation_hook(models, spec) -> models` — pass-through; replaced by W5

These are private (underscore-prefixed). Tests import them directly to exercise their
units without going through Delta.

## Tests (`tests/test_writers.py`)
- **Patchseq regression** (the headline): `write_models(DataSet(A))`, then
  `write_models(DataSet(B))` with the same `project_id` but different `id`, read the
  table back, assert both rows exist.
- **Idempotency**: writing the same models twice yields the same row count.
- **Append-new-by-id**: writing a batch with one new + one existing id appends only the
  new one.
- **Multi-scope-group dispatch**: a batch with two distinct scope tuples produces two
  predicates and two rows in the table; neither overwrites the other. Inspect
  `WriteResult.predicates` to assert the count.
- **Predicate construction**: call `_build_predicate` directly and verify the format
  (`col = 'val' AND col = 'val'`) by string match.
- **Per-class smoke**: iterate `WRITABLE_CLASSES` and round-trip a small instance of each
  through `write_models` — every registry entry exercised.
- **`write_projection_matrix`**: enriches the PMM (sets `region_coverage`) and writes
  successfully; the input is unmutated.

## Reporting
- The full list of registry entries at the end of W3 (table: class / subdir /
  partition_by / scope_columns / write_mode).
- Any class you skipped because no notebook writes it, and why.
- Any new `write_mode` you added beyond `overwrite_scoped` / `append_new_by_id` /
  `wide_parquet`, with a one-sentence justification.
- Any current notebook predicate you believe is wrong (do not fix the notebook here —
  W6 owns that).
- `pytest tests/ -q` summary (full suite, not just `test_writers.py`).

## Do not
- Add per-class `write_*` wrapper functions. Hardcode any predicate. Skip the prototype
  loop and bulk-add registry entries from intuition. Touch `models.py`, schemas, or any
  notebook. Re-export internal backends from `io/__init__.py` (W4 owns the public
  surface).
