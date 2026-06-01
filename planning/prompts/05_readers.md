# Agent prompt — Readers (predicate-based + cross-dataset)

> Prepend `00_shared_context.md`. Depends on `write_spec.py` (+ `config.py`).

## Goal
Create `src/connects_common_connectivity/io/readers.py`: convenient reads over the shared
Delta tables, scoped by the registry, plus flexible cross-dataset/cross-schema queries.
Readers are conveniences — users can always drop to raw `polars.read_delta`.

## Layer 1 — predicate-based readers
- `read_dataset(*, project_id=None, dataset_id=None, settings=None)`,
  `read_dataitem(...)`, `read_features(...)` etc.
- Resolve the path via the registry `subdir` + `table_path`; filter by the given scope
  columns; return a polars DataFrame (offer `.to_pandas()` convenience).
- Reuse `parquet_loader.load_parquet_to_models` where returning typed models is wanted.

## Layer 2 — cross-dataset / cross-schema reads
Flagship function (build this and design it to generalize):
`read_dataitems_for_clusters(cluster_ids, *, via=("membership","mapping"), project_id=None,
settings=None) -> DataFrame`:
- Returns the union of DataItems that have EITHER a `ClusterMembership` OR a
  `CellToClusterMapping` to any cluster in `cluster_ids`.
- Join the membership and mapping Delta tables on cluster id; collect distinct DataItem
  ids; optionally hydrate with DataItem rows. Cross-dataset and cross-modality by design —
  do not assume a single source dataset.
- Use `walk_ancestors` semantics so a query for a parent cluster also matches descendants
  if the membership/mapping tables are denormalized that way (check how the `_03`/cluster
  notebooks write the hierarchy before assuming).

## Layer 3 — fold in analysis utils
Port `populate_region_coverage(pmm, matrix)` and `compare_region_coverage(pmms)` from
`io/io_plans.md` into this module (or a sibling `analysis.py` if cleaner). Keep their
documented signatures and pure-function behavior.

## Tests (`tests/test_readers.py`)
- Round-trip: write models via the writers, read them back scoped, assert equality on
  scope columns.
- `read_dataitems_for_clusters` returns the correct union for a small synthetic
  membership + mapping fixture, including cross-dataset cases.

## Do not
- Touch `models.py` or schemas. Lock users out of raw polars (readers are additive).
