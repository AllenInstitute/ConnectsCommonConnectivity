# Plan: dataset-oriented reader and V1DD demonstration

## Goal

Add a `DatasetReader` API that opens a Common Connectivity dataset root and
returns one wide Polars table per requested dataset:

- one row per associated `DataItem`;
- one `dataitem_id` key column;
- feature columns from all selected related `CellFeatureSet` matrices;
- cluster-membership columns from all selected related cluster hierarchies,
  with one column per hierarchy level.

The implementation will use the existing on-disk schema and writer layout. No
files under `schemas/` will be edited.

## Proposed public API

Add `DatasetReader` to
`src/connects_common_connectivity/io/read.py`:

```python
reader = DatasetReader(dataset_root)

reader.display_dataset_names()
reader.display_featuresets(dataset_name)
reader.display_clustersets(dataset_name)

reader.read_dataset(
    dataset_name,
    featuresets=None,
    clustersets=None,
)
```

- `dataset_root` accepts `str | pathlib.Path`.
- `dataset_name` resolves an exact `DataSet.id`. The display method will show
  both the ID and human-readable `name`.
- `featuresets` and `clustersets` accept `None`, one name, or an iterable of
  names. `None` means all related sets; an empty iterable means none.
- Display methods return small `polars.DataFrame` objects so they render
  naturally in notebooks and remain programmatically useful.
- `read_dataset` returns a `polars.DataFrame`.

Export `DatasetReader` through
`connects_common_connectivity.io` and update the locked public-API test.

## Dataset and relation discovery

Implement private/helper methods on `DatasetReader` for the following steps:

1. Validate that the root and required Delta tables exist.
2. Resolve the requested dataset row and its `project_id`. If an ID is absent
   or ambiguous, raise an error that lists valid choices.
3. Read `dataitem_dataset_association/`, filter by dataset and project, and
   produce the distinct dataset `dataitem_id` values.
4. Discover feature sets:
   - read `cellfeaturematrix/` metadata for the dataset project;
   - locate each wide matrix at the root-relative writer location
     `cellfeatures/<feature_set_id>/`;
   - inspect its configured `cell_index_column`;
   - retain matrices whose IDs overlap the dataset's data items;
   - join descriptive metadata from `cellfeatureset/`.
5. Discover cluster sets (schema term: `ClusterHierarchy`):
   - filter `clustermembership/` by the dataset project and data-item IDs;
   - collect distinct `hierarchy_id` values;
   - join rows from `clusterhierarchy/` for display.

Discovery by data-item overlap is required because the schemas do not directly
link a `DataSet` to a `CellFeatureSet` or `ClusterHierarchy`.

## Building the wide dataset table

Start with all distinct associated IDs so data items are retained even when
they have no feature values or cluster annotations.

### Features

For each selected feature set:

1. Load its wide Delta matrix.
2. Filter to the dataset's data-item IDs and matching project where the matrix
   carries `project_id`.
3. Rename the configured index column to `dataitem_id`.
4. Remove matrix scope/metadata columns such as `project_id` and
   `feature_set_id`.
5. Left-join feature columns onto the base table.

Feature names will remain their schema-defined IDs when unique. If selected
feature sets contain duplicate feature-column names, fail with a clear error
rather than silently overwrite data; the error will identify the conflicting
sets and columns.

### Cluster memberships

For each selected hierarchy:

1. Filter memberships to the dataset items, project, and hierarchy.
2. Join `cluster/` on `(hierarchy_id, cluster)` to obtain each cluster's
   integer `level`.
3. Validate that an item has at most one cluster per hierarchy and level.
   Multiple memberships at the same level cannot be represented safely as one
   scalar column, so this case will raise a descriptive error instead of
   choosing a membership implicitly.
4. Pivot memberships to one column per level, named
   `<hierarchy_id>_level_<level>`, containing the cluster ID.
5. Left-join the pivoted columns onto the base table.

All represented levels, including level `0`, will be returned by default. In
the V1DD example this produces three columns for the root, coarse, and fine
levels. Unannotated data items remain present with null membership values.

The final table will be sorted by `dataitem_id` for deterministic output.

## Errors and validation

Use explicit errors for:

- missing dataset root or required Delta table;
- unknown or ambiguous dataset IDs;
- unknown requested feature-set or cluster-set names;
- missing feature-matrix storage;
- missing configured feature index columns;
- duplicate feature names across selected sets;
- cluster memberships referencing unknown clusters or clusters without levels;
- multiple cluster assignments for one item/hierarchy/level.

Optional tables will only be treated as empty when their absence semantically
means that no features or clustering data has been written. Missing tables
required to identify datasets and their items will always be errors.

## Tests

Add focused tests for `DatasetReader`, using temporary Delta roots:

1. Dataset listing and dataset-ID validation.
2. Feature-set and cluster-set discovery by overlapping data-item IDs.
3. A complete read with multiple feature sets and multiple hierarchy levels.
4. Optional selection of one or no feature/cluster sets.
5. Preservation of items with missing features or memberships.
6. Project isolation when IDs overlap across projects.
7. Clear failures for unknown selectors, duplicate feature columns, malformed
   matrices, and duplicate same-level memberships.
8. Public export through `connects_common_connectivity.io`.

Run the targeted reader/public-API tests, Ruff on changed Python files, and
then the existing full test suite if the targeted checks pass.

## Demonstration notebook

Create `code/wtl_v1dd_04_read.ipynb` using
`../scratch/v1dd_1196_v2/`, the output written near the end of
`code/etl_v1dd_02_cave.ipynb`.

The notebook will:

1. Import `DatasetReader` from `connects_common_connectivity.io`.
2. Construct the reader from the V1DD root.
3. Display available datasets.
4. Display feature sets and cluster sets related to `v1dd_1196_em`.
5. Read the full EM dataset and display its shape, columns, and sample rows.
6. Demonstrate subset selection for feature sets and cluster sets.
7. Assert the expected V1DD behavior:
   - 143,448 EM data-item rows;
   - the `v1dd_soma_spatial` feature columns are present;
   - the `v1dd_cell_types` level columns are present;
   - cells without annotations remain in the result with null cluster values.
8. Execute the notebook headlessly to ensure the example is reproducible.
9. use this dataset root: `scratch/v1dd_1196_v2`
10. load these two datasets as examples `v1dd_1196_proofread_axons` and `v1dd_1196_proofread_dendrites`

## Edit changelog
- add these changes to the changelog

## Files expected to change after approval

- `src/connects_common_connectivity/io/read.py`
- `src/connects_common_connectivity/io/__init__.py`
- `tests/test_read.py` (new)
- `tests/test_public_api.py`
- `code/wtl_v1dd_04_read.ipynb` (new)
- changelog

No schema or generated-model changes are planned.
