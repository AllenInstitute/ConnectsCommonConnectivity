# Add ETL notebooks for VISp Patch-seq, WNM, and Minnie65 datasets

## What's in this PR

ETL notebooks that load four neuroscience datasets into the shared Delta Lake, plus the schema and helper-code fixes they need. Re-running any notebook is mostly safe.

## Schema changes

- **`CellFeatureDefinition` and `CellFeatureSet`** now carry `project_id` (and `feature_set_id`). Without these, two feature sets in the same project couldn't be written without one wiping the other.
- **`Cluster`** is now global (no `project_id`); both `Cluster` and `ClusterMembership` carry `hierarchy_id`. Lets one taxonomy be shared across projects, and lets one project hold memberships against several taxonomies.
- **`MappingSet`** endpoints can be `DataSet` *or* `ClusterHierarchy`. Needed because `CellToClusterMapping` targets a hierarchy, not a dataset.

## New helpers in `write_utils.py`

- `append_new_dataitems` — adds only cells not already registered for a project. Safe when two notebooks share a `project_id`.
- `walk_ancestors` — yields each ancestor of a leaf cluster up to the root. Used by every `_03` notebook for parent propagation.

## Notebooks added

Each project follows the same phases (not all phases exist for every project):

| Phase | What it writes |
|---|---|
| `_01` | `DataSet`, `DataItem`, and their links |
| `_02` | Cell features (CSM, morphology, projections, etc.) |
| `_03` | Cluster membership and/or cell-to-cluster mappings |
| `_04` | Cell-to-cell connectivity or projection matrices |

Projects: `visp_patchseq` (exc + inh Patch-seq cells), `visp_wnm` (whole-neuron morphology), `minnie65` (EM nuclei + CSM cohort), plus two global reference taxonomies (`tasic_2018_visp_taxonomy`, `visp_met_types_taxonomy`).

All writes use `mode="overwrite"` with a two-level predicate (`project_id` + a discriminator like `dataset_id`, `feature_set_id`, `hierarchy_id`, or `mapping_set`). When several notebooks share the same predicate slice, they read existing rows, drop their own cohort, and overwrite the union — keeping every notebook independently re-runnable.

## Tests

51 tests pass (`uv run pytest -q`). Coverage is per-area: `tests/test_cell_features_schema.py`, `tests/test_clustering_schema.py`, `tests/test_mappings_schema.py`, `tests/test_write_utils.py`, plus pre-existing `test_basic.py` and `test_projection_schema.py`.

## Documentation

- `code/etl_examples_readme.ipynb` — overview of the registered datasets and feature sets for new readers.
- `etl_example_prompt.md` — guide for writing new ETL notebooks (file order, conventions, common mistakes).

Both are linked from `README.md`.

## Known gaps (not in this PR)

- `etl_wnm_exc_05_single_cell_recon_and_brain_region_assoc.ipynb` — blocked on `BrainRegionAssociation` / `SingleCellReconstruction` not being `ProjectScoped`, and on `brainregion/` not yet being populated.
- `etl_wnm_exc_04_projection_matrix.ipynb` open questions: `measurement_type` is inferred from value magnitudes (confirm with data owner); `region_index` stores raw acronyms (waiting on `BrainRegion`); `values` typed as `ZarrArray` but stored as a parquet path.
- `CellCellConnectivityLong` would benefit from a `connectome_id` discriminator so multiple connectomes per project can share one table.
- `HierarchyCategory` rows (`class`, `subclass`, etc.) are intentionally shared across taxonomies, so neither overwrite nor append is safe. Need a global-dedup append helper before any taxonomy notebook should write here. `etl_minnie_03` skips the write for this reason.
- `etl_minnie_03` reads its taxonomy from a legacy delta lake at `data/microns1412/` because the original CAVE source table no longer exists at materialization v1412 and the parquet feature file uses a different cell-type vocabulary. Temp solution; revisit once a current source is available.
