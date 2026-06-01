# Shared context — prepend to every IO-layer agent prompt

You are working in the `ConnectsCommonConnectivity` repo: a LinkML+pydantic data schema
holding multi-scale connectomics data (EM cell-to-cell, morphology cell-to-area, viral
area-to-area, patch-seq multimodal) in one format, plus taxonomies/clusters.

## Non-negotiable rules
1. **Never edit `src/connects_common_connectivity/models.py`** — it is auto-generated
   from `schemas/*.yaml`. Treat it as read-only.
2. **Never edit `schemas/*.yaml`** without explicit written permission from the maintainer
   (YY). If your task seems to require a new slot for safe writing, STOP and report what
   you need and why; do not change the schema.
3. **Single source of truth = the LinkML schema / generated models.** Read field
   definitions from `models.py`; do not restate them.
4. New IO code goes under `src/connects_common_connectivity/io/`. Do not move plotting
   code out of `code/utils.py`.
5. Read `planning/ARCHITECTURE.md` fully before starting. It governs the design.

## What already exists — reuse, don't rebuild
- `models.py`: generated pydantic v2 classes incl. `DataSet`, `DataItem`,
  `DataItemDataSetAssociation`, `Cluster`, `ClusterHierarchy`, `ClusterMembership`,
  `CellFeatureSet`, `CellFeatureDefinition`, `CellFeatureMatrix`, `CellFeatureMeasurement`,
  `MappingSet`, `CellToCellMapping`, `CellToClusterMapping`, `ClusterToClusterMapping`,
  `ProjectionMeasurementMatrix`. `ProjectScoped` mixin → `project_id`.
- `arrow_utils.py`: `build_arrow_schema`, `models_to_table`, `attach_linkml_metadata`,
  `build_cell_feature_matrix_schema`.
- `write_utils.py`: `append_new_dataitems(path, table, *, project_id, id_column="id")`,
  `walk_ancestors(leaf_id, parent_of)`.
- `parquet_loader.py`: `load_parquet_to_models(...)`. `cli.py`: LinkML full validation.
- `io/io_plans.md`: analysis-util specs to fold into readers.

## Conventions
- Python 3.10+, pydantic v2, polars + pyarrow + deltalake (already deps).
- Match existing style (ruff, line-length 100). Add docstrings like the existing modules.
- Add `pytest` tests under `tests/` for anything you implement.
- After implementing, run the relevant tests and report results. Do not mark work done
  with failing tests or partial implementation.

## Reference: the bug to keep in mind
`visp_exc_patchseq` and `visp_inh_patchseq` share `project_id='visp_patchseq'` with
different `dataset_id`. The current DataSet write uses predicate
`project_id = '<project>'`, so writing one wipes the other. The registry fixes this by
making DataSet's scope `(project_id, id)`. Any writer you build must derive its predicate
from the registry, never from a hardcoded string.
