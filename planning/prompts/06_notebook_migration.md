# Agent prompt — Notebook migration

> Prepend `00_shared_context.md`. Depends on writers (and readers for verification cells).

## Goal
Migrate the ETL notebooks in `code/etl_*.ipynb` to use the new IO API. Move bookkeeping
into the library; keep the science logic and verification cells.

## Per notebook
1. Replace the hardcoded `OUTPUT_ROOT = "../scratch/..."` with:
   ```python
   from connects_common_connectivity.io.config import Settings
   settings = Settings.load()
   print(settings)  # show resolved output_root at top
   ```
2. Replace each direct `write_deltalake(... mode=... predicate=... partition_by=...)` call
   with the matching typed writer (`write_dataset`, `write_dataitem`, `write_association`,
   `write_features`, `write_cluster`, `write_cell_to_cluster_mapping`,
   `write_projection_matrix`, ...). Delete the now-redundant `mode`/`predicate`/
   `partition_by` arguments and their explanatory comments — that logic now lives in the
   registry.
3. Keep verification cells; update their paths to use `table_path(settings, ...)`.

## Migrate in this order
1. `etl_*_01_dataset_dataitem.ipynb` (all of minnie, wnm, visp_exc/inh patchseq) — these
   carry the DataSet overwrite bug.
2. feature notebooks (`_02_cell_features`).
3. cluster / membership / mapping notebooks (`_03`, cluster files).
4. projection (`etl_wnm_exc_04_projection_matrix.ipynb`).

## Patchseq regression acceptance test (do this explicitly)
Run `etl_visp_exc_patchseq_01` then `etl_visp_inh_patchseq_01` (in that order), then read
the `dataset` table and assert BOTH `visp_exc_patchseq` and `visp_inh_patchseq` rows
exist under `project_id='visp_patchseq'`. Before the fix, the second run wiped the first.
Report the before/after row counts.

## Do not
- Change the science/ETL transformation logic. Fix the `etl_visp_inh_patchseq` data logic
  beyond the write path — the maintainer said the writer fix is enough for now.
- Touch `models.py` or schemas.
