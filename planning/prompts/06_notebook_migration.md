# Agent prompt — Notebook migration

> Prepend `00_shared_context.md`. Depends on writers (and readers for verification cells).

## Goal
Migrate the ETL notebooks in `code/etl_*.ipynb` to use the new IO API. Move bookkeeping
into the library; keep the science logic and verification cells. The output path lives in
ONE file (`ccc_config.yaml`) discovered automatically — notebooks carry no path and no
config cell.

## First: create the config file
Create `ccc_config.yaml` at the repo root (the single source of truth, version-controlled):
```yaml
output_root: ../scratch/em_patchseq_wnm_v1/   # match the value grep'd from code/*.ipynb
dry_run: false
```
To repoint local vs CodeOcean, edit this file (or set `CCC_OUTPUT_ROOT`); nothing else
changes. The library finds it by walking up from the notebook's working directory.

## Per ETL notebook
1. Delete the hardcoded `OUTPUT_ROOT = "../scratch/..."` entirely. There is no replacement
   config cell and no `%run` — the library discovers `ccc_config.yaml` on its own, so
   `write_*` / `read_*` calls need neither a path nor `settings=`. (If a cell wants to show
   the resolved config, it may `from connects_common_connectivity.io import get_settings;
   print(get_settings())`, but this is optional.)
2. Replace each direct `write_deltalake(... mode=... predicate=... partition_by=...)` call
   with the matching typed writer (`write_dataset`, `write_dataitem`, `write_association`,
   `write_features`, `write_cluster`, `write_cell_to_cluster_mapping`,
   `write_projection_matrix`, ...). Delete the now-redundant `mode`/`predicate`/
   `partition_by` arguments and their explanatory comments — that logic now lives in the
   registry.
3. Keep verification cells; update their paths to use
   `table_path(get_settings(), ...)`.

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

## After migration — hand off shim removal
Once every notebook imports from the `io/` paths, the re-export shims at `arrow_utils.py`,
`write_utils.py`, `parquet_loader.py` are dead weight. Do TODO 5.4: delete them and confirm
the no-shim test (`07_tests.md`) passes. Report which old paths were still referenced, if any.

## Do not
- Change the science/ETL transformation logic. Fix the `etl_visp_inh_patchseq` data logic
  beyond the write path — the maintainer said the writer fix is enough for now.
- Touch `models.py` or schemas.
