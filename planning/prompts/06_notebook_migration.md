# Agent prompt — Notebook migration

> Prepend `00_shared_context.md`. Depends on writers (and readers for verification cells).

## Goal
Migrate the ETL notebooks in `code/etl_*.ipynb` to use the new IO API. Replace the
hand-rolled `write_deltalake(... mode/predicate/partition_by ...)` calls with
`write_models` / `write_projection_matrix`, and replace the hardcoded
`OUTPUT_ROOT = "../scratch/..."` constant with a call to
`connects_common_connectivity.config.output_root()` — a cwd-aware helper that returns
the path string with trailing `/`, so it's a literal drop-in for the old constant.
Notebooks keep their per-dataset config cell (input paths, dataset/project ids,
versions, feature-set ids, etc.); only the output root and the manual write
bookkeeping move into the library.

## Required reading before touching any notebook
1. `etl_example_prompt.md` (repo root) — describes the **pre-migration** notebook patterns:
   write predicates, two-level overwrite rules, `append_new_dataitems`, the patchseq
   shared-partition bug, parent propagation, etc. Read this so you understand WHAT each
   notebook is doing scientifically and WHY the old write patterns were shaped that way.
   Treat its rules about ids, enums, schemas, and verification cells as still binding.
2. `src/connects_common_connectivity/io/` — the **post-migration** target. The functions
   `write_models`, `write_projection_matrix`, `get_settings`, `table_path` (re-exported
   from `connects_common_connectivity.io`) now own everything `etl_example_prompt.md`
   spelled out by hand: mode, predicate, partition_by, append-new-by-id, two-level scoping
   per class. Migration is the act of replacing those manual rules with these calls.
3. The config file `ccc_config.yaml` already exists at repo root — do NOT recreate it.
   Migration only edits notebooks.

## What changes between old and new
| Old (per `etl_example_prompt.md`) | New (this migration) |
|---|---|
| `OUTPUT_ROOT = "../scratch/..."` constant in cell 3 | `OUTPUT_ROOT = output_root()` — same string shape, sourced from `ccc_config.yaml` |
| `write_deltalake(path, table, mode="overwrite", predicate=..., partition_by=...)` | `write_models(instance_or_list)` — registry owns mode/predicate/partition |
| `append_new_dataitems(...)` for `dataitem/` | `write_models(dataitem_list)` — append-new-by-id is the registered mode |
| Manual two-level predicate strings | None in notebooks; the `WriteSpec` for each class encodes them |
| Verification cell hardcoded path string | `output_root() + "<table>/"` (or `table_path(get_settings(), "<table>")` for a typed `Path`) |
| `write_deltalake(...)` for projection matrix wide form | `write_projection_matrix(pmm, dense_matrix)` |

The model construction, ETL transforms, and verification assertions do not change.

## Per ETL notebook
1. Replace the hardcoded `OUTPUT_ROOT = "../scratch/..."` with
   `OUTPUT_ROOT = output_root()` (imported from
   `connects_common_connectivity.config`). The helper returns a cwd-relative path
   string with trailing `/`, so existing string concatenations like
   `OUTPUT_ROOT + "dataitem/"` keep working. `write_models(...)` calls need neither a
   path nor `settings=` — the library discovers `ccc_config.yaml` on its own.
2. Replace each direct `write_deltalake(... mode=... predicate=... partition_by=...)` call
   with `write_models(my_instance)` (or `write_models([inst1, inst2])`). The class is
   inferred from the argument; the registry owns mode / predicate / partition. Use
   `write_projection_matrix(pmm, matrix)` for the one projection notebook — it's the
   single non-`write_models` writer. Delete the now-redundant `mode`/`predicate`/
   `partition_by` arguments and their explanatory comments.
3. Keep verification cells; their `OUTPUT_ROOT + "<table>/"` reads continue to work
   unchanged once `OUTPUT_ROOT` is sourced from `output_root()`.

## Pilot first — do not fan out
Migrate ONE notebook end-to-end before touching any others. Pick
`etl_visp_inh_patchseq_01_dataset_dataitem.ipynb` as the pilot (small, exercises the
patchseq bug, uses both `DataSet` and `DataItem` writes). For the pilot:

1. Run the pre-migration version once and record the output Delta tables (row counts and
   `(project_id, id)` sets for `dataset/`, `dataitem/`, `dataitem_dataset_association/`).
2. Migrate the notebook per the rules above and run it against a **fresh** output root
   (point `ccc_config.yaml` or `CCC_OUTPUT_ROOT` somewhere new so the pre-migration data
   is preserved for comparison).
3. Diff: assert the post-migration tables match the pre-migration ones in row count and
   `(project_id, id)` set equality. Any drift is a registry/spec bug — STOP and report
   before migrating further notebooks.
4. Only after the pilot passes the diff, proceed in the order below.

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
Once every notebook imports from the `io/` paths, the write-side re-export shims at
`arrow_utils.py` and `write_utils.py` are dead weight. Do TODO 3.4: delete them and confirm
the no-shim test (`07_tests.md`) passes. Report which old paths were still referenced, if any.
(`parquet_loader.py` is untouched this round — it moves with the deferred read-side work.)

## Do not
- Change the science/ETL transformation logic. Fix the `etl_visp_inh_patchseq` data logic
  beyond the write path — the maintainer said the writer fix is enough for now.
- Touch `models.py` or schemas.
