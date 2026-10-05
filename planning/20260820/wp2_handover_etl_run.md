# WP2 Code Ocean ETL run handover

## Purpose

Validate the WP2 merge-scoped writer against the VISp patch-seq ETL notebooks in
Code Ocean, where the required data assets are attached. The local development
environment cannot perform this validation because it does not have those assets.

This is acceptance testing for GitHub issues #13, #14, and #15. Issue #16 is
covered by local automated tests and does not require a Code Ocean data asset.

## Preconditions

- Use the WP2 branch or PR commit after its local Python test suite has passed.
- Ensure the environment includes the WP1 changes on which WP2 is based.
- Attach the same VISp excitatory and inhibitory patch-seq data assets used by the
  existing notebooks.
- Ensure `etl_tasic_01_cluster.ipynb` and `etl_visp_met_types_01_cluster.ipynb`
   have populated their required taxonomy tables in the selected output root before
   running either patch-seq `_03` notebook.
- Configure `ccc_config.yaml` with a new, empty output root dedicated to this run.
- Do not point the run at a shared or production Delta root.
- Record the tested commit SHA, capsule/environment identifier, data asset names
  and versions, and configured output root in the results report.

## Notebook run

Run these notebooks from the `code/` directory in dependency order:

1. `etl_visp_exc_patchseq_01_dataset_dataitem.ipynb`
2. `etl_visp_exc_patchseq_02_cell_features.ipynb`
3. Populate the Tasic and VISp MET-type taxonomies if they are not already present.
4. `etl_visp_exc_patchseq_03_cluster_membership_and_mapping.ipynb`
5. `etl_visp_inh_patchseq_01_dataset_dataitem.ipynb`
6. `etl_visp_inh_patchseq_02_cell_features.ipynb`
7. `etl_visp_inh_patchseq_03_cluster_membership_and_mapping.ipynb`

Restart the kernel before each notebook if that matches the normal capsule
workflow. Run every cell and treat any exception or failed assertion as a failed
acceptance run. Do not restore any removed read-union-rewrite workaround.

## First-run acceptance checks

After all six notebooks finish, query the Delta tables under the configured output
root and verify:

1. `dataitem_dataset_association` retains all contributors to the scope
   `(project_id="visp_patchseq", dataset_id="visp_inh_patchseq")`. It must contain
   2,879 rows for the known asset versions and must not shrink to the final
   notebook's 495-row contribution.
2. `clustermembership` contains 2,637 rows in the scope
   `(project_id="visp_patchseq", hierarchy_id="visp_met_types_taxonomy")`:
   1,152 excitatory rows plus 1,485 inhibitory rows.
3. Those 2,637 membership rows represent 879 distinct `item` values: 384
   excitatory cells plus 495 inhibitory cells.
4. The merge keys are unique within each table scope. In particular, there must be
   no duplicate association key
   `(project_id, dataset_id, dataitem_id)` or membership key
   `(project_id, hierarchy_id, item, cluster)`.
5. Verification cells in the notebooks report persisted state, not reconstructed
   read-union batches, and all notebook assertions pass.

If attached asset versions legitimately change the expected totals, report both
the observed values and source-file row counts. Do not accept lower totals without
showing that the difference follows from changed inputs rather than a later writer
deleting earlier rows.

## Rerun acceptance checks

Without clearing the output root, rerun notebooks 2, 3, 5, and 6. Then repeat the
queries above and verify:

- Association and membership row counts are unchanged.
- Merge-key uniqueness is unchanged.
- Rows contributed by both excitatory and inhibitory notebooks still coexist.
- No notebook requires a manual read-union-rewrite step to remain idempotent.

## Failure capture

For any failure, preserve:

- The notebook name and visible cell number.
- The full exception or assertion message.
- Counts grouped by the relevant scope and merge key.
- The tested commit SHA and data asset versions.
- Whether the failure occurred on the clean first run or the idempotency rerun.

Do not patch the notebooks only inside Code Ocean. Report the failure back to the
WP2 implementation agent so the fix and regression coverage are committed in the
repository.

## Handover result

Return a short report containing:

- Commit SHA, capsule/environment identifier, and data asset versions.
- Pass/fail for each notebook.
- Final inhibitory association count.
- Final shared MET membership row and distinct-item counts.
- Duplicate-key counts for both checked tables.
- Counts after the rerun.
- Links or paths to retained execution logs.