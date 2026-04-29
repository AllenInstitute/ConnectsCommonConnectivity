"""Idempotent write helpers for Delta Lake tables shared across notebooks."""
from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
from deltalake import write_deltalake


def append_new_dataitems(
    output_path: str,
    table: pa.Table,
    *,
    project_id: str,
    id_column: str = "id",
) -> int:
    """Append only rows whose ``id`` is not already in the Delta table for this project.

    Safe to call from multiple notebooks that share the same ``project_id`` partition
    (e.g. ``visp_inh_patchseq_01`` and ``visp_exc_patchseq_01`` both write to
    ``dataitem/`` under ``project_id='visp_patchseq'``). Unlike a scoped overwrite,
    this function never removes rows written by another notebook.

    Idempotent: re-running with the same rows appends nothing and returns 0.
    Handles the case where the Delta table does not yet exist.

    Parameters
    ----------
    output_path:
        Path to the Delta table directory.
    table:
        PyArrow table of candidate rows to append.
    project_id:
        Value used to filter existing rows before checking for duplicates.
    id_column:
        Name of the id column to deduplicate on. Defaults to ``"id"``.

    Returns
    -------
    int
        Number of rows actually appended (0 if all were already present).
    """
    existing_ids: set[str] = set()
    try:
        import polars as pl

        existing_ids = set(
            pl.read_delta(output_path)
            .filter(pl.col("project_id") == project_id)[id_column]
            .to_list()
        )
    except Exception:
        # Table doesn't exist yet, or read failed — treat all rows as new.
        pass

    if existing_ids:
        id_array = table.column(id_column)
        existing_array = pa.array(list(existing_ids), type=id_array.type)
        in_existing = pc.is_in(id_array, value_set=existing_array)
        new_rows = table.filter(pc.invert(in_existing))
    else:
        new_rows = table

    if new_rows.num_rows == 0:
        return 0

    write_deltalake(output_path, new_rows, mode="append", partition_by=["project_id"])
    return new_rows.num_rows
