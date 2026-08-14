"""Idempotent write helpers for Delta Lake tables shared across notebooks."""
from __future__ import annotations

from typing import Iterator, Mapping, Optional, Tuple

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
from deltalake import write_deltalake
from numpy.typing import ArrayLike

from connects_common_connectivity.models import ProjectionMeasurementMatrix

__all__ = [
    "append_new_dataitems",
    "populate_region_coverage",
    "walk_ancestors",
]


def walk_ancestors(
    leaf_id: str,
    parent_of: Mapping[str, Optional[str]],
) -> Iterator[Tuple[str, bool]]:
    """Yield ``(cluster_id, is_leaf)`` from a leaf cluster up to the root.

    Used by cluster-membership / cell-to-cluster-mapping notebooks to
    denormalize the hierarchy into the membership/mapping table so that
    consumers can filter at any level without a recursive cluster join.
    The first yielded tuple has ``is_leaf=True``; all ancestors yield
    ``is_leaf=False``. The walk terminates when ``parent_of[current]`` is
    ``None`` (the root).

    Parameters
    ----------
    leaf_id:
        Cluster id to start from. Must be a key in ``parent_of``.
    parent_of:
        Mapping from cluster id to parent id, with ``None`` for the
        root. Typically built as
        ``dict(zip(cluster_df["id"], cluster_df["parent"]))`` filtered to
        a single ``hierarchy_id``.

    Yields
    ------
    tuple[str, bool]
        ``(cluster_id, is_leaf)`` pairs from leaf to root, inclusive.

    Raises
    ------
    KeyError
        If ``leaf_id`` is not a key in ``parent_of`` (the caller should
        validate cluster ids against the registered taxonomy first and
        fail loudly on unknowns).
    """
    if leaf_id not in parent_of:
        raise KeyError(leaf_id)
    cur: Optional[str] = leaf_id
    is_leaf = True
    while cur is not None:
        yield cur, is_leaf
        is_leaf = False
        cur = parent_of.get(cur)


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


def populate_region_coverage(
    pmm: ProjectionMeasurementMatrix, matrix: ArrayLike
) -> ProjectionMeasurementMatrix:
    """Return a copy of ``pmm`` with ``region_coverage`` derived from ``matrix``.

    ``region_coverage`` is the subset of ``pmm.region_index`` whose
    corresponding column in the dense ``matrix`` has at least one non-zero
    value. Pure function: the input ``pmm`` is not mutated.

    Parameters
    ----------
    pmm:
        A :class:`ProjectionMeasurementMatrix` instance with ``region_index``
        already populated.
    matrix:
        Dense numeric array of shape
        ``(len(pmm.data_item_index), len(pmm.region_index))`` — typically a
        NumPy ``ndarray``, but any input accepted by :func:`numpy.asarray`
        works.

    Returns
    -------
    ProjectionMeasurementMatrix
        A new instance equal to ``pmm`` except that ``region_coverage`` is
        the list of region ids with at least one non-zero entry, in the
        order they appear in ``region_index``.

    Raises
    ------
    ValueError
        If ``pmm.region_index`` is missing or its length does not match
        ``matrix.shape[1]``.
    """
    region_index = getattr(pmm, "region_index", None)
    if region_index is None:
        raise ValueError("pmm.region_index must be set before populating region_coverage")

    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError(
            f"matrix must be 2D (cells x regions); got shape {arr.shape!r}"
        )
    if arr.shape[1] != len(region_index):
        raise ValueError(
            f"matrix.shape[1] ({arr.shape[1]}) must equal len(region_index) "
            f"({len(region_index)})"
        )

    nonzero_cols = np.any(arr != 0, axis=0)
    coverage = [r for r, keep in zip(region_index, nonzero_cols.tolist()) if keep]
    return pmm.model_copy(update={"region_coverage": coverage})
