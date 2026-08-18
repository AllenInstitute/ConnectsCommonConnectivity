"""Write helpers for Delta Lake tables shared across ETL notebooks."""
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
    absent or ``None`` (normally the root).

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

    Notes
    -----
    The mapping is expected to describe an acyclic parent chain. Cycles are
    not detected and would make iteration non-terminating. After the initial
    leaf check, a missing ancestor key ends the walk after that ancestor has
    been yielded.
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
    """Append candidate rows whose ids are not stored for one project.

    On a sequential call where the existing Delta table can be read, rows
    whose ``id_column`` value already occurs in the selected ``project_id``
    partition are omitted. The append does not remove existing rows. If the
    table does not exist or the read fails for any reason, every candidate row
    is treated as new.

    Parameters
    ----------
    output_path:
        Complete path to the Delta table directory.
    table:
        Arrow table of candidate rows. It must contain ``id_column`` and a
        ``project_id`` column whose values match the ``project_id`` argument;
        duplicate ids within this batch are not removed.
    project_id:
        Existing-table partition to inspect before checking candidate ids.
    id_column:
        Candidate and existing-table column used for the id comparison.

    Returns
    -------
    int
        Number of candidate rows submitted to the Delta append, or zero when
        none remain after the existing-id check.

    Notes
    -----
    Repeating a batch is idempotent only for sequential calls where the
    existing Delta table can be read. This helper provides no transaction
    spanning the read and append, so concurrent writers can append the same
    id. A read failure is treated like a missing table and disables duplicate
    detection for that call.
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
        Two-dimensional numeric array whose columns correspond to
        ``pmm.region_index``. The row count is not validated. Typically this
        is a NumPy ``ndarray``, but any input accepted by
        :func:`numpy.asarray` works.

    Returns
    -------
    ProjectionMeasurementMatrix
        A new instance equal to ``pmm`` except that ``region_coverage`` is
        the list of region ids with at least one non-zero entry, in the
        order they appear in ``region_index``.

    Raises
    ------
    ValueError
        If ``pmm.region_index`` is missing, the matrix is not two-dimensional,
        or its column count does not match the region index length.
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
