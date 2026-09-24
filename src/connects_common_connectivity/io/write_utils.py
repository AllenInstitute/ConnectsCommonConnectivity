"""Write helpers for Delta Lake tables shared across ETL notebooks."""
from __future__ import annotations

from typing import Iterator, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from connects_common_connectivity.models import ProjectionMeasurementMatrix

__all__ = [
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
