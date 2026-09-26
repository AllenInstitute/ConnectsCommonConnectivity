"""Write helpers for Delta Lake tables shared across ETL notebooks."""
from __future__ import annotations

from typing import Iterator, Mapping, Optional, Tuple

import numpy as np
import polars as pl
import pyarrow as pa
from numpy.typing import ArrayLike

from connects_common_connectivity.io.arrow_utils import build_arrow_schema
from connects_common_connectivity.models import (
    CellCellConnectivityLong,
    Modality,
    ProjectionMeasurementMatrix,
    SynapticMeasurementType,
    Unit,
)

__all__ = [
    "derive_cell_cell_connectivity",
    "populate_region_coverage",
    "walk_ancestors",
]

_CELL_CELL_IDENTITY_COLUMNS = [
    "project_id",
    "presynaptic_cell",
    "postsynaptic_cell",
]
_CELL_CELL_ARROW_SCHEMA = build_arrow_schema(CellCellConnectivityLong)
_CELL_CELL_OUTPUT_SCHEMA = pl.from_arrow(
    pa.Table.from_batches([], schema=_CELL_CELL_ARROW_SCHEMA)
).schema


def derive_cell_cell_connectivity(
    synapses: pl.DataFrame,
    *,
    connectome_id: str,
    modality: Modality | str,
    size_column: str | None = None,
    size_unit: Unit | str | None = None,
) -> pl.DataFrame:
    """Aggregate single-synapse rows into cell-cell connectivity measurements.

    One ``SYNAPSE_COUNT`` row is emitted for every project and endpoint pair.
    Supplying both ``size_column`` and ``size_unit`` additionally emits a
    ``SUM_ANATOMICAL_SIZE`` row. Null sizes are rejected because ignoring them
    would report a partial sum as a total anatomical size.

    Output IDs are readable strings containing the connectome, endpoints, and
    measurement type. Project scope is carried separately by ``project_id``.
    If every input row has the same non-null ``synapse_table_id``, that value
    is preserved as optional provenance; it does not affect grouping or output
    identity.
    """
    missing = [
        column for column in _CELL_CELL_IDENTITY_COLUMNS if column not in synapses
    ]
    if missing:
        raise ValueError(f"Missing required synapse columns: {missing}")
    if not isinstance(connectome_id, str) or not connectome_id.strip():
        raise ValueError("connectome_id must be a non-empty string")
    if (size_column is None) != (size_unit is None):
        raise ValueError("size_column and size_unit must be supplied together")

    try:
        modality_value = Modality(modality).value
    except ValueError as error:
        raise ValueError(f"Unsupported modality: {modality!r}") from error

    size_unit_value: str | None = None
    if size_column is not None:
        if size_column not in synapses:
            raise ValueError(f"Size column not found: {size_column!r}")
        if not synapses.schema[size_column].is_numeric():
            raise TypeError(f"Size column {size_column!r} must be numeric")
        if synapses[size_column].null_count():
            raise ValueError(f"Size column {size_column!r} contains null values")
        try:
            size_unit_value = Unit(size_unit).value
        except ValueError as error:
            raise ValueError(f"Unsupported size unit: {size_unit!r}") from error

    null_columns = [
        column
        for column in _CELL_CELL_IDENTITY_COLUMNS
        if synapses[column].null_count()
    ]
    if null_columns:
        raise ValueError(f"Identity columns contain null values: {null_columns}")
    if synapses.is_empty():
        return pl.DataFrame(schema=_CELL_CELL_OUTPUT_SCHEMA)

    synapse_table_id = _single_synapse_table_id(synapses)
    normalized = synapses.with_columns(
        pl.col(_CELL_CELL_IDENTITY_COLUMNS).cast(pl.String)
    )
    count_rows = normalized.group_by(
        _CELL_CELL_IDENTITY_COLUMNS,
        maintain_order=True,
    ).len(name="value")
    count_rows = _shape_cell_cell_measurements(
        count_rows,
        connectome_id=connectome_id,
        synapse_table_id=synapse_table_id,
        modality=modality_value,
        measurement_type=SynapticMeasurementType.SYNAPSE_COUNT.value,
        unit=Unit.COUNT.value,
    )

    measurements = [count_rows]
    if size_column is not None and size_unit_value is not None:
        size_rows = normalized.group_by(
            _CELL_CELL_IDENTITY_COLUMNS,
            maintain_order=True,
        ).agg(pl.col(size_column).sum().cast(pl.Float64).alias("value"))
        measurements.append(
            _shape_cell_cell_measurements(
                size_rows,
                connectome_id=connectome_id,
                synapse_table_id=synapse_table_id,
                modality=modality_value,
                measurement_type=SynapticMeasurementType.SUM_ANATOMICAL_SIZE.value,
                unit=size_unit_value,
            )
        )

    return pl.concat(measurements)


def _shape_cell_cell_measurements(
    measurements: pl.DataFrame,
    *,
    connectome_id: str,
    synapse_table_id: str | None,
    modality: str,
    measurement_type: str,
    unit: str,
) -> pl.DataFrame:
    measurements = measurements.with_columns(
        pl.lit(None, dtype=pl.String).alias("description"),
        pl.lit(connectome_id).alias("connectome_id"),
        pl.lit(synapse_table_id, dtype=pl.String).alias("synapse_table_id"),
        pl.lit(measurement_type).alias("measurement_type"),
        pl.lit(modality).alias("modality"),
        pl.col("value").cast(pl.Float64),
        pl.lit(unit).alias("unit"),
    )
    return measurements.with_columns(
        pl.struct(
            [
                "project_id",
                "connectome_id",
                "presynaptic_cell",
                "postsynaptic_cell",
                "measurement_type",
            ]
        )
        .map_elements(_cell_cell_measurement_id, return_dtype=pl.String)
        .alias("id")
    ).select(_CELL_CELL_OUTPUT_SCHEMA.keys())


def _cell_cell_measurement_id(identity: dict[str, str]) -> str:
    return "_".join(
        [
            identity["connectome_id"],
            identity["presynaptic_cell"],
            identity["postsynaptic_cell"],
            identity["measurement_type"],
        ]
    )


def _single_synapse_table_id(synapses: pl.DataFrame) -> str | None:
    """Return one complete source-table value, otherwise no provenance."""
    if "synapse_table_id" not in synapses.columns:
        return None
    values = synapses["synapse_table_id"]
    if values.null_count():
        return None
    unique = values.unique().to_list()
    return str(unique[0]) if len(unique) == 1 else None


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
