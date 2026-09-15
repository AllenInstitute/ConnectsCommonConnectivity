"""Dispatch core for IO-layer Delta writers.

A single public entry point — :func:`write_models` — normalizes one model or
an iterable into a non-empty, homogeneous exact-type batch, then routes the
write through the :class:`~connects_common_connectivity.io.write_spec.WriteSpec`
registered for that concrete class. The only standalone writer is
:func:`write_projection_matrix`, which exists because its signature is
genuinely non-uniform (it accepts a dense matrix alongside the model).

Class-specific behavior lives in the registry, never here. Callers
discover what is writable via :data:`WRITABLE_CLASSES`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
from deltalake import write_deltalake
from numpy.typing import ArrayLike
from pydantic import BaseModel

from connects_common_connectivity.config import Settings, get_settings
from connects_common_connectivity.io.arrow_utils import (
    attach_linkml_metadata,
    build_arrow_schema,
    models_to_table,
)
from connects_common_connectivity.io.write_spec import REGISTRY, WriteSpec, get_spec
from connects_common_connectivity.io.write_utils import (
    append_new_dataitems,
    populate_region_coverage,
)
from connects_common_connectivity.io.write_validation import validate_for_write
from connects_common_connectivity.models import ProjectionMeasurementMatrix

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WrittenResult:
    """Return value of a single :func:`write_models` invocation.

    ``predicates`` is one entry per scope group for ``overwrite_scoped``
    writes; an empty tuple for ``append_new_by_id`` (no predicate is
    issued because existing ids are checked before the Delta append).
    """

    class_name: str
    path: Path
    mode: str
    predicates: tuple[str, ...]
    rows_written: int


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


WRITABLE_CLASSES: tuple[type[BaseModel], ...] = tuple(
    spec.model_cls for spec in REGISTRY.values()
)


# ---------------------------------------------------------------------------
# Helpers (private; tested directly)
# ---------------------------------------------------------------------------


def _normalize_models(models: BaseModel | Iterable[BaseModel]) -> list[BaseModel]:
    """Normalize one model or iterable to a non-empty exact-type model list.

    Iterable inputs, including one-shot generators, are materialized exactly
    once. This is the write path's only shape conversion boundary; downstream
    validation receives the resulting sequence without further coercion.

    Parameters
    ----------
    models:
        One pydantic model or an iterable whose members all have the same
        exact pydantic model type. Strings and bytes are not model iterables.

    Returns
    -------
    list[BaseModel]
        A new non-empty list. A single model is wrapped without copying it;
        iterable members retain their identities and input order.

    Raises
    ------
    TypeError
        If the input is neither a model nor an iterable of models, or if an
        iterable contains a non-model or mixed exact model types.
    ValueError
        If the iterable is empty.
    """
    if isinstance(models, BaseModel):
        return [models]
    if isinstance(models, (str, bytes)) or not isinstance(models, Iterable):
        raise TypeError(
            f"write_models expected a pydantic model or iterable of models; "
            f"got {type(models).__name__}"
        )
    materialized = list(models)
    if len(materialized) == 0:
        raise ValueError("write_models received an empty batch")

    items: list[BaseModel] = []
    batch_type: type[BaseModel] | None = None
    for index, item in enumerate(materialized):
        if not isinstance(item, BaseModel):
            raise TypeError(
                "write_models expected pydantic models; "
                f"item at index {index} has type {type(item).__name__}"
            )
        if batch_type is None:
            batch_type = type(item)
        elif type(item) is not batch_type:
            raise TypeError(
                "write_models requires homogeneous exact types; "
                f"item at index {index} has type {type(item).__name__}, "
                f"expected {batch_type.__name__}"
            )
        items.append(item)
    return items


def _format_value(v: Any) -> str:
    """Render one scope value as a Delta predicate literal.

    Parameters
    ----------
    v:
        Scope value to render. ``None`` represents SQL ``NULL``; every other
        value is stringified, single-quoted, and has embedded quotes escaped.

    Returns
    -------
    str
        The literal text inserted into a predicate expression.
    """
    if v is None:
        return "NULL"
    return "'" + str(v).replace("'", "''") + "'"


def _build_predicate(scope_columns: Sequence[str], row_values: Sequence[Any]) -> str:
    """Build an AND-joined ``col = 'val'`` predicate for ``write_deltalake``.

    The format is exactly ``col1 = 'val1' AND col2 = 'val2'`` — single
    quotes, AND-joined, no extra whitespace beyond the single space around
    each operator. Notebooks that compose predicates by hand use the same
    format; this helper is the canonical implementation.

    Parameters
    ----------
    scope_columns:
        Ordered column names that define one overwrite scope.
    row_values:
        Values for those columns in matching positional order.

    Returns
    -------
    str
        The conjunction of one equality expression per column, or an empty
        string when both sequences are empty.

    Raises
    ------
    ValueError
        If the column and value sequences have different lengths.
    """
    if len(scope_columns) != len(row_values):
        raise ValueError(
            f"scope_columns ({len(scope_columns)}) and row_values "
            f"({len(row_values)}) length mismatch"
        )
    parts = [f"{c} = {_format_value(v)}" for c, v in zip(scope_columns, row_values)]
    return " AND ".join(parts)


def _group_by_scope(
    table: pa.Table, scope_columns: Sequence[str]
) -> list[tuple[tuple, pa.Table]]:
    """Partition ``table`` into one ``(scope_tuple, sub_table)`` per scope group.

    Scope groups preserve row order within each group. Two rows belong to
    the same group iff they have equal values across every column in
    ``scope_columns``. Order of groups is the order of first appearance.

    Parameters
    ----------
    table:
        Arrow table containing every requested scope column.
    scope_columns:
        Non-empty ordered column names whose combined values define a group.

    Returns
    -------
    list[tuple[tuple, pyarrow.Table]]
        Scope-value tuples paired with the corresponding row subsets. An
        empty table produces an empty list.

    Raises
    ------
    ValueError
        If ``scope_columns`` is empty.
    """
    if len(scope_columns) == 0:
        raise ValueError("scope_columns must be non-empty for overwrite_scoped writes")

    cols = [table.column(c).to_pylist() for c in scope_columns]
    keys: list[tuple] = list(zip(*cols)) if cols else []

    seen: dict[tuple, list[int]] = {}
    for i, key in enumerate(keys):
        seen.setdefault(key, []).append(i)

    return [(key, table.take(pa.array(idxs))) for key, idxs in seen.items()]


# ---------------------------------------------------------------------------
# Dispatch branches
# ---------------------------------------------------------------------------


def _dispatch_overwrite_scoped(
    table: pa.Table, spec: WriteSpec, path: Path
) -> WrittenResult:
    """Replace the Delta rows in each scope represented by a prepared batch.

    Parameters
    ----------
    table:
        A write-ready Arrow table containing every column named by
        ``spec.scope_columns`` and ``spec.partition_by``. Validation, schema
        alignment, and LinkML metadata attachment must already be complete.
    spec:
        The batch's write policy. Its non-empty ``scope_columns`` define the
        row groups and overwrite predicates; its ``partition_by`` columns are
        forwarded to every Delta write.
    path:
        The complete Delta table directory, including the output root and
        ``spec.subdir``.

    Returns
    -------
    WrittenResult
        The model class and Delta path, ``"overwrite_scoped"`` mode, one
        predicate per scope group in first-appearance order, and the total
        number of rows submitted across all writes.

    Raises
    ------
    ValueError
        If the spec has no scope columns.

    Notes
    -----
    This function performs one predicated Delta overwrite per distinct scope
    tuple. Rows matching each predicate are replaced; other scopes remain
    untouched.
    """
    groups = _group_by_scope(table, spec.scope_columns)
    predicates: list[str] = []
    rows_written = 0
    partition_by = spec.partition_by or None
    for key, sub in groups:
        predicate = _build_predicate(spec.scope_columns, key)
        write_deltalake(
            str(path),
            sub,
            mode="overwrite",
            predicate=predicate,
            partition_by=partition_by,
        )
        predicates.append(predicate)
        rows_written += sub.num_rows
    return WrittenResult(
        class_name=spec.model_cls.__name__,
        path=path,
        mode="overwrite_scoped",
        predicates=tuple(predicates),
        rows_written=rows_written,
    )


def _dispatch_append_new_by_id(
    table: pa.Table, spec: WriteSpec, path: Path
) -> WrittenResult:
    """Append rows with ids not yet stored for one project.

    Parameters
    ----------
    table:
        A write-ready Arrow table for exactly one ``project_id``. Validation,
        schema alignment, and LinkML metadata attachment must already be
        complete, and the table must contain the id column named by the first
        entry in ``spec.scope_columns``.
    spec:
        The batch's write policy. Its first scope column is used as the id
        column for the existing-row check; additional scope columns, if any,
        do not participate in deduplication.
    path:
        The complete Delta table directory, including the output root and
        ``spec.subdir``.

    Returns
    -------
    WrittenResult
        The model class and Delta path, ``"append_new_by_id"`` mode, no
        predicates, and the number of rows actually appended.

    Raises
    ------
    ValueError
        If the spec has no scope columns, the table lacks ``project_id``, or
        the table does not contain exactly one distinct ``project_id``.

    Notes
    -----
    This function performs Delta IO through :func:`append_new_dataitems`.
    Duplicate prevention has that helper's sequential, readable-table
    guarantees; it does not provide concurrency control.
    """
    if len(spec.scope_columns) == 0:
        raise ValueError(
            f"{spec.model_cls.__name__}: scope_columns is empty for append_new_by_id "
            f"(expected the id column at index 0)"
        )
    id_column = spec.scope_columns[0]

    if "project_id" not in table.column_names:
        raise ValueError(
            f"{spec.model_cls.__name__}: append_new_by_id requires a 'project_id' "
            f"column on every row (got columns {table.column_names!r})"
        )
    project_ids = set(table.column("project_id").to_pylist())
    if len(project_ids) != 1:
        raise ValueError(
            f"{spec.model_cls.__name__}: append_new_by_id requires a single "
            f"project_id per call (got {sorted(project_ids)!r}). Split the "
            f"batch upstream."
        )
    (project_id,) = project_ids

    rows_written = append_new_dataitems(
        str(path), table, project_id=project_id, id_column=id_column
    )
    return WrittenResult(
        class_name=spec.model_cls.__name__,
        path=path,
        mode="append_new_by_id",
        predicates=(),
        rows_written=rows_written,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _resolve_output_root(
    settings: Settings | None,
    output_root: str | Path | None,
) -> tuple[Path, Settings | None]:
    """Resolve the output root and settings state used by one write call.

    Parameters
    ----------
    settings:
        Explicit configuration supplying the output root and write controls
        such as ``dry_run``. When omitted with no root override, configuration
        is discovered through :func:`get_settings`.
    output_root:
        Per-call root override. The caller later combines this path with
        ``spec.subdir`` to form the complete Delta table directory. It is
        mutually exclusive with ``settings``.

    Returns
    -------
    Path
        The effective output root later combined with ``spec.subdir``.
    Settings or None
        The explicit or discovered settings retained so the caller can honor
        ``dry_run``. This value is ``None`` when ``output_root`` is explicit,
        so no settings-based ``dry_run`` policy applies to a root override.

    Raises
    ------
    TypeError
        If both ``settings`` and ``output_root`` are supplied.
    """
    if output_root is not None and settings is not None:
        raise TypeError(
            "Pass either settings= or output_root=, not both. "
            "output_root= is the per-call override; settings= carries the "
            "full Settings object."
        )
    if output_root is not None:
        return Path(output_root), None
    resolved = settings or get_settings()
    return Path(resolved.output_root), resolved


def write_models(
    models: BaseModel | Iterable[BaseModel],
    *,
    settings: Settings | None = None,
    output_root: str | Path | None = None,
) -> WrittenResult:
    """Write generated pydantic models to the shared Delta lake.

    This public boundary normalizes input into a non-empty homogeneous batch
    whose members have one exact concrete type. That type is resolved through
    the global :class:`WriteSpec` registry, which remains authoritative for
    write dispatch and validation policy.

    Parameters
    ----------
    models:
        A single model instance or a non-empty iterable of instances of the
        same exact class. Iterables are materialized exactly once, and the
        concrete class must be one of :data:`WRITABLE_CLASSES`.
    settings:
        Optional explicit settings. Falls back to :func:`get_settings` when
        omitted; an explicit ``settings=`` always wins over the discovered
        config (matches the precedence documented in
        :mod:`connects_common_connectivity.config`).
    output_root:
        Optional per-call override of the on-disk root under which the
        canonical ``spec.subdir`` is written. Use this when a single
        notebook/dataset should write to a different location than the
        shared ``ccc_config.yaml`` ``output_root`` (e.g. an isolated test
        dataset). Mutually exclusive with ``settings=`` — passing both
        raises ``TypeError``. Because no settings object is resolved for an
        explicit root, settings-based ``dry_run`` handling does not apply.

    Returns
    -------
    WrittenResult
        Class name, on-disk path, dispatch mode, the predicates issued (one
        per scope group for ``overwrite_scoped``; empty for
        ``append_new_by_id``), and the number of rows written.

    Raises
    ------
    TypeError
        If the input cannot form one homogeneous exact-type model batch, or
        if both ``settings`` and ``output_root`` are supplied.
    ValueError
        If the batch is empty, fails write-required validation, violates a
        dispatch invariant, or resolves to an unsupported write mode.
    KeyError
        If the batch's exact model class has no registered write spec.

    Notes
    -----
    Unless settings enable ``dry_run``, this function writes to the Delta
    table at ``output_root / spec.subdir``. A dry run performs normalization,
    registry lookup, and validation but no Arrow conversion or Delta IO, and
    returns zero rows and no predicates. Use ``WRITABLE_CLASSES`` to enumerate
    supported exact model classes at runtime.
    """
    items = _normalize_models(models)
    cls = type(items[0])
    spec = get_spec(cls)

    items = validate_for_write(items, spec)

    root, resolved_settings = _resolve_output_root(settings, output_root)
    path = root / spec.subdir

    if resolved_settings is not None and resolved_settings.dry_run:
        return WrittenResult(
            class_name=spec.model_cls.__name__,
            path=path,
            mode=spec.write_mode,
            predicates=(),
            rows_written=0,
        )

    schema = build_arrow_schema(cls)
    table = models_to_table(items, schema=schema)
    table = attach_linkml_metadata(table, linkml_class=cls.__name__)

    if spec.write_mode == "overwrite_scoped":
        return _dispatch_overwrite_scoped(table, spec, path)
    if spec.write_mode == "append_new_by_id":
        return _dispatch_append_new_by_id(table, spec, path)
    raise ValueError(
        f"{cls.__name__}: unsupported write_mode {spec.write_mode!r}. "
        f"Add a dispatch branch in writers.py."
    )


def write_projection_matrix(
    pmm: ProjectionMeasurementMatrix,
    matrix: ArrayLike,
    *,
    settings: Settings | None = None,
    output_root: str | Path | None = None,
) -> WrittenResult:
    """Enrich ``pmm`` with derived ``region_coverage`` and write it.

    Parameters
    ----------
    pmm:
        Projection metadata whose ``region_index`` defines the matrix column
        order. The input model is not mutated.
    matrix:
        Dense cell-by-region values used to derive ``region_coverage`` before
        delegating the enriched copy to :func:`write_models`.
    settings:
        Optional write configuration with the same resolution and ``dry_run``
        semantics as :func:`write_models`.
    output_root:
        Optional per-call output-root override. Mutually exclusive with
        ``settings`` and combined with the projection write spec's subdir.

    Returns
    -------
    WrittenResult
        The result of writing the enriched projection model, including its
        Delta table path, predicates, and row count.

    Raises
    ------
    ValueError
        If ``region_index`` is absent, the matrix is not two-dimensional, or
        its column count differs from the region index length.
    TypeError
        If both ``settings`` and ``output_root`` are supplied.

    Notes
    -----
    The derived copy follows the same validation and Delta IO path as
    :func:`write_models`.
    """
    enriched = populate_region_coverage(pmm, matrix)
    return write_models(enriched, settings=settings, output_root=output_root)


def write_cellcellconnectivitylong(
    *args: Any, **kwargs: Any
) -> WrittenResult:
    """Placeholder writer for ``CellCellConnectivityLong`` rows.

    Not implemented. ``CellCellConnectivityLong`` has no ``WriteSpec`` entry
    yet, and the existing ETL notebooks (``etl_minnie_04_cell_cell.ipynb``,
    ``parse_minnie_clustering.ipynb``) write to non-canonical, run-specific
    subdirs (e.g. ``cellcellconnectivitylong_proofread_pre_to_csm_post/``)
    rather than the canonical ``cellcellconnectivitylong/`` subdir that
    ``write_models`` would resolve. Until either (a) those callers
    consolidate onto the canonical subdir and a ``WriteSpec`` is added, or
    (b) dispatch is extended to accept a per-call subdir override, those
    notebooks keep using ``write_deltalake`` directly. This stub exists as
    a reminder of that open work.

    Parameters
    ----------
    *args:
        Ignored positional arguments reserved for a future writer contract.
    **kwargs:
        Ignored keyword arguments reserved for a future writer contract.

    Raises
    ------
    NotImplementedError
        Always; this model has no registered write path.
    """
    raise NotImplementedError(
        "write_cellcellconnectivitylong is not implemented yet; "
        "see the docstring for the migration plan."
    )


__all__ = [
    "WRITABLE_CLASSES",
    "WrittenResult",
    "write_models",
    "write_projection_matrix",
    "write_cellcellconnectivitylong",
]
