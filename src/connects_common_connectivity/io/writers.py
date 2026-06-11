"""Dispatch core for IO-layer Delta writers.

A single public entry point — :func:`write_models` — accepts a homogeneous
batch of generated pydantic models and routes the write through the
:class:`~connects_common_connectivity.io.write_spec.WriteSpec` registered
for that class. The only standalone writer is
:func:`write_projection_matrix`, which exists because its signature is
genuinely non-uniform (it accepts a dense matrix alongside the model).

Class-specific behavior lives in the registry, never here. Callers
discover what is writable via :data:`WRITABLE_CLASSES`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow as pa
from deltalake import write_deltalake
from pydantic import BaseModel

from ..config import Settings, get_settings, table_path
from .arrow_utils import attach_linkml_metadata, build_arrow_schema, models_to_table
from .write_spec import REGISTRY, WriteSpec, get_spec
from .write_utils import append_new_dataitems, populate_region_coverage

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WriteResult:
    """Return value of a single :func:`write_models` invocation.

    ``predicates`` is one entry per scope group for ``overwrite_scoped``
    writes; an empty tuple for ``append_new_by_id`` (no predicate is
    issued — Delta append + id-dedupe handles idempotency).
    """

    class_name: str
    path: Path
    mode: str
    predicates: tuple[str, ...]
    rows_written: int


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


WRITABLE_CLASSES: tuple[type, ...] = tuple(
    spec.model_cls for spec in REGISTRY.values()
)


# ---------------------------------------------------------------------------
# Validation hook (replaced by W5)
# ---------------------------------------------------------------------------


def _validation_hook(models: Sequence[BaseModel], spec: WriteSpec) -> Sequence[BaseModel]:
    """Pass-through identity hook; W5 monkey-patches this to enforce invariants."""
    return models


# ---------------------------------------------------------------------------
# Helpers (private; tested directly)
# ---------------------------------------------------------------------------


def _normalize_models(models: Any) -> list[BaseModel]:
    """Coerce ``models`` to a list, accepting a single model or any iterable.

    Requires homogeneous type. Empty input is rejected — callers always
    know which class they are writing.
    """
    if isinstance(models, BaseModel):
        return [models]
    if isinstance(models, (str, bytes)) or not isinstance(models, Iterable):
        raise TypeError(
            f"write_models expected a pydantic model or iterable of models; "
            f"got {type(models).__name__}"
        )
    items = list(models)
    if not items:
        raise ValueError("write_models received an empty batch")
    cls = type(items[0])
    for m in items:
        if type(m) is not cls:
            raise TypeError(
                f"write_models requires homogeneous types; got "
                f"{cls.__name__} and {type(m).__name__}"
            )
    return items


def _format_value(v: Any) -> str:
    """Render ``v`` as a single-quoted SQL literal for the Delta predicate."""
    if v is None:
        return "NULL"
    return "'" + str(v).replace("'", "''") + "'"


def _build_predicate(scope_columns: Sequence[str], row_values: Sequence[Any]) -> str:
    """Build an AND-joined ``col = 'val'`` predicate for ``write_deltalake``.

    The format is exactly ``col1 = 'val1' AND col2 = 'val2'`` — single
    quotes, AND-joined, no extra whitespace beyond the single space around
    each operator. Notebooks that compose predicates by hand use the same
    format; this helper is the canonical implementation.
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
    """
    if not scope_columns:
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
) -> WriteResult:
    """Group by scope, issue one predicated overwrite per group."""
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
    return WriteResult(
        class_name=spec.model_cls.__name__,
        path=path,
        mode="overwrite_scoped",
        predicates=tuple(predicates),
        rows_written=rows_written,
    )


def _dispatch_append_new_by_id(
    table: pa.Table, spec: WriteSpec, path: Path
) -> WriteResult:
    """Append only rows whose id is new, scoped to a single ``project_id``."""
    if not spec.scope_columns:
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
    return WriteResult(
        class_name=spec.model_cls.__name__,
        path=path,
        mode="append_new_by_id",
        predicates=(),
        rows_written=rows_written,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_models(models: Any, *, settings: Settings | None = None) -> WriteResult:
    """Write a batch of generated pydantic models to the shared Delta lake.

    The class is inferred from ``models`` and dispatched through its
    :class:`WriteSpec` (see :mod:`connects_common_connectivity.io.write_spec`).
    No per-class wrapper functions exist; renaming this function eight times
    would add no behavior, only drift surface.

    Parameters
    ----------
    models:
        A single model instance or a non-empty iterable of instances of the
        same class. The class must be one of :data:`WRITABLE_CLASSES`.
    settings:
        Optional explicit settings. Falls back to :func:`get_settings` when
        omitted; an explicit ``settings=`` always wins (matches the
        precedence documented in :mod:`connects_common_connectivity.config`).

    Returns
    -------
    WriteResult
        Class name, on-disk path, dispatch mode, the predicates issued (one
        per scope group for ``overwrite_scoped``; empty for
        ``append_new_by_id``), and the number of rows written.

    Notes
    -----
    Writable classes (the registry, in order):
    ``DataSet``, ``DataItem``, ``DataItemDataSetAssociation``,
    ``Cluster``, ``ClusterHierarchy``, ``ClusterMembership``,
    ``MappingSet``, ``CellToClusterMapping``,
    ``CellFeatureSet``, ``CellFeatureDefinition``, ``CellFeatureMatrix``,
    ``ProjectionMeasurementMatrix``.
    Use ``WRITABLE_CLASSES`` to enumerate at runtime.
    """
    items = _normalize_models(models)
    cls = type(items[0])
    spec = get_spec(cls)

    items = list(_validation_hook(items, spec))

    settings = settings or get_settings()
    schema = build_arrow_schema(cls)
    table = models_to_table(items, schema=schema)
    table = attach_linkml_metadata(table, linkml_class=cls.__name__)

    path = table_path(settings, spec.subdir)

    if spec.write_mode == "overwrite_scoped":
        return _dispatch_overwrite_scoped(table, spec, path)
    if spec.write_mode == "append_new_by_id":
        return _dispatch_append_new_by_id(table, spec, path)
    raise ValueError(
        f"{cls.__name__}: unsupported write_mode {spec.write_mode!r}. "
        f"Add a dispatch branch in writers.py."
    )


def write_projection_matrix(
    pmm: Any, matrix: Any, *, settings: Settings | None = None
) -> WriteResult:
    """Enrich ``pmm`` with derived ``region_coverage`` and write it.

    The single non-:func:`write_models` public writer, justified by the
    non-uniform signature: callers must hand in the dense ``matrix``
    alongside the model so coverage can be derived from it. The input
    ``pmm`` is not mutated — :func:`populate_region_coverage` returns a
    new instance.
    """
    enriched = populate_region_coverage(pmm, matrix)
    return write_models(enriched, settings=settings)


__all__ = [
    "WRITABLE_CLASSES",
    "WriteResult",
    "write_models",
    "write_projection_matrix",
]
