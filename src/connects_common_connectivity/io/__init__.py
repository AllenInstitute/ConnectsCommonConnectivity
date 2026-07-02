"""IO layer for ConnectsCommonConnectivity.

The IO layer owns the write/read path between generated pydantic models
and the shared Delta lake. This module is the curated public surface:
import from here for stable user code; everything else under ``io/`` is
internal plumbing.

Example::

    from connects_common_connectivity.io import write_models, write_projection_matrix
    from connects_common_connectivity.models import DataSet

    write_models(DataSet(id="ds1", name="example", project_id="p1"))
    write_projection_matrix(pmm, dense_matrix)
"""

from __future__ import annotations

from connects_common_connectivity.config import Settings, get_settings, table_path
from connects_common_connectivity.io.writers import (
    WRITABLE_CLASSES,
    WrittenResult,
    write_models,
    write_projection_matrix,
)

__all__ = [
    "get_settings",
    "Settings",
    "table_path",
    "write_models",
    "write_projection_matrix",
    "WrittenResult",
    "WRITABLE_CLASSES",
]
