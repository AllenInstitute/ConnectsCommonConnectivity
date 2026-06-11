"""Smoke test asserting public IO names are importable from BOTH paths.

The shims at the package root re-export from ``io/`` to keep notebooks
working through W6. This test pins that contract: anything notebooks
import today must keep working.
"""

from __future__ import annotations


def test_public_names_from_io_paths():
    from connects_common_connectivity.io.arrow_utils import (  # noqa: F401
        attach_linkml_metadata,
        build_arrow_schema,
        build_cell_feature_matrix_schema,
        models_to_table,
    )
    from connects_common_connectivity.io.write_utils import (  # noqa: F401
        append_new_dataitems,
        populate_region_coverage,
        walk_ancestors,
    )


def test_public_names_from_shim_paths():
    from connects_common_connectivity.arrow_utils import (  # noqa: F401
        attach_linkml_metadata,
        build_arrow_schema,
        build_cell_feature_matrix_schema,
        models_to_table,
    )
    from connects_common_connectivity.write_utils import (  # noqa: F401
        append_new_dataitems,
        populate_region_coverage,
        walk_ancestors,
    )


def test_shim_and_io_resolve_to_same_object():
    from connects_common_connectivity import arrow_utils as shim_arrow
    from connects_common_connectivity import write_utils as shim_write
    from connects_common_connectivity.io import arrow_utils as io_arrow
    from connects_common_connectivity.io import write_utils as io_write

    assert shim_arrow.build_arrow_schema is io_arrow.build_arrow_schema
    assert shim_arrow.models_to_table is io_arrow.models_to_table
    assert shim_arrow.attach_linkml_metadata is io_arrow.attach_linkml_metadata
    assert shim_arrow.build_cell_feature_matrix_schema is io_arrow.build_cell_feature_matrix_schema
    assert shim_write.append_new_dataitems is io_write.append_new_dataitems
    assert shim_write.walk_ancestors is io_write.walk_ancestors
    assert shim_write.populate_region_coverage is io_write.populate_region_coverage
