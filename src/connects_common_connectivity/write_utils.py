"""Deprecated re-export shim — moved to :mod:`connects_common_connectivity.io.write_utils`.

This module exists to avoid breaking notebooks and external imports while the
codebase is mid-migration to the ``io/`` layer. It will be removed in W6.
"""
from .io.write_utils import *  # noqa: F401,F403  (deprecated; removed in W6)
from .io.write_utils import (  # noqa: F401
    append_new_dataitems,
    populate_region_coverage,
    walk_ancestors,
)
