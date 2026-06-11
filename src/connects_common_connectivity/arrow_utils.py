"""Deprecated re-export shim — moved to :mod:`connects_common_connectivity.io.arrow_utils`.

This module exists to avoid breaking notebooks and external imports while the
codebase is mid-migration to the ``io/`` layer. It will be removed in W6.
"""
from .io.arrow_utils import *  # noqa: F401,F403  (deprecated; removed in W6)
from .io.arrow_utils import __all__  # noqa: F401
