"""IO layer for ConnectsCommonConnectivity.

This package owns write/read backends and (re-)exports a few package-wide
helpers for convenience. The settings live in :mod:`connects_common_connectivity.config`;
they are re-exported here so IO callers can ``from connects_common_connectivity.io
import get_settings, table_path``.
"""

from __future__ import annotations

from ..config import Settings, get_settings, output_root, table_path

__all__ = ["Settings", "get_settings", "output_root", "table_path"]
