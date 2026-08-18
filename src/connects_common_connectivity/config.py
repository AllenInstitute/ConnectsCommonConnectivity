"""Package-wide settings discovered from a repo-root ``ccc_config.yaml``.

Configuration is a *mechanism* here; the *values* live in a single
version-controlled ``ccc_config.yaml`` at the repo root. Every entry point
(CLI, writers/readers, notebooks, future plotting/analysis) calls
:func:`get_settings`, which walks up from ``cwd`` to find that file,
validates it with pydantic, and returns a cached :class:`Settings`.

Resolution precedence (highest wins):

1. An explicit ``settings=`` argument passed by a caller.
2. ``CCC_OUTPUT_ROOT`` environment variable (overrides ``output_root`` only;
   it cannot express structured knobs like ``dry_run``).
3. The discovered ``ccc_config.yaml``.
4. Otherwise: a clear, actionable error.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

CONFIG_FILENAME = "ccc_config.yaml"


class Settings(BaseModel):
    """Validated, package-wide settings loaded from ``ccc_config.yaml``."""

    output_root: Path = Field(
        ...,
        description="Root directory under which Delta/Parquet tables are written.",
    )
    dry_run: bool = Field(
        default=False,
        description="If True, callers should log intended writes instead of executing them.",
    )

    model_config = {"extra": "forbid"}


def find_config_file(
    start: Optional[Path] = None,
    filename: str = CONFIG_FILENAME,
) -> Optional[Path]:
    """Walk up from ``start`` (default: ``cwd``) to the filesystem root looking
    for ``filename``.

    Returns the resolved path to the first match, or ``None`` if not found.
    Mirrors the discovery pattern used by ``pyproject.toml``, ``ruff``, and
    ``pytest`` — a notebook in ``code/`` finds the repo-root config with zero
    config code.
    """
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        path = candidate / filename
        if path.is_file():
            return path
    return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Discover ``ccc_config.yaml``, validate it, and return cached settings.

    Raises ``RuntimeError`` with an actionable message if no config file is
    discoverable from the current working directory.

    Tests can call ``get_settings.cache_clear()`` to force re-discovery.
    """
    path = find_config_file()
    if path is None:
        raise RuntimeError(
            f"No {CONFIG_FILENAME} found — create one at the repo root with "
            "output_root: <path>. Discovery walks up from the current working "
            "directory, like pyproject.toml/ruff/pytest."
        )

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"{path}: expected a YAML mapping at the top level, got {type(raw).__name__}."
        )

    config_dir = path.parent

    if "output_root" in raw and raw["output_root"] is not None:
        raw["output_root"] = _anchor_path(raw["output_root"], config_dir)

    env_override = os.environ.get("CCC_OUTPUT_ROOT")
    if env_override:
        # Env values come from the user's shell; anchor to cwd so they are
        # cwd-independent thereafter (matches shell intuition).
        raw["output_root"] = _anchor_path(env_override, Path.cwd())

    return Settings(**raw)


def _anchor_path(value, base: Path) -> Path:
    """Return ``value`` as an absolute :class:`Path`, anchored at ``base`` if relative.

    Uses :func:`os.path.abspath` rather than :meth:`Path.resolve`: abspath
    normalizes the path without following symlinks, so a symlinked
    ``scratch -> /scratch`` doesn't suddenly point outside the repo and
    relative-path output stays sensible (e.g. ``../scratch/x`` from ``code/``).
    """
    p = Path(value)
    if not p.is_absolute():
        p = base / p
    return Path(os.path.abspath(p))


__all__ = [
    "CONFIG_FILENAME",
    "Settings",
    "find_config_file",
    "get_settings",
]
