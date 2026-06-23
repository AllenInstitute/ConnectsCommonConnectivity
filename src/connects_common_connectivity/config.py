"""Package-wide settings discovered from a repo-root ``ccc_config.yaml``.

Configuration is a *mechanism* here; the *values* live in a single
version-controlled ``ccc_config.yaml`` at the repo root. Every entry point
(CLI, writers/readers, notebooks, future plotting/analysis) calls
:func:`get_settings`, which walks up from ``cwd`` to find that file,
validates it with pydantic, and returns a cached :class:`Settings`.

No notebook setup cell, no ``%run``, no process-global mutation.

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

    def describe(self) -> str:
        """Return a human-readable summary of the resolved settings."""
        return (
            f"Settings(output_root={self.output_root!s}, dry_run={self.dry_run})"
        )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return self.describe()


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


def table_path(settings: Settings, table: str) -> Path:
    """Resolve the on-disk path for a named Delta/Parquet table subdir.

    ``table`` should be one of the canonical subdir names used by the
    notebooks (e.g. ``"dataset"``, ``"dataitem"``,
    ``"dataitem_dataset_association"``, ``"cellfeatureset"``,
    ``"cellfeaturematrix"``, ``"cluster"``, ``"clusterhierarchy"``,
    ``"clustermembership"``, ``"mappingset"``, ``"celltoclustermapping"``,
    ``"projectionmeasurementmatrix"``). Callers pass the exact name so
    nothing concatenates path strings ad hoc.
    """
    return Path(settings.output_root) / table


def output_root(settings: Optional[Settings] = None, *, absolute: bool = False) -> str:
    """Return ``output_root`` as a string with a trailing ``/``.

    Resolution rule (the bit that makes notebooks Just Work): a relative
    ``output_root`` in ``ccc_config.yaml`` is anchored at the config file's
    directory (the repo root), not at ``cwd``. So a notebook running in
    ``code/`` and a script running at the repo root both point at the same
    place. By default this function then returns the path **relative to the
    current working directory**, so a notebook in ``code/`` sees
    ``"../scratch/<project>/"`` while a process at the repo root sees
    ``"scratch/<project>/"``. Pass ``absolute=True`` to get the fully
    resolved absolute path instead.

    Prefer :func:`table_path` for new code — it returns a typed :class:`Path`
    for a named table subdir and is cwd-independent.
    """
    s = settings if settings is not None else get_settings()
    abs_path = Path(s.output_root)
    if not abs_path.is_absolute():
        abs_path = Path(os.path.abspath(abs_path))
    if absolute:
        text = str(abs_path)
    else:
        try:
            text = os.path.relpath(abs_path, Path.cwd())
        except ValueError:
            # Different drives on Windows — fall back to absolute.
            text = str(abs_path)
    return text if text.endswith("/") else text + "/"


__all__ = [
    "CONFIG_FILENAME",
    "Settings",
    "find_config_file",
    "get_settings",
    "output_root",
    "table_path",
]
