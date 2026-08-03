"""Connects Common Connectivity package.


"""
from __future__ import annotations

import inspect
from enum import Enum
from pathlib import Path
from typing import Dict, Type

from pydantic import BaseModel

from connects_common_connectivity import models as _models

__version__ = "0.3.0"

_INTERNAL_CLASSES = frozenset({"ConfiguredBaseModel", "LinkMLMeta"})


def generate_pydantic_models() -> Dict[str, Type]:
    """Return a dict mapping class/enum names to their types from the generated models module."""
    result: Dict[str, Type] = {}
    for name, obj in inspect.getmembers(_models, inspect.isclass):
        if name.startswith("_") or name in _INTERNAL_CLASSES:
            continue
        if issubclass(obj, (BaseModel, Enum)):
            result[name] = obj
    return result


def get_schema_path(schema_name: str = "connectivity_schema.yaml") -> str:
    """Resolve a schema filename to its absolute path in the schemas/ directory."""
    schemas_dir = Path(__file__).resolve().parent.parent.parent / "schemas"
    return str(schemas_dir / schema_name)

