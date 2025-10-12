"""Connects Common Connectivity package.

Provides utilities to load LinkML schemas and dynamically produce Pydantic models for use in
Python data processing. See `schemas/` for the authoritative model definitions.
"""
from __future__ import annotations

from importlib import resources as _resources
from pathlib import Path as _Path
from functools import lru_cache as _lru_cache
from typing import Any, Dict, Type

__all__ = ["get_schema_path", "load_schema_text", "generate_pydantic_models", "__version__"]

__version__ = "0.1.0"


def get_schema_path(name: str = "connectivity_schema.yaml") -> str:
    """Return absolute path to a schema file stored under the package's schemas directory.

    Parameters
    ----------
    name: str
        Schema filename relative to the installed package schema directory.
    """
    with _resources.as_file(_resources.files(__package__) / "../.." / "schemas" / name) as p:
        return str(p.resolve())


def load_schema_text(name: str = "connectivity_schema.yaml") -> str:
    path = get_schema_path(name)
    return _Path(path).read_text(encoding="utf-8")


@_lru_cache(maxsize=1)
def generate_pydantic_models(name: str = "connectivity_schema.yaml") -> Dict[str, Type[Any]]:
    """Generate Pydantic models from the LinkML schema.

    Notes
    -----
    This provides a very thin wrapper. For complex cases you may instead want to run
    the linkml code generation CLI and import static code. For early prototyping we
    create models dynamically.
    """
    from linkml_runtime.loaders.yaml_loader import YAMLLoader  # type: ignore
    from linkml_runtime import SchemaView  # type: ignore
    from pydantic import BaseModel, create_model
    import yaml

    schema_text = load_schema_text(name)
    schema_obj = yaml.safe_load(schema_text)
    sv = SchemaView(schema_obj)

    models: Dict[str, Type[BaseModel]] = {}

    # Iterate through classes in schema and create simplistic Pydantic models
    for cname, cdef in sv.all_classes().items():  # type: ignore[attr-defined]
        # Collect attributes / slots
        fields: Dict[str, tuple[Any, Any]] = {}
        for slot_name in sv.class_slots(cname):  # type: ignore[attr-defined]
            slot = sv.induced_slot(slot_name, cname)  # type: ignore[attr-defined]
            py_type: Any = (str if slot.range in ("string", None) else float if slot.range == "float" else int if slot.range == "integer" else Any)
            default = None if slot.required else None
            fields[slot.name] = (py_type, default)
        model = create_model(cname, **fields)  # type: ignore[arg-type]
        models[cname] = model

    return models
