"""Connects Common Connectivity package.

Provides utilities to load LinkML schemas and dynamically produce Pydantic models for use in
Python data processing. See `schemas/` for the authoritative model definitions.
"""
from __future__ import annotations

from importlib import resources as _resources
from pathlib import Path as _Path
from functools import lru_cache as _lru_cache
from typing import Any, Dict, Type, List, Optional, get_args, get_origin
from datetime import datetime as _Datetime
from enum import Enum as _Enum

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
    """Generate richer Pydantic v2 models dynamically from the LinkML schema.

    Features:
      * Proper import resolution (pass schema path to SchemaView)
      * Enum generation for LinkML enums
      * Primitive & custom type constraints (patterns, min/max)
      * Multivalued slot -> List[T]
      * Required slot enforcement (Field(...))
      * Cross-class references resolved via forward refs

    Limitations:
      * Does not implement full LinkML inlining/containment semantics
      * Does not coerce JsonObject strings to dict automatically
      * Numeric bounds on Probability applied if present
    """
    from linkml_runtime import SchemaView  # type: ignore
    from pydantic import BaseModel, Field, create_model

    schema_path = get_schema_path(name)
    sv = SchemaView(schema_path)  # ensure imports are resolved via file path

    models: Dict[str, Type[BaseModel]] = {}
    enum_types: Dict[str, Type[_Enum]] = {}

    # Build enums
    for ename, edef in sv.all_enums().items():  # type: ignore[attr-defined]
        if not getattr(edef, 'permissible_values', None):
            continue
        # Map each permissible value key to itself (simple enumeration)
        enum_dict = {pv: pv for pv in edef.permissible_values.keys()}
        enum_types[ename] = _Enum(ename, enum_dict)  # type: ignore[arg-type]

    # Primitive map & custom types
    primitive_map: Dict[str, Any] = {
        "string": str,
        "integer": int,
        "float": float,
        "boolean": bool,
        "datetime": _Datetime,
    }

    # Pre-create empty classes (first pass) so forward references can resolve
    for cname in sv.all_classes().keys():  # type: ignore[attr-defined]
        models[cname] = type(cname, (BaseModel,), {})

    # Helper to resolve a LinkML range to a Python type
    def _resolve_type(range_name: Optional[str], multivalued: bool) -> Any:
        rng = range_name or "string"
        py_type: Any
        if rng in primitive_map:
            py_type = primitive_map[rng]
        elif rng in enum_types:
            py_type = enum_types[rng]
        elif rng in models:  # class reference
            py_type = models[rng]
        else:
            # Try type facets (custom types) else fallback to str
            tdef = sv.get_type(rng) if sv.get_type(rng) else None  # type: ignore[attr-defined]
            if tdef and tdef.base in primitive_map:
                py_type = primitive_map[tdef.base]
            else:
                py_type = Any
        if multivalued:
            from typing import List as _List
            py_type = _List[py_type]
        return py_type

    # Second pass: build models with fields
    for cname, cdef in sv.all_classes().items():  # type: ignore[attr-defined]
        field_defs: Dict[str, tuple[Any, Any]] = {}
        for slot_name in sv.class_slots(cname):  # type: ignore[attr-defined]
            # LinkML runtime SchemaView.induced_slot signature expects (slot_name, class_name)
            slot = sv.induced_slot(slot_name, cname)  # type: ignore[attr-defined]
            py_type = _resolve_type(slot.range, bool(getattr(slot, 'multivalued', False)))

            # Constraints from slot or its type
            field_kwargs: Dict[str, Any] = {}
            if getattr(slot, 'description', None):
                field_kwargs['description'] = slot.description
            # Pattern from slot or type
            pattern = getattr(slot, 'pattern', None)
            tdef = sv.get_type(slot.range) if slot.range else None  # type: ignore[attr-defined]
            if not pattern and tdef is not None:
                pattern = getattr(tdef, 'pattern', None)
            if pattern:
                field_kwargs['pattern'] = pattern
            # Numeric bounds (e.g., Probability type)
            if tdef is not None:
                minv = getattr(tdef, 'minimum_value', None)
                maxv = getattr(tdef, 'maximum_value', None)
                if minv is not None:
                    field_kwargs['ge'] = minv
                if maxv is not None:
                    field_kwargs['le'] = maxv

            required = bool(getattr(slot, 'required', False))
            default = ... if required else None
            field_defs[slot.name] = (py_type, Field(default, **field_kwargs))

        # Re-create model with collected field definitions
        models[cname] = create_model(cname, __base__=models[cname], **field_defs)  # type: ignore[arg-type]

    # Resolve forward references & rebuild to apply annotations (Pydantic v2)
    for m in models.values():
        try:
            m.model_rebuild()
        except Exception:  # pragma: no cover
            pass

    # Attach enums in output too (optional convenience)
    models.update(enum_types)

    return models
