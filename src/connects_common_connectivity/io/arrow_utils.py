"""Utilities for converting Pydantic (LinkML-derived) models to PyArrow Tables efficiently.

Design goals:
- Avoid JSON serialization; work directly with Python-native values.
- Normalize Enum instances, nested Pydantic models, and object references.
- Provide schema construction from Pydantic field annotations for stability.
- Support column-oriented batch conversion for speed and memory efficiency.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, List, Dict, Union, get_origin, get_args
import hashlib
from datetime import datetime, date

import pyarrow as pa

try:
    from pydantic import BaseModel  # type: ignore
except ImportError:  # pragma: no cover
    class BaseModel:  # minimal placeholder to satisfy type hints
        pass

# Primitive Python -> Arrow type mapping (extend as needed)
PRIMITIVE_TYPE_MAP: Dict[Any, pa.DataType] = {
    int: pa.int64(),
    float: pa.float64(),
    bool: pa.bool_(),
    str: pa.string(),
    bytes: pa.binary(),
    datetime: pa.timestamp("us"),
    date: pa.date32(),
}


def normalize_value(v: Any) -> Any:
    """Recursively normalize a value so Arrow can ingest it.

    - Enum -> underlying value (typically str)
    - Pydantic model -> dict of normalized fields
    - list/dict -> element-wise normalization
    - Other primitives unchanged
    """
    if isinstance(v, Enum):
        return v.value
    if hasattr(v, "model_dump") and callable(getattr(v, "model_dump")):
        return {k: normalize_value(x) for k, x in v.model_dump(mode="python").items()}
    if isinstance(v, list):
        return [normalize_value(x) for x in v]
    if isinstance(v, dict):
        return {k: normalize_value(x) for k, x in v.items()}
    return v


def flatten_refs(row: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten object reference dicts to id form when they contain an identifier field.

    Example: {'parent_identifier': {'id': 'BR123'}} -> {'parent_identifier_id': 'BR123'}
    Leaves the original key removed for simpler Arrow schema.
    """
    for key, val in list(row.items()):
        # Single embedded reference -> promote to *_id
        if isinstance(val, dict):
            ident = val.get("id") or val.get("identifier")
            if ident is not None:  # heuristic: treat as ref not complex struct
                row[f"{key}"] = ident
                continue
        # List of simple embedded references -> replace with list of ids
        if isinstance(val, list) and val and all(isinstance(x, dict) for x in val):
            ids: List[Any] = []
            simple = True
            for x in val:
                ident = x.get("id") or x.get("identifier") if isinstance(x, dict) else None
                if ident is None:
                    simple = False
                    break
                ids.append(ident)
            if simple:
                row[key] = ids
    return row


def model_to_row(model: Any, *, flatten: bool = True) -> Dict[str, Any]:
    """Convert a single Pydantic model instance to a normalized row dict.

    Parameters
    ----------
    model: BaseModel
        Pydantic instance.
    flatten: bool
        If True, attempt to flatten object references to *_id columns.
    """
    raw = model.model_dump(mode="python", exclude_none=True)
    norm = {k: normalize_value(v) for k, v in raw.items()}
    return flatten_refs(norm) if flatten else norm


def _arrow_field_for(name: str, annotation: Any, required: bool) -> pa.Field:
    """Infer an Arrow Field from a Python type annotation.

    Handles Optional[T], List[T], Enums, and basic primitives.
    Fallback is string.
    """
    nullable = not required
    origin = get_origin(annotation)
    args = get_args(annotation)

    # Optional / Union[T, None]
    if origin is Union and type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            annotation = non_none[0]
        nullable = True
        origin = get_origin(annotation)
        args = get_args(annotation)

    # List[T]
    if origin is list and args:
        inner = args[0]
        inner_field = _arrow_field_for(name, inner, required=True)
        return pa.field(name, pa.list_(inner_field.type), nullable=True)

    # Enum
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return pa.field(name, pa.string(), nullable=nullable)

    # Primitive direct map
    if annotation in PRIMITIVE_TYPE_MAP:
        return pa.field(name, PRIMITIVE_TYPE_MAP[annotation], nullable=nullable)

    # Fallback struct for embedded BaseModel? treat as JSON/string for now.
    return pa.field(name, pa.string(), nullable=nullable)


def build_arrow_schema(model_cls: Any) -> pa.Schema:
    """Build a stable Arrow schema from a Pydantic model class.

    Uses model_fields (Pydantic v2) annotations; unknown types default to string.
    """
    fields: List[pa.Field] = []
    model_fields = getattr(model_cls, "model_fields", {})
    for fname, finfo in model_fields.items():
        annotation = getattr(finfo, "annotation", Any)
        required = finfo.is_required()
        fields.append(_arrow_field_for(fname, annotation, required))
    return pa.schema(fields)


def models_to_table(models: Iterable[Any], schema: Union[pa.Schema, None] = None, *, flatten: bool = True) -> pa.Table:
    """Convert an iterable of Pydantic models to a PyArrow Table.

    If no schema is provided, one is generated from the class of the first model.
    Column-oriented assembly for speed (single pass). Nested dicts become stringified.
    """
    models = list(models)
    if not models:
        return pa.Table.from_arrays([], schema=schema or pa.schema([]))

    # Build schema if absent
    if schema is None:
        schema = build_arrow_schema(models[0].__class__)
    # Schema must be a pyarrow.Schema now
    assert isinstance(schema, pa.Schema), "Schema construction failed"

    # Initialize column buffers
    buffers: Dict[str, List[Any]] = {field.name: [] for field in schema}  # type: ignore[arg-type]

    for m in models:
        row: Dict[str, Any] = model_to_row(m, flatten=flatten)
        for field in schema:  # type: ignore[arg-type]
            val = row.get(field.name)
            if isinstance(val, dict):
                val = str(val)
            # If list contains dicts, attempt to reduce each dict to id/identifier, else stringify
            if isinstance(val, list) and val and any(isinstance(x, dict) for x in val):
                reduced: List[Any] = []
                for x in val:
                    if isinstance(x, dict):
                        ident = x.get("id") or x.get("identifier")
                        reduced.append(ident if ident is not None else str(x))
                    else:
                        reduced.append(x)
                val = reduced
            buffers[field.name].append(val)
    # Build arrays now that buffers are filled
    arrays: List[pa.Array] = []
    for field in schema:  # type: ignore[arg-type]
        arrays.append(pa.array(buffers[field.name], type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)
def _schema_fingerprint(schema: pa.Schema) -> str:
    """Create a stable fingerprint for an Arrow schema (field names + types)."""
    parts = [f"{f.name}:{f.type}" for f in schema]
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return f"sha256:{digest}"


def attach_linkml_metadata(table: pa.Table, *, linkml_class: str, linkml_schema_version: str | None = None) -> pa.Table:
    """Attach LinkML metadata (class, optional schema version, schema fingerprint) to an Arrow table.

    Parameters
    ----------
    table : pa.Table
        Table to decorate.
    linkml_class : str
        Name of the LinkML class represented by rows.
    linkml_schema_version : str, optional
        Version string of the LinkML schema.
    """
    meta = dict(table.schema.metadata or {})
    meta.setdefault(b"linkml_class", linkml_class.encode())
    if linkml_schema_version is None:
        try:  # pragma: no cover
            from connects_common_connectivity import __version__  # type: ignore
            linkml_schema_version = __version__
        except Exception:
            linkml_schema_version = None
    if linkml_schema_version:
        meta.setdefault(b"linkml_schema_version", str(linkml_schema_version).encode())
    meta.setdefault(b"schema_fingerprint", _schema_fingerprint(table.schema).encode())
    new_schema = table.schema.with_metadata(meta)  # type: ignore[arg-type]
    return table.replace_schema_metadata(new_schema.metadata)


__all__ = [
    "normalize_value",
    "flatten_refs",
    "model_to_row",
    "build_arrow_schema",
    "models_to_table",
    "attach_linkml_metadata",
    "build_cell_feature_matrix_schema",
]

# ---------------------------------------------------------------------------
# Feature Matrix Schema Construction (Wide Parquet)
# ---------------------------------------------------------------------------
def _numpy_typestr_to_arrow(dtype_str: str) -> pa.DataType:
    """Map a NumPy typestr (e.g. '<f4', '<i8', '|u1') to a PyArrow DataType.

    Supported kinds: t (bool), b (int8), i (signed int), u (unsigned int), f (float),
    m (timedelta -> duration[ns]), M (datetime -> timestamp[ns]), S (bytes), U (unicode string),
    O (object -> string), V (void/raw -> binary).
    Complex 'c' not natively supported; raise for now.
    """
    if not isinstance(dtype_str, str):  # fail-safe
        return pa.string()
    dtype_str = dtype_str.strip()
    # Basic validation: must match pattern like <f4 etc.
    if len(dtype_str) < 3:
        return pa.string()
    kind = dtype_str[1] if dtype_str[0] in ('<', '>', '|', '=') else dtype_str[0]
    # Extract item size digits
    import re
    m = re.search(r'(\d+)$', dtype_str)
    size = int(m.group(1)) if m else None
    # Map kind
    if kind == 't':
        return pa.bool_()
    if kind == 'b':
        return pa.int8()
    if kind == 'i':
        # Signed integers
        if size == 1:
            return pa.int8()
        if size == 2:
            return pa.int16()
        if size == 4:
            return pa.int32()
        if size == 8:
            return pa.int64()
        return pa.int32()  # default
    if kind == 'u':
        if size == 1:
            return pa.uint8()
        if size == 2:
            return pa.uint16()
        if size == 4:
            return pa.uint32()
        if size == 8:
            return pa.uint64()
        return pa.uint32()
    if kind == 'f':
        if size == 2:  # half precision not always supported; treat as float32
            return pa.float32()
        if size == 4:
            return pa.float32()
        if size == 8:
            return pa.float64()
        return pa.float32()
    if kind == 'm':  # timedelta
        return pa.duration('ns')
    if kind == 'M':  # datetime
        return pa.timestamp('ns')
    if kind == 'S':  # bytes length-limited; store as binary
        return pa.binary()
    if kind == 'U':  # unicode string
        return pa.string()
    if kind == 'O':  # generic object -> string
        return pa.string()
    if kind == 'V':  # raw data
        return pa.binary()
    if kind == 'c':  # complex number not directly supported: represent as fixed-size list of two floats?
        return pa.list_(pa.float64(), list_size=2)
    return pa.string()

def build_cell_feature_matrix_schema(cell_feature_set: Any, feature_definitions: Iterable[Any], *, cell_index_column: str = "id") -> pa.Schema:
    """Construct a PyArrow schema for a wide CellFeatureMatrix Parquet file.

    Parameters
    ----------
    cell_feature_set : CellFeatureSet (Pydantic/LinkML instance or object with 'id')
        The feature set defining which features appear.
    feature_definitions : Iterable[CellFeatureDefinition]
        Collection of feature definition instances (must have id, data_type, unit, description).
    cell_index_column : str, default 'id'
        Name of the column holding DataItem identifiers (row index semantics).

    Returns
    -------
    pyarrow.Schema
        Schema with first field the cell index column (string) followed by one column per feature id
        using mapped Arrow types and embedding metadata (feature_id, unit, dtype, description).
    """
    fields: List[pa.Field] = []
    # Cell index column always string (DataItemId ultimately string in model)
    fields.append(pa.field(cell_index_column, pa.string(), nullable=False, metadata={"role": "cell_index"}))
    fields.append(pa.field('project_id', pa.string(), nullable=False, metadata={"description": "Project identifier"}))
    fields.append(pa.field('feature_set_id', pa.string(), nullable=False, metadata={"description": "CellFeatureSet identifier"}))
    for fd in feature_definitions:
        fid = getattr(fd, 'id', None)
        dtype_str = getattr(fd, 'data_type', None) or ''
        unit = getattr(fd, 'unit', None)
        desc = getattr(fd, 'description', None)
        if not fid:
            continue  # skip invalid definition
        arrow_type = _numpy_typestr_to_arrow(dtype_str)
        meta: Dict[str, bytes] = {}
        meta['feature_id'] = str(fid).encode()
        if unit:
            meta['unit'] = str(unit).encode()
        if dtype_str:
            meta['dtype'] = str(dtype_str).encode()
        if desc:
            meta['description'] = str(desc).encode()
        fields.append(pa.field(str(fid), arrow_type, nullable=True, metadata=meta))

    schema = pa.schema(fields, metadata={
        b'linkml_class': b'CellFeatureMatrix',
        b'feature_set_id': str(getattr(cell_feature_set, 'id', 'UNKNOWN')).encode(),
        b'schema_fingerprint': _schema_fingerprint(pa.schema(fields)).encode(),
    })
    return schema

