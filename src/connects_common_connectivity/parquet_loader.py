"""Generic utilities for loading Parquet rows into LinkML-derived dynamic Pydantic models.

Design Goals:
  * Schema-driven column→slot mapping using declared aliases (no hard-coded heuristics)
  * Support scalar and multivalued slots (list coercion + JSON / comma parsing)
  * Two-phase resolution for object reference slots (slots whose range is a class) so that
    forward references and hierarchical links are established after all objects are created.
  * Provide structured validation feedback (errors + warnings) without stopping ingestion.

Core Function: load_parquet_to_models(schema_name, class_name, parquet_path)
Returns: (instances, report_dict)

The report contains:
  mapping: slot→column selected (or None)
  errors: list of {row_index, id?, message}
  warnings: list of textual warnings (e.g., unresolved references)
  counts: summary numbers

NOTE: This is intentionally minimal and synchronous. For large tables consider chunked reading
      or streaming conversion of RecordBatches.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import json
import sys

import pyarrow.parquet as pq
import pyarrow as pa
from linkml_runtime import SchemaView  # type: ignore

from . import generate_pydantic_models, get_schema_path


def _coerce_list(raw: Any) -> List[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, tuple):
        return list(raw)
    if isinstance(raw, str):
        txt = raw.strip()
        if not txt:
            return []
        if (txt.startswith("[") and txt.endswith("]")) or (txt.startswith("{") and txt.endswith("}")):
            try:
                parsed = json.loads(txt)
                return parsed if isinstance(parsed, list) else [parsed]
            except Exception:  # noqa: BLE001
                pass
        if "," in txt:
            return [p.strip() for p in txt.split(",") if p.strip()]
        return [txt]
    return [raw]


def _build_alias_map(sv: SchemaView, class_name: str) -> Dict[str, List[str]]:
    slot_map: Dict[str, List[str]] = {}
    for slot_name in sv.class_slots(class_name):
        slot = sv.get_slot(slot_name)
        if not slot:
            continue
        aliases = list(getattr(slot, "aliases", []) or [])
        slot_map[slot_name] = [slot_name] + aliases
    return slot_map


def _map_columns(columns: List[str], alias_map: Dict[str, List[str]]) -> Dict[str, Optional[str]]:
    lower = {c.lower(): c for c in columns}
    mapping: Dict[str, Optional[str]] = {}
    for slot, aliases in alias_map.items():
        mapping[slot] = next((lower[a.lower()] for a in aliases if a.lower() in lower), None)
    return mapping


def load_parquet_to_models(
    schema_name: str,
    class_name: str,
    parquet_path: str,
    *,
    strict_required: bool = True,
    max_errors: Optional[int] = 100,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Load Parquet rows into dynamic Pydantic model instances for a LinkML class.

    Parameters
    ----------
    schema_name: Name of aggregator schema yaml (e.g. 'connectivity_schema.yaml').
    class_name: LinkML class to instantiate.
    parquet_path: Local path or s3:// URI.
    strict_required: If True, rows missing required slots are dropped and recorded as errors.
    max_errors: Optional cap to stop ingestion early if too many errors accumulate.
    """
    sv = SchemaView(get_schema_path(schema_name))
    cls_def = sv.get_class(class_name)
    if not cls_def:
        raise ValueError(f"Class '{class_name}' not found in schema '{schema_name}'")

    table: pa.Table = pq.read_table(parquet_path)
    models = generate_pydantic_models()
    Model = models[class_name]

    alias_map = _build_alias_map(sv, class_name)
    mapping = _map_columns(table.schema.names, alias_map)

    # Identify object-reference slots (range is a class) vs primitive/multivalued
    object_slots: Dict[str, str] = {}
    multivalued: Dict[str, bool] = {}
    required: Dict[str, bool] = {}
    # Use induced_slot to capture class-specific overrides reliably
    for slot_name in sv.class_slots(class_name):
        slot = sv.induced_slot(slot_name, class_name)
        if slot.range and sv.get_class(slot.range):  # class range (object reference)
            object_slots[slot_name] = slot.range
        multivalued[slot_name] = bool(getattr(slot, "multivalued", False))
        required[slot_name] = bool(getattr(slot, "required", False))

    records = table.to_pydict()
    n = table.num_rows
    instances: List[Any] = []
    by_id: Dict[str, Any] = {}
    pending_refs: Dict[str, Dict[str, List[str]]] = {}
    errors: List[Dict[str, Any]] = []
    warnings: List[str] = []

    # Determine identifier slot (explicit or fall back to 'id' if present)
    identifier_slot = next(
        (s for s in sv.class_slots(class_name) if getattr(sv.induced_slot(s, class_name), "identifier", False)),
        "id" if "id" in sv.class_slots(class_name) else None,
    )

    # Pass 1: instantiate without object references
    for row_idx in range(n):
        kwargs: Dict[str, Any] = {}
        row_id_val: Any = None
        try:
            for slot_name in sv.class_slots(class_name):
                col = mapping.get(slot_name)
                if not col:
                    continue
                raw_val = records[col][row_idx]
                slot = sv.induced_slot(slot_name, class_name)
                if slot_name in object_slots:
                    # Collect identifiers only; always treat as list pipeline then resolve later
                    id_list = _coerce_list(raw_val)
                    pending_refs.setdefault(row_idx, {})[slot_name] = [v for v in id_list if v is not None]
                else:
                    if multivalued[slot_name]:
                        coerced = _coerce_list(raw_val)
                        if coerced:  # only assign if non-empty
                            kwargs[slot_name] = coerced
                    else:
                        # Collapse single-element lists erroneously provided
                        if isinstance(raw_val, list) and len(raw_val) == 1:
                            raw_val = raw_val[0]
                        if raw_val is not None:
                            kwargs[slot_name] = raw_val
                    # Track identifier candidate
                    if identifier_slot == slot_name or getattr(slot, "identifier", False):
                        row_id_val = kwargs.get(slot_name, row_id_val)
            # Required checks
            if strict_required:
                for req_slot, is_req in required.items():
                    if is_req and req_slot not in kwargs:
                        raise ValueError(f"Missing required slot '{req_slot}'")
            # If identifier slot exists but not populated, drop row
            if identifier_slot and identifier_slot not in kwargs:
                raise ValueError("Missing identifier value")
            inst = Model(**kwargs)
            instances.append(inst)
            if row_id_val:
                by_id[str(row_id_val)] = inst
        except Exception as e:  # noqa: BLE001
            errors.append({"row": row_idx, "id": row_id_val, "message": f"instantiation error: {e}"})
            if max_errors and len(errors) >= max_errors:
                warnings.append(f"Aborting after reaching max_errors={max_errors}")
                break

    # Pass 2: resolve references
    unresolved_counts: Dict[str, int] = {}
    unresolved_references: Dict[str, List[int, str]] = {}
    for row_idx, ref_map in pending_refs.items():
        if row_idx >= len(instances):
            continue  # skipped row
        inst = instances[row_idx]
        for slot_name, id_list in ref_map.items():
            target_range = object_slots[slot_name]
            if multivalued[slot_name]:
                resolved_objs = []
                for rid in id_list:
                    obj = by_id.get(str(rid))
                    if obj is None:
                        unresolved_counts[slot_name] = unresolved_counts.get(slot_name, 0) + 1
                        unresolved_references.setdefault(slot_name, []).append((row_idx, str(rid)))
                    else:
                        resolved_objs.append(obj)
                try:
                    setattr(inst, slot_name, resolved_objs)
                except Exception as e:  # noqa: BLE001
                    errors.append({"row": row_idx, "id": getattr(inst, identifier_slot, None), "message": f"linking error {slot_name}: {e}"})
            else:
                rid = id_list[0] if id_list else None
                if rid is None:
                    continue
                obj = by_id.get(str(rid))
                if obj is None:
                    unresolved_counts[slot_name] = unresolved_counts.get(slot_name, 0) + 1
                    unresolved_references.setdefault(slot_name, []).append((row_idx, str(rid)))
                else:
                    try:
                        setattr(inst, slot_name, obj)
                    except Exception as e:  # noqa: BLE001
                        errors.append({"row": row_idx, "id": getattr(inst, identifier_slot, None), "message": f"linking error {slot_name}: {e}"})

    for slot_name, count in unresolved_counts.items():
        warnings.append(f"Unresolved references for slot '{slot_name}': {count}")
        warnings.append(f"Examples: {unresolved_references.get(slot_name, [])[:5]}")

    report = {
        "mapping": mapping,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "rows": n,
            "instances": len(instances),
            "errors": len(errors),
            "warnings": len(warnings),
        },
        "unresolved": unresolved_counts,
    }
    return instances, report


def _flatten_instance(obj: Any, hierarchy_slots: Optional[List[Tuple[str, Optional[str]]]] = None) -> Dict[str, Any]:
    """Return a plain dict for a dynamic Pydantic object, flattening object references to IDs.

    hierarchy_slots: list of (slot_name, parallel_id_slot) pairs.
    """
    data = obj.model_dump()
    if not hierarchy_slots:
        return data
    for slot_name, parallel in hierarchy_slots:
        val = getattr(obj, slot_name, None)
        if val is None:
            continue
        if isinstance(val, list):
            if val and all(hasattr(v, "id") for v in val):
                ids = [v.id for v in val]
                data[slot_name] = ids
                if parallel and parallel not in data:
                    data[parallel] = ids
        else:
            if hasattr(val, "id"):
                data[slot_name] = val.id
    return data


if __name__ == "__main__":  # pragma: no cover
    # Basic CLI
    import argparse
    ap = argparse.ArgumentParser(description="Generic Parquet→LinkML loader")
    ap.add_argument("schema", help="Aggregator schema filename (e.g. connectivity_schema.yaml)")
    ap.add_argument("class_name", help="LinkML class to instantiate")
    ap.add_argument("parquet_path", help="Local or s3:// path to parquet")
    ap.add_argument("--out", help="Output file for serialized instances (YAML or JSONL)")
    ap.add_argument("--format", choices=["yaml", "jsonl"], default="yaml")
    ap.add_argument("--no-strict", action="store_true", help="Disable strict required slot checking")
    ap.add_argument("--max-errors", type=int, default=100, help="Abort after this many instantiation errors")
    ap.add_argument("--flatten", action="store_true", help="Flatten object reference slots to identifier lists")
    args = ap.parse_args()

    objs, rep = load_parquet_to_models(
        args.schema,
        args.class_name,
        args.parquet_path,
        strict_required=not args.no_strict,
        max_errors=args.max_errors,
    )
    print(f"Loaded {len(objs)} objects; errors={rep['counts']['errors']} warnings={rep['counts']['warnings']}")
    if rep["warnings"]:
        for w in rep["warnings"]:
            print("WARNING:", w, file=sys.stderr)
    if rep["errors"]:
        for e in rep["errors"][:10]:
            print("ERROR:", e, file=sys.stderr)

    if args.out:
        # Determine hierarchy slots for flattening if requested
        hierarchy_slots: List[Tuple[str, Optional[str]]] = []
        if args.flatten and args.class_name == "BrainRegion":
            hierarchy_slots = [
                ("parent_identifier", None),
                ("child_identifiers", None),
                ("descendants", None)
            ]
        serializable = [
            _flatten_instance(o, hierarchy_slots) if args.flatten else o.model_dump()
            for o in objs
        ]
        if args.format == "jsonl":
            with open(args.out, "w", encoding="utf-8") as f:
                for row in serializable:
                    f.write(json.dumps(row) + "\n")
        else:
            import yaml
            with open(args.out, "w", encoding="utf-8") as f:
                yaml.safe_dump(serializable, f, sort_keys=False)
        print(f"Wrote {args.out} ({args.format})")