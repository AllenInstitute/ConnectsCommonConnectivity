"""ETL and validation of BrainRegion records from a Parquet file.

Usage:
        uv run python examples/etl_brain_regions.py \
                s3://allen-atlas-assets/terminologies/allen-adult-mouse-terminology/2020/terminology.parquet \
                --out brain_regions.yaml

This script:
    1. Reads a Parquet file (local path or s3:// URI) with pyarrow
    2. Infers column mappings to BrainRegion slots using only aliases declared in the schema
    3. Directly ingests hierarchical columns (parent_identifier, child_identifiers, descendants,
         descendant_annotation_values) if present, instead of reconstructing hierarchy.
    4. Builds Pydantic BrainRegion models (dynamic) and reports validation issues
    5. Writes a YAML list or JSONL of valid BrainRegion objects

Hierarchy modeling:
        BrainRegion now exposes dual hierarchy representation:
            * Object slots: parent_identifier (BrainRegion), child_identifiers (List[BrainRegion]), descendants (List[BrainRegion])
            * Denormalized ID lists: child_identifier_ids, descendant_ids for quick lookup / export
        ETL performs a two-pass process:
            1. Instantiate all BrainRegion objects without object reference slots (capture raw ID lists)
            2. Resolve object references by looking up IDs in the instantiated map; unresolved IDs are reported as warnings

Column mapping relies solely on schema-declared aliases. If a mapping target cannot be found, the
field is left blank (except id/name which are required by the schema).

Handling list-valued columns:
    If a multivalued slot column is detected and contains:
        * A Python list/array (e.g. loaded from Parquet list type) -> used as-is.
        * A string representing JSON (e.g. "[\"A\", \"B\"]") -> parsed via json.loads.
        * A comma-separated string (e.g. "A,B,C") -> split on commas.

Note: For public S3 access without credentials you may need:
        export AWS_NO_SIGN_REQUEST=YES
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

import pyarrow.parquet as pq
import pyarrow as pa

from connects_common_connectivity import generate_pydantic_models, get_schema_path
from linkml_runtime import SchemaView  # type: ignore


def build_slot_alias_map(schema_name: str = "connectivity_schema.yaml") -> Dict[str, List[str]]:
    """Derive slot→aliases map exclusively from the schema (no hard-coded synonyms).

    Includes all BrainRegion slots now ingested directly, including multivalued hierarchy slots.
    """
    sv = SchemaView(get_schema_path(schema_name))
    br = sv.get_class("BrainRegion")
    if not br:
        return {}
    slot_map: Dict[str, List[str]] = {}
    for slot_name in sv.class_slots("BrainRegion"):
        slot = sv.get_slot(slot_name)
        if not slot:
            continue
        # Accept everything; multivalued handled downstream.
        aliases = list(getattr(slot, "aliases", []) or [])
        slot_map[slot_name] = [slot_name] + aliases
    return slot_map


def map_columns(columns: List[str], alias_map: Dict[str, List[str]]) -> Dict[str, Optional[str]]:
    """Map DataFrame columns to slot names using aliases defined only in schema."""
    lower_cols = {c.lower(): c for c in columns}
    mapping: Dict[str, Optional[str]] = {}
    for slot, aliases in alias_map.items():
        found = next((lower_cols[a.lower()] for a in aliases if a.lower() in lower_cols), None)
        mapping[slot] = found
    return mapping


def load_table(path: str) -> pa.Table:
    # pyarrow can usually infer S3; rely on build with AWS support.
    try:
        return pq.read_table(path)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR reading parquet: {e}", file=sys.stderr)
        raise SystemExit(2)


def _coerce_list_value(raw: Any) -> List[Any]:
    """Coerce a raw parquet cell into a list.

    Acceptable inputs:
      * list/tuple -> returned as list
      * string representing JSON list -> parsed
      * comma-separated string -> split
      * None -> []
      * scalar (non-string) -> [scalar]
    """
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
        # Try JSON list
        if (txt.startswith("[") and txt.endswith("]")) or (txt.startswith("{") and txt.endswith("}")):
            import json
            try:
                parsed = json.loads(txt)
                if isinstance(parsed, list):
                    return parsed
                return [parsed]
            except Exception:  # noqa: BLE001
                # Fall back to comma split
                pass
        if "," in txt:
            return [p.strip() for p in txt.split(",") if p.strip()]
        return [txt]
    # Fallback scalar
    return [raw]


def build_brain_regions(table: pa.Table, mapping: Dict[str, Optional[str]]):
    """Two-pass BrainRegion construction with object reference resolution.

    Pass 1: Instantiate BrainRegion objects capturing scalar + list ID data.
    Pass 2: Link object reference hierarchy (parent_identifier, child_identifiers, descendants).
    """
    models = generate_pydantic_models()
    BrainRegion = models["BrainRegion"]

    records = table.to_pydict()
    n = table.num_rows
    regions: List[Any] = []  # first pass objects; BrainRegion type resolved at runtime
    by_id: Dict[str, Any] = {}
    pending_children: Dict[str, List[str]] = {}
    pending_descendants: Dict[str, List[str]] = {}
    pending_parent: Dict[str, Optional[str]] = {}
    errors: List[tuple] = []

    for i in range(n):
        rid_col = mapping.get("id")
        name_col = mapping.get("name")
        acronym_col = mapping.get("acronym")
        color_col = mapping.get("color_hex_triplet")
        parent_col = mapping.get("parent_identifier")
        children_col = mapping.get("child_identifiers") or mapping.get("child_identifier_ids")
        descendants_col = mapping.get("descendants") 
        child_ids_col = mapping.get("child_identifier_ids")
        desc_ann_vals_col = mapping.get("descendant_annotation_values")
        ann_val_col = mapping.get("annotation_value")
        term_set_col = mapping.get("term_set_name")

        try:
            rid = records[rid_col][i] if rid_col else None
            name = records[name_col][i] if name_col else None
            acronym = records[acronym_col][i] if acronym_col else None
            color = records[color_col][i] if color_col else None
            parent_id = records[parent_col][i] if parent_col else None
            child_raw = records[children_col][i] if children_col else None
            descendants_raw = records[descendants_col][i] if descendants_col else None
            desc_ann_raw = records[desc_ann_vals_col][i] if desc_ann_vals_col else None
            ann_val = records[ann_val_col][i] if ann_val_col else None
            # Some parquet extractions may yield single-element lists for scalar integer fields
            if isinstance(ann_val, list):
                if len(ann_val) == 1:
                    ann_val = ann_val[0]
                else:
                    # Multi-valued annotation_value encountered; skip row (ambiguous scalar field)
                    raise ValueError(f"annotation_value has multiple entries: {ann_val}")
            if isinstance(ann_val, str) and ann_val.isdigit():
                ann_val = int(ann_val)
            term_set = records[term_set_col][i] if term_set_col else None

            if isinstance(color, str) and color and not color.startswith("#"):
                color = f"#{color}"

            child_ids = _coerce_list_value(child_raw) if children_col else []
            descendant_ann_vals_raw = _coerce_list_value(desc_ann_raw) if desc_ann_vals_col else []
            # Cast descendant annotation values to int when feasible
            descendant_ann_vals: List[int] = []
            for v in descendant_ann_vals_raw:
                if v is None or v == "":
                    continue
                try:
                    descendant_ann_vals.append(int(v))
                except Exception:
                    # Non-integer value encountered; skip but could log.
                    continue

            region_kwargs = dict(id=rid, name=name)
            if acronym is not None:
                region_kwargs["acronym"] = acronym
            if color is not None:
                region_kwargs["color_hex_triplet"] = color
            # Store denormalized ID lists; object linking deferred to pass 2
            if parent_id is not None and parent_id != rid:
                pending_parent[rid] = parent_id
            else:
                pending_parent[rid] = None
            if child_ids:
                region_kwargs["child_identifier_ids"] = child_ids
                pending_children[rid] = child_ids
            if descendant_ann_vals:
                region_kwargs["descendant_annotation_values"] = descendant_ann_vals
            if ann_val is not None:
                region_kwargs["annotation_value"] = ann_val
            if term_set is not None:
                region_kwargs["term_set_name"] = term_set

            # Validate required id and name presence before instantiation
            if not rid or not name:
                raise ValueError("Missing required id or name")

            region = BrainRegion(**region_kwargs)
            regions.append(region)
            by_id[rid] = region
        except Exception as e:  # noqa: BLE001
            errors.append((records.get(rid_col, [None])[i] if rid_col else None, f"instantiation error: {e}"))

    # Pass 2: resolve object references
    unresolved_parent = 0
    unresolved_children = 0
    unresolved_descendants = 0
    for rid, region in by_id.items():
        pid = pending_parent.get(rid)
        if pid:
            parent_obj = by_id.get(pid)
            if parent_obj:
                try:
                    region.parent_identifier = parent_obj  # type: ignore[attr-defined]
                except Exception as e:  # noqa: BLE001
                    errors.append((rid, f"parent linking error: {e}"))
            else:
                unresolved_parent += 1
        for cid in pending_children.get(rid, []):
            child_obj = by_id.get(cid)
            if child_obj:
                try:
                    current = getattr(region, "child_identifiers", []) or []
                    current.append(child_obj)
                    region.child_identifiers = current  # type: ignore[attr-defined]
                except Exception as e:  # noqa: BLE001
                    errors.append((rid, f"child linking error: {e}"))
            else:
                unresolved_children += 1
        for did in pending_descendants.get(rid, []):
            desc_obj = by_id.get(did)
            if desc_obj:
                try:
                    current = getattr(region, "descendants", []) or []
                    current.append(desc_obj)
                    region.descendants = current  # type: ignore[attr-defined]
                except Exception as e:  # noqa: BLE001
                    errors.append((rid, f"descendant linking error: {e}"))
            else:
                unresolved_descendants += 1

    if unresolved_parent or unresolved_children or unresolved_descendants:
        print(
            f"Hierarchy linking warnings: parent={unresolved_parent} children={unresolved_children} descendants={unresolved_descendants} unresolved",
            file=sys.stderr,
        )

    return regions, errors


def serialize(objects, fmt: str) -> str:
    if fmt == "jsonl":
        import json
        lines = []
        for obj in objects:
            data = obj.model_dump()
            # Flatten object references to identifiers
            if isinstance(data.get("parent_identifier"), dict):
                # parent may have been serialized as nested dict; replace with its id if present
                parent_obj = getattr(obj, "parent_identifier", None)
                if parent_obj is not None and hasattr(parent_obj, "id"):
                    data["parent_identifier"] = parent_obj.id
            # Lists: child_identifiers, descendants
            for list_slot in [("child_identifiers"), ("descendants")]:
                val = getattr(obj, list_slot, None)
                if val and isinstance(val, list) and all(hasattr(v, "id") for v in val):
                    data[list_slot] = [v.id for v in val]
                    # Ensure denormalized id list also present
                    if id_slot not in data:
                        data[id_slot] = data[list_slot]
                elif list_slot in data and isinstance(data[list_slot], list):
                    # If pydantic produced list of empty dicts, replace using object attributes
                    fixed_ids = []
                    for v in getattr(obj, list_slot, []) or []:
                        if hasattr(v, "id"):
                            fixed_ids.append(v.id)
                    if fixed_ids:
                        data[list_slot] = fixed_ids
                        if id_slot not in data:
                            data[id_slot] = fixed_ids
            lines.append(json.dumps(data))
        return "\n".join(lines)
    # yaml list
    import yaml
    out_list = []
    for obj in objects:
        data = obj.model_dump()
        # Parent
        parent_obj = getattr(obj, "parent_identifier", None)
        if parent_obj is not None and hasattr(parent_obj, "id"):
            data["parent_identifier"] = parent_obj.id
        # Children / descendants
        for list_slot, id_slot in [("child_identifiers", "child_identifier_ids"), ("descendants", "descendant_ids")]:
            val = getattr(obj, list_slot, None)
            if val and isinstance(val, list) and all(hasattr(v, "id") for v in val):
                data[list_slot] = [v.id for v in val]
                if id_slot not in data:
                    data[id_slot] = data[list_slot]
            elif list_slot in data and isinstance(data[list_slot], list):
                fixed_ids = []
                for v in getattr(obj, list_slot, []) or []:
                    if hasattr(v, "id"):
                        fixed_ids.append(v.id)
                if fixed_ids:
                    data[list_slot] = fixed_ids
                    if id_slot not in data:
                        data[id_slot] = fixed_ids
        out_list.append(data)
    return yaml.safe_dump(out_list, sort_keys=False)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ETL BrainRegion from Parquet and validate")
    ap.add_argument("parquet_path", help="Local path or s3:// URI to Parquet file")
    ap.add_argument("--out", help="Output file (YAML or JSONL)")
    ap.add_argument("--format", choices=["yaml", "jsonl"], default="yaml")
    # Optional explicit column mappings
    args = ap.parse_args(argv)

    table = load_table(args.parquet_path)
    alias_map = build_slot_alias_map()
    mapping = map_columns(table.schema.names, alias_map)
    # Report mapping for transparency
    print("Column mapping:")
    for slot, col in mapping.items():
        print(f"  {slot} -> {col}")

    regions, errors = build_brain_regions(table, mapping)
    print(f"Loaded {len(regions)} BrainRegion objects ({len(errors)} errors)")
    if errors:
        for rid, msg in errors[:10]:  # show sample
            print(f"  ERROR region_id={rid}: {msg}", file=sys.stderr)
        if len(errors) > 10:
            print(f"  ... {len(errors)-10} more errors", file=sys.stderr)

    output_text = serialize(regions, args.format)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"Wrote {args.out}")
    else:
        print(output_text)
    return 0 if not errors else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
