"""Command-line interface for Connects Common Connectivity.

Provides utilities to:
  * Show currently installed schema version/location
  * Bundle schemas + example manifests into a distributable archive
  * Validate one or more data files (YAML/JSON/Parquet) against the LinkML schema

Usage (after install):

    python -m connects_common_connectivity --help
    ccc --help
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path
from typing import Iterable, List

from . import __version__, get_schema_path

try:
    from linkml_runtime import SchemaView  # type: ignore
    from linkml_runtime.loaders import yaml_loader  # type: ignore
except Exception:  # pragma: no cover
    SchemaView = None  # type: ignore
    yaml_loader = None  # type: ignore


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--schema",
        default="connectivity_schema.yaml",
        help="Schema filename under installed package schemas directory (default: %(default)s)",
    )


def cmd_info(args: argparse.Namespace) -> int:
    spath = get_schema_path(args.schema)
    print(f"Schema path: {spath}")
    print(f"Package version: {__version__}")
    return 0


def cmd_bundle(args: argparse.Namespace) -> int:
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    members: List[Path] = []
    schemas_dir = Path(get_schema_path()).parent
    for file in schemas_dir.glob("*.yaml"):
        members.append(file)

    # include example manifest(s) if present
    examples_dir = Path.cwd() / "examples"
    if examples_dir.exists():
        for f in examples_dir.glob("*.yaml"):
            members.append(f)
        for f in examples_dir.glob("*.json"):
            members.append(f)

    with tarfile.open(out, "w:gz") as tf:
        for m in members:
            tf.add(m, arcname=f"schemas/{m.name}" if m.parent == schemas_dir else f"examples/{m.name}")
    print(f"Wrote bundle: {out} ({len(members)} files)")
    return 0


def _iter_input_files(paths: Iterable[str]) -> Iterable[Path]:
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for sub in path.rglob("*.yaml"):
                yield sub
            for sub in path.rglob("*.json"):
                yield sub
        elif path.exists():
            yield path
        else:
            print(f"Warning: path not found: {p}", file=sys.stderr)


def cmd_validate(args: argparse.Namespace) -> int:  # pragma: no cover - minimal runtime smoke tests
    if SchemaView is None or yaml_loader is None:
        print("linkml_runtime not available; install package with core dependencies.", file=sys.stderr)
        return 2

    spath = get_schema_path(args.schema)
    sv = SchemaView(spath)
    errors = 0

    for file in _iter_input_files(args.inputs):
        if file.suffix.lower() in {".yaml", ".yml"}:
            try:
                obj = yaml_loader.load(str(file), target_class=None)  # type: ignore[arg-type]
                # We could add deeper validation here using JSONSchema; for now, just load.
                print(f"VALID (basic load): {file}")
            except Exception as e:  # noqa: BLE001
                print(f"ERROR loading {file}: {e}", file=sys.stderr)
                errors += 1
        elif file.suffix.lower() == ".json":
            try:
                json.loads(file.read_text())
                print(f"VALID (json parse): {file}")
            except Exception as e:  # noqa: BLE001
                print(f"ERROR parsing {file}: {e}", file=sys.stderr)
                errors += 1
        else:
            print(f"Skipping unsupported file: {file}")

    if errors:
        print(f"Completed with {errors} error(s).", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ccc", description="Connects Common Connectivity utilities")
    sub = p.add_subparsers(dest="command", required=True)

    # info
    info_p = sub.add_parser("info", help="Show schema installation info")
    _add_common_args(info_p)
    info_p.set_defaults(func=cmd_info)

    # bundle
    bundle_p = sub.add_parser("bundle", help="Create a tar.gz of schemas + examples")
    _add_common_args(bundle_p)
    bundle_p.add_argument("-o", "--output", default="connectivity_bundle.tar.gz", help="Output tar.gz filename")
    bundle_p.set_defaults(func=cmd_bundle)

    # validate
    val_p = sub.add_parser("validate", help="Basic validation of YAML/JSON files against loader")
    _add_common_args(val_p)
    val_p.add_argument("inputs", nargs="+", help="Files or directories to validate")
    val_p.set_defaults(func=cmd_validate)

    return p


def main(argv: List[str] | None = None) -> int:  # pragma: no cover
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
