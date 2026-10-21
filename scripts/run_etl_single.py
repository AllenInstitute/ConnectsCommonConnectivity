"""Run selected notebooks with a configurable CCC data destination.

Set NOTEBOOKS and OUTPUT_SUFFIX below (defaults to V1DD and ``_v1dd``).
The suffix is appended to the configured output root and execution timestamp.
Override destinations with ``--data-output-root PATH`` and ``--output-dir PATH``;
the latter controls executed notebook copies only. Use ``--dry-run`` to preview
both destinations without running the ETL. The configuration file and source
notebooks are never modified.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from run_etl import REPO_ROOT, run_notebooks

from connects_common_connectivity.config import get_settings

NOTEBOOKS = ("code/EM/etl_v1dd1196_01.ipynb",)
OUTPUT_SUFFIX = "_v1dd"


def main(argv: Sequence[str] | None = None) -> int:
    """Run selected notebooks with a temporary output-root override inherited by their kernels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-output-root", type=Path,
        help=f"CCC data destination (default: configured output root plus {OUTPUT_SUFFIX!r}).",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="New directory for executed notebook copies, not the CCC data destination.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the notebook and destinations without executing or writing anything.",
    )
    args = parser.parse_args(argv)
    data_output_root = args.data_output_root
    if data_output_root is None:
        data_output_root = Path(f"{get_settings().output_root}{OUTPUT_SUFFIX}")
    data_output_root = data_output_root.expanduser().resolve()
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output_dir = REPO_ROOT / "scratch" / "etl_runs" / f"{timestamp}{OUTPUT_SUFFIX}"
    output_dir = output_dir.expanduser().resolve()
    print(f"ETL notebook: {NOTEBOOKS[0]}", flush=True)
    print(f"CCC data destination: {data_output_root}", flush=True)
    print(f"Executed notebooks: {output_dir}", flush=True)
    if args.dry_run:
        return 0

    previous_output_root = os.environ.get("CCC_OUTPUT_ROOT")
    os.environ["CCC_OUTPUT_ROOT"] = str(data_output_root)
    try:
        run_notebooks(REPO_ROOT, output_dir, NOTEBOOKS)
    finally:
        if previous_output_root is None:
            os.environ.pop("CCC_OUTPUT_ROOT", None)
        else:
            os.environ["CCC_OUTPUT_ROOT"] = previous_output_root
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
