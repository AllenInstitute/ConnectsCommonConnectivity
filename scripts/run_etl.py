"""Execute the active ETL notebooks sequentially, preserving source notebooks."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import nbformat
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "code/cell_types/etl_ttype_01.ipynb",
    "code/cell_types/etl_met_type_01.ipynb",
    "code/cell_types/etl_abc_01.ipynb",
    "code/BARseq/etl_barseq_01.ipynb",
    "code/INH_patchseq/etl_patchseq_inh_01.ipynb",
    "code/EXC_WNM_patchseq/etl_patchseq_exc_01.ipynb",
    "code/EXC_WNM_patchseq/etl_wnm_exc_01.ipynb",
    "code/EM/etl_minnie1822_01.ipynb",
    "code/EM/etl_minnie1822_02_umap_features.ipynb",
    "code/EM/etl_v1dd1196_01.ipynb",
)


def run_notebooks(repo_root: Path, output_dir: Path, notebooks: Sequence[str]) -> None:
    """Run notebooks in order, saving partial output and propagating the first failure."""
    for relative_path in notebooks:
        path = repo_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Required ETL notebook is missing: {path}")
    output_dir.mkdir(parents=True, exist_ok=False)

    for index, relative_path in enumerate(notebooks, start=1):
        source = repo_root / relative_path
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        notebook = nbformat.read(source, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        client = NotebookClient(
            notebook,
            kernel_name="python3",
            timeout=None,
            allow_errors=False,
            force_raise_errors=True,
        )
        manager = client.create_kernel_manager()
        spec = manager.kernel_spec
        if spec is None:
            raise RuntimeError("The python3 kernel specification is unavailable.")
        # Use this process's environment, not an unrelated installed Python kernel.
        spec.argv = [
            sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}",
        ]
        print(f"[{index}/{len(notebooks)}] Running {relative_path}", flush=True)
        try:
            client.execute(cwd=str(repo_root))
        finally:
            nbformat.write(notebook, destination)
            print(f"Saved execution record: {destination}", flush=True)
        print(f"[{index}/{len(notebooks)}] Passed {relative_path}", flush=True)
    print(f"All {len(notebooks)} ETL notebooks completed.", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse runner options; dry-run prints the order without creating files or kernels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the ordered notebook list without executing or writing anything.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="New directory for executed copies (default: scratch/etl_runs/<UTC timestamp>). "
             "Does not change the configured CCC data destination.",
    )
    parser.add_argument(
        "--from", dest="from_notebook", metavar="NOTEBOOK",
        help="Start at this notebook filename or repository-relative path, inclusive. "
             "Earlier dependencies must already exist in the configured CCC output.",
    )
    args = parser.parse_args(argv)
    notebooks: tuple[str, ...] = NOTEBOOKS
    if args.from_notebook is not None:
        matches = [
            index for index, notebook in enumerate(NOTEBOOKS)
            if args.from_notebook in (notebook, Path(notebook).name)
        ]
        if len(matches) != 1:
            parser.error("--from must identify exactly one active ETL notebook")
        notebooks = NOTEBOOKS[matches[0]:]
    print("ETL order:")
    for index, notebook in enumerate(notebooks, start=1):
        print(f"  {index}. {notebook}")
    if args.dry_run:
        return 0
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output_dir = REPO_ROOT / "scratch" / "etl_runs" / timestamp
    output_dir = output_dir.expanduser().resolve()
    print(f"Executed notebooks: {output_dir}", flush=True)
    run_notebooks(REPO_ROOT, output_dir, notebooks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
