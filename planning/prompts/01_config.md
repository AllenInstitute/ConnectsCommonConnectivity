# Agent prompt — Config module (global output path)

> Prepend `00_shared_context.md`.

## Goal
Create `src/connects_common_connectivity/io/config.py` providing a single, version-
controlled, human-readable global output path, with an optional env override. **No new
dependency** — plain pydantic `BaseModel` only (NOT pydantic-settings).

## Requirements
1. A `Settings(BaseModel)` class with:
   - `output_root: Path` — default `Path("../scratch/em_patchseq_wnm_v1/")` (current value
     used across notebooks; confirm by grepping `OUTPUT_ROOT` in `code/*.ipynb`).
   - A `load()` classmethod that returns `Settings`, using
     `os.environ.get("CCC_OUTPUT_ROOT", <default>)` so CodeOcean can override the path via
     env without editing tracked code.
   - Designed so more knobs (e.g. `dry_run`, `schema_version_pin`) can be added later.
2. A helper `table_path(settings: Settings, table: str) -> Path` that joins
   `output_root / table` (e.g. `"dataset"`, `"dataitem"`,
   `"dataitem_dataset_association"`) so notebooks never concatenate path strings. Use the
   exact subdir names currently in the notebooks.
3. A `describe()` / `__repr__` that prints the resolved config so notebooks can show it at
   the top instead of relying on hidden state.

## Tests (`tests/test_config.py`)
- Default `output_root` is the expected path when env var unset.
- `CCC_OUTPUT_ROOT` env var overrides the default.
- `table_path` joins correctly and returns a `Path`.

## Do not
- Add pydantic-settings or any new dependency.
- Touch `models.py` or schemas.

## Report
List the subdir names you found in the notebooks and confirm the default matches.
