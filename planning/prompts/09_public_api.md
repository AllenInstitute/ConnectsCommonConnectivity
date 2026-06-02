# Agent prompt — Public API (`io/__init__.py`)

> Prepend `00_shared_context.md`. Depends on writers (3.1/3.2); readers can be added later.

## Why
`io/__init__.py` is the single most important file for "user-friendly": it defines what a
user types after `from connects_common_connectivity.io import …` and what shows up in
autocomplete. It also decouples the public surface from internal module layout, so seed
sections can later be split into `transforms.py` / `analysis.py` without breaking imports.

## Requirements
1. A concise module docstring: one paragraph on the IO layer (note settings come from a
   discovered `ccc_config.yaml`) + a 3–5 line usage example (a `write_*` call, a `read_*`
   call — no config ceremony needed).
2. Curated re-exports — only the names users should touch:
   - config (from the package root, `from ..config import ...`): `get_settings`, `Settings`,
     `table_path`
   - writers: `write_models` + the generated typed wrappers
   - readers: `read_dataset`, `read_dataitem`, `read_features`,
     `read_dataitems_for_clusters`, and (when present) `compare_region_coverage`
   Do NOT re-export backends (`arrow`, `write_utils`) or internal helpers.
3. Define `__all__` to match exactly the curated list (keeps `dir()` and `*` imports clean).
4. Keep it import-light: no heavy work at import time; just imports + `__all__`.

## Test (`tests/test_public_api.py`)
- Every name in `__all__` is importable from `connects_common_connectivity.io`.
- No backend/internal module name leaks into `__all__`.

## Do not
- Re-export internal backends. Touch `models.py` or schemas.
