# Agent prompt — Public API (`io/__init__.py`)

> Prepend `00_shared_context.md`. Depends on writers (W3); reader exports added later when
> the read-side work happens.

## Why
`io/__init__.py` is the single most important file for "user-friendly": it defines what a
user types after `from connects_common_connectivity.io import …` and what shows up in
autocomplete. It also decouples the public surface from internal module layout.

## Requirements
1. A concise module docstring: one paragraph on the IO layer (note settings come from a
   discovered `ccc_config.yaml`) + a 3–5 line usage example using `write_models(...)` and
   `write_projection_matrix(...)` — no config ceremony needed.
2. Curated re-exports — only the names users should touch (write-side for now):
   - config (from the package root, `from ..config import ...`): `get_settings`, `Settings`,
     `table_path`
   - writers: `write_models`, `write_projection_matrix`, `WriteResult`, `WRITABLE_CLASSES`
   - reader names are added here when readers land (deferred) — leave a clear TODO comment.
   Do NOT re-export backends (`arrow_utils`, `write_utils`) or internal helpers.
   Do NOT add per-class wrappers (`write_dataset`, etc.) — they don't exist; `write_models`
   infers the class.
3. Define `__all__` to match exactly the curated list (keeps `dir()` and `*` imports clean).
4. Keep it import-light: no heavy work at import time; just imports + `__all__`.

## Test (`tests/test_public_api.py`)
- Every name in `__all__` is importable from `connects_common_connectivity.io`.
- No backend/internal module name leaks into `__all__`.
- `__all__` does NOT contain any `write_dataset` / `write_dataitem` / etc. — those
  wrappers don't exist by design.

## Do not
- Re-export internal backends. Add per-class wrappers. Touch `models.py` or schemas.
