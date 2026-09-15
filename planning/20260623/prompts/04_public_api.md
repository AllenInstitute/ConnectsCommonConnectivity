# Agent prompt — Public API (`io/__init__.py`)

> Prepend `00_shared_context.md`. Depends on writers (W3).

## Why
`io/__init__.py` defines what users type after `from connects_common_connectivity.io
import …` and what shows up in autocomplete. It is the file that decides whether the
package feels curated or sprawling.

## Requirements
1. Re-export, and only re-export, the curated names below. Source paths in parentheses.
   - `get_settings`, `Settings`, `table_path` — from `..config`
   - `write_models`, `write_projection_matrix`, `WriteResult`, `WRITABLE_CLASSES`
     — from `.writers`
2. Define `__all__` to exactly that list (no more, no less).
3. Add a module docstring: one short paragraph on the IO layer + a 3–5 line usage
   example using `write_models(...)` and `write_projection_matrix(...)`. No config
   ceremony in the example — `get_settings()` is implicit.
4. Leave a single `# TODO(W8): reader exports` comment at the bottom of the imports
   block, so the reader slot is obvious when W8 lands.

## Test (`tests/test_public_api.py`)
- Import every name in `__all__` from `connects_common_connectivity.io` and assert it
  resolves to a non-`None` object.
- Assert no name in `__all__` starts with `_`.

## Do not
- Re-export `arrow_utils`, `write_utils`, or any private helper.
- Touch `models.py` or schemas.