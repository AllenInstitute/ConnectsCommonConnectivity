# planning/ — IO layer design & agent prompts

How we're building the user-friendly IO layer (write / read / validation) for
ConnectsCommonConnectivity. Created 2026-06-01.

- `ARCHITECTURE.md` — the design (source of truth).
- `TODO.md` — ordered, dependency-aware task list.
- `prompts/` — one prompt per work item (`00_shared_context.md` is prepended to every other
  prompt and holds the **hard rules**: don't edit `models.py` or `schemas/*.yaml`):
  `01_config` · `02_write_spec` · `03_validation` · `04_writers` · `05_readers` ·
  `06_notebook_migration` · `07_tests` · `08_analysis` (read-side analysis + opt-in
  referential check) · `09_public_api` (`io/__init__.py`).

## How to run an item
Hand the implementing agent `00_shared_context.md` + the specific prompt, point it at
`ARCHITECTURE.md`, and follow the order in `TODO.md`.
