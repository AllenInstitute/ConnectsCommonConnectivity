# Shared context — prepend to every IO-layer agent prompt

You are working in the `ConnectsCommonConnectivity` repo (LinkML+pydantic schema for
multi-scale connectomics). **Read `planning/ARCHITECTURE.md` before starting** — it owns
the design, the existing-file inventory, the target `io/` layout, and the motivating bug.
This file is only the rules of the room.

## Non-negotiable rules
1. **Never edit `src/connects_common_connectivity/models.py`** — auto-generated from
   `schemas/*.yaml`. Read-only.
2. **Never edit `schemas/*.yaml`** without explicit written permission from YY. If your
   task seems to need a new slot, STOP and report what you need and why.
3. **Single source of truth = the LinkML schema / generated models.** Read field
   definitions from `models.py`; never restate them.
4. **IO code lives under `src/connects_common_connectivity/io/`.** Existing root
   modules (`arrow_utils.py`, `write_utils.py`, `parquet_loader.py`) are MOVED there and
   wrapped as backends — never reimplemented. `cli.py` and `models.py` stay at root; so does
   `config.py` (package-wide settings, not IO-specific) and plotting stays in
   `code/utils.py`. Exact layout: ARCHITECTURE.md → "Target io/ structure".
5. When you move a module, leave a one-line re-export shim at its old path until notebook
   migration is done, so nothing breaks mid-transition.

## Conventions
- Python 3.10+, pydantic v2; polars + pyarrow + deltalake (already deps). No new deps
  without asking.
- Match existing style (ruff, line-length 100); docstring like the existing modules.
- Add `pytest` tests under `tests/` for anything you implement.
- Run the relevant tests and report results. Never mark work done with failing tests or a
  partial implementation.
