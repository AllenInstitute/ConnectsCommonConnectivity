# planning/ — IO layer design & agent prompts

How we're building the user-friendly IO layer (write / read / validation) for
ConnectsCommonConnectivity. Created 2026-06-01.

- `ARCHITECTURE.md` — the design (source of truth).
- `TODO.md` — ordered, flat task list (W1–W8).
- `prompts/` — one prompt per work item. `00_shared_context.md` is prepended to every other
  prompt and holds the **hard rules** (don't edit `models.py` or `schemas/*.yaml`).
- `prompts/_deferred/` — designs kept for reference; not actionable this round.

## TODO ↔ prompt map

| TODO | Prompt                              | What it owns                                    |
|------|-------------------------------------|-------------------------------------------------|
| W1   | `01_config.md`                      | `config.py` + `ccc_config.yaml` discovery       |
| W2   | `02_write_spec.md`                  | Registry (seed 3 classes) + drift test          |
| W3   | `03_writers.md`                     | Relocation, writers, per-class prototyping      |
| W4   | `04_public_api.md`                  | `io/__init__.py` curated surface                |
| W5   | `05_validation.md`                  | Strict submodels + hook swap                    |
| W6   | `06_notebook_migration.md`          | Migrate notebooks, regression, shim removal     |
| W7   | `07_tests.md`                       | Write-side suite gaps                           |
| W8   | (no prompt)                         | README / usage docs update                      |
| L1   | `_deferred/08_readers.md`           | Readers (deferred)                              |
| L2   | `_deferred/09_analysis.md`          | Read-side analysis + `check_refs` (deferred)    |

## How to run an item
Hand the implementing agent `00_shared_context.md` + the specific prompt, point it at
`ARCHITECTURE.md`, and follow the order in `TODO.md`.
