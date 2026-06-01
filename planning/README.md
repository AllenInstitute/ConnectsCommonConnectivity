# planning/ — IO layer design & agent prompts

This folder documents how we're building the user-friendly IO layer (write / read /
validation) for ConnectsCommonConnectivity, and holds ready-to-run prompts for the agents
that implement each piece. Created 2026-06-01.

## Contents
- `ARCHITECTURE.md` — the design (source of truth). Registry-centric write/read/validation.
- `TODO.md` — ordered, dependency-aware task list.
- `prompts/` — one prompt per work item, to hand to implementing agents:
  - `00_shared_context.md` — **prepend to every prompt below.** Hard rules + repo facts.
  - `01_config.md` — global output path (`Settings`, no new dep).
  - `02_write_spec.md` — the registry (source of truth) + drift test.
  - `03_validation.md` — auto-derived strict submodels.
  - `04_writers.md` — write dispatch + typed wrappers + `io/transforms.py` (fixes the
    patchseq bug; relocates `arrow_utils`/`write_utils` into `io/`).
  - `05_readers.md` — predicate-based + cross-dataset reads (folds in `parquet_loader`).
  - `06_notebook_migration.md` — migrate ETL notebooks to the new API.
  - `07_tests.md` — safe-writing test suite.
  - `08_analysis.md` — read-side analysis (`compare_region_coverage`).

## Two hard rules (repeated everywhere on purpose)
1. **Never edit `src/connects_common_connectivity/models.py`** — auto-generated from LinkML.
2. **Never edit `schemas/*.yaml`** without explicit permission from YY.

## Locked decisions
- Config: plain pydantic, version-controlled default + `CCC_OUTPUT_ROOT` env override.
- Write spec: explicit registry, schema-checked for drift.
- Validation: auto-derived strict submodels (single source of truth).

## How to run an item
Hand the implementing agent: `00_shared_context.md` + the specific prompt, and point it at
`ARCHITECTURE.md`. Follow the order in `TODO.md` (config → registry → validation →
writers → readers → notebook migration → tests).
