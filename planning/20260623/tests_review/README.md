# Tests Review — Findings & Implementation Plan

Review of `tests/` (12 files, ~1,540 LOC) on branch `ingestion-v2`.

## Documents

- [`findings.md`](./findings.md) — Numbered review report: high / medium / low priority issues, plus what's working well.
- [`plan.md`](./plan.md) — Sequential implementation plan (5 work packages) for an agent to execute end-to-end in one go, with code snippets and per-package guardrails.

## TL;DR

Suite is solid (good docstrings, parametrization, regression tests named after the bug). Main gaps:

1. No `conftest.py` → duplicated helpers, cache-pollution risk.
2. `pytest.raises(Exception)` used in several places → too broad.
3. Regression assertions lack failure messages.
4. `WRITABLE_CLASSES` ↔ `_make_instance` drift is silent.
5. Missing coverage for `cli.py`, `parquet_loader.py`, and `dry_run` semantics.

Five sequential work packages proposed for end-to-end agent execution. WPs 1–4 are pure test refactors; WP 5 is the only one likely to surface production bugs (fix in-place).
