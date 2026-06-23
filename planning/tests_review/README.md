# Tests Review — Findings & Implementation Plan

Review of `tests/` (12 files, ~1,540 LOC) on branch `ingestion-v2`.

## Documents

- [`findings.md`](./findings.md) — Numbered review report: high / medium / low priority issues, plus what's working well.
- [`plan.md`](./plan.md) — 5-PR implementation plan for the next steps, with code snippets and sequencing.

## TL;DR

Suite is solid (good docstrings, parametrization, regression tests named after the bug). Main gaps:

1. No `conftest.py` → duplicated helpers, cache-pollution risk.
2. `pytest.raises(Exception)` used in several places → too broad.
3. Regression assertions lack failure messages.
4. `WRITABLE_CLASSES` ↔ `_make_instance` drift is silent.
5. Missing coverage for `cli.py`, `parquet_loader.py`, and `dry_run` semantics.

Five small PRs proposed, sequenced so PRs 1–4 are pure test refactors and PR 5 is the only one likely to surface production bugs.
