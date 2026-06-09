# Agent prompt — Read-side analysis + referential check

> **DEFERRED — not actionable this round.** Rides with the read-side work, after config →
> write IO → validation → notebook migration. Design kept for reference.
>
> Prepend `00_shared_context.md`. Depends on `readers.py` (uses read outputs).

Two things land here, both requiring readers to exist:

## A. Read-side analysis — `compare_region_coverage`
Add as a clearly-marked section in `io/readers.py` (NOT a new `io/analysis.py` yet — single
function = premature module; relocate to `io/analysis.py` only when a second analysis
function arrives, a pure move with no public-API change). It reads finished data and
summarizes; it never writes or mutates inputs.

Port `compare_region_coverage(pmms)` from `io/io_plans.md`:
- Input: list of `ProjectionMeasurementMatrix` instances with `region_index` and
  `region_coverage` populated.
- Compute `shared_regions` (intersection of `region_index`), `shared_coverage`
  (intersection of `region_coverage`), and, for every non-empty subset of the inputs, the
  count of regions exclusively covered by that combination.
- Print the summary table shown in `io_plans.md` and return a dict with keys
  `shared_regions`, `shared_coverage`, `exclusive_counts`.

## B. Opt-in referential check — `check_refs`
This is the home for the referential rule deliberately kept off the hot path in
`05_validation.md`. Implement it as an opt-in step invoked by writers:
- `write_models(..., check_refs=False)` — when True, before writing a
  `DataItemDataSetAssociation`, read the `dataset` table (via the readers) and assert each
  `dataset_id` exists for that `project_id`; raise a clear error naming the missing id.
- It reads other tables, so it is NOT a strict-submodel validator and never runs on the
  default write path. Default `check_refs=False` keeps writes fast.

## Tests
- `compare_region_coverage`: small synthetic PMM set gives expected shared/exclusive counts;
  inputs are not mutated.
- `check_refs`: writing an association whose `dataset_id` is absent raises with
  `check_refs=True`, and succeeds (no check) with the default.

## Do not
- Write to disk in the analysis function. Put referential checks on the default write path.
  Touch `models.py` or schemas.
