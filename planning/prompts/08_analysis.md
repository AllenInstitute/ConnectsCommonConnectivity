# Agent prompt — Analysis module (read-side)

> Prepend `00_shared_context.md`. Depends on `readers.py` (uses read outputs).

## Goal
Create `src/connects_common_connectivity/io/analysis.py` for read-side analysis over
already-written tables. This is distinct from `io/transforms.py` (write-side enrichment):
analysis reads finished data and summarizes; it never writes or mutates inputs.

## Seed function
Port `compare_region_coverage(pmms)` from `io/io_plans.md`:
- Input: list of `ProjectionMeasurementMatrix` instances with `region_index` and
  `region_coverage` populated.
- Compute `shared_regions` (intersection of `region_index`), `shared_coverage`
  (intersection of `region_coverage`), and, for every non-empty subset of the inputs, the
  count of regions exclusively covered by that combination.
- Print the summary table shown in `io_plans.md` and return a dict with keys
  `shared_regions`, `shared_coverage`, `exclusive_counts`.

## Tests (`tests/test_analysis.py`)
- Small synthetic set of PMMs gives the expected shared/exclusive counts.
- Pure: inputs are not mutated.

## Do not
- Write to disk here. Touch `models.py` or schemas.
