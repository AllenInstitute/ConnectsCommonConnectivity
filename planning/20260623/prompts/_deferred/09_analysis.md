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

Spec for `compare_region_coverage(pmms) → dict` (moved here from the old
`src/connects_common_connectivity/io/io_plans.md`; source-tree file deleted):

- **Input:** `pmms` — list of `ProjectionMeasurementMatrix` instances, each with
  `region_index` and `region_coverage` populated. (`region_coverage` is produced by
  `populate_region_coverage`, already shipped in `io/write_utils.py`.)
- **Computes:**
  - `shared_regions`: intersection of all `region_index` across inputs (what regions can
    we compare at all?).
  - `shared_coverage`: intersection of all `region_coverage` across inputs (where do all
    datasets have signal?).
  - For every non-empty subset of the input PMMs (powerset, size 1 through N): count of
    regions that are in that subset's `region_coverage` intersection but **not** in any
    other PMM's `region_coverage` (exclusive to that combination).
- **Prints:** A summary table showing, for each subset combination, how many regions are
  exclusively covered by that combination. Example for 3 datasets A, B, C:
  ```
  Only in A:           12
  Only in B:            5
  Only in C:            8
  Only in A ∩ B:        3
  Only in A ∩ C:        2
  Only in B ∩ C:        1
  In all (A ∩ B ∩ C):  45
  ```
- **Returns:** dict with keys `shared_regions`, `shared_coverage`, and
  `exclusive_counts` (mapping subset labels to region counts).
- **Properties:** Pure function, no side effects. Does not modify inputs.

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
