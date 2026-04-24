# IO Utility Functions — Plans

## `populate_region_coverage(pmm, matrix) → ProjectionMeasurementMatrix`

Automatically populates the `region_coverage` field on a `ProjectionMeasurementMatrix` from the dense values array.

- **Input:**
  - `pmm`: a `ProjectionMeasurementMatrix` instance with `region_index` already set.
  - `matrix`: dense numeric array of shape `(len(data_item_index), len(region_index))` — numpy ndarray or similar.
- **Logic:** For each column index `i`, check `any(matrix[:, i] != 0)`. Collect the corresponding `pmm.region_index[i]` entries where the column has at least one non-zero value.
- **Output:** Returns a copy of `pmm` with `region_coverage` set to the non-zero-column subset of `region_index`.
- **Properties:** Pure function, no side effects. Does not modify the input `pmm`.

---

## `compare_region_coverage(pmms) → dict`

Compares region index and region coverage across multiple `ProjectionMeasurementMatrix` instances. Answers: "which regions are shared, and which are exclusive to specific dataset combinations?"

- **Input:**
  - `pmms`: list of `ProjectionMeasurementMatrix` instances, each with `region_index` and `region_coverage` populated.
- **Computes:**
  - `shared_regions`: intersection of all `region_index` across inputs (what regions can we compare at all?).
  - `shared_coverage`: intersection of all `region_coverage` across inputs (where do all datasets have signal?).
  - For every non-empty subset of the input PMMs (powerset, size 1 through N): count of regions that are in that subset's `region_coverage` intersection but **not** in any other PMM's `region_coverage` (exclusive to that combination).
- **Prints:** A summary table showing, for each subset combination, how many regions are exclusively covered by that combination. Example for 3 datasets A, B, C:
  ```
  Only in A:           12
  Only in B:            5
  Only in C:            8
  Only in A ∩ B:        3
  Only in A ∩ C:        2
  Only in B ∩ C:        1
  In all (A ∩ B ∩ C):  45
  ```
- **Returns:** dict with keys `shared_regions`, `shared_coverage`, and `exclusive_counts` (mapping subset labels to region counts).
