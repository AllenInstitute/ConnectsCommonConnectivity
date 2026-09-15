# Plan: Type Projection Writer Inputs

Replace the projection writer's broad `Any` annotations with types that match its existing
implementation and callers. Do not expand this reviewer request into new runtime validation.

**Steps**

- [ ] In `src/connects_common_connectivity/io/write_utils.py`, type `pmm` and the return value of `populate_region_coverage` as `ProjectionMeasurementMatrix`. Type `matrix` as `numpy.typing.ArrayLike`, matching the existing `np.asarray` conversion.
- [ ] Apply the same `ProjectionMeasurementMatrix` and `ArrayLike` parameter types to `write_projection_matrix` in `src/connects_common_connectivity/io/writers.py`; retain its existing `WrittenResult` return type.
- [ ] Preserve current behavior: `populate_region_coverage` continues to require `region_index`, convert with `np.asarray`, reject non-2D input, validate the column count, derive non-zero coverage, and return a copied model. Do not add dtype rules, row-count validation, or new index requirements.
- [ ] Update only focused tests or docstrings needed to reflect the typed signatures. Retain the existing NumPy `ndarray` writer coverage and add one `populate_region_coverage` case using a nested Python list to confirm a non-NumPy `ArrayLike` input still works. The current WNM notebook already passes NumPy arrays and should require no change.
- [ ] Do not edit schemas or generated models. Any broader projection-model concern is separate work, not a reason to add an IO workaround here.

**Verification**

1. `uv run pytest -q tests/test_write_utils.py tests/test_writers.py`
2. `uv run ruff check src/connects_common_connectivity/io/write_utils.py src/connects_common_connectivity/io/writers.py tests/test_write_utils.py tests/test_writers.py`
3. `uv run mypy src/connects_common_connectivity/io/write_utils.py src/connects_common_connectivity/io/writers.py`
4. `uv run pytest -q`