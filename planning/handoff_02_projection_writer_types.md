# Handoff 02 — Projection writer types

## Context

`write_projection_matrix` and `populate_region_coverage` currently accept `Any` for both the
projection model and dense matrix even though their implementation expects a specific Pydantic
model and a numeric two-dimensional array. The reviewer requested stricter types for both
parameters.

## Working agreement

This is a prompt for planning, not an implementation plan. Inspect current callers and tests,
then discuss the static and runtime contracts with the user before editing any files.

## Relevant files

- `src/connects_common_connectivity/io/writers.py`
- `src/connects_common_connectivity/io/write_utils.py`
- `src/connects_common_connectivity/models.py` (read-only generated model definitions)
- `tests/test_writers.py`
- `tests/test_write_utils.py`
- `tests/test_projection_schema.py`
- `code/etl_wnm_exc_04_projection_matrix.ipynb`
- `planning/20260623/prompts/03_writers.md`

## Reviewer comments

- [Use a stricter type for `pmm`](https://github.com/AllenInstitute/ConnectsCommonConnectivity/pull/5#discussion_r3483848659)
- [Use a stricter type for `matrix`](https://github.com/AllenInstitute/ConnectsCommonConnectivity/pull/5#discussion_r3483849488)

## Tasks to plan with the user

1. **Type the projection model explicitly.** Evaluate using
   `ProjectionMeasurementMatrix` for the `pmm` parameter and return type in both the public
   writer and enrichment helper. The motivation is to expose the fields the implementation
   requires and catch invalid callers before runtime.
2. **Choose the matrix type contract.** Compare `numpy.typing.ArrayLike`,
   `NDArray[np.number]`, or a deliberately narrower alternative against actual notebook and
   test callers. The motivation is to improve type checking without rejecting useful inputs
   that are intentionally converted through `np.asarray`.
3. **Define runtime validation.** Agree on behavior for nonnumeric data, object arrays,
   one-dimensional inputs, more-than-two-dimensional inputs, and row/column shape mismatches
   with `data_item_index` and `region_index`. The motivation is that type hints alone do not
   protect untyped notebook callers or guarantee scientific shape consistency.
4. **Assess caller and import impact.** Check whether importing the generated model into
   `write_utils.py` introduces cycles and whether any notebook passes lists, Polars objects, or
   other array-like values. The motivation is to select a contract that is strict, practical,
   and compatible with the existing IO module graph.
5. **Plan focused tests.** Cover a valid numeric matrix, accepted array-like inputs if any,
   nonnumeric values, wrong dimensionality, mismatched dimensions, and non-mutation of the
   original model. The motivation is to pin both the type decision and runtime behavior.

## Planning outcome

Produce the recommended type signatures, runtime checks, compatibility assessment, minimal
implementation sequence, and focused validation commands. Stop for user approval before edits.