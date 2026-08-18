# Plan: Improve Write-Path Docstrings

Document the write-path helper contracts requested in review, then make a bounded consistency
pass across the functions in `writers.py` and the `write_*.py` modules. The work should help a
maintainer understand inputs, outputs, and side effects without reconstructing the call graph.
It must not change runtime behavior.

## Scope

The required review targets are:

- `validate_for_write`
- `_dispatch_overwrite_scoped`
- `_dispatch_append_new_by_id`
- `_resolve_output_root`

The consistency pass covers module-level functions in:

- `src/connects_common_connectivity/io/writers.py`
- `src/connects_common_connectivity/io/write_validation.py`
- `src/connects_common_connectivity/io/write_spec.py`
- `src/connects_common_connectivity/io/write_utils.py`

This is a documentation-only change. Signatures, implementation, tests, schemas, generated
models, registry entries, and IO behavior are outside scope.

## Docstring Rules

- Describe the function's contract beyond what its name and type annotations already say.
- Start with purpose and responsibility, not a restatement of the implementation.
- Define each parameter by its semantic role and explain important interactions between
  parameters. For policy or configuration objects, clarify what they control in this function.
- Give paths precise meanings, such as output root or Delta table directory, and state how the
  function uses them.
- Explain each return value independently, especially tuple members and optional values.
- State caller-relevant preconditions, invariants, and side effects that are not evident from the
  signature.
- Document errors callers can reasonably act on; omit incidental dependency exceptions.
- Keep detail proportional to ambiguity. Consistency requires shared terminology and equivalent
  contract coverage, not identical formatting or length.
- Verify every claim against the implementation and avoid promising behavior the code does not
  guarantee.

## Tasks

- [ ] Confirm the current signatures and behavior of the four required helpers before editing
  their docstrings. Treat the implementation after the batch-validation and projection-typing
  work as authoritative.
- [ ] Address the four reviewer comments. The revised docstrings should make parameter roles and
  interactions clear, explain meaningful return values, and identify relevant side effects and
  invariants. In particular, document that:
  - `validate_for_write` uses `spec.model_cls` for exact-type validation and
    `spec.required_for_write` for strict field validation, returns the original instances in a
    new list, and performs no IO;
  - the dispatch helpers receive a write-ready Arrow table and the complete Delta table path;
    `_dispatch_overwrite_scoped` uses the spec's scope and partition columns, while
    `_dispatch_append_new_by_id` uses its first scope column as the id column and requires one
    `project_id` in the table; and
  - `_resolve_output_root` returns both the output root later combined with `spec.subdir` and the
    optional resolved `Settings` retained for `dry_run` handling. The settings value is `None`
    when an explicit `output_root` is used.
- [ ] Review all module-level function docstrings in the scoped write modules for consistency.
  Update only docstrings that remain ambiguous or conflict with the standard established by the
  four required helpers. Keep detail proportional to the function; do not expand adequate
  docstrings merely to make their format identical.
- [ ] Use consistent terminology for model batches, Arrow tables, scopes, predicates, output
  roots, Delta table paths, validation, and write results. Prefer contract information that is
  not already obvious from the signature over implementation narration.
- [ ] **Correct inaccurate claims.** Verify documentation against actual behavior. In particular,
  narrow `append_new_dataitems` claims about safety and idempotency to the conditions the
  implementation guarantees: sequential calls where the existing Delta table can be read. Do
  not turn documentation discrepancies into behavioral fixes.
- [ ] Review the completed documentation against the original comments and remove unnecessary
  repetition. The final diff should contain only docstring changes in the scoped production
  modules.

## Acceptance Criteria

- Each of the four reviewer comments is directly addressed.
- Parameter descriptions explain semantic purpose and important interactions, rather than
  repeating annotations.
- Structured and optional return values are understandable without reading the function body.
- Delta IO side effects and caller-relevant invariants are documented where they matter.
- All functions in the scoped write modules have been considered for consistency, while only
  incomplete or inaccurate docstrings are changed.
- No documentation promises behavior the implementation does not guarantee.
- No runtime behavior or public signature changes are included.

## Verification

1. Run syntax and lint checks for the scoped modules.
2. Run the focused write-path tests as a behavior-preservation check.
3. Inspect the final diff to confirm it is documentation-only and satisfies the four review
   comments without unrelated documentation churn.
