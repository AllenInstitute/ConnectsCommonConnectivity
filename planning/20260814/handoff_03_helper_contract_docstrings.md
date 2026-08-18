# Handoff 03 — Helper contract docstrings

## Context

Several write-path helpers have short docstrings that describe the operation but not the role
of each parameter, the return contract, or how `WriteSpec` controls the Arrow table and output
path. The reviewer requested maintainable API documentation for these boundaries.

## Working agreement

This is a prompt for planning, not an implementation plan. Confirm the final signatures from
the batch-validation and projection-typing work with the user before proposing docstring edits,
and do not change behavior as part of this package.

## Relevant files

- `src/connects_common_connectivity/io/writers.py`
- `src/connects_common_connectivity/io/write_validation.py`
- `src/connects_common_connectivity/io/write_spec.py`
- `src/connects_common_connectivity/io/write_utils.py`
- `tests/test_writers.py`
- `tests/test_write_validation.py`
- `planning/handoff_01_batch_validation_contract.md`
- `planning/handoff_02_projection_writer_types.md`

## Reviewer comments

- [Document `validate_for_write` parameters and the purpose of `spec`](https://github.com/AllenInstitute/ConnectsCommonConnectivity/pull/5#discussion_r3483335247)
- [Document overwrite-dispatch parameters](https://github.com/AllenInstitute/ConnectsCommonConnectivity/pull/5#discussion_r3483815561)
- [Document append-dispatch parameters](https://github.com/AllenInstitute/ConnectsCommonConnectivity/pull/5#discussion_r3483822893)
- [Explain `_resolve_output_root` return values](https://github.com/AllenInstitute/ConnectsCommonConnectivity/pull/5#discussion_r3483830599)

## Tasks to plan with the user

1. **Inventory the helper contracts that need expansion.** Start with
   `validate_for_write`, `_dispatch_overwrite_scoped`, `_dispatch_append_new_by_id`, and
   `_resolve_output_root`, then identify directly related helpers whose contracts would remain
   ambiguous. The motivation is to address the reviewer comments consistently without creating
   broad documentation churn.
2. **Document parameters in terms of ownership and interaction.** Explain what the Arrow table
   contains, which `WriteSpec` fields each dispatcher consumes, what `path` points to, and what
   validation has already occurred. The motivation is to make assumptions at module boundaries
   visible to future maintainers.
3. **Document return values and side effects.** Define `WrittenResult`, explain why
   `_resolve_output_root` returns both a `Path` and optional `Settings`, and state which helpers
   perform Delta IO. The motivation is to let readers understand dry-run behavior and dispatch
   outcomes without reconstructing the call graph.
4. **Document errors and invariants.** List meaningful exceptions for empty scope columns,
   mixed projects, model/spec mismatches, and invalid output-root combinations after confirming
   the final contracts. The motivation is to turn current implementation details into stable,
   reviewable behavior.
5. **Correct inaccurate claims.** Verify the statement that strict runtime subclasses preserve
   parent field metadata, because replacing a field with `Field(...)` may drop descriptions and
   aliases. The motivation is to avoid documentation that promises behavior the implementation
   does not provide; discuss whether to fix the behavior or narrow the wording.

## Planning outcome

Produce a bounded list of docstrings to update, the information each one must contain, any
inaccurate claims needing a user decision, and documentation-focused validation steps. Stop for
user approval before editing.