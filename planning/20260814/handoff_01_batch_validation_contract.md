# Handoff 01 — Batch validation contract

## Context

The write path currently normalizes model inputs in both `writers.py` and
`write_validation.py`, which leaves ownership of input shape, homogeneity, and spec selection
unclear. The reviewer identified this duplication and questioned whether validation safely
handles every item in a heterogeneous batch.

## Working agreement

This is a prompt for planning, not an implementation plan. Read the relevant code and tests,
then work with the user to agree on the desired public and internal contracts before editing
any files; call out compatibility choices and unresolved questions explicitly.

## Relevant files

- `src/connects_common_connectivity/io/writers.py`
- `src/connects_common_connectivity/io/write_validation.py`
- `src/connects_common_connectivity/io/write_spec.py`
- `tests/test_writers.py`
- `tests/test_write_validation.py`
- `tests/test_write_spec.py`

## Reviewer comments

- [Consolidate `_normalize_models` and `_coerce_iterable`](https://github.com/AllenInstitute/ConnectsCommonConnectivity/pull/5#discussion_r3483744511)
- [Clarify the `Any` input and output contract](https://github.com/AllenInstitute/ConnectsCommonConnectivity/pull/5#discussion_r3483325430)
- [Do not infer batch validity from only the first item](https://github.com/AllenInstitute/ConnectsCommonConnectivity/pull/5#discussion_r3483344302)
- [Explain the purpose of the `spec` parameter](https://github.com/AllenInstitute/ConnectsCommonConnectivity/pull/5#discussion_r3483335247)

## Tasks to plan with the user

1. **Choose one owner for input normalization.** Decide whether `write_models` alone should
   accept a single model or iterable, reject empty inputs, and enforce homogeneous Pydantic
   types. The motivation is to remove duplicated shape handling and give every downstream
   function one reliable batch contract.
2. **Define the validation API after normalization.** Decide whether `validate_for_write`
   should accept only a non-empty `Sequence[BaseModel]`, what it should return, and whether it
   remains public. The motivation is to replace ambiguous `Any -> Any` behavior and unreachable
   shape-preservation branches with a typeable contract.
3. **Validate the entire batch against the selected spec.** Specify how every item is checked
   against `spec.model_cls`, including the errors for heterogeneous or non-Pydantic inputs. The
   motivation is to ensure that selecting a class from the first item cannot let later items
   reach Arrow conversion under the wrong schema.
4. **Make spec authority explicit.** Decide whether strict-model construction should consume
   the supplied `WriteSpec` or whether validation should remove the `spec` argument and use the
   global registry exclusively. The motivation is that the current function accepts a spec but
   reloads `required_for_write` from `REGISTRY`, so a caller-supplied spec can be silently ignored.
5. **Plan focused regression coverage.** Include cases for a single model, a homogeneous batch,
   an empty batch, a heterogeneous batch, a non-Pydantic item, a mismatched spec, and a spec with
   custom requirements. The motivation is to make the agreed boundary executable and prevent
   the two normalization paths from reappearing.

## Planning outcome

Produce a proposed contract, a minimal ordered implementation plan, compatibility notes, and
the exact focused tests to run. Stop for user approval before changing production or test code.