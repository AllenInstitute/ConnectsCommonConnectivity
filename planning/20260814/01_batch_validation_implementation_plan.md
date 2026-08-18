## Plan: Consolidate Batch Validation Contract

Refactor the write path so `write_models` owns public input normalization and every downstream function receives one non-empty homogeneous exact-type Pydantic batch. Keep `REGISTRY` as the source of truth for normal writes, but make the `WriteSpec` explicitly passed to validation authoritative for that validation call, including custom specs. Remove `_coerce_iterable`, replace ambiguous `Any -> Any` validation behavior with an enforced sequence-to-list contract, preserve the original model instances after successful strict checking, and add regression coverage for every reviewer concern.

**Steps**

### Phase 1: Establish the batch boundary

- [ ] 1. In `/workspaces/ConnectsCommonConnectivity/src/connects_common_connectivity/io/writers.py`, type the public input as `BaseModel | Iterable[BaseModel]` and retain `_normalize_models` as the only shape-conversion helper. This includes one-shot generators, which are materialized before validation and Arrow conversion.
- [ ] 2. Strengthen `_normalize_models` so it converts one `BaseModel` or materializes an iterable exactly once, rejects strings/bytes and other invalid outer values, rejects empty batches, verifies every item is a `BaseModel`, and requires every item to have the exact same concrete type (`type(item) is batch_type`). Error messages should identify the failing index/type where useful.
- [ ] 3. Preserve exact-type semantics rather than allowing subclasses, because registry lookup and Arrow schema construction are based on the concrete generated model class.
- [ ] 4. Keep `write_models` responsible for calling `get_spec(type(items[0]))`; arbitrary custom specs must not be accepted by the public writer, so `REGISTRY` remains authoritative for write behavior.

### Phase 2: Make validation sequence-only and spec-driven

- [ ] 5. In `/workspaces/ConnectsCommonConnectivity/src/connects_common_connectivity/io/write_validation.py`, delete `_coerce_iterable` and remove the `Any`, `Iterable`, and shape-preservation logic it required.
- [ ] 6. Change `validate_for_write` to accept `Sequence[BaseModel]` and return `list[BaseModel]`. It must not accept or return a shape-dependent single model. Keep it exported for focused use and testing, while documenting that callers must provide a normalized non-empty sequence.
- [ ] 7. Enforce that contract at runtime before inspecting the batch: reject strings/bytes and any non-`Sequence` value, including a single model or generator, with a clear `TypeError`, and reject an empty sequence with a clear `ValueError`. This boundary check must not materialize or otherwise normalize arbitrary iterables; `write_models` owns that conversion.
- [ ] 8. Before strict revalidation, inspect every batch member and require `type(model) is spec.model_cls`, reporting the failing row index and actual type. This prevents a later member from reaching `model_dump` or Arrow conversion under a spec selected from only the first item.
- [ ] 9. Change `strict_model_for` to accept a `WriteSpec`, using `spec.model_cls` and `spec.required_for_write` rather than consulting global `REGISTRY`. Preserve caching with a private cached builder keyed by the model class and an immutable tuple of required field names, since mutable Pydantic specs/lists are unsuitable cache keys.
- [ ] 10. Remove the `REGISTRY` import from `write_validation.py`. Document the authority split: `write_models` obtains its spec from `REGISTRY`, while `validate_for_write` honors the exact supplied spec so isolated/custom validation remains possible.
- [ ] 11. In `/workspaces/ConnectsCommonConnectivity/src/connects_common_connectivity/io/write_spec.py`, narrow `WriteSpec.model_cls` to `type[BaseModel]` and align `get_spec` annotations with Pydantic model classes/instances. Do not add a custom-spec parameter to `write_models` or otherwise change registry dispatch.
- [ ] 12. Use each derived strict model only to validate `model.model_dump()`; after successful checking, append the original `model` to the returned list rather than the generated strict-subclass instance. This preserves object identity and guarantees `type(result[i]) is spec.model_cls` for downstream Arrow conversion.
- [ ] 13. In `writers.py`, remove the now-redundant `_validation_hook` forwarding wrapper and call `validate_for_write(items, spec)` directly. Continue passing the returned original-model list to Arrow conversion and dispatch.
- [ ] 14. Update every changed production function and module docstring to describe the final contract accurately. In particular, document normalization ownership, non-empty homogeneous exact-type batches, runtime-enforced sequence-only validation, original-instance list return values, supplied-spec authority, and registry authority for `write_models`; remove stale shape-preservation or registry-lookup descriptions.

### Phase 3: Update focused regression tests

- [ ] 15. Update `/workspaces/ConnectsCommonConnectivity/tests/test_write_validation.py` so all valid direct `validate_for_write` calls pass sequences and assert list returns. Adapt missing-field, happy-path, no-requirements passthrough, and class-mismatch tests to the new contract. For successful validation with and without tightened fields, assert each returned object `is` the corresponding input and has exact type `spec.model_cls`.
- [ ] 16. Update `strict_model_for` tests to pass a `WriteSpec`, retaining checks for parent non-mutation, same-key caching/identity, required-field tightening, and returning the parent model when no fields are tightened.
- [ ] 17. Add one self-contained cache/spec-authority regression using the same model class with its registry spec and a copied/constructed custom spec whose `required_for_write` differs. Assert that equal `(model_cls, required-fields tuple)` keys reuse a class, different required-field tuples produce different classes, and a model accepted by the registry spec is rejected by the custom spec. This proves both the supplied `spec` authority and the complete cache key without depending on test order.
- [ ] 18. Add validation-boundary tests for an empty sequence; a direct single model; a direct one-shot generator; a valid tuple that returns a list containing the original objects; and a later mismatched member whose first item matches `spec.model_cls`. Assert non-sequences are rejected rather than materialized and that the member-mismatch error identifies the later row/type.
- [ ] 19. Update `/workspaces/ConnectsCommonConnectivity/tests/test_writers.py` contract coverage for: one model accepted, homogeneous list, tuple, and one-shot generator inputs accepted, empty input rejected, heterogeneous Pydantic types rejected, a homogeneous iterable of non-Pydantic objects rejected, a mixed model/non-model batch rejected, and an unregistered Pydantic model still rejected by registry lookup. Add a one-shot generator with a later invalid member so the test proves `_normalize_models` materializes exactly once before validation. Prefer assertions through `write_models` so the public boundary is tested; retain existing write/round-trip tests.
- [ ] 20. Review `/workspaces/ConnectsCommonConnectivity/tests/test_write_spec.py` after narrowing `model_cls`; retain existing registry/model identity tests and add or adjust only the minimum type/behavior assertion needed. Do not duplicate validation-policy tests here.
- [ ] 21. Add or update a brief behavioral docstring on every test method changed or added in the three focused test files. Each docstring should state the contract or regression protected, so a failure indicates what behavior needs investigation; avoid implementation narration or generic phrases such as “test that this works.”
- [ ] 22. Amend the existing `Unreleased / Added` write-validation entry in `/workspaces/ConnectsCommonConnectivity/CHANGELOG.md` to describe the final helper signatures, including `validate_for_write([model], spec)` and `strict_model_for(spec)`. Do not add a migration or deprecation entry: these helpers were introduced and refined within the same unreleased PR.

### Phase 4: Verify and review

- [ ] 23. Synchronize the repository environment from the committed lockfile, including the development extra, with `uv sync --locked --extra dev`.
- [ ] 24. Run the focused behavior suite first: `uv run pytest -q tests/test_writers.py tests/test_write_validation.py tests/test_write_spec.py`.
- [ ] 25. Run static checks on the touched production and test files: `uv run ruff check src/connects_common_connectivity/io/writers.py src/connects_common_connectivity/io/write_validation.py src/connects_common_connectivity/io/write_spec.py tests/test_writers.py tests/test_write_validation.py tests/test_write_spec.py` and `uv run mypy src/connects_common_connectivity/io/writers.py src/connects_common_connectivity/io/write_validation.py src/connects_common_connectivity/io/write_spec.py`.
- [ ] 26. Run the full suite in the same environment: `uv run pytest -q`.
- [ ] 27. Review the final diff against the four reviewer comments plus the strengthened contract: one normalization owner, runtime enforcement of sequence-only validation, no validation `Any -> Any`, every item checked, supplied `WriteSpec` requirements honored, original instances/exact types preserved after validation, and custom-spec cache isolation. Include a documentation audit confirming changed production docstrings match the code, every touched test method has a concise behavioral docstring, and the changelog describes the final unreleased API accurately without implying a released migration.

**Relevant files**

- `/workspaces/ConnectsCommonConnectivity/src/connects_common_connectivity/io/writers.py` — own input normalization, resolve the registry spec once, remove `_validation_hook`, and call typed validation directly.
- `/workspaces/ConnectsCommonConnectivity/src/connects_common_connectivity/io/write_validation.py` — remove `_coerce_iterable`, define the sequence-only validation API, validate every member, and build strict models from the supplied spec.
- `/workspaces/ConnectsCommonConnectivity/src/connects_common_connectivity/io/write_spec.py` — narrow the `model_cls` type while preserving registry authority for `write_models`.
- `/workspaces/ConnectsCommonConnectivity/tests/test_writers.py` — exercise the complete public normalization contract and preserve existing dispatch/IO regressions.
- `/workspaces/ConnectsCommonConnectivity/tests/test_write_validation.py` — migrate direct callers to sequence input and cover custom specs, empty input, and later-member mismatch.
- `/workspaces/ConnectsCommonConnectivity/tests/test_write_spec.py` — preserve registry drift checks and align only where the narrowed model-class contract requires it.
- `/workspaces/ConnectsCommonConnectivity/CHANGELOG.md` — amend the existing unreleased write-validation entry to describe the final public helper signatures.

**Verification**

1. `uv sync --locked --extra dev`
2. `uv run pytest -q tests/test_writers.py tests/test_write_validation.py tests/test_write_spec.py`
3. `uv run ruff check src/connects_common_connectivity/io/writers.py src/connects_common_connectivity/io/write_validation.py src/connects_common_connectivity/io/write_spec.py tests/test_writers.py tests/test_write_validation.py tests/test_write_spec.py`
4. `uv run mypy src/connects_common_connectivity/io/writers.py src/connects_common_connectivity/io/write_validation.py src/connects_common_connectivity/io/write_spec.py`
5. `uv run pytest -q`

**Decisions**

- `write_models` accepts either one Pydantic model or an arbitrary iterable, including a one-shot generator; it materializes iterable input exactly once. Downstream validation accepts only the normalized sequence and always returns a list.
- `_normalize_models` is the sole conversion/shape owner. Validation enforces its runtime sequence and non-empty preconditions but does not coerce inputs or preserve caller shape.
- Exact concrete types are required throughout; subclass acceptance is deliberately excluded because registry and Arrow schema selection use the concrete generated class.
- Strict derived subclasses are validation-only implementation details. Successful validation returns the original model objects in a new list, preserving both identity and exact concrete type for downstream consumers.
- `REGISTRY` remains the sole authority for actual `write_models` dispatch. Custom specs are supported only when calling validation explicitly.
- Within `validate_for_write`, the supplied `WriteSpec` is authoritative; global registry lookup is removed from strict-model construction.
- `validate_for_write` and `strict_model_for` remain exported with the final signatures established during their unreleased PR. The changelog describes those final forms without documenting a migration or deprecation from unreleased intermediate signatures.
- No changes to write modes, paths, predicates, generated models, or Delta IO behavior are included.
