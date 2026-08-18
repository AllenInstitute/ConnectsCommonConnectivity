# PR review changes

The following changes were pushed after the
review.

### Imports and public API cleanup

- Moved runtime and test imports to module scope and converted package-relative
	imports to absolute `connects_common_connectivity...` imports. The relocation
	tests retain dynamic imports only where they intentionally assert that removed
	modules cannot be imported. (`3790a39`, `a0dcae7`)
- Renamed the result dataclass and all public exports, annotations, and tests
	from `WriteResult` to `WrittenResult`, reflecting that it describes an
	already-completed write. (`a520841`)
- Removed the custom `Settings.describe()` and `Settings.__repr__()` methods;
	Pydantic's existing model representation is used instead. (`2f6d2c2`)
- Removed the unused `table_path()` helper and its `io` re-export. Writers and
	their `WriteSpec` entries continue to own canonical table subdirectories.
	(`093549c`)
- Removed the string-returning `output_root()` helper. ETL notebooks and the ETL
	prompt now use the absolute `Path` from `get_settings().output_root` and join
	subpaths with `/`. Associated helper tests and exports were removed.
	(`537dcb2`)

### Batch normalization and write validation

- Made `write_models()` the only input-normalization boundary. It accepts one
	Pydantic model or an iterable, materializes iterables once, rejects empty or
	non-Pydantic inputs, and requires every item to have the same exact concrete
	model type. (`57b9c4c`)
- Changed `validate_for_write()` to accept only a non-empty
	`Sequence[BaseModel]`, validate every member against the exact
	`spec.model_cls`, and return a new list containing the original instances.
	It no longer normalizes single models or generators. (`57b9c4c`)
- Made the supplied `WriteSpec` authoritative for direct validation.
	`strict_model_for(spec)` no longer consults the global registry, and its
	bounded cache is keyed by the model class and sorted required-field policy.
	`write_models()` still obtains its spec from the registry. (`57b9c4c`)
- Narrowed `WriteSpec.model_cls`, `get_spec()`, and `WRITABLE_CLASSES` to
	Pydantic model classes and removed the redundant validation forwarding hook.
	(`57b9c4c`)
- Added regression coverage for supported writer input shapes, invalid and
	heterogeneous batches, later-member failures, exact-type validation, custom
	spec authority, cache isolation, and preservation of input object identity.
	(`57b9c4c`)
- Updated the unreleased changelog entry to describe the final
	`strict_model_for(spec)` and `validate_for_write(models, spec)` contracts.
	(`57b9c4c`)
- Replaced truthiness-based empty checks with explicit `len(...) == 0` checks at
	the reviewed sequence and scope boundaries. (`ac3b830`, `f2fd6ea`)

### Projection writer typing

- Replaced `Any` with `ProjectionMeasurementMatrix` for projection metadata and
	`numpy.typing.ArrayLike` for matrix inputs in both
	`write_projection_matrix()` and `populate_region_coverage()`.
	(`f18e278`)
- Kept the existing `numpy.asarray` conversion and runtime shape checks, and
	added a nested-list test to confirm that non-NumPy array-like inputs remain
	supported without mutating the original model. (`f18e278`)

### Contract and test documentation

- Expanded write-path docstrings to describe parameter roles, return values,
	errors, invariants, and IO side effects. This includes the reviewed contracts
	for `validate_for_write()`, `_dispatch_overwrite_scoped()`,
	`_dispatch_append_new_by_id()`, and `_resolve_output_root()`. (`a04579e`)
- Corrected the `append_new_dataitems()` documentation to limit duplicate
	prevention to sequential calls where the existing Delta table can be read;
	it does not claim concurrency protection or idempotency after read failures.
	(`a04579e`)
- Removed the stale hardcoded writable-class list from `write_models()`
	documentation in favor of runtime discovery through `WRITABLE_CLASSES`.
	(`ac3b830`)
- Added concise behavioral docstrings across 13 test modules, including the
	requested config and Parquet-loader tests. (`429c9c8`)

### Supporting repository changes

- Added review-planning documents and implementation plans. These record the
	review topics but are not the basis for the implementation claims above;
	those claims come from the production and test diffs. (`d39a31a`, `0848c73`,
	`a04579e`)
- Added a repository-local docstring writing and auditing skill.
	(`1dc361a`)
- Added a future configuration architecture document. That commit changed only
	planning documentation; it did not untrack `ccc_config.yaml` or alter runtime
	configuration behavior. (`f4be99f`)
- Added the Code Ocean secret declaration used for V1DD authentication.
	(`8563537`)
