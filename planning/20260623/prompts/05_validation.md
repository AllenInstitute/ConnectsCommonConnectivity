# Agent prompt — Write-validation (auto-derived strict submodels)

> Prepend `00_shared_context.md`. Depends on `write_spec.py` (W2) and `writers.py` (W3).
> Wires into the pass-through `_validation_hook(models, spec) -> models` left in
> `write_models`.

## Naming
File is `io/write_validation.py` — write-time, pydantic-only, registry-coupled. The
generic word "validation" is already used by `cli.py`'s LinkML conformance check; the
two are intentionally distinct.

## What W5 ships
1. **Populate `required_for_write`** on the registry entries that need it. Driven by
   the same prototype loop as W3: read the corresponding notebook's write call, identify
   the slots the predicate / partition / append-id depend on, and list them. Empty list
   is a valid answer — only add slots a real write actually relies on.
2. `strict_model_for(model_cls) -> type[BaseModel]`:
   - Subclass the generated model at runtime; do NOT mutate `models.py` classes.
   - For each name in `spec.required_for_write`, override the field to be required
     (no default, not Optional). Use any pydantic v2 mechanism that doesn't touch the
     parent class.
   - Cache by class so the derived type is built once.
3. `validate_for_write(models, spec) -> models` — accepts the same shape `_validation_hook`
   already does (single instance OR iterable, returns the same shape). Runs each instance
   through the strict submodel; on failure, raise an error naming the class and the
   failing slot. Pydantic-only, no I/O.
4. **Wire it in.** In `write_models`, replace the pass-through `_validation_hook` with
   `validate_for_write`. This is the only edit to `writers.py`.

## Out of scope (deferred, not skipped)
- Cross-field rules. `WriteSpec.cross_field_rules` exists as an empty list; until a real
  invariant needs one, do not introduce a rule registry. Add the dict + `model_validator`
  scaffolding when the first rule is actually written, not before.
- Referential checks (e.g. "association.dataset_id exists in DataSet"). These read other
  tables and belong with the read-side opt-in `check_refs` (`_deferred/09_analysis.md`),
  not on the write path.

## Tests (`tests/test_write_validation.py`)
- A model with a missing `required_for_write` slot fails before any IO.
- A model with all slots present passes and is returned unchanged (field-by-field equal).
- The class object in `models.py` has the same `model_fields` after `strict_model_for`
  runs as before — proving no in-place mutation.
- `validate_for_write([m1, m2], spec)` accepts a list (same shape contract as the hook).

## Do not
- Edit `models.py` or schemas. Restate field types from the schema. Call the LinkML
  validator on the write path. Add cross-field rules speculatively.