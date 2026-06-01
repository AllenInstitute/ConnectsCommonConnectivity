# Agent prompt — Validation (auto-derived strict submodels)

> Prepend `00_shared_context.md`. Depends on `write_spec.py`.

## Goal
Create `src/connects_common_connectivity/io/validation.py` that derives a STRICT pydantic
submodel per class **at runtime** from (a) the generated model in `models.py` and (b) the
registry's `required_for_write` + `cross_field_rules`. Single source of truth: nothing
is restated from the schema.

## Requirements
1. `strict_model_for(model_cls) -> type[BaseModel]`:
   - Subclass the generated model.
   - For each slot in the registry's `required_for_write`, make it required (no default /
     not Optional). Use pydantic v2 mechanisms (`model_fields` overrides via
     `create_model` or field re-annotation) — do NOT edit the generated class in place.
   - Attach each named `cross_field_rule` as a `@model_validator(mode="after")`.
   - Cache the derived class (e.g. `functools.lru_cache`) so it's built once.
2. `validate_for_write(model) -> model` (or list): run the instance through the strict
   submodel, raising a clear error that names the class, the failing slot/rule, and the
   offending value. This runs on the hot write path, so keep it pydantic-only (fast); do
   NOT call the LinkML/`cli.py` validator here.
3. Implement a starter cross-field rule registry (a dict name → callable) including:
   - `association_dataset_exists`: a `DataItemDataSetAssociation`'s `dataset_id` must
     exist among written DataSets for that `project_id`. (May need a reader/lookup; if the
     reader module isn't ready, implement the hook and mark it TODO without breaking the
     import.)
   Add others only as the registry references them.

## Tests (`tests/test_validation.py`)
- A model missing a `required_for_write` slot fails `validate_for_write` before any IO.
- A valid model passes and is returned unchanged (round-trip equality on fields).
- The generated `models.py` class is unchanged after deriving the strict model
  (no in-place mutation).

## Do not
- Edit `models.py`. Restate schema field definitions. Put LinkML validation on the write path.
