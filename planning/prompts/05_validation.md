# Agent prompt — Write-validation (auto-derived strict submodels)

> Prepend `00_shared_context.md`. Depends on `write_spec.py` and `writers.py` (built after
> the write path; wires into the pass-through validation hook left in `write_models`).

## Naming
File is `io/write_validation.py`, NOT `io/validation.py`: this is specifically write-safety
validation coupled to `write_spec`. The generic word "validation" is already claimed by
`cli.py`'s LinkML full-conformance check — keep the two distinct.

## Goal
Create `src/connects_common_connectivity/io/write_validation.py` that derives a STRICT pydantic
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
   offending value. This runs on the hot write path, so keep it pydantic-only (fast, **no
   I/O**); do NOT call the LinkML/`cli.py` validator here.
3. **Wire it into `write_models`:** replace the pass-through validation hook left by
   `03_writers.md` with `validate_for_write`. This is the only change to the writer.
4. Implement a starter cross-field rule registry (a dict name → callable). Rules here MUST
   be pure: they inspect only the model instance in hand, do no I/O, and never read other
   tables. Add rules only as the registry references them.
   - Do NOT implement `association_dataset_exists` here. It reads written DataSets, so it is
     a referential check, not a structural one — it is deferred with the read-side work as an
     opt-in `check_refs` (`09_analysis.md`). Keeping it out keeps validation free of any
     dependency on readers.

## Tests (`tests/test_write_validation.py`)
- A model missing a `required_for_write` slot fails `validate_for_write` before any IO.
- A valid model passes and is returned unchanged (round-trip equality on fields).
- The generated `models.py` class is unchanged after deriving the strict model
  (no in-place mutation).

## Do not
- Edit `models.py`. Restate schema field definitions. Put LinkML validation on the write path.
