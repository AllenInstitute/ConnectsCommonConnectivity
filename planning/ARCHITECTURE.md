# IO Layer Architecture — write / read / validation

Status: design agreed 2026-06-01. Implementation to be done by follow-up agents.
This document is the source of truth for the design. The runnable agent prompts
live in `planning/prompts/`. The task breakdown lives in `planning/TODO.md`.

## Hard constraints (read before any work)

1. **Do not edit `src/connects_common_connectivity/models.py`.** It is auto-generated
   from the LinkML YAMLs in `schemas/`. Any change to the data model happens in the
   YAMLs and is regenerated — never hand-edited.
2. **Do not change the LinkML schemas without explicit permission from YY.** If safe
   writing turns out to need a new slot (e.g. a clearer project/dataset scoping key),
   stop and ask first. Propose the change in writing; do not edit `schemas/*.yaml`
   pre-emptively.
3. **Single source of truth = the LinkML schema.** The registry and the derived
   validators read from the generated models; they never restate field definitions.
4. New IO code lives under `src/connects_common_connectivity/io/`. Plotting stays in
   `code/utils.py`. Notebooks are migrated to call the new API, not to embed logic.

## What exists today (do not rebuild)

- `models.py` — LinkML-generated pydantic v2 models. Key classes:
  `DataSet`, `DataItem`, `DataItemDataSetAssociation`, `Cluster`, `ClusterHierarchy`,
  `ClusterMembership`, `CellFeatureSet`, `CellFeatureDefinition`, `CellFeatureMatrix`,
  `CellFeatureMeasurement`, `MappingSet`, `CellToCellMapping`, `CellToClusterMapping`,
  `ClusterToClusterMapping`, `ProjectionMeasurementMatrix`, `BrainRegionAssociation`,
  `ZarrDataset`, `ParquetDataset`. A `ProjectScoped` mixin supplies `project_id`.
- `arrow_utils.py` — `build_arrow_schema(model_cls)`, `models_to_table(models, schema)`,
  `attach_linkml_metadata(table, linkml_class=...)`, `build_cell_feature_matrix_schema(...)`.
  These already convert pydantic models → Arrow tables with LinkML metadata. **Reuse.**
- `write_utils.py` — `append_new_dataitems(path, table, project_id=...)` (id-deduped
  append) and `walk_ancestors(leaf_id, parent_of)` (hierarchy denormalization). **Reuse;
  the new writers wrap/generalize these rather than replacing them.**
- `parquet_loader.py` — `load_parquet_to_models(...)` (Parquet → models with a report).
- `cli.py` — LinkML `SchemaView`-based full validation (the `ccc` command). Kept as the
  occasional heavyweight conformance check, **not** on the hot write path.
- `io/io_plans.md` — pre-existing analysis-util plans (`populate_region_coverage`,
  `compare_region_coverage`). Keep; fold into the read/analysis module.

## The bug this design fixes

In every `_01_dataset_dataitem` notebook the DataSet is written with:

```python
write_deltalake(root+"dataset/", table_ds, mode="overwrite",
                predicate=f"project_id = '{PROJECT_ID}'", partition_by=["project_id"])
```

`visp_exc_patchseq` and `visp_inh_patchseq` **share** `project_id = 'visp_patchseq'` but
have different `dataset_id`. So writing the inhibitory dataset overwrites the excitatory
dataset's row (and vice versa). The association write already does the right thing
(`predicate = "project_id = '...' AND dataset_id = '...'"`). The fix is structural: the
correct scope columns must come from a **per-class registry**, not be retyped by hand in
each notebook. `DataSet`'s scope is `(project_id, id)`; the association's scope is
`(project_id, dataset_id)`; `DataItem` is append-by-id; etc.

## Design overview

One registry entry per class is the hub. It drives four things so they can never drift
apart: partitioning, the overwrite predicate (scope columns), which slots are required
for a safe write, and the auto-derived strict validator.

```
                 ┌─────────────────────────────┐
LinkML schema ──▶│  models.py (generated)      │
                 └─────────────────────────────┘
                              │ read-only
                              ▼
        ┌───────────────────────────────────────────────┐
        │  write_spec registry  (one entry per class)    │
        │  partition_by · scope_columns · write_mode ·   │
        │  required_for_write · cross_field_rules        │
        └───────────────────────────────────────────────┘
            │              │                 │
            ▼              ▼                 ▼
      validation      write module        read module
   (strict submodel  (write_dataset,    (predicate-based +
    derived per      write_dataitem,     flexible cross-dataset
    class)           write_features...)  reads)
                              │
                              ▼
                    Settings (global output_root)
```

## Module 1 — `config.py` (global output path)

Decision: **plain pydantic `BaseModel`**, version-controlled default in code, optional
env override. No new dependency (no pydantic-settings).

```python
class Settings(BaseModel):
    output_root: Path = Path("../scratch/em_patchseq_wnm_v1/")
    # add knobs here later (dry_run, schema_version_pin, ...) as needed

    @classmethod
    def load(cls) -> "Settings":
        default = cls.model_fields["output_root"].default
        return cls(output_root=os.environ.get("CCC_OUTPUT_ROOT", default))
```

Rationale: the default is readable in git without running anything and adds no
dependency; the `CCC_OUTPUT_ROOT` env override is the escape hatch for CodeOcean, where
the write location differs from local. Notebooks replace the hardcoded `OUTPUT_ROOT`
string with `settings = Settings.load()` and print the resolved value at the top.
A `table_path(settings, "dataset")` helper resolves per-table subdirectories so notebooks
never concatenate path strings.

## Module 2 — `write_spec.py` (the registry)

An explicit, hand-maintained lookup, one entry per writable class, seeded from the schema
and refined from early experience. It is the source of truth for write/validation
behavior. A test cross-checks it against the LinkML schema so drift fails loudly (the
class names and `project_id`/identifier slots must exist in the generated models).

Each entry declares:

- `subdir` — Delta table subdirectory under `output_root` (e.g. `"dataset"`).
- `partition_by` — Delta partition columns (e.g. `["project_id"]`).
- `scope_columns` — columns that define the overwrite predicate (the identity within the
  shared table). DataSet → `["project_id", "id"]`; DataItemDataSetAssociation →
  `["project_id", "dataset_id"]`.
- `write_mode` — `"overwrite_scoped"` (scoped idempotent overwrite) or
  `"append_new_by_id"` (the `append_new_dataitems` behavior for DataItem).
- `required_for_write` — slots that must be present/non-null for a safe write (may be
  stricter than the schema's own `required`).
- `cross_field_rules` — names of cross-field checks to attach to the strict validator.

Predicate is built from `scope_columns` + the row values, e.g.
`"project_id = 'visp_patchseq' AND id = 'visp_exc_patchseq'"`. This is exactly the bug
fix: DataSet now carries `id` in its scope.

## Module 3 — `validation.py` (auto-derived strict submodels)

Decision: **auto-derived** strict submodels — single source of truth.

`strict_model_for(cls)` takes the generated pydantic model and returns a subclass that
(a) flips each slot in the registry's `required_for_write` to required, and (b) attaches
the registry's `cross_field_rules` as pydantic `model_validator`s. No field definitions
are restated; everything is read from `models.py` + the registry. `models.py` is never
touched. Validation runs on the hot write path (fast, pydantic-only). The LinkML/`cli.py`
validator remains the separate, occasional full-conformance check.

Example cross-field rule: an association's `dataset_id` must refer to a DataSet already
present for that `project_id` (referential safety before write).

## Module 4 — `writers.py` (+ keep `write_utils.py`)

A single dispatch core plus thin typed wrappers:

- `write_models(models, *, settings=None)` — infers the class, looks up the registry,
  validates each model via the strict submodel, converts via `arrow_utils`, attaches
  LinkML metadata, then writes per `write_mode` (scoped overwrite with the
  registry-built predicate, or `append_new_by_id` via the existing helper).
- Wrappers for ergonomics and discoverability: `write_dataset`, `write_dataitem`,
  `write_association`, `write_features`, `write_cluster`, `write_cluster_membership`,
  `write_cell_to_cluster_mapping`, `write_projection_matrix`, etc. Each is a one-liner
  over `write_models`.
- `write_utils.py` stays: `append_new_dataitems` becomes the `append_new_by_id` backend;
  `walk_ancestors` stays for membership/mapping denormalization. Generalize
  `append_new_dataitems` only if needed (e.g. parametrize the partition column), without
  breaking its current callers.

Wide feature matrices (`CellFeatureMatrix`) use `build_cell_feature_matrix_schema` and a
matrix-specific writer path, since they are wide Parquet, not row-modeled Delta tables.

## Module 5 — `readers.py` (+ fold in `io_plans.md`)

Two layers:

- Thin predicate-based readers mirroring the write spec: `read_dataset`, `read_dataitem`,
  `read_features`, scoped by `project_id`/`dataset_id`, returning polars/pandas.
- Flexible cross-dataset / cross-schema reads now that datasets share tables. Flagship
  example: "read all DataItems that have either a ClusterMembership or a
  CellToClusterMapping to a given set of clusters" — a cross-table query joining
  membership/mapping tables on cluster ids and returning the union of matching DataItems,
  regardless of source dataset/modality. Users can still drop to raw
  `polars.read_delta` for ad-hoc queries; the readers are conveniences, not a wall.
- Fold the `io_plans.md` analysis utilities (`populate_region_coverage`,
  `compare_region_coverage`) into this module.

## Notebook migration (no logic, no schema, no models.py changes)

For each ETL notebook: replace hardcoded `OUTPUT_ROOT` with `Settings.load()`, replace
direct `write_deltalake(...)` calls with the typed writers, and delete the per-cell
`mode`/`predicate`/`partition_by` bookkeeping (now owned by the registry). Verification
cells stay. The `visp_*_patchseq` bug is fixed automatically once DataSet writes go
through the registry (scope = project_id + id). Confirm exc + inh DataSet rows coexist
after a re-run as the migration's acceptance test.

## Testing

- Registry-vs-schema drift test (class names + scope/identifier slots exist in models).
- Idempotency: writing the same models twice yields no duplicates and no row loss.
- Shared-partition safety: writing dataset B does not remove dataset A's rows when they
  share a `project_id` (the patchseq regression test).
- Strict-validator tests: missing `required_for_write` slot or failing cross-field rule
  raises before any write touches disk.
- Round-trip: write models → read back via readers → equality on scope columns.
