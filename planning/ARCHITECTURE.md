# IO Layer Architecture — write / read / validation

Status: design agreed 2026-06-01. Implementation to be done by follow-up agents.
This document is the source of truth for the design. The runnable agent prompts
live in `planning/prompts/`. The task breakdown lives in `planning/TODO.md`.

## Hard constraints (read before any work)

The non-negotiable rules live in `prompts/00_shared_context.md` and are not restated here:
never edit `models.py` (generated) or `schemas/*.yaml` (ask YY first); the LinkML schema is
the single source of truth; all IO code lives under `src/connects_common_connectivity/io/`.
This document assumes those and adds the design on top.

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
- `io/io_plans.md` — two pre-existing ideas that are **different concerns** and must land in
  different modules (see below):
  - `populate_region_coverage(pmm, matrix)` — derives `region_coverage` from the dense
    values **before** a matrix is written → a **write-side transform**.
  - `compare_region_coverage(pmms)` — summarizes overlap across already-written matrices →
    **read/analysis**.

## Target `io/` structure (clean is the goal)

The existing IO files are scattered at the package root. The target is a single tidy `io/`
package; the existing modules are **relocated into it and become backends** the new files
call. "Do not rebuild" means *move and wrap, never reimplement*.

```
src/connects_common_connectivity/
  models.py            # generated, UNTOUCHED, stays at root
  cli.py               # CLI entry point, stays at root; full LinkML conformance check
  config.py            # NEW  package-wide Settings (output_root, dry_run, ...) — see below
  io/
    __init__.py        # NEW  curated public API (what users import); __all__ + docstring
    write_spec.py      # NEW  registry — source of truth
    write_validation.py# NEW  auto-derived strict submodels (write-safety validation)
    arrow_utils.py     # MOVED from root (no rename)  (models <-> Arrow conversion)
    writers.py         # NEW  write_models() + typed wrappers + write-side transforms
    write_utils.py     # MOVED from root  (append-by-id backend, walk_ancestors)
    readers.py         # MOVED + folds parquet_loader.py + predicate/cross-dataset reads
```

`config.py` lives at the **package root**, not in `io/`: configuration is package-wide
(`cli.py` and future plotting/analysis code read it too), so the general name belongs in the
general namespace next to `models.py`. Conversely the io validator is named
`write_validation.py`, not `validation.py`: it is specifically write-safety validation
coupled to `write_spec`, and the bare word "validation" is already claimed by `cli.py`'s
LinkML conformance check — two different validations, so neither owns the generic name.

Seed-stage modules are NOT split out prematurely. Write-side enrichment
(`populate_region_coverage`) starts as a section at the top of `writers.py`; read-side
analysis (`compare_region_coverage`) starts in `readers.py`. Promote either to its own
module (`transforms.py` / `analysis.py`) only when a second function arrives — that move is
a pure relocation with no public-API change because users import from `io/__init__.py`.

Where each existing file goes:
- `arrow_utils.py` → `io/arrow_utils.py`. Conversion layer used by `writers.py`. Pure move.
- `write_utils.py` → `io/write_utils.py`. `append_new_dataitems` becomes the
  `append_new_by_id` backend; `walk_ancestors` is used by membership/mapping writers and by
  cross-dataset reads. Pure move.
- `parquet_loader.py` → folded into `io/readers.py` (Parquet→models with report becomes the
  typed-read backend). Pure move/merge.
- `cli.py` stays at the package root as the `ccc` entry point; it owns the occasional full
  LinkML conformance check (separate from `io/write_validation.py`, which is the fast
  write-path check).
- `config.py` is NEW at the package root (package-wide settings; see structure note above).
- `models.py` stays at root, generated, never edited.

Migration safety: while notebooks are being migrated, the moved modules may keep one-line
re-export shims at their old import paths (e.g. `from .io.arrow_utils import *`) so nothing breaks
mid-transition. Shim removal is a tracked task (TODO 5.4), gated by a test that asserts no
old import path is referenced anywhere once migration is complete — otherwise the two import
paths linger and become exactly the clutter this redesign removes.

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

## Module 1 — `config.py` (package root; discovered config file)

Decision: settings live in a **declarative, version-controlled `ccc_config.yaml`** at the
repo root, discovered by walking up from the working directory (the `pyproject.toml` /
`ruff` / `pytest` pattern) and loaded into a validated pydantic `Settings`. No `%run`, no
process-global mutation, no per-notebook setup. No new dependency (pydantic + PyYAML, the
latter already in the tree via LinkML).

```yaml
# ccc_config.yaml  (repo root — the ONE place values live)
output_root: ../scratch/em_patchseq_wnm_v1/
dry_run: false
```

```python
class Settings(BaseModel):
    output_root: Path          # required, no default
    dry_run: bool = False
    # room for more knobs (schema_version_pin, ...) later

@lru_cache
def get_settings() -> Settings:
    path = find_config_file("ccc_config.yaml")        # walk cwd → parents
    if path is None:
        raise RuntimeError("No ccc_config.yaml found — create one at the repo root "
                           "with output_root: ...")
    data = yaml.safe_load(path.read_text())
    if env := os.environ.get("CCC_OUTPUT_ROOT"):      # developer escape hatch, path only
        data["output_root"] = env
    return Settings(**data)
```

Resolution precedence: **explicit `settings=` arg (per call) > `CCC_OUTPUT_ROOT` env >
`ccc_config.yaml` > error.** The file is the source of truth and is validated by pydantic on
load; the env var is a subordinate developer override for `output_root` only (it cannot
express structured knobs like `dry_run`). There is no built-in default path — a missing file
fails loudly rather than writing somewhere arbitrary. `get_settings()` is a pure, cached
function of the filesystem (clearable in tests), not a mutable global.

How the ETL uses it (kills the per-notebook setup): there is no config cell at all. A
notebook just imports and calls `write_dataset(...)` / `read_dataset(...)`; the library
discovers `ccc_config.yaml` on its own. Writers/readers do `settings = settings or
get_settings()`. To repoint local vs CodeOcean, edit the one file (or set `CCC_OUTPUT_ROOT`).
A `table_path(settings, "dataset")` helper resolves per-table subdirectories so nothing
concatenates path strings.

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

## Module 3 — `io/write_validation.py` (auto-derived strict submodels)

Decision: **auto-derived** strict submodels — single source of truth.

`strict_model_for(cls)` takes the generated pydantic model and returns a subclass that
(a) flips each slot in the registry's `required_for_write` to required, and (b) attaches
the registry's `cross_field_rules` as pydantic `model_validator`s. No field definitions
are restated; everything is read from `models.py` + the registry. `models.py` is never
touched. Validation runs on the hot write path (fast, pydantic-only, **no I/O**). The
LinkML/`cli.py` validator remains the separate, occasional full-conformance check.

Hot-path validation is purely structural: required-slot enforcement plus pure cross-field
rules that only inspect the model in hand. **Referential checks that read other tables do
NOT belong on the hot path.** Example: "an association's `dataset_id` must refer to a
DataSet already present for that `project_id`" requires a reader, so it is an opt-in check
(`write_models(..., check_refs=True)`) implemented after readers exist (Phase 4b), not a
strict-submodel validator. This keeps Phase 2 free of any dependency on Phase 4.

## Module 4 — `writers.py` (+ `io/write_utils.py`, `io/arrow_utils.py`)

A single dispatch core plus thin typed wrappers:

- `write_models(models, *, settings=None)` — infers the class, looks up the registry,
  validates each model via the strict submodel, converts via `io/arrow_utils.py`, attaches
  LinkML metadata, then writes per `write_mode` (scoped overwrite with the
  registry-built predicate, or `append_new_by_id` via the backend).
- Typed wrappers for discoverability (`write_dataset`, `write_dataitem`,
  `write_association`, `write_features`, `write_cluster`, `write_cluster_membership`,
  `write_cell_to_cluster_mapping`, `write_projection_matrix`). `write_models` is the one
  real entry point; the wrappers are sugar. To avoid hand-maintaining eight one-liners that
  must stay in lockstep with the registry, **generate them from the registry** (a small
  factory binding the class) and re-export the generated names from `io/__init__.py`. A
  hand-written wrapper is justified only where a class needs a non-uniform signature
  (e.g. `write_projection_matrix` taking the dense matrix for enrichment).
- `io/write_utils.py` (moved from root): `append_new_dataitems` is the `append_new_by_id`
  backend; `walk_ancestors` is used by membership/mapping writers. Generalize
  `append_new_dataitems` only if needed (e.g. parametrize the partition column), without
  breaking callers.
- **Write-side enrichment** lives as a section at the top of `writers.py` (not yet its own
  module): `populate_region_coverage(pmm, matrix)` from `io_plans.md` derives
  `region_coverage` from the dense values. `write_projection_matrix` calls it (or accepts
  an already-enriched matrix). Keep it a pure function (no IO, no mutation of input).
  Split into `io/transforms.py` only when a second transform appears.

Wide feature matrices (`CellFeatureMatrix`) use `build_cell_feature_matrix_schema` (in
`io/arrow_utils.py`) and a matrix-specific writer path, since they are wide Parquet, not
row-modeled Delta tables.

## Module 5 — `readers.py` (folds `parquet_loader.py`)

Two layers:

- Thin predicate-based readers mirroring the write spec: `read_dataset`, `read_dataitem`,
  `read_features`, scoped by `project_id`/`dataset_id`, returning polars/pandas. Typed
  reads (Parquet→models) use the folded-in `load_parquet_to_models`.
- Flexible cross-dataset / cross-schema reads now that datasets share tables. Flagship
  example: "read all DataItems that have either a ClusterMembership or a
  CellToClusterMapping to a given set of clusters" — a cross-table query joining
  membership/mapping tables on cluster ids and returning the union of matching DataItems,
  regardless of source dataset/modality. Users can still drop to raw
  `polars.read_delta` for ad-hoc queries; the readers are conveniences, not a wall.
- **Read-side analysis** lives as a section in `readers.py` to start:
  `compare_region_coverage(pmms)` from `io_plans.md` (shared vs exclusive region coverage
  across matrices). It reads finished data and summarizes — the mirror image of write-side
  enrichment, which augments data on the way in. Split into `io/analysis.py` only when a
  second analysis function appears.

## Notebook migration (no logic, no schema, no models.py changes)

For each ETL notebook: delete hardcoded `OUTPUT_ROOT` (no config cell — the library
discovers `ccc_config.yaml`), replace
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
