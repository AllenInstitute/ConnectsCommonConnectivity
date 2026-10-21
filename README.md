# ConnectsCommonConnectivity

Common connectivity data models (LinkML) + dynamic Pydantic models for BRAIN CONNECTS pilot work.

## Goals
This repository is designed to help create a common connectivity "matrix" for cross comparison of data about connections and cells in the brain, focused on methods which have single axon resolution in the mouse brain.  Because connectivity data and the data we want to relate to is highly multi-modal, there is not a singular kind of matrix which can represent it.  Instead, there are a set of inter-connected concepts which share some common data shapes.  For example, single cell synaptic connectivity data measured by EM will measure connections between individual cells and give detailed morphology information (at least locally).  Single cell morphology reconstructions will contain long range projection information, but also skeleton based morphological information (both local and long range).  Patch-seq data can have local morphology data, but also gene expression and electrophysiology features.  Bar-seq can have projection distributions along with gene expression. So on and so forth across the methods.  Setting up a framework where different measurements of projections, or single cell morphology can all have the same data shape and be accessed in a single location through a common api will allow for integrative analysis that can transcend the impact of each individual dataset. 

The pilot of the Common Connectivity Pilot is focused on developing a framework that could be extended to the whole mouse brain, while importing dataset from mouse visual cortex, where there are examples of data from many of these modalities to demonstrate the power of integrative analysis. 

## Features

- Modular LinkML schema (aggregated by `schemas/connectivity_schema.yaml`)
- On-demand generation of Pydantic models (no pre-generated code needed for quick iteration)
- Example script in `examples/generate_and_use.py`
- Projection measurement modeling (per-cell vectors & aggregated matrices) example in `examples/projection_measurements_example.yaml`
- Packaged with `pyproject.toml` and intended to be managed via `uv`
- BrainRegion ETL example from Parquet (S3/local) via `examples/etl_brain_regions.py` or CLI `ccc etl-brain-regions`
- Generic Parquet→LinkML loader utility (`parquet_loader.py`) for any class in the schema
- Curated IO layer (`connects_common_connectivity.io`) for writing generated pydantic models to a shared Delta lake — `write_models(...)` / `write_projection_matrix(...)` dispatched via a `WriteSpec` registry, with output location resolved from `ccc_config.yaml`

## Getting Started (with uv)

Install uv if you haven't:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a virtual environment and install this project in editable mode:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .[dev]
```

Run tests:

```bash
uv run pytest -q
```

Run the example:

```bash
uv run python examples/generate_and_use.py
```

## Using the Dynamic Models

```python
from connects_common_connectivity import generate_pydantic_models
models = generate_pydantic_models()
BrainRegion = models["BrainRegion"]
br = BrainRegion(id="BR1", name="Region 1", species="mouse")
```

## Generic Parquet Loading

For a more reusable ingestion path, use the generic loader in `connects_common_connectivity.parquet_loader`.
It auto-maps columns to slots using schema-declared aliases and performs a two-pass resolution of object
reference slots (those whose range is another class) to establish links.

```python
from connects_common_connectivity.parquet_loader import load_parquet_to_models

instances, report = load_parquet_to_models(
		schema_name="connectivity_schema.yaml",
		class_name="BrainRegion",
		parquet_path="s3://allen-atlas-assets/terminologies/allen-adult-mouse-terminology/2020/terminology.parquet",
)
print(report["counts"], report["warnings"][:3])
```

CLI form (module execution):

```bash
uv run python -m connects_common_connectivity.parquet_loader connectivity_schema.yaml BrainRegion \
	s3://allen-atlas-assets/terminologies/allen-adult-mouse-terminology/2020/terminology.parquet
```

Report dictionary keys:

| Key | Meaning |
|-----|---------|
| mapping | slot→column chosen |
| errors | instantiation or linking problems (row, id, message) |
| warnings | unresolved references summary lines |
| counts | summary numbers (rows, instances, errors, warnings) |
| unresolved | per-slot count of unresolved object references |

To customize behavior (e.g., tolerate missing required fields) set `strict_required=False`.
You can also cap error processing via `max_errors=N`.

Extensibility ideas:
* Add per-slot coercion hooks (e.g., transform color codes, parse nested JSON)
* Support chunked reading for very large Parquet datasets
* Emit provenance metadata (ingest timestamp, source URI) alongside objects
* Integrate with the CLI (`ccc`) for generic class ingestion

This generic loader complements the specialized example in `examples/etl_brain_regions.py` which
adds dual hierarchy denormalization; prefer the generic approach for new classes.
## ETL BrainRegion from Parquet

You can ingest a Parquet file containing region ontology rows and validate them against the dynamic `BrainRegion` model.

Example (S3):

```bash
uv run python examples/etl_brain_regions.py \
	s3://allen-atlas-assets/terminologies/allen-adult-mouse-terminology/2020/terminology.parquet \
	--out brain_regions.yaml
```

Or via CLI subcommand:

```bash
uv run ccc etl-brain-regions \
	s3://allen-atlas-assets/terminologies/allen-adult-mouse-terminology/2020/terminology.parquet \
	--out brain_regions.yaml
```

Column auto-mapping heuristics look for common names like `structure_id`, `structure_name`, `parent_id`, `rgb_hex`. Override with flags: `--id-col`, `--name-col`, `--acronym-col`, `--color-col`, `--parent-col`.

Install `pyarrow` (already in dependencies) and ensure your environment permits anonymous S3 access (e.g., set `AWS_NO_SIGN_REQUEST=YES`).


## Evolving the Schema

The schema has been split into logical modules for clarity:

```
schemas/
	base_schema.yaml            # prefixes, types, enums, global slots
	core_schema.yaml            # DataSet, DataItem
    spatial_schema.yaml         # ReferenceSpace, Default2DView, AnatomicalOrientation, SpatialLocation
	clustering_schema.yaml      # AlgorithmRun, ClusterHierarchy, Cluster, ClusterMembership
	brain_region_schema.yaml    # BrainRegion hierarchy
	projection_schema.yaml     # ProjectionMeasurement* + ProjectionMeasurementTypeMetadata
	connectivity_schema.yaml    # aggregator (imports all above) – primary entry point
```

Edit the specific module most closely related to your change. For cross-cutting slot/enums, modify `base_schema.yaml`.
Consumers should continue to reference only the aggregator (`connectivity_schema.yaml`) to obtain the full model.

After editing, re-run any code using `generate_pydantic_models()`. Because results are cached, restart your Python process (or call with a different filename) to see changes.

Regenerate the checked-in Pydantic models from the repository root with
`bash scripts/generate_models.sh`. Never edit the generated model file manually.

## ETL Notebooks

A set of ETL Jupyter notebooks in `code/` registers real datasets into a shared Delta Lake store via the `connects_common_connectivity.io` layer (`write_models`, `write_projection_matrix`). The output location is resolved from `ccc_config.yaml` at the repo root (or the `CCC_OUTPUT_ROOT` environment variable), so notebooks do not hard-code a destination path. These serve as concrete working examples for every schema class.

- **Excitatory morphology:** run [the excitatory Patch-seq notebook](code/EXC_WNM_patchseq/etl_patchseq_exc_01.ipynb), then [the WNM notebook](code/EXC_WNM_patchseq/etl_wnm_exc_01.ipynb), after the shared T-type and MET-type taxonomy ETLs. Both datasets use project `visp_exc_morphology`. Patch-seq owns the canonical 50-feature `exc_visp_morph_features` set; WNM reuses its definitions for 44 available features without replacing the shared vocabulary. Their payloads live in separate Delta tables at `cellfeatures/<dataset_id>/<feature_set_id>`, with dataset-specific matrix IDs.
  The shared definitions promote `apical_dendrite_max_branch_order`, `apical_dendrite_num_branches`, `basal_dendrite_calculate_number_of_stems`, `basal_dendrite_max_branch_order`, and `basal_dendrite_num_branches` to float64 because WNM supplies fractional values. Other CSV datatypes are retained; the source CSV is not modified.
- Active ETLs store `CellFeatureMatrix.parquet_path`, `SynapseFeatureMatrix.parquet_path`, and `ProjectionMeasurementMatrix.values` relative to the configured output root, without a `file://` prefix. Resolve these pointers against the release root, not the working directory; moving the complete release folder preserves them. Existing remote and absolute file URIs remain supported, but absolute file URIs must be updated if payloads are relocated.
- `DatasetReader` follows each `CellFeatureMatrix.parquet_path` to discover and read its payload, resolving relative pointers against `dataset_root`, so matrices sharing a feature set need not share a table or contain every defined feature. Discovery includes the matrix's `parquet_path`. Records without a pointer retain the legacy `cellfeatures/<feature_set_id>` lookup; a broken explicit pointer is reported rather than silently replaced with legacy data.

### Run all active ETLs

[scripts/run_etl.py](scripts/run_etl.py) runs the ten active `etl_*.ipynb`
notebooks in dependency order: shared Tasic, MET-type and ABC taxonomies, BARseq, inhibitory Patch-seq,
excitatory Patch-seq, WNM, Minnie base/features, and V1DD.
It excludes `code/old/`, backup notebooks, CAVE downloads, and BARseq raw-data
extraction. Prepare the local input files before running.

```bash
# Preview the order without executing notebooks or writing files.
uv run python scripts/run_etl.py --dry-run

# Execute all ten notebooks, stopping on the first error.
uv run python scripts/run_etl.py

# Restart at V1DD without repeating previously completed notebooks.
uv run python scripts/run_etl.py --from etl_v1dd1196_01.ipynb
```

`--from` accepts an active notebook's filename or repository-relative path and
runs it and all later notebooks. It restarts that notebook from its first cell;
earlier dependencies must already exist in the configured CCC output.
Combine it with `--dry-run` to preview the selected notebooks.

Each notebook gets a fresh Python kernel using the runner's interpreter and the
repository root as its working directory. Cells have no execution timeout, and
errors stop the run even if a cell is tagged to allow errors. The notebooks retain
their normal CCC configuration (`ccc_config.yaml` or `CCC_OUTPUT_ROOT`); completed
writes are not rolled back if a later notebook fails.

After each notebook, its kernel is shut down and execution output is saved.
The runner then collects its Python garbage before starting the next notebook.
Cleanup also runs on failure. This releases between-notebook resources, but does
not reduce peak memory inside an individual notebook.

Source notebooks are not modified. Executed copies, including partial output
from a failed notebook, are saved under `scratch/etl_runs/<UTC timestamp>/code/`.
Use `--output-dir /path/to/new-directory` to choose a different execution-record
directory, not a different CCC data destination. The directory must not already
exist. Open the saved copies to inspect cell output, errors, and plots.

V1DD connectivity reads only the needed Feather columns one record batch at a
time, retaining selected synapses with numeric IDs. The
`iter_cell_cell_connectivity` helper aggregates counts and sizes together and
expands the compact pair aggregates into bounded measurement batches. These
batches feed one atomic Delta overwrite, avoiding a full in-memory copy of the
long-form connectivity table.

- **`code/etl_examples_readme.ipynb`** — markdown-only overview of all registered datasets and feature sets: what each dataset contains, why cell counts differ between sources, and how shared feature sets work across projects. Start here if you're new to the data.

- **`etl_example_prompt.md`** (repo root) — prompt guide for AI assistants creating new ETL notebooks. Covers which files to read first, hard rules (never edit `models.py`, schemas are truth, no id casting), the canonical notebook structure, the idempotent write pattern for each table type, and a common-mistakes reference table.

## License

MIT License. See `LICENSE`.


```mermaid
erDiagram
AlgorithmRun {
    string id  
    string algorithm_name  
    string algorithm_version  
    JsonObject parameters_json  
    datetime run_timestamp  
    string score_description  
    string distance_description  
}
BarcodingExperimentMetadata {
    string experiment_type  
    string sequencing_platform  
    string data_processing_pipeline  
    string normalization_method  
}
BrainRegion {
    string id  
    string name  
    string acronym  
    HexColor color_hex_triplet  
    string term_set_name  
    integer annotation_value  
    integerList descendant_annotation_values  
}
CellFeatureDefinition {
    string id  
    string feature_name  
    string description  
    string unit  
    string data_type  
    float range_min  
    float range_max  
}
CellFeatureMatrix {
    string id  
    string description  
    Unit unit  
}
CellFeatureMeasurement {
    string id  
    string dtype  
    float value_float  
    integer value_int  
    boolean value_bool  
    string value_string  
    string value_bytes  
    datetime value_datetime  
}
CellFeatureSet {
    string id  
    string name  
    string description  
    string extraction_method  
}
CellGeneData {
    string id  
}
CellMetadata {
    string cell_id  
    float quality_score  
    integer total_counts  
    integer n_genes_detected  
}
CellToCellMapping {
    string id  
    float score  
    Probability probability  
    string notes  
}
CellToClusterMapping {
    string id  
    float score  
    Probability probability  
    string notes  
}
Cluster {
    string id  
    string name  
    integer level  
    integer size  
    float score  
    float distance_to_parent  
    string members  
}
ClusterHierarchy {
    string id  
    string name  
}
ClusterMembership {
    float membership_score  
    Probability membership_probability  
    float distance  
}
ClusterToClusterMapping {
    string id  
    float score  
    Probability probability  
    string notes  
}
DataItem {
    string id  
    string name  
    string neuroglancer_link  
}
DataSet {
    string id  
    string name  
    string publication  
    Modality modality  
}
GeneMetadata {
    string gene_id  
    string gene_symbol  
}
MappingSet {
    string id  
    string name  
    string description  
    string method_name  
    string method_version  
    string author  
    datetime created_at  
    JsonObject parameters_json  
}
ParquetDataset {
    string id  
    string path  
}
ProjectionMeasurementMatrix {
    string id  
    string description  
    ProjectionMeasurementType measurement_type  
    Modality modality  
    Unit unit  
}
SingleCellReconstruction {
    string id  
}
SpatialLocation {
    string project_id
    string dataitem_id
    LocationType location_type
    float x  
    float y  
    float z  
    string reference_space  
}
ReferenceSpace {
    string id
    string project_id
    string name
    string description
}
Default2DView {
    SignedAxis left_to_right
    SignedAxis bottom_to_top
}
AnatomicalOrientation {
    SignedAxis anterior_to_posterior
    SignedAxis rostral_to_caudal
    SignedAxis superior_to_inferior
    SignedAxis dorsal_to_ventral
    SignedAxis subject_left_to_right
    SignedAxis medial_to_lateral
    SignedAxis pia_to_white_matter
}
ZarrArray {
    string id  
    string path  
}
ZarrDataset {
    string id  
    string path  
}

AlgorithmRun ||--|o DataSet : "input_dataset"
AlgorithmRun ||--}o ClusterHierarchy : "produced_hierarchies"
BrainRegion ||--|o BrainRegion : "parent_identifier"
BrainRegion ||--}o BrainRegion : "child_identifiers"
BrainRegion ||--}o BrainRegion : "descendants"
CellFeatureMatrix ||--|o ParquetDataset : "backing_store"
CellFeatureMatrix ||--|o ZarrDataset : "feature_values"
CellFeatureMatrix ||--|| CellFeatureSet : "feature_set"
CellFeatureMatrix ||--}o CellFeatureDefinition : "feature_index"
CellFeatureMatrix ||--}o DataItem : "cell_index"
CellFeatureMeasurement ||--|o CellFeatureSet : "feature_set"
CellFeatureMeasurement ||--|| CellFeatureDefinition : "feature"
CellFeatureMeasurement ||--|| DataItem : "data_item"
CellFeatureSet ||--}o CellFeatureDefinition : "feature_definitions"
CellGeneData ||--|o BarcodingExperimentMetadata : "experiment_metadata"
CellGeneData ||--|o ZarrArray : "gene_metadata"
CellGeneData ||--|| DataItem : "data_item"
CellGeneData ||--|| ZarrArray : "cell_gene_matrix"
CellGeneData ||--}o DataItem : "cell_index"
SpatialLocation }o--|| DataItem : "dataitem_id"
SpatialLocation }o--|| ReferenceSpace : "reference_space"
ReferenceSpace ||--o| Default2DView : "default_2d_view"
ReferenceSpace ||--o| AnatomicalOrientation : "anatomical_orientation"
CellToCellMapping ||--|| DataItem : "source_cell"
CellToCellMapping ||--|| DataItem : "target_cell"
CellToCellMapping ||--|| MappingSet : "mapping_set"
CellToClusterMapping ||--|| Cluster : "target_cluster"
CellToClusterMapping ||--|| DataItem : "source_cell"
CellToClusterMapping ||--|| MappingSet : "mapping_set"
Cluster ||--|o Cluster : "parent"
Cluster ||--}o Cluster : "children"
ClusterHierarchy ||--|o AlgorithmRun : "run"
ClusterHierarchy ||--|o Cluster : "root"
ClusterHierarchy ||--}o Cluster : "clusters"
ClusterMembership ||--|o Cluster : "cluster"
ClusterMembership ||--|o DataItem : "item"
ClusterToClusterMapping ||--|| Cluster : "source_cluster"
ClusterToClusterMapping ||--|| Cluster : "target_cluster"
ClusterToClusterMapping ||--|| MappingSet : "mapping_set"
DataItem ||--|| DataSet : "dataset"
MappingSet ||--|| DataSet : "source_dataset"
MappingSet ||--|| DataSet : "target_dataset"
ProjectionMeasurementMatrix ||--|o ZarrArray : "values"
ProjectionMeasurementMatrix ||--}o BrainRegion : "region_index"
ProjectionMeasurementMatrix ||--}o DataItem : "data_item_index"
SingleCellReconstruction ||--|o ZarrArray : "ccf_registered_file"
SingleCellReconstruction ||--|| DataItem : "data_item"

```