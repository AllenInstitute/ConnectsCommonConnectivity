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

## Evolving the Schema

The schema has been split into logical modules for clarity:

```
schemas/
	base_schema.yaml            # prefixes, types, enums, global slots
	core_schema.yaml            # DataSet, DataItem
	clustering_schema.yaml      # AlgorithmRun, ClusterHierarchy, Cluster, ClusterMembership
	brain_region_schema.yaml    # BrainRegion hierarchy
	projection_schema.yaml     # ProjectionMeasurement* + ProjectionMeasurementTypeMetadata
	connectivity_schema.yaml    # aggregator (imports all above) – primary entry point
```

Edit the specific module most closely related to your change. For cross-cutting slot/enums, modify `base_schema.yaml`.
Consumers should continue to reference only the aggregator (`connectivity_schema.yaml`) to obtain the full model.

After editing, re-run any code using `generate_pydantic_models()`. Because results are cached, restart your Python process (or call with a different filename) to see changes.

For production / performance you may eventually wish to use LinkML's code generation to create static
Pydantic models; this repository currently favors agility for early design.

## License

MIT License. See `LICENSE`.
