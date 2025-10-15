# ConnectsCommonConnectivity

Common connectivity data models (LinkML) + dynamic Pydantic models for BRAIN CONNECTS pilot work.

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
