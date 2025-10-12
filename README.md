# ConnectsCommonConnectivity

Common connectivity data models (LinkML) + dynamic Pydantic models for BRAIN CONNECTS pilot work.

## Features

- LinkML schema in `schemas/connectivity_schema.yaml`
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
br = BrainRegion(id="BR1", label="Region 1", species="mouse")
```

## Evolving the Schema

Edit `schemas/connectivity_schema.yaml`, then re-run any code using `generate_pydantic_models()`.
Because results are cached, restart your Python process (or call with a different filename) to see changes.

For production / performance you may eventually wish to use LinkML's code generation to create static
Pydantic models; this repository currently favors agility for early design.

## License

MIT License. See `LICENSE`.
