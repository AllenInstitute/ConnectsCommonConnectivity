# ConnectsCommonConnectivity

Common LinkML schemas for relating multimodal neuroscience connectivity data in
the BRAIN CONNECTS pilot. The initial focus is mouse visual cortex, bringing
together synaptic connectivity, single-cell morphology, projections, cell
features, and cell-type annotations.

**This is a work in progress.** The schemas are the primary focus; the Python
models, IO helpers, and notebooks are supporting tools. Schema coverage and
maturity vary, and definitions and APIs may change as more datasets are integrated.

## Schemas

The [LinkML schemas](schemas/) are the source of truth.
[connectivity_schema.yaml](schemas/connectivity_schema.yaml) imports the modules
into a common model.

| Area | Schema modules |
| --- | --- |
| Datasets, data items, and features | [Core](schemas/core_schema.yaml), [cell features](schemas/cell_features_schema.yaml) |
| Connectivity and projections | [Synapses](schemas/synapse_schema.yaml), [cell-cell connectivity](schemas/cell_cell_schema.yaml), [projections](schemas/projection_schema.yaml) |
| Morphology and coordinates | [Single-cell reconstructions](schemas/single_cell_schema.yaml), [spatial coordinates](schemas/spatial_schema.yaml), [embeddings](schemas/embedding_schema.yaml) |
| Cell types and relationships | [Clustering](schemas/clustering_schema.yaml), [mappings](schemas/mappings_schema.yaml) |
| Brain regions and gene expression | [Brain regions](schemas/brain_region_schema.yaml), [cell-gene data](schemas/cell_gene_schema.yaml) |

Some modules are less developed than others. In particular, the **cell-gene and
brain-region schemas are under active development**, with coverage being expanded
as additional use cases and datasets are incorporated.

## Prototype data

We have ingested previously published datasets into a common representation to
exercise the schemas and support cross-dataset exploration.

- **Data summary:** [ccm_release_202610_v7 overview](https://alleninstitute.github.io/ccm-data-overview/pages/ccm_release_202610_v7.html)
- **Data location:** [s3://connects-common-connectivity/ccm_release_202610_v7/](https://connects-common-connectivity.s3.us-west-2.amazonaws.com/ccm_release_202610_v7/)

### Accessing the data

The bucket is public, so no AWS account or credentials are needed. Install the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) and add `--no-sign-request` to every command.

**Browse**

```bash
# List the top level directories
aws s3 ls s3://connects-common-connectivity/ccm_release_202610_v7/ --no-sign-request

# List everything with sizes and a total
aws s3 ls s3://connects-common-connectivity/ccm_release_202610_v7/ --no-sign-request --recursive --human-readable --summarize
```

**Download**

```bash
# Whole release (preserves directory layout)
aws s3 sync s3://connects-common-connectivity/ccm_release_202610_v7/ ./ccm_release_202610_v7/ --no-sign-request
```

When copying a release locally, preserve its directory layout: relative payload
references are resolved against the release root, not the working directory.

## Working with the data

Use **Polars** and the Python **`deltalake`** package (including `DeltaTable`) to
read, transform, and write the Delta and Parquet tables directly. The CCC IO
package is optional for working with these tables.

[Read and write helpers](src/connects_common_connectivity/io/) are available for
selected models, not every schema. The
[write-spec registry](src/connects_common_connectivity/io/write_spec.py) defines
supported model writers, validation classes, and write policies. The
[path specification](src/connects_common_connectivity/io/path_spec.py) defines
the shared table layout. Direct dataframe writes do not automatically apply the
helpers' model and write-time validation.

## Examples

- [ETL notebooks](examples/ETL/) show how source datasets are mapped to the
  schemas and written into the common representation.
- [Reading and plotting notebooks](examples/read_plot/) illustrate how to
  explore ingested data. More examples will be added soon and over time.

The examples are also a work in progress. Prepare the source inputs required by
each ETL and follow its prerequisites; shared taxonomies must be ingested before
dependent datasets. Reading existing data does not require rerunning the ETLs,
but individual examples may expect a local copy of the relevant tables.

### ETL setup

With [uv](https://docs.astral.sh/uv/getting-started/installation/) installed, run
this from the repository root to install the project:

```bash
uv sync
```

Some notebooks need additional dependencies not yet declared by the project,
such as `umap-learn` and `scikit-learn` for the Minnie UMAP example. Check the
notebook imports and input requirements before running it.

Create a local `ccc_config.yaml` at the repository root:

```yaml
output_root: scratch/my_ccc_etl/
dry_run: false
```

- `output_root` is the **local filesystem destination** for generated tables.
  Relative paths resolve against the configuration file's directory. Do not use
  the S3 release URL as this setting.
- Configuration discovery walks upward from the working directory, so notebooks
  run from within the repository can find the root configuration.
- `dry_run: true` controls writes through the CCC helpers. It is **not a
  notebook-wide no-write guarantee**: direct Delta writes bypass this setting.

To override the destination for a session:

```bash
export CCC_OUTPUT_ROOT=/absolute/path/to/my_ccc_etl
```

The configuration file is still required when using this override. Settings are
cached within a Python process; restart the notebook kernel after changing the
file or environment, or call `get_settings.cache_clear()` after importing
`get_settings` from `connects_common_connectivity.config`.

## Developing the schemas

Edit the appropriate module under [schemas/](schemas/), then regenerate the
checked-in [Pydantic models](src/connects_common_connectivity/models.py) from the
repository root:

```bash
bash scripts/generate_models.sh
```

Do not edit the generated models manually. To run the tests:

```bash
uv run --extra dev pytest -q
```

## License

This repository is licensed under the [MIT License](LICENSE). Refer to each
source dataset's terms and citation requirements when using its data.
