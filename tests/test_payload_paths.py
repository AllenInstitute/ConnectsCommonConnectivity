import pytest
from pydantic import ValidationError

from connects_common_connectivity import models as checked_in_models


@pytest.fixture(params=["schema", "checked_in"])
def payload_models(request, models):
    """Exercise both schema-generated and checked-in payload reference models."""
    return models if request.param == "schema" else vars(checked_in_models)


@pytest.mark.parametrize("class_name,field,required", [
    ("CellFeatureMatrix", "parquet_path", {"project_id": "p", "feature_set_id": "f"}),
    ("SynapseFeatureMatrix", "parquet_path", {"project_id": "p", "synapse_table_id": "s"}),
    ("ParquetDataset", "path", {}),
])
@pytest.mark.parametrize("pointer", [
    "cellfeatures/features",
    "./synapsefeatures/features/",
    "projectionmeasurementmatrix/matrix",
    "cellfeatures/feature set/",
    "s3://bucket/features",
    "gs://bucket/features",
    "https://example.org/features",
    "http://example.org/features",
    "file:///tmp/features",
])
def test_payload_paths_accept_relative_paths_and_uris(
    payload_models, class_name, field, required, pointer,
):
    """Relative payload paths and supported URIs retain their exact stored value."""
    model = payload_models[class_name](id="matrix", **required, **{field: pointer})
    assert getattr(model, field) == pointer


@pytest.mark.parametrize("class_name,field,required", [
    ("CellFeatureMatrix", "parquet_path", {"project_id": "p", "feature_set_id": "f"}),
    ("SynapseFeatureMatrix", "parquet_path", {"project_id": "p", "synapse_table_id": "s"}),
    ("ParquetDataset", "path", {}),
])
@pytest.mark.parametrize("pointer", ["", " ", "/tmp/features", "ftp://host/features",
                                    "C:\\features", "features\\matrix"])
def test_payload_paths_reject_invalid_locations(
    payload_models, class_name, field, required, pointer,
):
    """Empty paths, bare absolute paths, unsupported URIs, and backslashes are rejected."""
    with pytest.raises(ValidationError, match=field):
        payload_models[class_name](id="matrix", **required, **{field: pointer})


def test_feature_pointer_requiredness_is_unchanged(payload_models):
    """Cell pointers remain optional while synapse pointers remain required."""
    cell = payload_models["CellFeatureMatrix"](id="m", project_id="p", feature_set_id="f")
    assert cell.parquet_path is None
    with pytest.raises(ValidationError, match="parquet_path"):
        payload_models["SynapseFeatureMatrix"](id="m", project_id="p", synapse_table_id="s")


@pytest.mark.parametrize("pointer", [
    "projectionmeasurementmatrix/matrix", "file:///tmp/matrix", "s3://bucket/matrix",
])
def test_projection_pointer_retains_relative_paths_and_uris(payload_models, pointer):
    """Projection references retain relative payload paths and existing absolute URIs."""
    matrix = payload_models["ProjectionMeasurementMatrix"](
        id="matrix", project_id="p", measurement_type="NUMBER_OF_TIPS",
        modality="MORPHOLOGY", laterality="IPSILATERAL", values=pointer,
    )
    assert matrix.values == pointer
