from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from connects_common_connectivity.models import (
    EmbeddingLocation,
    EmbeddingMethod,
    EmbeddingSpace,
)


@pytest.mark.parametrize("method", list(EmbeddingMethod))
def test_embedding_space_methods_and_optional_metadata(method):
    """Each supported embedding method is accepted without optional metadata."""
    space = EmbeddingSpace(project_id="p", id="embedding", embedding_method=method)
    assert space.embedding_method == method
    for field in (
        "name", "description", "parameters_json", "input_feature_set_id",
        "input_features_description", "creation_date",
    ):
        assert getattr(space, field) is None


def test_embedding_method_vocabulary():
    """The method vocabulary is UMAP, TSNE, PCA, MDS, and OTHER; unknown values fail."""
    assert {method.value for method in EmbeddingMethod} == {
        "UMAP", "TSNE", "PCA", "MDS", "OTHER",
    }
    with pytest.raises(ValidationError, match="embedding_method"):
        EmbeddingSpace(project_id="p", id="embedding", embedding_method="UNKNOWN")


@pytest.mark.parametrize("model, values, required", [
    (
        EmbeddingSpace,
        dict(project_id="p", id="embedding", embedding_method="UMAP"),
        ("project_id", "id", "embedding_method"),
    ),
    (
        EmbeddingLocation,
        dict(project_id="p", dataitem_id="injection", embedding_space="embedding", x=1, y=2),
        ("project_id", "dataitem_id", "embedding_space", "x", "y"),
    ),
])
def test_embedding_required_fields_reject_missing_and_null(model, values, required):
    """Required embedding fields reject both omission and explicit null values."""
    for field in required:
        incomplete = {name: value for name, value in values.items() if name != field}
        with pytest.raises(ValidationError, match=field):
            model(**incomplete)
        with pytest.raises(ValidationError, match=field):
            model(**{**values, field: None})


@pytest.mark.parametrize("feature_set_id", [None, "projection_features"])
def test_embedding_metadata_json_round_trip(feature_set_id):
    """Metadata and dates survive JSON round trips with or without a feature-set ID."""
    space = EmbeddingSpace(
        project_id="p", id="embedding", embedding_method="UMAP",
        name="Projection embedding", description="Embedding of injection experiments.",
        parameters_json='{"random_state": 42}',
        input_feature_set_id=feature_set_id,
        input_features_description="Normalized projection measurements.",
        creation_date="2026-10-07",
    )
    assert space.input_feature_set_id == feature_set_id
    assert space.creation_date == date(2026, 10, 7)
    assert EmbeddingSpace.model_validate_json(space.model_dump_json()) == space


def test_embedding_location_is_two_dimensional():
    """Locations expose only x/y, preserve numeric coordinates, and reject nonnumeric input."""
    location = EmbeddingLocation(
        project_id="p", dataitem_id="injection", embedding_space="embedding", x=-1.5, y=2.5,
    )
    assert set(EmbeddingLocation.model_fields) == {
        "project_id", "dataitem_id", "embedding_space", "x", "y",
    }
    assert location.x == -1.5 and location.y == 2.5
    assert EmbeddingLocation.model_validate_json(location.model_dump_json()) == location
    for field in ("x", "y"):
        with pytest.raises(ValidationError, match=field):
            EmbeddingLocation(**{**location.model_dump(), field: "not-a-number"})


def test_embedding_schema_identities_and_feature_set_reference():
    """The imported embedding schema declares scoped identities and a CellFeatureSet reference."""
    schemas = Path(__file__).resolve().parents[1] / "schemas"
    schema = yaml.safe_load((schemas / "embedding_schema.yaml").read_text())
    for class_name, key, fields in (
        ("EmbeddingSpace", "embedding_space_identity", ["project_id", "id"]),
        (
            "EmbeddingLocation", "embedding_location_identity",
            ["project_id", "dataitem_id", "embedding_space"],
        ),
    ):
        assert schema["classes"][class_name]["unique_keys"][key]["unique_key_slots"] == fields
    assert schema["slots"]["input_feature_set_id"]["range"] == "CellFeatureSet"
    aggregator = yaml.safe_load((schemas / "connectivity_schema.yaml").read_text())
    assert "embedding_schema" in aggregator["imports"]