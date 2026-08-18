import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# CellFeatureDefinition
# ---------------------------------------------------------------------------


def test_cell_feature_definition_project_id_required(models):
    """Cell feature definitions must require a project identifier."""
    CellFeatureDefinition = models["CellFeatureDefinition"]
    with pytest.raises(ValidationError, match=r"(?s)project_id.*Field required"):
        CellFeatureDefinition(id="nucleus_volume_um", description="Nucleus volume")


def test_cell_feature_definition_valid(models):
    """A complete cell feature definition must preserve its values."""
    CellFeatureDefinition = models["CellFeatureDefinition"]
    cfd = CellFeatureDefinition(
        id="nucleus_volume_um",
        description="Nucleus volume in cubic microns",
        unit="MICRONS_CUBED",
        data_type="<f4",
        range_min=0.0,
        project_id="minnie65",
    )
    assert cfd.id == "nucleus_volume_um"
    assert cfd.project_id == "minnie65"
    assert cfd.range_max is None  # optional


def test_cell_feature_definition_range_min_max_optional(models):
    """Cell feature definition range bounds must remain optional."""
    CellFeatureDefinition = models["CellFeatureDefinition"]
    # Both range fields absent — should not raise
    cfd = CellFeatureDefinition(id="some_feature", project_id="minnie65")
    assert cfd.range_min is None
    assert cfd.range_max is None


def test_cell_feature_definition_data_type_pattern_valid(models):
    """Cell feature definitions must accept valid NumPy data-type strings."""
    CellFeatureDefinition = models["CellFeatureDefinition"]
    for dt in ["<f4", "<f8", "<i4", "<i2", "|u1", ">f8", "=i4"]:
        cfd = CellFeatureDefinition(id="feat", data_type=dt, project_id="p1")
        assert cfd.data_type == dt


def test_cell_feature_definition_data_type_pattern_invalid(models):
    """Cell feature definitions must reject malformed data-type strings."""
    CellFeatureDefinition = models["CellFeatureDefinition"]
    for bad in ["float32", "f4", "<float4", "f", "<f"]:
        with pytest.raises(ValidationError, match=r"(?s)data_type"):
            CellFeatureDefinition(id="feat", data_type=bad, project_id="p1")


def test_cell_feature_definition_feature_set_id_optional(models):
    """Cell feature definitions must allow an omitted feature-set identifier."""
    CellFeatureDefinition = models["CellFeatureDefinition"]
    # feature_set_id is optional — valid to omit
    cfd = CellFeatureDefinition(id="some_feat", project_id="minnie65")
    assert cfd.feature_set_id is None


def test_cell_feature_definition_feature_set_id_set(models):
    """Cell feature definitions must retain a supplied feature-set identifier."""
    CellFeatureDefinition = models["CellFeatureDefinition"]
    cfd = CellFeatureDefinition(
        id="x_medial-lateral",
        project_id="minnie65",
        feature_set_id="minnie65_std_transform_coordinates",
    )
    assert cfd.feature_set_id == "minnie65_std_transform_coordinates"


# ---------------------------------------------------------------------------
# CellFeatureSet
# ---------------------------------------------------------------------------


def test_cell_feature_set_project_id_required(models):
    """Cell feature sets must require a project identifier."""
    CellFeatureSet = models["CellFeatureSet"]
    with pytest.raises(ValidationError, match=r"(?s)project_id.*Field required"):
        CellFeatureSet(id="csm_cluster_features")


def test_cell_feature_set_valid(models):
    """A complete cell feature set must preserve its values."""
    CellFeatureSet = models["CellFeatureSet"]
    cfs = CellFeatureSet(
        id="csm_cluster_features",
        description="CSM dendrite ultrastructure features",
        feature_definition_ids=["nucleus_volume_um", "soma_volume_um"],
        extraction_method="custom",
        project_id="minnie65",
    )
    assert cfs.id == "csm_cluster_features"
    assert cfs.project_id == "minnie65"
    assert len(cfs.feature_definition_ids) == 2


def test_cell_feature_set_optional_fields(models):
    """A minimal cell feature set must allow optional metadata to be omitted."""
    CellFeatureSet = models["CellFeatureSet"]
    # description, feature_definition_ids, extraction_method are all optional
    cfs = CellFeatureSet(id="minimal_set", project_id="minnie65")
    assert cfs.description is None
    assert cfs.extraction_method is None
