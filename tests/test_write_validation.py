"""Tests for write-time validation (auto-derived strict submodels)."""

from __future__ import annotations

import pytest

from connects_common_connectivity.config import Settings
from connects_common_connectivity.io.write_spec import (
    DATA_AXIS_BY_SIGNED_AXIS,
    REGISTRY,
    ClusterWrite,
)
from connects_common_connectivity.io.write_validation import validate_for_write
from connects_common_connectivity.io.writers import write_models
from connects_common_connectivity.models import (
    CellFeatureDefinition,
    Cluster,
    ClusterMembership,
    DataSet,
    Default2DView,
    ReferenceSpace,
    SignedAxis,
)


@pytest.mark.parametrize("horizontal", list(SignedAxis))
@pytest.mark.parametrize("vertical", list(SignedAxis))
def test_reference_space_axes_checked_before_io(horizontal, vertical, tmp_path):
    """Repeated underlying axes fail before IO regardless of sign; distinct axes remain valid."""
    space = ReferenceSpace(id="frame", default_2d_view=Default2DView(
        left_to_right=horizontal, bottom_to_top=vertical,
    ))
    if DATA_AXIS_BY_SIGNED_AXIS[horizontal] == DATA_AXIS_BY_SIGNED_AXIS[vertical]:
        root = tmp_path / "must-not-exist"
        with pytest.raises(ValueError, match="different data axes") as ei:
            write_models(space, output_root=root)
        assert "id=frame" in str(ei.value)
        assert not root.exists()
    else:
        assert validate_for_write([space], REGISTRY["ReferenceSpace"])[0] is space


def test_every_signed_axis_maps_to_a_data_axis():
    """A new enum member must be mapped, not silently parsed from its name."""
    assert set(DATA_AXIS_BY_SIGNED_AXIS) == set(SignedAxis)


def test_reference_space_revalidates_constructed_view(tmp_path):
    """An incomplete view created without Pydantic validation must still fail before writing."""
    space = ReferenceSpace.model_construct(
        id="invalid", default_2d_view={"left_to_right": "PLUS_X"}
    )
    root = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="bottom_to_top"):
        write_models(space, output_root=root)
    assert not root.exists()


def test_constructed_row_revalidated_without_a_write_class(tmp_path):
    """Schema re-validation must not depend on a spec declaring a write class."""
    spec = REGISTRY["DataSet"]
    assert spec.write_cls is None
    bad = DataSet.model_construct(id="d1", name="d")  # project_id missing
    root = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="project_id"):
        write_models(bad, output_root=root)
    assert not root.exists()


# ---------------------------------------------------------------------------
# write classes
# ---------------------------------------------------------------------------


def test_write_class_tightens_field_without_mutating_parent():
    """A write class must narrow its slot without touching the generated model."""
    assert issubclass(ClusterWrite, Cluster)
    assert not Cluster.model_fields["hierarchy_id"].is_required()
    assert ClusterWrite.model_fields["hierarchy_id"].is_required()


def test_spec_without_write_class_validates_against_generated_model():
    """A spec with no extra constraints must validate rows against its model class."""
    spec = REGISTRY["DataSet"]

    assert spec.write_cls is None
    assert spec.validation_cls is DataSet


def test_custom_spec_controls_validation():
    """The supplied spec, not the registry, must decide what a row needs."""
    registry_spec = REGISTRY["Cluster"]

    class StricterCluster(ClusterWrite):
        level: int

    custom_spec = registry_spec.model_copy(update={"write_cls": StricterCluster})

    model = Cluster(id="c1", hierarchy_id="h1")
    result = validate_for_write([model], registry_spec)
    assert result == [model]
    assert result[0] is model

    with pytest.raises(ValueError, match="level"):
        validate_for_write([model], custom_spec)


# ---------------------------------------------------------------------------
# validate_for_write — failure path
# ---------------------------------------------------------------------------


def test_missing_required_for_write_slot_raises_before_io():
    """A missing write-required field must fail validation before writer IO."""
    spec = REGISTRY["Cluster"]
    bad = Cluster(id="c1")  # hierarchy_id missing
    with pytest.raises(
        ValueError, match=r"invalid slot\(s\): hierarchy_id"
    ):
        validate_for_write([bad], spec)


def test_missing_slot_names_class_in_error():
    """A required-field failure must identify the model class in its message."""
    spec = REGISTRY["CellFeatureDefinition"]
    bad = CellFeatureDefinition(id="f1", project_id="p1")  # feature_set_id missing
    with pytest.raises(ValueError, match="CellFeatureDefinition"):
        validate_for_write([bad], spec)


@pytest.mark.parametrize(
    "missing_field,values",
    [
        ("item", {"item": None, "cluster": "c1"}),
        ("cluster", {"item": "cell_1", "cluster": None}),
    ],
)
def test_cluster_membership_merge_keys_are_required_for_write(
    missing_field, values
):
    """Nullable schema fields used as merge keys must fail before IO."""
    membership = ClusterMembership(
        project_id="p1",
        hierarchy_id="h1",
        **values,
    )

    with pytest.raises(ValueError, match=missing_field):
        validate_for_write([membership], REGISTRY["ClusterMembership"])


# ---------------------------------------------------------------------------
# validate_for_write — happy path
# ---------------------------------------------------------------------------


def test_valid_model_returns_original_instance():
    """Successful strict validation must return the original exact-type object."""
    spec = REGISTRY["Cluster"]
    good = Cluster(id="c1", hierarchy_id="h1", level=2)
    result = validate_for_write([good], spec)

    assert isinstance(result, list)
    assert result[0] is good
    assert type(result[0]) is spec.model_cls


def test_validate_for_write_accepts_tuple_and_returns_originals_in_list():
    """A valid tuple must return a list preserving each input object's identity."""
    spec = REGISTRY["Cluster"]
    items = (
        Cluster(id="c1", hierarchy_id="h1"),
        Cluster(id="c2", hierarchy_id="h1"),
    )
    result = validate_for_write(items, spec)

    assert isinstance(result, list)
    assert all(actual is expected for actual, expected in zip(result, items))
    assert all(type(model) is spec.model_cls for model in result)


def test_validate_for_write_list_reports_failing_row():
    """A later required-field failure must identify that row and its id."""
    spec = REGISTRY["Cluster"]
    items = [
        Cluster(id="c1", hierarchy_id="h1"),
        Cluster(id="c2"),  # missing hierarchy_id
    ]
    with pytest.raises(ValueError, match="hierarchy_id") as ei:
        validate_for_write(items, spec)
    assert "c2" in str(ei.value), f"error should name failing row; got: {ei.value}"


def test_validate_for_write_passthrough_when_required_is_empty():
    """A spec without a write class must still return a list of the original objects."""
    spec = REGISTRY["DataSet"]
    ds = DataSet(id="d1", name="d", project_id="p1")
    result = validate_for_write([ds], spec)

    assert result == [ds]
    assert result[0] is ds
    assert type(result[0]) is spec.model_cls


def test_validate_for_write_rejects_class_mismatch():
    """Every sequence member must have the exact class declared by the spec."""
    spec = REGISTRY["Cluster"]
    not_a_cluster = DataSet(id="d1", name="d", project_id="p1")
    with pytest.raises(TypeError, match="Cluster"):
        validate_for_write([not_a_cluster], spec)


def test_validate_for_write_rejects_empty_sequence():
    """Validation requires a non-empty normalized sequence."""
    with pytest.raises(ValueError, match="empty"):
        validate_for_write([], REGISTRY["Cluster"])


def test_validate_for_write_rejects_single_model():
    """Validation must reject a direct model instead of normalizing its shape."""
    model = Cluster(id="c1", hierarchy_id="h1")

    with pytest.raises(TypeError, match="sequence"):
        validate_for_write(model, REGISTRY["Cluster"])  # type: ignore[arg-type]


def test_validate_for_write_rejects_generator_without_consuming_it():
    """Validation must reject a one-shot generator without materializing it."""
    consumed = False

    def models():
        nonlocal consumed
        consumed = True
        yield Cluster(id="c1", hierarchy_id="h1")

    with pytest.raises(TypeError, match="sequence"):
        validate_for_write(models(), REGISTRY["Cluster"])  # type: ignore[arg-type]

    assert not consumed


def test_validate_for_write_rejects_later_mismatched_member():
    """A later exact-type mismatch must report its row and actual class."""
    items = [
        Cluster(id="c1", hierarchy_id="h1"),
        DataSet(id="d1", name="d", project_id="p1"),
    ]

    with pytest.raises(TypeError, match=r"row 1.*DataSet"):
        validate_for_write(items, REGISTRY["Cluster"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Wired into write_models
# ---------------------------------------------------------------------------


def test_write_models_calls_validation_before_io(tmp_path):
    """Public writes must stop on strict validation failure before creating a table."""
    settings = Settings(output_root=tmp_path)
    bad = Cluster(id="c1")  # hierarchy_id missing
    with pytest.raises(ValueError, match="hierarchy_id"):
        write_models(bad, settings=settings)
    # No table directory created — IO never happened.
    assert not (tmp_path / "cluster").exists(), (
        "validation failure should short-circuit before any IO; "
        "cluster/ directory was created anyway"
    )
