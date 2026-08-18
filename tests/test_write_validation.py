"""Tests for write-time validation (auto-derived strict submodels)."""

from __future__ import annotations

import pytest

from connects_common_connectivity.config import Settings
from connects_common_connectivity.io.write_spec import REGISTRY
from connects_common_connectivity.io.write_validation import (
    strict_model_for,
    validate_for_write,
)
from connects_common_connectivity.io.writers import write_models
from connects_common_connectivity.models import (
    CellFeatureDefinition,
    Cluster,
    DataSet,
)

# ---------------------------------------------------------------------------
# strict_model_for
# ---------------------------------------------------------------------------


def test_strict_model_subclasses_parent_without_mutating_it():
    """Tightening fields must derive a subclass without mutating generated fields."""
    before = dict(Cluster.model_fields)
    strict = strict_model_for(REGISTRY["Cluster"])
    after = dict(Cluster.model_fields)

    assert before.keys() == after.keys()
    for k in before:
        assert before[k].is_required() == after[k].is_required(), (
            f"Cluster.model_fields[{k!r}] was mutated"
        )
    assert issubclass(strict, Cluster)
    assert strict is not Cluster


def test_strict_model_for_is_cached():
    """Equivalent model and required-field policies must reuse one strict class."""
    spec = REGISTRY["Cluster"]
    equivalent_spec = spec.model_copy(deep=True)

    a = strict_model_for(spec)
    b = strict_model_for(equivalent_spec)

    assert a is b


def test_strict_model_returns_parent_when_no_required_for_write():
    """A spec with no tightened fields must reuse its generated parent model."""
    spec = REGISTRY["DataSet"]

    assert spec.required_for_write == []
    assert strict_model_for(spec) is DataSet


def test_strict_model_flips_optional_field_to_required():
    """A write-required optional field must become required only on the strict class."""
    strict = strict_model_for(REGISTRY["Cluster"])

    assert not Cluster.model_fields["hierarchy_id"].is_required()
    assert strict.model_fields["hierarchy_id"].is_required()


def test_custom_spec_controls_validation_and_cache_policy():
    """Custom required fields must control validation and remain cache-isolated."""
    registry_spec = REGISTRY["Cluster"]
    equivalent_registry_spec = registry_spec.model_copy(deep=True)
    custom_spec = registry_spec.model_copy(
        update={"required_for_write": ["hierarchy_id", "level"]}
    )
    equivalent_custom_spec = custom_spec.model_copy(
        update={"required_for_write": ["level", "hierarchy_id"]}
    )

    registry_strict = strict_model_for(registry_spec)
    custom_strict = strict_model_for(custom_spec)

    assert strict_model_for(equivalent_registry_spec) is registry_strict
    assert strict_model_for(equivalent_custom_spec) is custom_strict
    assert custom_strict is not registry_strict

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
        ValueError, match=r"invalid required_for_write slot\(s\): hierarchy_id"
    ):
        validate_for_write([bad], spec)


def test_missing_slot_names_class_in_error():
    """A required-field failure must identify the model class in its message."""
    spec = REGISTRY["CellFeatureDefinition"]
    bad = CellFeatureDefinition(id="f1", project_id="p1")  # feature_set_id missing
    with pytest.raises(ValueError, match="CellFeatureDefinition"):
        validate_for_write([bad], spec)


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
    """A no-requirements spec must still return a list of the original objects."""
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
