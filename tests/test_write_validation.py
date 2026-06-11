"""Tests for write-time validation (auto-derived strict submodels)."""

from __future__ import annotations

import pytest

from connects_common_connectivity.io.write_spec import REGISTRY, WriteSpec
from connects_common_connectivity.io.write_validation import (
    strict_model_for,
    validate_for_write,
)
from connects_common_connectivity.models import (
    CellFeatureDefinition,
    Cluster,
    DataSet,
)


# ---------------------------------------------------------------------------
# strict_model_for
# ---------------------------------------------------------------------------


def test_strict_model_subclasses_parent_without_mutating_it():
    before = dict(Cluster.model_fields)
    strict = strict_model_for(Cluster)
    after = dict(Cluster.model_fields)

    assert before.keys() == after.keys()
    for k in before:
        assert before[k].is_required() == after[k].is_required(), (
            f"Cluster.model_fields[{k!r}] was mutated"
        )
    assert issubclass(strict, Cluster)
    assert strict is not Cluster


def test_strict_model_for_is_cached():
    a = strict_model_for(Cluster)
    b = strict_model_for(Cluster)
    assert a is b


def test_strict_model_returns_parent_when_no_required_for_write():
    # DataSet has empty required_for_write; the strict subclass is just the parent.
    assert REGISTRY["DataSet"].required_for_write == []
    assert strict_model_for(DataSet) is DataSet


def test_strict_model_flips_optional_field_to_required():
    strict = strict_model_for(Cluster)
    # On the parent, hierarchy_id is optional.
    assert not Cluster.model_fields["hierarchy_id"].is_required()
    # On the strict subclass, hierarchy_id is required.
    assert strict.model_fields["hierarchy_id"].is_required()


# ---------------------------------------------------------------------------
# validate_for_write — failure path
# ---------------------------------------------------------------------------


def test_missing_required_for_write_slot_raises_before_io():
    spec = REGISTRY["Cluster"]
    bad = Cluster(id="c1")  # hierarchy_id missing
    with pytest.raises(ValueError, match="hierarchy_id"):
        validate_for_write(bad, spec)


def test_missing_slot_names_class_in_error():
    spec = REGISTRY["CellFeatureDefinition"]
    bad = CellFeatureDefinition(id="f1", project_id="p1")  # feature_set_id missing
    with pytest.raises(ValueError, match="CellFeatureDefinition"):
        validate_for_write(bad, spec)


# ---------------------------------------------------------------------------
# validate_for_write — happy path
# ---------------------------------------------------------------------------


def test_valid_model_passes_and_round_trips_field_by_field():
    spec = REGISTRY["Cluster"]
    good = Cluster(id="c1", hierarchy_id="h1", level=2)
    result = validate_for_write(good, spec)
    # Field-by-field equality with the input.
    for name in Cluster.model_fields:
        assert getattr(result, name) == getattr(good, name)


def test_validate_for_write_accepts_a_list():
    spec = REGISTRY["Cluster"]
    items = [
        Cluster(id="c1", hierarchy_id="h1"),
        Cluster(id="c2", hierarchy_id="h1"),
    ]
    result = validate_for_write(items, spec)
    assert isinstance(result, list)
    assert [m.id for m in result] == ["c1", "c2"]


def test_validate_for_write_passthrough_when_required_is_empty():
    spec = REGISTRY["DataSet"]
    ds = DataSet(id="d1", name="d", project_id="p1")
    result = validate_for_write(ds, spec)
    # No revalidation needed; identity-equal.
    assert result is ds


def test_validate_for_write_rejects_class_mismatch():
    spec = REGISTRY["Cluster"]
    not_a_cluster = DataSet(id="d1", name="d", project_id="p1")
    with pytest.raises(TypeError, match="Cluster"):
        validate_for_write(not_a_cluster, spec)


# ---------------------------------------------------------------------------
# Wired into write_models
# ---------------------------------------------------------------------------


def test_write_models_calls_validation_before_io(tmp_path):
    from connects_common_connectivity.config import Settings
    from connects_common_connectivity.io.writers import write_models

    settings = Settings(output_root=tmp_path)
    bad = Cluster(id="c1")  # hierarchy_id missing
    with pytest.raises(ValueError, match="hierarchy_id"):
        write_models(bad, settings=settings)
    # No table directory created — IO never happened.
    assert not (tmp_path / "cluster").exists()
