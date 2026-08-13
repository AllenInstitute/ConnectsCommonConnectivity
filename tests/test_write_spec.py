"""Drift tests for the WriteSpec registry.

These tests guard against the registry getting out of sync with
``models.py`` — e.g., a renamed field silently breaking a writer's
predicate.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from connects_common_connectivity import models as models_module
from connects_common_connectivity.io.write_spec import REGISTRY, WriteSpec, get_spec


def test_registry_contains_seed_entries():
    seed = {"DataSet", "DataItem", "DataItemDataSetAssociation"}
    assert seed.issubset(set(REGISTRY))


@pytest.mark.parametrize("key", list(REGISTRY))
def test_registry_key_matches_model_cls(key):
    spec = REGISTRY[key]
    cls = getattr(models_module, key, None)
    assert cls is not None, f"models.py has no class named {key!r}"
    assert spec.model_cls is cls, (
        f"REGISTRY[{key!r}].model_cls is {spec.model_cls!r}, expected {cls!r}"
    )
    assert spec.model_cls.__name__ == key


@pytest.mark.parametrize("key", list(REGISTRY))
def test_spec_columns_exist_on_model(key):
    spec: WriteSpec = REGISTRY[key]
    fields = set(spec.model_cls.model_fields)
    for col in spec.scope_columns + spec.partition_by + spec.required_for_write:
        assert col in fields, (
            f"{spec.model_cls.__name__}: column {col!r} is not a field "
            f"(have: {sorted(fields)})"
        )


def test_get_spec_accepts_class_and_instance():
    ds_cls = REGISTRY["DataSet"].model_cls
    instance = ds_cls(id="d1", name="example", project_id="p1")
    assert get_spec(ds_cls) is REGISTRY["DataSet"]
    assert get_spec(instance) is REGISTRY["DataSet"]


def test_get_spec_unknown_class_raises():
    class NotRegistered:
        pass

    with pytest.raises(KeyError):
        get_spec(NotRegistered)


def test_write_spec_requires_pydantic_model_class():
    """WriteSpec must reject classes outside the Pydantic model hierarchy."""
    class NotAModel:
        pass

    with pytest.raises(ValidationError):
        WriteSpec(
            model_cls=NotAModel,
            subdir="invalid",
            partition_by=[],
            scope_columns=["id"],
            write_mode="overwrite_scoped",
        )
