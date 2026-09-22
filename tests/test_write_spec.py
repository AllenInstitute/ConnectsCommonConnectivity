"""Drift tests for the WriteSpec registry.

These tests guard against the registry getting out of sync with
``models.py`` — e.g., a renamed field silently breaking a writer's
predicate.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from connects_common_connectivity import models as models_module
from connects_common_connectivity.io.write_spec import REGISTRY, WriteSpec, get_spec

EXPECTED_MERGE_KEYS = {
    "DataSet": ["project_id", "id"],
    "DataItem": ["project_id", "id"],
    "DataItemDataSetAssociation": ["project_id", "dataset_id", "dataitem_id"],
    "Cluster": ["hierarchy_id", "id"],
    "ClusterHierarchy": ["id"],
    "ClusterMembership": ["project_id", "hierarchy_id", "item", "cluster"],
    "MappingSet": ["project_id", "id"],
    "CellToClusterMapping": ["project_id", "id"],
    "CellFeatureSet": ["project_id", "id"],
    "CellFeatureDefinition": ["project_id", "feature_set_id", "id"],
    "CellFeatureMatrix": ["project_id", "id"],
    "ProjectionMeasurementMatrix": ["project_id", "id"],
    "AlgorithmRun": ["id"],
    "HierarchyCategory": ["hierarchy_id", "id"],
    "SynapseFeatureMatrix": ["project_id", "id"],
}


def test_registry_contains_seed_entries():
    """The writer registry must contain its foundational model entries."""
    seed = {"DataSet", "DataItem", "DataItemDataSetAssociation"}
    assert seed.issubset(set(REGISTRY))


def test_milestone_scopes_use_taxonomy_and_project_identity():
    """Hierarchy categories and projection matrices must not collide across scopes."""
    hierarchy = REGISTRY["HierarchyCategory"]
    assert hierarchy.partition_by == ["hierarchy_id"]
    assert hierarchy.scope_columns == ["hierarchy_id", "id"]
    assert hierarchy.required_for_write == ["hierarchy_id"]

    projection = REGISTRY["ProjectionMeasurementMatrix"]
    assert projection.partition_by == ["project_id"]
    assert projection.scope_columns == ["project_id", "id"]


def test_registered_metadata_classes_use_declared_merge_keys():
    """Every WP2 metadata writer must declare its complete row identity."""
    assert set(REGISTRY) == set(EXPECTED_MERGE_KEYS)
    for class_name, merge_on in EXPECTED_MERGE_KEYS.items():
        spec = REGISTRY[class_name]
        assert spec.write_mode == "merge_scoped"
        assert spec.merge_on == merge_on

    assert REGISTRY["DataItem"].scope_columns == ["project_id", "id"]


def test_cluster_membership_merge_keys_are_required_only_for_write():
    """Nullable schema keys must be tightened at the IO boundary."""
    spec = REGISTRY["ClusterMembership"]
    assert spec.required_for_write == ["hierarchy_id", "item", "cluster"]


@pytest.mark.parametrize("key", list(REGISTRY))
def test_registry_key_matches_model_cls(key):
    """Each registry key must match its generated model class."""
    spec = REGISTRY[key]
    cls = getattr(models_module, key, None)
    assert cls is not None, f"models.py has no class named {key!r}"
    assert spec.model_cls is cls, (
        f"REGISTRY[{key!r}].model_cls is {spec.model_cls!r}, expected {cls!r}"
    )
    assert spec.model_cls.__name__ == key


@pytest.mark.parametrize("key", list(REGISTRY))
def test_spec_columns_exist_on_model(key):
    """Every configured writer column must exist on its model."""
    spec: WriteSpec = REGISTRY[key]
    fields = set(spec.model_cls.model_fields)
    for col in (
        spec.scope_columns
        + spec.partition_by
        + spec.required_for_write
        + spec.merge_on
    ):
        assert col in fields, (
            f"{spec.model_cls.__name__}: column {col!r} is not a field "
            f"(have: {sorted(fields)})"
        )


def test_get_spec_accepts_class_and_instance():
    """Writer specs must resolve from either a model class or instance."""
    ds_cls = REGISTRY["DataSet"].model_cls
    instance = ds_cls(id="d1", name="example", project_id="p1")
    assert get_spec(ds_cls) is REGISTRY["DataSet"]
    assert get_spec(instance) is REGISTRY["DataSet"]


def test_get_spec_unknown_class_raises():
    """Spec lookup must reject unregistered classes."""
    class NotRegistered:
        pass

    with pytest.raises(KeyError):
        get_spec(NotRegistered)


def test_get_spec_same_named_class_raises():
    """Spec lookup must require identity, not only a matching class name."""
    class DataSet(BaseModel):
        pass

    with pytest.raises(KeyError, match="exact class"):
        get_spec(DataSet)


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


def test_merge_scoped_requires_merge_keys():
    """A merge writer without an identity must be rejected at registration."""
    with pytest.raises(ValidationError, match="merge_on"):
        WriteSpec(
            model_cls=models_module.DataSet,
            subdir="dataset",
            partition_by=["project_id"],
            scope_columns=["project_id", "id"],
            write_mode="merge_scoped",
        )


def test_non_merge_mode_rejects_merge_keys():
    """Merge keys must not be silently ignored by another write mode."""
    with pytest.raises(ValidationError, match="merge_on"):
        WriteSpec(
            model_cls=models_module.DataSet,
            subdir="dataset",
            partition_by=["project_id"],
            scope_columns=["project_id", "id"],
            write_mode="overwrite_scoped",
            merge_on=["project_id", "id"],
        )
