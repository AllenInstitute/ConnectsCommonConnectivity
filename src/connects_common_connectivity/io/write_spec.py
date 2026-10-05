"""Write-spec registry for IO-layer Delta writers.

A :class:`WriteSpec` describes how a generated pydantic model is persisted into
the shared Delta lake: which subdirectory, which partition columns, which scope
columns, and which write mode the backend should dispatch on. :data:`REGISTRY`
is the source of truth for which classes are writable; add an entry here to
make a new class writable through :func:`write_models`.
"""

from __future__ import annotations

from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, model_validator

from connects_common_connectivity.models import (
    AlgorithmRun,
    CellFeatureDefinition,
    CellFeatureMatrix,
    CellFeatureSet,
    CellToClusterMapping,
    Cluster,
    ClusterHierarchy,
    ClusterMembership,
    DataItem,
    DataItemDataSetAssociation,
    DataSet,
    HierarchyCategory,
    MappingSet,
    ProjectionMeasurementMatrix,
    SynapseFeatureMatrix,
)


def _allows_none(annotation: Any) -> bool:
    """Return whether a field annotation accepts ``None``."""
    if annotation is type(None):
        return True
    origin = get_origin(annotation)
    return origin in (Union, UnionType) and type(None) in get_args(annotation)


class WriteSpec(BaseModel):
    """Declarative policy for validating and writing one model class to Delta.

    Attributes
    ----------
    model_cls:
        Exact generated Pydantic model class accepted by this policy.
    subdir:
        Delta table directory relative to the configured output root.
    partition_by:
        Columns used to partition the Delta table. Merge writes may also use
        qualifying partition columns to narrow the target scan.
    scope_columns:
        Columns defining replacement groups for ``overwrite_scoped`` writes.
        They do not determine identity for ``merge_scoped`` writes.
    write_mode:
        Dispatch strategy used by :func:`~connects_common_connectivity.io.writers.write_models`.
    merge_on:
        Complete row identity for ``merge_scoped`` writes. It must be non-empty
        only for that mode, and every key must be non-null at write time.
    required_for_write:
        Model fields made required and non-null by IO-layer validation without
        changing the shared LinkML schema.
    cross_field_rules:
        Reserved names for cross-field validation rules. The current write path
        does not consume them.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_cls: type[BaseModel]
    subdir: str
    partition_by: list[str]
    scope_columns: list[str]
    write_mode: Literal["overwrite_scoped", "merge_scoped"]
    merge_on: list[str] = Field(default_factory=list)
    required_for_write: list[str] = Field(default_factory=list)
    cross_field_rules: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_merge_on(self) -> WriteSpec:
        """Require merge keys exactly when the selected mode consumes them."""
        if self.write_mode == "merge_scoped" and not self.merge_on:
            raise ValueError("merge_on must be non-empty for merge_scoped writes")
        if self.write_mode != "merge_scoped" and self.merge_on:
            raise ValueError("merge_on is only valid for merge_scoped writes")

        missing_keys = [
            name for name in self.merge_on if name not in self.model_cls.model_fields
        ]
        if missing_keys:
            raise ValueError(
                f"merge_on fields are not declared by {self.model_cls.__name__}: "
                f"{missing_keys!r}"
            )

        unsafe_keys = []
        for name in self.merge_on:
            field = self.model_cls.model_fields[name]
            schema_enforces_non_null = (
                field.is_required() and not _allows_none(field.annotation)
            )
            if not schema_enforces_non_null and name not in self.required_for_write:
                unsafe_keys.append(name)
        if unsafe_keys:
            raise ValueError(
                "merge_on fields must be non-null at write time; make each field "
                "schema-required and non-nullable or add it to required_for_write: "
                f"{unsafe_keys!r}"
            )
        return self


REGISTRY: dict[str, WriteSpec] = {
    "DataSet": WriteSpec(
        model_cls=DataSet,
        subdir="dataset",
        partition_by=["project_id"],
        # Scoped on (project_id, id) so DataSet rows from sibling notebooks
        # sharing a project_id (e.g. patchseq exc/inh) do not overwrite each
        # other.
        scope_columns=["project_id", "id"],
        write_mode="merge_scoped",
        merge_on=["project_id", "id"],
    ),
    "DataItem": WriteSpec(
        model_cls=DataItem,
        subdir="dataitem",
        partition_by=["project_id"],
        scope_columns=["project_id", "id"],
        write_mode="merge_scoped",
        merge_on=["project_id", "id"],
    ),
    "DataItemDataSetAssociation": WriteSpec(
        model_cls=DataItemDataSetAssociation,
        subdir="dataitem_dataset_association",
        partition_by=["project_id"],
        scope_columns=["project_id", "dataset_id"],
        write_mode="merge_scoped",
        merge_on=["project_id", "dataset_id", "dataitem_id"],
    ),
    # Cluster taxonomy is project-agnostic in the schema — Cluster and
    # ClusterHierarchy do not carry project_id. Scope is the hierarchy id
    # (Cluster) or the row id (ClusterHierarchy), matching the existing
    # cluster ETL notebooks.
    "Cluster": WriteSpec(
        model_cls=Cluster,
        subdir="cluster",
        partition_by=["hierarchy_id"],
        scope_columns=["hierarchy_id"],
        write_mode="merge_scoped",
        merge_on=["hierarchy_id", "id"],
        required_for_write=["hierarchy_id"],
    ),
    "ClusterHierarchy": WriteSpec(
        model_cls=ClusterHierarchy,
        subdir="clusterhierarchy",
        partition_by=[],
        scope_columns=["id"],
        write_mode="merge_scoped",
        merge_on=["id"],
    ),
    "ClusterMembership": WriteSpec(
        model_cls=ClusterMembership,
        subdir="clustermembership",
        partition_by=["project_id", "hierarchy_id"],
        scope_columns=["project_id", "hierarchy_id"],
        write_mode="merge_scoped",
        merge_on=["project_id", "hierarchy_id", "item", "cluster"],
        required_for_write=["hierarchy_id", "item", "cluster"],
    ),
    "MappingSet": WriteSpec(
        model_cls=MappingSet,
        subdir="mappingset",
        partition_by=["project_id"],
        scope_columns=["project_id", "id"],
        write_mode="merge_scoped",
        merge_on=["project_id", "id"],
    ),
    "CellToClusterMapping": WriteSpec(
        model_cls=CellToClusterMapping,
        subdir="celltoclustermapping",
        partition_by=["project_id"],
        scope_columns=["project_id", "mapping_set"],
        write_mode="merge_scoped",
        merge_on=["project_id", "mapping_set", "id"],
    ),
    "CellFeatureSet": WriteSpec(
        model_cls=CellFeatureSet,
        subdir="cellfeatureset",
        partition_by=["project_id"],
        scope_columns=["project_id", "id"],
        write_mode="merge_scoped",
        merge_on=["project_id", "id"],
    ),
    "CellFeatureDefinition": WriteSpec(
        model_cls=CellFeatureDefinition,
        subdir="cellfeaturedefinition",
        partition_by=["project_id", "feature_set_id"],
        scope_columns=["project_id", "feature_set_id"],
        write_mode="merge_scoped",
        merge_on=["project_id", "feature_set_id", "id"],
        required_for_write=["feature_set_id"],
    ),
    "CellFeatureMatrix": WriteSpec(
        model_cls=CellFeatureMatrix,
        subdir="cellfeaturematrix",
        partition_by=["project_id"],
        scope_columns=["project_id", "feature_set_id"],
        # CellFeatureMatrix rows are metadata pointers (one row per matrix);
        # the wide-form numeric Parquet at ``cellfeatures/{feature_set_id}/``
        # is built from raw dataframes in the notebook, not from a model
        # instance, so it does not flow through ``write_models`` and stays
        # outside the registry.
        write_mode="merge_scoped",
        merge_on=["project_id", "feature_set_id", "id"],
    ),
    "ProjectionMeasurementMatrix": WriteSpec(
        model_cls=ProjectionMeasurementMatrix,
        subdir="projectionmeasurementmatrix",
        partition_by=["project_id"],
        scope_columns=["project_id", "id"],
        write_mode="merge_scoped",
        merge_on=["project_id", "id"],
    ),
    # AlgorithmRun and HierarchyCategory are project-agnostic taxonomy metadata
    # (no project_id slot). Notebook predicates are id-only, matching scope=["id"].
    "AlgorithmRun": WriteSpec(
        model_cls=AlgorithmRun,
        subdir="algorithmrun",
        partition_by=[],
        scope_columns=["id"],
        write_mode="merge_scoped",
        merge_on=["id"],
    ),
    "HierarchyCategory": WriteSpec(
        model_cls=HierarchyCategory,
        subdir="hierarchycategory",
        partition_by=["hierarchy_id"],
        scope_columns=["hierarchy_id", "id"],
        write_mode="merge_scoped",
        merge_on=["hierarchy_id", "id"],
        required_for_write=["hierarchy_id"],
    ),
#    "SynapseConnectivityLong": WriteSpec(
#        model_cls=SynapseConnectivityLong,
#        subdir="synapse",
#        partition_by=["project_id"],
#        scope_columns=["project_id", "dataset_id"],
#        write_mode="overwrite_scoped",
#    ),
    "SynapseFeatureMatrix": WriteSpec(
        model_cls=SynapseFeatureMatrix,
        subdir="synapsefeaturematrix",
        partition_by=["project_id"],
        scope_columns=["project_id", "id"],
        write_mode="merge_scoped",
        merge_on=["project_id", "id"],
    ),
}


def get_spec(model_or_cls: type[BaseModel] | BaseModel) -> WriteSpec:
    """Resolve the registered write policy for an exact model class.

    Parameters
    ----------
    model_or_cls:
        Generated pydantic model class or instance. Instances are resolved to
        their concrete type. The registry entry must match that exact class;
        subclasses and unrelated classes with the same name are rejected.

    Returns
    -------
    WriteSpec
        The registry's existing policy object for the exact class.

    Raises
    ------
    KeyError
        If no policy is registered for the exact class. The error lists the
        currently known registry keys.
    """
    cls = model_or_cls if isinstance(model_or_cls, type) else type(model_or_cls)
    spec = REGISTRY.get(cls.__name__)
    if spec is None or spec.model_cls is not cls:
        raise KeyError(
            f"No WriteSpec registered for exact class {cls!r}. "
            f"Known: {sorted(REGISTRY)}"
        )
    return spec


__all__ = ["WriteSpec", "REGISTRY", "get_spec"]
