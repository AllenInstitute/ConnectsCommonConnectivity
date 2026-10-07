"""Write-spec registry for IO-layer Delta writers.

A :class:`WriteSpec` describes how a generated pydantic model is persisted into
the shared Delta lake: which subdirectory, which partition columns, which scope
columns, and which write mode the backend should dispatch on. :data:`REGISTRY`
is the source of truth for which classes are writable; add an entry here to
make a new class writable through :func:`write_models`.

Row constraints live in :mod:`write_validation` as ``*Write`` subclasses of
the generated models. A spec names its subclass through ``write_cls``, and
every row is validated against it before IO.
"""

from __future__ import annotations

from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, model_validator

from connects_common_connectivity.io.path_spec import MODEL_TABLE_PATHS
from connects_common_connectivity.io.write_validation import (
    CellFeatureDefinitionWrite,
    ClusterMembershipWrite,
    ClusterWrite,
    HierarchyCategoryWrite,
    ReferenceSpaceWrite,
)
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
    EmbeddingLocation,
    EmbeddingSpace,
    HierarchyCategory,
    MappingSet,
    ProjectionMeasurementMatrix,
    ReferenceSpace,
    SpatialLocation,
    SynapseFeatureMatrix,
)


def _allows_none(annotation: Any) -> bool:
    """Return whether a field annotation accepts ``None``."""
    if annotation is type(None):
        return True
    origin = get_origin(annotation)
    return origin in (Union, UnionType) and type(None) in get_args(annotation)


def _enforces_non_null(model_cls: type[BaseModel], name: str) -> bool:
    """Return whether ``model_cls`` rejects a missing or null value for ``name``."""
    field = model_cls.model_fields[name]
    return field.is_required() and not _allows_none(field.annotation)


class WriteSpec(BaseModel):
    """Declarative policy for validating and writing one model class to Delta.

    Attributes
    ----------
    model_cls:
        Exact generated Pydantic model class accepted by this policy.
    write_cls:
        Subclass of ``model_cls`` that every row is validated against before
        IO, carrying write-time constraints not enforced by the generated
        model. ``None`` validates rows against ``model_cls`` itself. It may
        not declare fields absent from ``model_cls``.
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
        only for that mode. Keys must be non-null unless explicitly opted in
        through ``nullable_merge_on``.
    nullable_merge_on:
        Subset of merge keys for which null is a valid identity value. These
        keys use null-safe equality and cannot also be required by
        ``validation_cls``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_cls: type[BaseModel]
    write_cls: type[BaseModel] | None = None
    subdir: str
    partition_by: list[str]
    scope_columns: list[str]
    write_mode: Literal["overwrite_scoped", "merge_scoped"]
    merge_on: list[str] = Field(default_factory=list)
    nullable_merge_on: list[str] = Field(default_factory=list)

    @property
    def validation_cls(self) -> type[BaseModel]:
        """Class every row is validated against, defaulting to ``model_cls``."""
        return self.write_cls or self.model_cls

    @model_validator(mode="after")
    def validate_merge_on(self) -> WriteSpec:
        """Require merge keys exactly when the selected mode consumes them."""
        if self.write_mode == "merge_scoped" and not self.merge_on:
            raise ValueError("merge_on must be non-empty for merge_scoped writes")
        if self.write_mode != "merge_scoped" and self.merge_on:
            raise ValueError("merge_on is only valid for merge_scoped writes")

        if self.write_cls is not None:
            if not issubclass(self.write_cls, self.model_cls):
                raise ValueError(
                    f"write_cls {self.write_cls.__name__} must subclass "
                    f"{self.model_cls.__name__}"
                )
            # Catches a write class left behind by a renamed generated slot.
            undeclared = sorted(
                set(self.write_cls.model_fields) - set(self.model_cls.model_fields)
            )
            if undeclared:
                raise ValueError(
                    f"write_cls {self.write_cls.__name__} declares fields absent "
                    f"from {self.model_cls.__name__}: {undeclared!r}"
                )

        if not set(self.nullable_merge_on).issubset(self.merge_on):
            raise ValueError("nullable_merge_on must be a subset of merge_on")

        missing_keys = [
            name for name in self.merge_on if name not in self.model_cls.model_fields
        ]
        if missing_keys:
            raise ValueError(
                f"merge_on fields are not declared by {self.model_cls.__name__}: "
                f"{missing_keys!r}"
            )

        contradictory_keys = [
            name
            for name in self.nullable_merge_on
            if _enforces_non_null(self.validation_cls, name)
        ]
        if contradictory_keys:
            raise ValueError(
                "nullable_merge_on cannot name keys that "
                f"{self.validation_cls.__name__} requires to be non-null: "
                f"{contradictory_keys!r}"
            )

        unsafe_keys = [
            name
            for name in self.merge_on
            if not _enforces_non_null(self.validation_cls, name)
            and name not in self.nullable_merge_on
        ]
        if unsafe_keys:
            raise ValueError(
                "merge_on fields must be non-null at write time; make each field "
                "schema-required and non-nullable, require it on a write_cls, "
                "or explicitly allow null identity values with nullable_merge_on: "
                f"{unsafe_keys!r}"
            )
        return self


REGISTRY: dict[str, WriteSpec] = {
    # Reference-space IDs are scoped by project; null identifies the global scope.
    "ReferenceSpace": WriteSpec(
        model_cls=ReferenceSpace,
        write_cls=ReferenceSpaceWrite,
        subdir=MODEL_TABLE_PATHS["ReferenceSpace"],
        partition_by=[],
        scope_columns=["project_id", "id"],
        write_mode="merge_scoped",
        merge_on=["project_id", "id"],
        nullable_merge_on=["project_id"],
    ),
    "SpatialLocation": WriteSpec(
        model_cls=SpatialLocation,
        subdir=MODEL_TABLE_PATHS["SpatialLocation"],
        partition_by=["project_id"],
        scope_columns=["project_id", "reference_space"],
        write_mode="merge_scoped",
        merge_on=["project_id", "dataitem_id", "reference_space", "location_type"],
    ),
    "EmbeddingSpace": WriteSpec(
        model_cls=EmbeddingSpace,
        subdir=MODEL_TABLE_PATHS["EmbeddingSpace"],
        partition_by=["project_id"],
        scope_columns=["project_id", "id"],
        write_mode="merge_scoped",
        merge_on=["project_id", "id"],
    ),
    "EmbeddingLocation": WriteSpec(
        model_cls=EmbeddingLocation,
        subdir=MODEL_TABLE_PATHS["EmbeddingLocation"],
        partition_by=["project_id"],
        scope_columns=["project_id", "embedding_space"],
        write_mode="merge_scoped",
        merge_on=["project_id", "dataitem_id", "embedding_space"],
    ),
    "DataSet": WriteSpec(
        model_cls=DataSet,
        subdir=MODEL_TABLE_PATHS["DataSet"],
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
        subdir=MODEL_TABLE_PATHS["DataItem"],
        partition_by=["project_id"],
        scope_columns=["project_id", "id"],
        write_mode="merge_scoped",
        merge_on=["project_id", "id"],
    ),
    "DataItemDataSetAssociation": WriteSpec(
        model_cls=DataItemDataSetAssociation,
        subdir=MODEL_TABLE_PATHS["DataItemDataSetAssociation"],
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
        write_cls=ClusterWrite,
        subdir=MODEL_TABLE_PATHS["Cluster"],
        partition_by=["hierarchy_id"],
        scope_columns=["hierarchy_id"],
        write_mode="merge_scoped",
        merge_on=["hierarchy_id", "id"],
    ),
    "ClusterHierarchy": WriteSpec(
        model_cls=ClusterHierarchy,
        subdir=MODEL_TABLE_PATHS["ClusterHierarchy"],
        partition_by=[],
        scope_columns=["id"],
        write_mode="merge_scoped",
        merge_on=["id"],
    ),
    "ClusterMembership": WriteSpec(
        model_cls=ClusterMembership,
        write_cls=ClusterMembershipWrite,
        subdir=MODEL_TABLE_PATHS["ClusterMembership"],
        partition_by=["project_id", "hierarchy_id"],
        scope_columns=["project_id", "hierarchy_id"],
        write_mode="merge_scoped",
        merge_on=["project_id", "hierarchy_id", "item", "cluster"],
    ),
    "MappingSet": WriteSpec(
        model_cls=MappingSet,
        subdir=MODEL_TABLE_PATHS["MappingSet"],
        partition_by=["project_id"],
        scope_columns=["project_id", "id"],
        write_mode="merge_scoped",
        merge_on=["project_id", "id"],
    ),
    "CellToClusterMapping": WriteSpec(
        model_cls=CellToClusterMapping,
        subdir=MODEL_TABLE_PATHS["CellToClusterMapping"],
        partition_by=["project_id"],
        scope_columns=["project_id", "mapping_set"],
        write_mode="merge_scoped",
        merge_on=["project_id", "mapping_set", "id"],
    ),
    "CellFeatureSet": WriteSpec(
        model_cls=CellFeatureSet,
        subdir=MODEL_TABLE_PATHS["CellFeatureSet"],
        partition_by=["project_id"],
        scope_columns=["project_id", "id"],
        write_mode="merge_scoped",
        merge_on=["project_id", "id"],
    ),
    "CellFeatureDefinition": WriteSpec(
        model_cls=CellFeatureDefinition,
        write_cls=CellFeatureDefinitionWrite,
        subdir=MODEL_TABLE_PATHS["CellFeatureDefinition"],
        partition_by=["project_id", "feature_set_id"],
        scope_columns=["project_id", "feature_set_id"],
        write_mode="merge_scoped",
        merge_on=["project_id", "feature_set_id", "id"],
    ),
    "CellFeatureMatrix": WriteSpec(
        model_cls=CellFeatureMatrix,
        subdir=MODEL_TABLE_PATHS["CellFeatureMatrix"],
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
        subdir=MODEL_TABLE_PATHS["ProjectionMeasurementMatrix"],
        partition_by=["project_id"],
        scope_columns=["project_id", "id"],
        write_mode="merge_scoped",
        merge_on=["project_id", "id"],
    ),
    # AlgorithmRun and HierarchyCategory are project-agnostic taxonomy metadata
    # (no project_id slot). Notebook predicates are id-only, matching scope=["id"].
    "AlgorithmRun": WriteSpec(
        model_cls=AlgorithmRun,
        subdir=MODEL_TABLE_PATHS["AlgorithmRun"],
        partition_by=[],
        scope_columns=["id"],
        write_mode="merge_scoped",
        merge_on=["id"],
    ),
    "HierarchyCategory": WriteSpec(
        model_cls=HierarchyCategory,
        write_cls=HierarchyCategoryWrite,
        subdir=MODEL_TABLE_PATHS["HierarchyCategory"],
        partition_by=["hierarchy_id"],
        scope_columns=["hierarchy_id", "id"],
        write_mode="merge_scoped",
        merge_on=["hierarchy_id", "id"],
    ),
#    "SynapseConnectivityLong": WriteSpec(
#        model_cls=SynapseConnectivityLong,
#        subdir=MODEL_TABLE_PATHS["SynapseConnectivityLong"],
#        partition_by=["project_id"],
#        scope_columns=["project_id", "synapse_table_id"],
#        write_mode="overwrite_scoped",
#    ),
    "SynapseFeatureMatrix": WriteSpec(
        model_cls=SynapseFeatureMatrix,
        subdir=MODEL_TABLE_PATHS["SynapseFeatureMatrix"],
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


__all__ = [
    "REGISTRY",
    "CellFeatureDefinitionWrite",
    "ClusterMembershipWrite",
    "ClusterWrite",
    "HierarchyCategoryWrite",
    "ReferenceSpaceWrite",
    "WriteSpec",
    "get_spec",
]
