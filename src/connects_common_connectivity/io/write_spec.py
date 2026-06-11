"""Write-spec registry for IO-layer Delta writers.

A :class:`WriteSpec` describes how a generated pydantic model is persisted into
the shared Delta lake: which subdirectory, which partition columns, which scope
columns, and which write mode the backend should dispatch on.

Only the seed entries needed to unblock W3 are registered here
(``DataSet``, ``DataItem``, ``DataItemDataSetAssociation``). Additional classes
are added in W3 as their writers are prototyped — see
``planning/prompts/03_writers.md``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..models import (
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
    MappingSet,
    ProjectionMeasurementMatrix,
)


class WriteSpec(BaseModel):
    """Declarative description of how a model class is written to Delta."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_cls: type
    subdir: str
    partition_by: list[str]
    scope_columns: list[str]
    write_mode: Literal["overwrite_scoped", "append_new_by_id"]
    required_for_write: list[str] = []
    cross_field_rules: list[str] = []


REGISTRY: dict[str, WriteSpec] = {
    "DataSet": WriteSpec(
        model_cls=DataSet,
        subdir="dataset",
        partition_by=["project_id"],
        # patchseq fix: today's notebooks predicate only on project_id, which
        # is why visp_inh_patchseq overwrites visp_exc_patchseq. Scoping on
        # (project_id, id) keeps each DataSet row independent.
        scope_columns=["project_id", "id"],
        write_mode="overwrite_scoped",
    ),
    "DataItem": WriteSpec(
        model_cls=DataItem,
        subdir="dataitem",
        partition_by=["project_id"],
        scope_columns=["id"],
        write_mode="append_new_by_id",
    ),
    "DataItemDataSetAssociation": WriteSpec(
        model_cls=DataItemDataSetAssociation,
        subdir="dataitem_dataset_association",
        partition_by=["project_id"],
        scope_columns=["project_id", "dataset_id"],
        write_mode="overwrite_scoped",
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
        write_mode="overwrite_scoped",
    ),
    "ClusterHierarchy": WriteSpec(
        model_cls=ClusterHierarchy,
        subdir="clusterhierarchy",
        partition_by=[],
        scope_columns=["id"],
        write_mode="overwrite_scoped",
    ),
    "ClusterMembership": WriteSpec(
        model_cls=ClusterMembership,
        subdir="clustermembership",
        partition_by=["project_id", "hierarchy_id"],
        scope_columns=["project_id", "hierarchy_id"],
        write_mode="overwrite_scoped",
    ),
    "MappingSet": WriteSpec(
        model_cls=MappingSet,
        subdir="mappingset",
        partition_by=["project_id"],
        scope_columns=["project_id", "id"],
        write_mode="overwrite_scoped",
    ),
    "CellToClusterMapping": WriteSpec(
        model_cls=CellToClusterMapping,
        subdir="celltoclustermapping",
        partition_by=["project_id"],
        # Notebooks predicate on (project_id, mapping_set), which is the
        # mapping-set foreign key on the row.
        scope_columns=["project_id", "mapping_set"],
        write_mode="overwrite_scoped",
    ),
    "CellFeatureSet": WriteSpec(
        model_cls=CellFeatureSet,
        subdir="cellfeatureset",
        partition_by=["project_id"],
        scope_columns=["project_id", "id"],
        write_mode="overwrite_scoped",
    ),
    "CellFeatureDefinition": WriteSpec(
        model_cls=CellFeatureDefinition,
        subdir="cellfeaturedefinition",
        partition_by=["project_id", "feature_set_id"],
        scope_columns=["project_id", "feature_set_id"],
        write_mode="overwrite_scoped",
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
        # outside the registry. See planning/prompts/03_writers.md report.
        write_mode="overwrite_scoped",
    ),
    "ProjectionMeasurementMatrix": WriteSpec(
        model_cls=ProjectionMeasurementMatrix,
        subdir="projectionmeasurementmatrix",
        # ProjectionMeasurementMatrix is not ProjectScoped (schema gap noted
        # in etl_wnm_exc_04). The notebook predicate is therefore ``id IN (...)``
        # only, with no partition columns. Once the schema gains
        # ``ProjectScoped``, partition_by/scope_columns should be widened.
        partition_by=[],
        scope_columns=["id"],
        write_mode="overwrite_scoped",
    ),
}


def get_spec(model_or_cls: type | BaseModel) -> WriteSpec:
    """Look up the :class:`WriteSpec` for a model class or instance.

    Accepts either the generated pydantic class itself or an instance of it,
    keyed by ``__name__`` of the class.
    """
    cls = model_or_cls if isinstance(model_or_cls, type) else type(model_or_cls)
    try:
        return REGISTRY[cls.__name__]
    except KeyError as err:
        raise KeyError(
            f"No WriteSpec registered for {cls.__name__!r}. "
            f"Known: {sorted(REGISTRY)}"
        ) from err


__all__ = ["WriteSpec", "REGISTRY", "get_spec"]
