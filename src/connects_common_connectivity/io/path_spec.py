"""Canonical storage paths shared by IO readers, writers, and ETLs."""

from __future__ import annotations

from types import MappingProxyType

_MODEL_TABLE_PATHS = {
    "AlgorithmRun": "algorithmrun",
    "CellCellConnectivityLong": "cellcellconnectivitylong",
    "CellFeatureDefinition": "cellfeaturedefinition",
    "CellFeatureMatrix": "cellfeaturematrix",
    "CellFeatureSet": "cellfeatureset",
    "CellToClusterMapping": "celltoclustermapping",
    "Cluster": "cluster",
    "ClusterHierarchy": "clusterhierarchy",
    "ClusterMembership": "clustermembership",
    "DataItem": "dataitem",
    "DataItemDataSetAssociation": "dataitem_dataset_association",
    "DataSet": "dataset",
    "HierarchyCategory": "hierarchycategory",
    "MappingSet": "mappingset",
    "ProjectionMeasurementMatrix": "projectionmeasurementmatrix",
    "SynapseConnectivityLong": "synapse",
    "SynapseFeatureMatrix": "synapsefeaturematrix",
}

MODEL_TABLE_PATHS = MappingProxyType(_MODEL_TABLE_PATHS)

ALGORITHM_RUN_SUBDIR = MODEL_TABLE_PATHS["AlgorithmRun"]
CELL_CELL_CONNECTIVITY_SUBDIR = MODEL_TABLE_PATHS["CellCellConnectivityLong"]
CELL_FEATURE_DEFINITION_SUBDIR = MODEL_TABLE_PATHS["CellFeatureDefinition"]
CELL_FEATURE_MATRIX_SUBDIR = MODEL_TABLE_PATHS["CellFeatureMatrix"]
CELL_FEATURE_SET_SUBDIR = MODEL_TABLE_PATHS["CellFeatureSet"]
CELL_TO_CLUSTER_MAPPING_SUBDIR = MODEL_TABLE_PATHS["CellToClusterMapping"]
CLUSTER_SUBDIR = MODEL_TABLE_PATHS["Cluster"]
CLUSTER_HIERARCHY_SUBDIR = MODEL_TABLE_PATHS["ClusterHierarchy"]
CLUSTER_MEMBERSHIP_SUBDIR = MODEL_TABLE_PATHS["ClusterMembership"]
DATAITEM_SUBDIR = MODEL_TABLE_PATHS["DataItem"]
DATAITEM_DATASET_ASSOCIATION_SUBDIR = MODEL_TABLE_PATHS[
    "DataItemDataSetAssociation"
]
DATASET_SUBDIR = MODEL_TABLE_PATHS["DataSet"]
HIERARCHY_CATEGORY_SUBDIR = MODEL_TABLE_PATHS["HierarchyCategory"]
MAPPING_SET_SUBDIR = MODEL_TABLE_PATHS["MappingSet"]
PROJECTION_MEASUREMENT_MATRIX_SUBDIR = MODEL_TABLE_PATHS[
    "ProjectionMeasurementMatrix"
]
SYNAPSE_SUBDIR = MODEL_TABLE_PATHS["SynapseConnectivityLong"]
SYNAPSE_FEATURE_MATRIX_SUBDIR = MODEL_TABLE_PATHS["SynapseFeatureMatrix"]

# Wide payloads are dataframe-backed rather than model-table-backed.
CELL_FEATURES_SUBDIR = "cellfeatures"
SYNAPSE_FEATURES_SUBDIR = "synapsefeatures"

__all__ = [
    "ALGORITHM_RUN_SUBDIR",
    "CELL_CELL_CONNECTIVITY_SUBDIR",
    "CELL_FEATURE_DEFINITION_SUBDIR",
    "CELL_FEATURE_MATRIX_SUBDIR",
    "CELL_FEATURE_SET_SUBDIR",
    "CELL_FEATURES_SUBDIR",
    "CELL_TO_CLUSTER_MAPPING_SUBDIR",
    "CLUSTER_HIERARCHY_SUBDIR",
    "CLUSTER_MEMBERSHIP_SUBDIR",
    "CLUSTER_SUBDIR",
    "DATAITEM_DATASET_ASSOCIATION_SUBDIR",
    "DATAITEM_SUBDIR",
    "DATASET_SUBDIR",
    "HIERARCHY_CATEGORY_SUBDIR",
    "MAPPING_SET_SUBDIR",
    "MODEL_TABLE_PATHS",
    "PROJECTION_MEASUREMENT_MATRIX_SUBDIR",
    "SYNAPSE_FEATURE_MATRIX_SUBDIR",
    "SYNAPSE_FEATURES_SUBDIR",
    "SYNAPSE_SUBDIR",
]
