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

# Wide payloads are dataframe-backed rather than model-table-backed.
WIDE_PAYLOAD_PATHS = MappingProxyType(
    {
        "cell_features": "cellfeatures",
        "synapse_features": "synapsefeatures",
    }
)

__all__ = [
    "MODEL_TABLE_PATHS",
    "WIDE_PAYLOAD_PATHS",
]
