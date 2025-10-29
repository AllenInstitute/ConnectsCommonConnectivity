from __future__ import annotations 

import re
import sys
from datetime import (
    date,
    datetime,
    time
)
from decimal import Decimal 
from enum import Enum 
from typing import (
    Any,
    ClassVar,
    Literal,
    Optional,
    Union
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    field_validator
)


metamodel_version = "None"
version = "None"


class ConfiguredBaseModel(BaseModel):
    model_config = ConfigDict(
        validate_assignment = True,
        validate_default = True,
        extra = "forbid",
        arbitrary_types_allowed = True,
        use_enum_values = True,
        strict = False,
    )
    pass




class LinkMLMeta(RootModel):
    root: dict[str, Any] = {}
    model_config = ConfigDict(frozen=True)

    def __getattr__(self, key:str):
        return getattr(self.root, key)

    def __getitem__(self, key:str):
        return self.root[key]

    def __setitem__(self, key:str, value):
        self.root[key] = value

    def __contains__(self, key:str) -> bool:
        return key in self.root


linkml_meta = LinkMLMeta({'default_prefix': 'cc',
     'default_range': 'string',
     'description': 'Aggregator schema that imports all modular components of the '
                    'connectivity data model.',
     'id': 'https://brain-connects.org/ic3-common-connectivity-schema',
     'imports': ['linkml:types',
                 'core_schema',
                 'clustering_schema',
                 'brain_region_schema',
                 'projection_schema',
                 'cell_features_schema',
                 'cell_gene_schema',
                 'single_cell_schema',
                 'mappings_schema',
                 'cell_cell_schema'],
     'name': 'connectivity_schema',
     'prefixes': {'cc': {'prefix_prefix': 'cc',
                         'prefix_reference': 'https://brain-connects.org/cc/'},
                  'linkml': {'prefix_prefix': 'linkml',
                             'prefix_reference': 'https://w3id.org/linkml/'},
                  'schema': {'prefix_prefix': 'schema',
                             'prefix_reference': 'http://schema.org/'},
                  'skos': {'prefix_prefix': 'skos',
                           'prefix_reference': 'http://www.w3.org/2004/02/skos/core#'}},
     'source_file': 'schemas/connectivity_schema.yaml'} )

class Modality(str, Enum):
    BARCODED = "BARCODED"
    """
    Barcoded projection mapping methods (e.g., MAPseq, BARseq) producing molecule detection counts.
    """
    MORPHOLOGY = "MORPHOLOGY"
    """
    Morphological reconstructions providing structural projection features.
    """
    TRACER = "TRACER"
    """
    Traditional tracer experiments.
    """
    ELECTRON_MICROSCOPY = "ELECTRON_MICROSCOPY"
    """
    Electron microscopy based connectivity mapping.
    """
    XRAY_MICROSCOPY = "XRAY_MICROSCOPY"
    """
    X-ray microscopy based connectivity mapping.
    """
    EXPANSION_MICROSCOPY = "EXPANSION_MICROSCOPY"
    """
    Expansion microscopy based connectivity mapping.
    """
    OTHER = "OTHER"
    """
    Other modality.
    """


class ProjectionMeasurementType(str, Enum):
    """
    Types of projection-related measurements. Extendable per modality.
    """
    DETECTIONS = "DETECTIONS"
    """
    Count of detected RNA/barcode molecules in a region (barcoded modalities).
    """
    NUMBER_OF_TIPS = "NUMBER_OF_TIPS"
    """
    Number of axonal terminal tips reconstructed in a region (morphology derived metric).
    """
    MICRONS_OF_AXON = "MICRONS_OF_AXON"
    """
    Microns of axon contained in a region (morphology derived metric).
    """
    SIGNAL_INTENSITY = "SIGNAL_INTENSITY"
    """
    General normalized signal intensity (e.g., tracer fluorescence).
    """


class SynapticMeasurementType(str, Enum):
    """
    Types of synaptic connectivity-related measurements. Extendable per modality.
    """
    EXISTENCE = "EXISTENCE"
    """
    Binary existence of a synaptic connection between two cells, with no additional information about weight.
    """
    SYNAPSE_COUNT = "SYNAPSE_COUNT"
    """
    Count of synapses detected between two cells.
    """
    SUM_ANATOMICAL_SIZE = "SUM_ANATOMICAL_SIZE"
    """
    Anatomical size of synapse for synapses detected between two cells, summed over all synapses.
    """
    SUMMED_SIGNAL = "SUMMED_SIGNAL"
    """
    Summed signal intensity of synapses detected between two cells. For example PSD-95 or gephyrin signal.
    """
    SYNAPTIC_STRENGTH = "SYNAPTIC_STRENGTH"
    """
    Estimate of the synaptic strength between two cells,
    measured or estimated from physiology.
    For example, the size of a post-synaptic potential between two neurons (Unit here would be nA or mV)
    or could be an estimate of the effective weight of a synapse based on a model 
    that takes into account dendritic location, receptor density, etc.
    (Unit would then be AU).
    """


class Unit(str, Enum):
    COUNT = "COUNT"
    """
    Plain count unit.
    """
    MICRONS_LENGTH = "MICRONS_LENGTH"
    """
    Micrometers length measurement.
    """
    MICRONS_SQUARE = "MICRONS_SQUARE"
    """
    Micrometers squared area measurement.
    """
    MICRONS_CUBED = "MICRONS_CUBED"
    """
    Micrometers cubed volume measurement.
    """
    MICRONS_INVERSE = "MICRONS_INVERSE"
    """
    Inverse micrometers (e.g., density per micron).
    """
    COUNT_PER_MICRON = "COUNT_PER_MICRON"
    """
    Count per micrometer (e.g., linear density).
    """
    COUNT_PER_MICRONS_SQUARED = "COUNT_PER_MICRONS_SQUARED"
    """
    Count per micrometers squared (e.g., area density).
    """
    COUNT_PER_MICRONS_CUBED = "COUNT_PER_MICRONS_CUBED"
    """
    Count per micrometers cubed (e.g., volume density).
    """
    RATIO = "RATIO"
    """
    Dimensionless ratio.
    """
    NANOAMPS = "NANOAMPS"
    """
    Nanoamps electrical current.
    """
    MILLIVOLTS = "MILLIVOLTS"
    """
    Millivolts electrical potential.
    """
    ARBITRARY_UNIT = "ARBITRARY_UNIT"
    """
    Standardized measure with unspecified units. (e.g. summed pixel intensity).
    """
    BINARY = "BINARY"
    """
    Binary unit (0 or 1).
    """
    NONE = "NONE"
    """
    Dimensionless.
    """



class SpatialLocation(ConfiguredBaseModel):
    """
    3D spatial coordinates in a reference space.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-core-schema',
         'slot_usage': {'reference_space': {'description': 'Reference space (e.g., '
                                                           "'CCF_v3', 'CCF_v4').",
                                            'name': 'reference_space',
                                            'range': 'string',
                                            'required': True},
                        'x': {'description': 'X coordinate in the reference space.',
                              'name': 'x',
                              'range': 'float',
                              'required': True},
                        'y': {'description': 'Y coordinate in the reference space.',
                              'name': 'y',
                              'range': 'float',
                              'required': True},
                        'z': {'description': 'Z coordinate in the reference space.',
                              'name': 'z',
                              'range': 'float',
                              'required': True}}})

    x: float = Field(default=..., description="""X coordinate in the reference space.""", json_schema_extra = { "linkml_meta": {'alias': 'x', 'domain_of': ['SpatialLocation']} })
    y: float = Field(default=..., description="""Y coordinate in the reference space.""", json_schema_extra = { "linkml_meta": {'alias': 'y', 'domain_of': ['SpatialLocation']} })
    z: float = Field(default=..., description="""Z coordinate in the reference space.""", json_schema_extra = { "linkml_meta": {'alias': 'z', 'domain_of': ['SpatialLocation']} })
    reference_space: str = Field(default=..., description="""Reference space (e.g., 'CCF_v3', 'CCF_v4').""", json_schema_extra = { "linkml_meta": {'alias': 'reference_space', 'domain_of': ['SpatialLocation']} })


class AlgorithmRun(ConfiguredBaseModel):
    """
    Metadata about a single execution of a clustering algorithm.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-clustering-schema',
         'slot_usage': {'algorithm_name': {'description': 'Name of the clustering '
                                                          'algorithm (e.g., k-means, '
                                                          'hierarchical, DBSCAN).',
                                           'name': 'algorithm_name',
                                           'range': 'string',
                                           'required': True},
                        'algorithm_version': {'description': 'Version of the algorithm '
                                                             'implementation used.',
                                              'name': 'algorithm_version',
                                              'range': 'string'},
                        'distance_description': {'description': 'Description of the '
                                                                'distance metric used '
                                                                '(e.g., Euclidean, '
                                                                'cosine).',
                                                 'name': 'distance_description',
                                                 'range': 'string'},
                        'input_dataset': {'description': 'The dataset that was '
                                                         'clustered.',
                                          'name': 'input_dataset',
                                          'range': 'DataSet'},
                        'json_object': {'description': 'Arbitrary algorithm parameters '
                                                       'encoded as JSON object string.',
                                        'name': 'json_object'},
                        'produced_hierarchies': {'inlined': True,
                                                 'multivalued': True,
                                                 'name': 'produced_hierarchies',
                                                 'range': 'ClusterHierarchy'},
                        'run_timestamp': {'description': 'Timestamp when the algorithm '
                                                         'was executed.',
                                          'name': 'run_timestamp',
                                          'range': 'datetime'},
                        'score_description': {'description': 'Description of the '
                                                             'scoring metric used for '
                                                             'clusters (e.g., '
                                                             'silhouette score).',
                                              'name': 'score_description',
                                              'range': 'string'}}})

    id: str = Field(default=..., description="""Unique identifier within the class context.""", json_schema_extra = { "linkml_meta": {'alias': 'id',
         'aliases': ['identifier', 'structure_id', 'brain_region_id'],
         'domain_of': ['DataSet',
                       'DataItem',
                       'AlgorithmRun',
                       'ClusterHierarchy',
                       'Cluster',
                       'BrainRegion',
                       'ZarrArray',
                       'ZarrDataset',
                       'ParquetDataset',
                       'ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'CellFeatureMatrix',
                       'CellFeatureMeasurement',
                       'CellGeneData',
                       'SingleCellReconstruction',
                       'MappingSet',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    algorithm_name: str = Field(default=..., description="""Name of the clustering algorithm (e.g., k-means, hierarchical, DBSCAN).""", json_schema_extra = { "linkml_meta": {'alias': 'algorithm_name', 'domain_of': ['AlgorithmRun']} })
    algorithm_version: Optional[str] = Field(default=None, description="""Version of the algorithm implementation used.""", json_schema_extra = { "linkml_meta": {'alias': 'algorithm_version', 'domain_of': ['AlgorithmRun']} })
    json_object: Optional[str] = Field(default=None, description="""Arbitrary algorithm parameters encoded as JSON object string.""", json_schema_extra = { "linkml_meta": {'alias': 'json_object', 'domain_of': ['AlgorithmRun', 'MappingSet']} })
    run_timestamp: Optional[datetime ] = Field(default=None, description="""Timestamp when the algorithm was executed.""", json_schema_extra = { "linkml_meta": {'alias': 'run_timestamp', 'domain_of': ['AlgorithmRun']} })
    input_dataset: Optional[str] = Field(default=None, description="""The dataset that was clustered.""", json_schema_extra = { "linkml_meta": {'alias': 'input_dataset', 'domain_of': ['AlgorithmRun']} })
    produced_hierarchies: Optional[dict[str, ClusterHierarchy]] = Field(default=None, description="""Hierarchies produced in this run.""", json_schema_extra = { "linkml_meta": {'alias': 'produced_hierarchies', 'domain_of': ['AlgorithmRun']} })
    score_description: Optional[str] = Field(default=None, description="""Description of the scoring metric used for clusters (e.g., silhouette score).""", json_schema_extra = { "linkml_meta": {'alias': 'score_description', 'domain_of': ['AlgorithmRun']} })
    distance_description: Optional[str] = Field(default=None, description="""Description of the distance metric used (e.g., Euclidean, cosine).""", json_schema_extra = { "linkml_meta": {'alias': 'distance_description', 'domain_of': ['AlgorithmRun']} })

    @field_validator('json_object')
    def pattern_json_object(cls, v):
        pattern=re.compile(r"^\s*\{.*\}\s*$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid json_object format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid json_object format: {v}"
            raise ValueError(err_msg)
        return v


class ClusterHierarchy(ConfiguredBaseModel):
    """
    A complete clustering result (tree or DAG) from one algorithm run.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-clustering-schema',
         'slot_usage': {'clusters': {'inlined': False,
                                     'multivalued': True,
                                     'name': 'clusters',
                                     'range': 'Cluster'},
                        'root': {'name': 'root', 'range': 'Cluster'},
                        'run': {'name': 'run', 'range': 'AlgorithmRun'}}})

    id: str = Field(default=..., description="""Unique identifier within the class context.""", json_schema_extra = { "linkml_meta": {'alias': 'id',
         'aliases': ['identifier', 'structure_id', 'brain_region_id'],
         'domain_of': ['DataSet',
                       'DataItem',
                       'AlgorithmRun',
                       'ClusterHierarchy',
                       'Cluster',
                       'BrainRegion',
                       'ZarrArray',
                       'ZarrDataset',
                       'ParquetDataset',
                       'ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'CellFeatureMatrix',
                       'CellFeatureMeasurement',
                       'CellGeneData',
                       'SingleCellReconstruction',
                       'MappingSet',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    run: Optional[str] = Field(default=None, description="""The AlgorithmRun producing this hierarchy.""", json_schema_extra = { "linkml_meta": {'alias': 'run', 'domain_of': ['ClusterHierarchy']} })
    root: Optional[str] = Field(default=None, description="""The root cluster of the hierarchy.""", json_schema_extra = { "linkml_meta": {'alias': 'root', 'domain_of': ['ClusterHierarchy']} })
    clusters: Optional[list[str]] = Field(default=None, description="""All clusters in the hierarchy.""", json_schema_extra = { "linkml_meta": {'alias': 'clusters', 'domain_of': ['ClusterHierarchy']} })


class BrainRegion(ConfiguredBaseModel):
    """
    A brain region (node) in the common coordinate framework hierarchy.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-brain-region-schema',
         'slot_usage': {'annotation_value': {'description': 'Source-specific '
                                                            'annotation value for this '
                                                            'region. Should align to '
                                                            'an annotation volume for '
                                                            'a particular atlas.',
                                             'name': 'annotation_value'},
                        'child_identifiers': {'inlined': False,
                                              'multivalued': True,
                                              'name': 'child_identifiers',
                                              'range': 'BrainRegion',
                                              'slot_uri': 'skos:narrower'},
                        'descendant_annotation_values': {'description': 'Collected '
                                                                        'annotation '
                                                                        'values for '
                                                                        'descendant '
                                                                        'regions, '
                                                                        'Descendants '
                                                                        'meaning '
                                                                        'anything down '
                                                                        'the tree from '
                                                                        'this node. '
                                                                        'Useful when '
                                                                        'relabelling '
                                                                        'an annotation '
                                                                        'volume which '
                                                                        'is often '
                                                                        'labelled with '
                                                                        'the voxel '
                                                                        'farthest down '
                                                                        'the tree.',
                                                         'name': 'descendant_annotation_values'},
                        'descendants': {'description': 'Direct object references to '
                                                       'all descendant regions '
                                                       'anywhere below this node.',
                                        'inlined': False,
                                        'multivalued': True,
                                        'name': 'descendants',
                                        'range': 'BrainRegion'},
                        'hex_color': {'description': 'Hex RGB color assigned to this '
                                                     'region (from source ontology).',
                                      'name': 'hex_color'},
                        'parent_identifier': {'inlined': False,
                                              'name': 'parent_identifier',
                                              'range': 'BrainRegion',
                                              'slot_uri': 'skos:broader'},
                        'term_set_name': {'description': 'Name of the term set this '
                                                         'row came from. Term sets are '
                                                         'cuts across the hierachy '
                                                         'that represent a similar '
                                                         'level of biological detail '
                                                         'and cover the entire tree.',
                                          'name': 'term_set_name'}}})

    id: str = Field(default=..., description="""Unique identifier within the class context.""", json_schema_extra = { "linkml_meta": {'alias': 'id',
         'aliases': ['identifier', 'structure_id', 'brain_region_id'],
         'domain_of': ['DataSet',
                       'DataItem',
                       'AlgorithmRun',
                       'ClusterHierarchy',
                       'Cluster',
                       'BrainRegion',
                       'ZarrArray',
                       'ZarrDataset',
                       'ParquetDataset',
                       'ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'CellFeatureMatrix',
                       'CellFeatureMeasurement',
                       'CellGeneData',
                       'SingleCellReconstruction',
                       'MappingSet',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    name: str = Field(default=..., description="""A human-readable name or title.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'aliases': ['structure_name', 'region_name'],
         'domain_of': ['DataSet', 'DataItem', 'BrainRegion', 'MappingSet']} })
    parent_identifier: Optional[str] = Field(default=None, description="""Reference to broader brain region.""", json_schema_extra = { "linkml_meta": {'alias': 'parent_identifier',
         'aliases': ['parent_id', 'parent_structure_id'],
         'domain_of': ['BrainRegion'],
         'slot_uri': 'skos:broader'} })
    child_identifiers: Optional[list[str]] = Field(default=None, description="""Narrower brain regions nested under this region.""", json_schema_extra = { "linkml_meta": {'alias': 'child_identifiers',
         'aliases': ['child_ids', 'child_structure_ids'],
         'domain_of': ['BrainRegion'],
         'slot_uri': 'skos:narrower'} })
    acronym: Optional[str] = Field(default=None, description="""Short region acronym.""", json_schema_extra = { "linkml_meta": {'alias': 'acronym',
         'aliases': ['structure_acronym', 'region_acronym', 'abbreviation'],
         'domain_of': ['BrainRegion']} })
    hex_color: Optional[str] = Field(default=None, description="""Hex RGB color assigned to this region (from source ontology).""", json_schema_extra = { "linkml_meta": {'alias': 'hex_color',
         'aliases': ['rgb_hex', 'hex_color'],
         'domain_of': ['BrainRegion']} })
    term_set_name: Optional[str] = Field(default=None, description="""Name of the term set this row came from. Term sets are cuts across the hierachy that represent a similar level of biological detail and cover the entire tree.""", json_schema_extra = { "linkml_meta": {'alias': 'term_set_name', 'domain_of': ['BrainRegion']} })
    annotation_value: Optional[int] = Field(default=None, description="""Source-specific annotation value for this region. Should align to an annotation volume for a particular atlas.""", json_schema_extra = { "linkml_meta": {'alias': 'annotation_value', 'domain_of': ['BrainRegion']} })
    descendants: Optional[list[str]] = Field(default=None, description="""Direct object references to all descendant regions anywhere below this node.""", json_schema_extra = { "linkml_meta": {'alias': 'descendants', 'domain_of': ['BrainRegion']} })
    descendant_annotation_values: Optional[list[int]] = Field(default=None, description="""Collected annotation values for descendant regions, Descendants meaning anything down the tree from this node. Useful when relabelling an annotation volume which is often labelled with the voxel farthest down the tree.""", json_schema_extra = { "linkml_meta": {'alias': 'descendant_annotation_values', 'domain_of': ['BrainRegion']} })

    @field_validator('hex_color')
    def pattern_hex_color(cls, v):
        pattern=re.compile(r"^#?[0-9A-Fa-f]{6}$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid hex_color format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid hex_color format: {v}"
            raise ValueError(err_msg)
        return v


class ProjectScoped(ConfiguredBaseModel):
    """
    Mixin providing project scoping to entities that belong to a specific project.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-base-schema'})

    project_id: str = Field(default=..., description="""Identifier for the project or acquisition program context for this record.""", json_schema_extra = { "linkml_meta": {'alias': 'project_id',
         'aliases': ['project', 'program_id'],
         'domain_of': ['ProjectScoped']} })


class DataSet(ProjectScoped):
    """
    A collection of DataItems (e.g., all neurons from a study, all barcoded detections in a volume).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-core-schema',
         'mixins': ['ProjectScoped']})

    id: str = Field(default=..., description="""Unique identifier within the class context.""", json_schema_extra = { "linkml_meta": {'alias': 'id',
         'aliases': ['identifier', 'structure_id', 'brain_region_id'],
         'domain_of': ['DataSet',
                       'DataItem',
                       'AlgorithmRun',
                       'ClusterHierarchy',
                       'Cluster',
                       'BrainRegion',
                       'ZarrArray',
                       'ZarrDataset',
                       'ParquetDataset',
                       'ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'CellFeatureMatrix',
                       'CellFeatureMeasurement',
                       'CellGeneData',
                       'SingleCellReconstruction',
                       'MappingSet',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    name: str = Field(default=..., description="""A human-readable name or title.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'aliases': ['structure_name', 'region_name'],
         'domain_of': ['DataSet', 'DataItem', 'BrainRegion', 'MappingSet']} })
    publication: Optional[str] = Field(default=None, description="""Reference to publication describing the dataset.""", json_schema_extra = { "linkml_meta": {'alias': 'publication', 'domain_of': ['DataSet']} })
    modality: Optional[Modality] = Field(default=None, description="""Source modality for the data item (if relevant).""", json_schema_extra = { "linkml_meta": {'alias': 'modality',
         'domain_of': ['DataSet',
                       'ProjectionMeasurementMatrix',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    project_id: str = Field(default=..., description="""Identifier for the project or acquisition program context for this record.""", json_schema_extra = { "linkml_meta": {'alias': 'project_id',
         'aliases': ['project', 'program_id'],
         'domain_of': ['ProjectScoped']} })


class DataItem(ProjectScoped):
    """
    An individual datum (e.g., neuron reconstruction, barcoded detection, injection experiment).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-core-schema',
         'mixins': ['ProjectScoped'],
         'slot_usage': {'neuroglancer_link': {'description': 'A link that illustrates '
                                                             'this data item '
                                                             'visualized in a common '
                                                             'coordinate framework in '
                                                             'neuroglancer.',
                                              'multivalued': False,
                                              'name': 'neuroglancer_link',
                                              'range': 'string',
                                              'required': False}}})

    id: str = Field(default=..., description="""Unique identifier within the class context.""", json_schema_extra = { "linkml_meta": {'alias': 'id',
         'aliases': ['identifier', 'structure_id', 'brain_region_id'],
         'domain_of': ['DataSet',
                       'DataItem',
                       'AlgorithmRun',
                       'ClusterHierarchy',
                       'Cluster',
                       'BrainRegion',
                       'ZarrArray',
                       'ZarrDataset',
                       'ParquetDataset',
                       'ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'CellFeatureMatrix',
                       'CellFeatureMeasurement',
                       'CellGeneData',
                       'SingleCellReconstruction',
                       'MappingSet',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    name: str = Field(default=..., description="""A human-readable name or title.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'aliases': ['structure_name', 'region_name'],
         'domain_of': ['DataSet', 'DataItem', 'BrainRegion', 'MappingSet']} })
    neuroglancer_link: Optional[str] = Field(default=None, description="""A link that illustrates this data item visualized in a common coordinate framework in neuroglancer.""", json_schema_extra = { "linkml_meta": {'alias': 'neuroglancer_link', 'domain_of': ['DataItem']} })
    project_id: str = Field(default=..., description="""Identifier for the project or acquisition program context for this record.""", json_schema_extra = { "linkml_meta": {'alias': 'project_id',
         'aliases': ['project', 'program_id'],
         'domain_of': ['ProjectScoped']} })


class DataItemDataSetAssociation(ProjectScoped):
    """
    Many-to-many link between DataItem and DataSet.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-core-schema',
         'mixins': ['ProjectScoped'],
         'slot_usage': {'dataitem_id': {'description': 'Identifier of the DataItem you '
                                                       'are linking',
                                        'name': 'dataitem_id',
                                        'range': 'DataItem',
                                        'required': True},
                        'dataset_id': {'description': 'Identifier of the DataSet you '
                                                      'are linking',
                                       'name': 'dataset_id',
                                       'range': 'DataSet',
                                       'required': True}}})

    dataitem_id: str = Field(default=..., description="""Identifier of the DataItem you are linking""", json_schema_extra = { "linkml_meta": {'alias': 'dataitem_id',
         'domain_of': ['DataItemDataSetAssociation',
                       'CellFeatureMeasurement',
                       'CellGeneData',
                       'SingleCellReconstruction']} })
    dataset_id: str = Field(default=..., description="""Identifier of the DataSet you are linking""", json_schema_extra = { "linkml_meta": {'alias': 'dataset_id', 'domain_of': ['DataItemDataSetAssociation']} })
    project_id: str = Field(default=..., description="""Identifier for the project or acquisition program context for this record.""", json_schema_extra = { "linkml_meta": {'alias': 'project_id',
         'aliases': ['project', 'program_id'],
         'domain_of': ['ProjectScoped']} })


class Cluster(ProjectScoped):
    """
    A node (cluster) in a hierarchical clustering structure.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-clustering-schema',
         'mixins': ['ProjectScoped'],
         'slot_usage': {'centroid_values': {'description': 'Numeric centroid vector '
                                                           '(ordered feature list).',
                                            'multivalued': True,
                                            'name': 'centroid_values',
                                            'range': 'float'},
                        'children': {'description': 'Child clusters.',
                                     'multivalued': True,
                                     'name': 'children',
                                     'range': 'Cluster',
                                     'slot_uri': 'skos:narrower'},
                        'distance_to_parent': {'description': 'Distance metric to '
                                                              'parent centroid.',
                                               'name': 'distance_to_parent',
                                               'range': 'float'},
                        'level': {'description': 'Depth of the cluster in the '
                                                 'hierarchy where 0 is the root '
                                                 'cluster.',
                                  'name': 'level',
                                  'range': 'integer'},
                        'parent': {'description': 'Direct parent cluster (omit for '
                                                  'root).',
                                   'name': 'parent',
                                   'range': 'Cluster',
                                   'slot_uri': 'skos:broader'},
                        'score': {'description': 'Cluster quality metric (e.g., '
                                                 'silhouette score).',
                                  'name': 'score',
                                  'range': 'float'}}})

    id: str = Field(default=..., description="""Unique identifier within the class context.""", json_schema_extra = { "linkml_meta": {'alias': 'id',
         'aliases': ['identifier', 'structure_id', 'brain_region_id'],
         'domain_of': ['DataSet',
                       'DataItem',
                       'AlgorithmRun',
                       'ClusterHierarchy',
                       'Cluster',
                       'BrainRegion',
                       'ZarrArray',
                       'ZarrDataset',
                       'ParquetDataset',
                       'ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'CellFeatureMatrix',
                       'CellFeatureMeasurement',
                       'CellGeneData',
                       'SingleCellReconstruction',
                       'MappingSet',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    parent: Optional[str] = Field(default=None, description="""Direct parent cluster (omit for root).""", json_schema_extra = { "linkml_meta": {'alias': 'parent', 'domain_of': ['Cluster'], 'slot_uri': 'skos:broader'} })
    children: Optional[list[str]] = Field(default=None, description="""Child clusters.""", json_schema_extra = { "linkml_meta": {'alias': 'children', 'domain_of': ['Cluster'], 'slot_uri': 'skos:narrower'} })
    level: Optional[int] = Field(default=None, description="""Depth of the cluster in the hierarchy where 0 is the root cluster.""", json_schema_extra = { "linkml_meta": {'alias': 'level', 'domain_of': ['Cluster']} })
    score: Optional[float] = Field(default=None, description="""Cluster quality metric (e.g., silhouette score).""", json_schema_extra = { "linkml_meta": {'alias': 'score',
         'domain_of': ['Cluster',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping']} })
    distance_to_parent: Optional[float] = Field(default=None, description="""Distance metric to parent centroid.""", json_schema_extra = { "linkml_meta": {'alias': 'distance_to_parent', 'domain_of': ['Cluster']} })
    project_id: str = Field(default=..., description="""Identifier for the project or acquisition program context for this record.""", json_schema_extra = { "linkml_meta": {'alias': 'project_id',
         'aliases': ['project', 'program_id'],
         'domain_of': ['ProjectScoped']} })


class ClusterMembership(ProjectScoped):
    """
    Association linking a DataItem to a Cluster, with optional soft membership or distance metrics.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-clustering-schema',
         'mixins': ['ProjectScoped'],
         'slot_usage': {'cluster': {'inlined': False, 'name': 'cluster'},
                        'distance': {'description': 'Distance between the item and the '
                                                    'cluster centroid. (Smaller is '
                                                    'better).',
                                     'name': 'distance',
                                     'range': 'float'},
                        'membership_score': {'description': 'Algorithm-defined '
                                                            'membership strength. '
                                                            '(Optional, does not need '
                                                            'to be normalized)',
                                             'name': 'membership_score',
                                             'range': 'float'},
                        'probability': {'description': 'Normalized probability of '
                                                       'membership (sums to 1 across '
                                                       'clusters for a given item). '
                                                       'Optional, assume 100% if '
                                                       'misisng.',
                                        'name': 'probability'}}})

    item: Optional[str] = Field(default=None, description="""A DataItem that is a member of a Cluster.""", json_schema_extra = { "linkml_meta": {'alias': 'item', 'domain_of': ['ClusterMembership']} })
    cluster: Optional[str] = Field(default=None, description="""Cluster referenced in a membership association.""", json_schema_extra = { "linkml_meta": {'alias': 'cluster', 'domain_of': ['ClusterMembership']} })
    membership_score: Optional[float] = Field(default=None, description="""Algorithm-defined membership strength. (Optional, does not need to be normalized)""", json_schema_extra = { "linkml_meta": {'alias': 'membership_score', 'domain_of': ['ClusterMembership']} })
    probability: Optional[float] = Field(default=None, description="""Normalized probability of membership (sums to 1 across clusters for a given item). Optional, assume 100% if misisng.""", ge=0.0, le=1.0, json_schema_extra = { "linkml_meta": {'alias': 'probability',
         'domain_of': ['ClusterMembership',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping']} })
    distance: Optional[float] = Field(default=None, description="""Distance between the item and the cluster centroid. (Smaller is better).""", json_schema_extra = { "linkml_meta": {'alias': 'distance', 'domain_of': ['ClusterMembership']} })
    project_id: str = Field(default=..., description="""Identifier for the project or acquisition program context for this record.""", json_schema_extra = { "linkml_meta": {'alias': 'project_id',
         'aliases': ['project', 'program_id'],
         'domain_of': ['ProjectScoped']} })


class ZarrArray(ConfiguredBaseModel):
    """
    Reference to a zarr array containing matrix data with associated metadata.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-zarr-schema',
         'slot_usage': {'path': {'description': 'Path or URL to the zarr array. Must '
                                                'include s3://, gs://, https://, '
                                                'http://, or file:// prefix\n'
                                                'indicating protocol for remote or '
                                                'local storage. Examples:\n'
                                                '- '
                                                's3://my-bucket/data/connectivity.zarr\n'
                                                '- '
                                                'gs://my-project/data/connectivity.zarr  \n'
                                                '- '
                                                'https://example.com/data/connectivity.zarr\n'
                                                '- '
                                                'file:///local/path/connectivity.zarr',
                                 'name': 'path',
                                 'pattern': '^(s3://|gs://|https?://|file://).+',
                                 'range': 'string',
                                 'required': True}}})

    id: str = Field(default=..., description="""Unique identifier within the class context.""", json_schema_extra = { "linkml_meta": {'alias': 'id',
         'aliases': ['identifier', 'structure_id', 'brain_region_id'],
         'domain_of': ['DataSet',
                       'DataItem',
                       'AlgorithmRun',
                       'ClusterHierarchy',
                       'Cluster',
                       'BrainRegion',
                       'ZarrArray',
                       'ZarrDataset',
                       'ParquetDataset',
                       'ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'CellFeatureMatrix',
                       'CellFeatureMeasurement',
                       'CellGeneData',
                       'SingleCellReconstruction',
                       'MappingSet',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    path: str = Field(default=..., description="""Path or URL to the zarr array. Must include s3://, gs://, https://, http://, or file:// prefix
indicating protocol for remote or local storage. Examples:
- s3://my-bucket/data/connectivity.zarr
- gs://my-project/data/connectivity.zarr  
- https://example.com/data/connectivity.zarr
- file:///local/path/connectivity.zarr""", json_schema_extra = { "linkml_meta": {'alias': 'path', 'domain_of': ['ZarrArray', 'ZarrDataset', 'ParquetDataset']} })

    @field_validator('path')
    def pattern_path(cls, v):
        pattern=re.compile(r"^(s3://|gs://|https?://|file://).+")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid path format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid path format: {v}"
            raise ValueError(err_msg)
        return v


class ZarrDataset(ConfiguredBaseModel):
    """
    A collection of named zarr arrays stored under a common path (group or store).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-zarr-schema',
         'slot_usage': {'path': {'description': 'Path or URL to the zarr dataset root. '
                                                'Must include s3://, gs://, https://, '
                                                'http://, or file:// prefix.',
                                 'name': 'path',
                                 'pattern': '^(s3://|gs://|https?://|file://).+',
                                 'range': 'string',
                                 'required': True}}})

    id: str = Field(default=..., description="""Unique identifier within the class context.""", json_schema_extra = { "linkml_meta": {'alias': 'id',
         'aliases': ['identifier', 'structure_id', 'brain_region_id'],
         'domain_of': ['DataSet',
                       'DataItem',
                       'AlgorithmRun',
                       'ClusterHierarchy',
                       'Cluster',
                       'BrainRegion',
                       'ZarrArray',
                       'ZarrDataset',
                       'ParquetDataset',
                       'ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'CellFeatureMatrix',
                       'CellFeatureMeasurement',
                       'CellGeneData',
                       'SingleCellReconstruction',
                       'MappingSet',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    path: str = Field(default=..., description="""Path or URL to the zarr dataset root. Must include s3://, gs://, https://, http://, or file:// prefix.""", json_schema_extra = { "linkml_meta": {'alias': 'path', 'domain_of': ['ZarrArray', 'ZarrDataset', 'ParquetDataset']} })

    @field_validator('path')
    def pattern_path(cls, v):
        pattern=re.compile(r"^(s3://|gs://|https?://|file://).+")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid path format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid path format: {v}"
            raise ValueError(err_msg)
        return v


class ParquetDataset(ConfiguredBaseModel):
    """
    A dataset of parquet files (possibly partitioned) accessible via a URL or directory.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-zarr-schema',
         'slot_usage': {'path': {'description': 'Path or URL to the parquet dataset '
                                                'root (directory or file). Supports '
                                                'partitioned layouts.',
                                 'name': 'path',
                                 'pattern': '^(s3://|gs://|https?://|file://).+',
                                 'range': 'string',
                                 'required': True}}})

    id: str = Field(default=..., description="""Unique identifier within the class context.""", json_schema_extra = { "linkml_meta": {'alias': 'id',
         'aliases': ['identifier', 'structure_id', 'brain_region_id'],
         'domain_of': ['DataSet',
                       'DataItem',
                       'AlgorithmRun',
                       'ClusterHierarchy',
                       'Cluster',
                       'BrainRegion',
                       'ZarrArray',
                       'ZarrDataset',
                       'ParquetDataset',
                       'ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'CellFeatureMatrix',
                       'CellFeatureMeasurement',
                       'CellGeneData',
                       'SingleCellReconstruction',
                       'MappingSet',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    path: str = Field(default=..., description="""Path or URL to the parquet dataset root (directory or file). Supports partitioned layouts.""", json_schema_extra = { "linkml_meta": {'alias': 'path', 'domain_of': ['ZarrArray', 'ZarrDataset', 'ParquetDataset']} })

    @field_validator('path')
    def pattern_path(cls, v):
        pattern=re.compile(r"^(s3://|gs://|https?://|file://).+")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid path format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid path format: {v}"
            raise ValueError(err_msg)
        return v


class ProjectionMeasurementMatrix(ConfiguredBaseModel):
    """
    Aggregated projection measurements for a cohort (e.g., all cells) for a single measurement type. The rows of this matrix could be a set of cells, set of injections. The columns will be how those cells/injections distribute themselves across brain regions. This could mean for single cell reconstructions the outputs of axons. For rabies injections they could be distribution of input cells. For antereograde injections they could be fluorescence intensity of outputs For retrograde injections they could be cell body counts of inputs.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-measurement-schema',
         'slot_usage': {'data_item_index': {'description': 'Ordered data items '
                                                           'defining rows (or columns) '
                                                           'of the matrix.',
                                            'inlined': False,
                                            'multivalued': True,
                                            'name': 'data_item_index',
                                            'range': 'DataItem'},
                        'measurement_type': {'name': 'measurement_type',
                                             'range': 'ProjectionMeasurementType'},
                        'modality': {'name': 'modality', 'range': 'Modality'},
                        'region_index': {'description': 'Ordered regions defining '
                                                        'columns (or rows) of the '
                                                        'matrix.',
                                         'inlined': False,
                                         'multivalued': True,
                                         'name': 'region_index',
                                         'range': 'BrainRegion'},
                        'unit': {'name': 'unit', 'range': 'Unit'},
                        'values': {'description': 'Zarr array containing matrix values '
                                                  'with shape (data_item_index x '
                                                  'region_index).',
                                   'name': 'values',
                                   'range': 'ZarrArray'}}})

    id: str = Field(default=..., description="""Unique identifier within the class context.""", json_schema_extra = { "linkml_meta": {'alias': 'id',
         'aliases': ['identifier', 'structure_id', 'brain_region_id'],
         'domain_of': ['DataSet',
                       'DataItem',
                       'AlgorithmRun',
                       'ClusterHierarchy',
                       'Cluster',
                       'BrainRegion',
                       'ZarrArray',
                       'ZarrDataset',
                       'ParquetDataset',
                       'ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'CellFeatureMatrix',
                       'CellFeatureMeasurement',
                       'CellGeneData',
                       'SingleCellReconstruction',
                       'MappingSet',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    description: Optional[str] = Field(default=None, description="""Free-text human-readable description.""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'MappingSet',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    measurement_type: Optional[ProjectionMeasurementType] = Field(default=None, description="""The specific projection measurement type (enum) for this set.""", json_schema_extra = { "linkml_meta": {'alias': 'measurement_type',
         'domain_of': ['ProjectionMeasurementMatrix',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    modality: Optional[Modality] = Field(default=None, description="""Source modality for the data item (if relevant).""", json_schema_extra = { "linkml_meta": {'alias': 'modality',
         'domain_of': ['DataSet',
                       'ProjectionMeasurementMatrix',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    region_index: Optional[list[str]] = Field(default=None, description="""Ordered regions defining columns (or rows) of the matrix.""", json_schema_extra = { "linkml_meta": {'alias': 'region_index', 'domain_of': ['ProjectionMeasurementMatrix']} })
    data_item_index: Optional[list[str]] = Field(default=None, description="""Ordered data items defining rows (or columns) of the matrix.""", json_schema_extra = { "linkml_meta": {'alias': 'data_item_index', 'domain_of': ['ProjectionMeasurementMatrix']} })
    values: Optional[str] = Field(default=None, description="""Zarr array containing matrix values with shape (data_item_index x region_index).""", json_schema_extra = { "linkml_meta": {'alias': 'values',
         'domain_of': ['ProjectionMeasurementMatrix', 'CellCellMeasurementMatrix']} })
    unit: Optional[Unit] = Field(default=None, description="""Unit of measure for values.""", json_schema_extra = { "linkml_meta": {'alias': 'unit',
         'domain_of': ['ProjectionMeasurementMatrix',
                       'CellFeatureDefinition',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })


class CellFeatureSet(ConfiguredBaseModel):
    """
    A defined set of cell features with their descriptions and metadata.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-morphology-features-schema',
         'slot_usage': {'description': {'description': 'Longer human description of '
                                                       'what this feature set measures '
                                                       'and where it came from.',
                                        'name': 'description',
                                        'range': 'string'},
                        'extraction_method': {'description': 'Method used to extract '
                                                             'these features (e.g., '
                                                             "'L-Measure', "
                                                             "'NeuroMorpho', "
                                                             "'custom').",
                                              'name': 'extraction_method',
                                              'range': 'string'},
                        'feature_definition_ids': {'description': 'Individual feature '
                                                                  'definitions within '
                                                                  'this set.',
                                                   'multivalued': True,
                                                   'name': 'feature_definition_ids',
                                                   'range': 'CellFeatureDefinition'},
                        'id': {'description': 'Human-readable short name for this '
                                              "feature set (e.g., 'AllenFeatureSet1', "
                                              "'NeuroMorpho', 'AuthorYearSet').",
                               'name': 'id',
                               'range': 'string',
                               'required': True}}})

    id: str = Field(default=..., description="""Human-readable short name for this feature set (e.g., 'AllenFeatureSet1', 'NeuroMorpho', 'AuthorYearSet').""", json_schema_extra = { "linkml_meta": {'alias': 'id',
         'aliases': ['identifier', 'structure_id', 'brain_region_id'],
         'domain_of': ['DataSet',
                       'DataItem',
                       'AlgorithmRun',
                       'ClusterHierarchy',
                       'Cluster',
                       'BrainRegion',
                       'ZarrArray',
                       'ZarrDataset',
                       'ParquetDataset',
                       'ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'CellFeatureMatrix',
                       'CellFeatureMeasurement',
                       'CellGeneData',
                       'SingleCellReconstruction',
                       'MappingSet',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    description: Optional[str] = Field(default=None, description="""Longer human description of what this feature set measures and where it came from.""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'MappingSet',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    feature_definition_ids: Optional[list[str]] = Field(default=None, description="""Individual feature definitions within this set.""", json_schema_extra = { "linkml_meta": {'alias': 'feature_definition_ids', 'domain_of': ['CellFeatureSet']} })
    extraction_method: Optional[str] = Field(default=None, description="""Method used to extract these features (e.g., 'L-Measure', 'NeuroMorpho', 'custom').""", json_schema_extra = { "linkml_meta": {'alias': 'extraction_method', 'domain_of': ['CellFeatureSet']} })


class CellFeatureDefinition(ConfiguredBaseModel):
    """
    Definition of a single feature with metadata.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-morphology-features-schema',
         'slot_usage': {'data_type': {'description': 'Data type as NumPy typestr '
                                                     '(byteorder + code + bytes), '
                                                     "e.g., '<i2', '<f4', '|u1'.",
                                      'name': 'data_type',
                                      'pattern': '^([<>|=])[tbiufcmMOSUV]\\d+$',
                                      'range': 'string'},
                        'description': {'description': 'Detailed description of what '
                                                       'this feature measures.',
                                        'name': 'description',
                                        'range': 'string'},
                        'range_max': {'description': 'Expected maximum value for this '
                                                     'feature.',
                                      'name': 'range_max',
                                      'range': 'float'},
                        'range_min': {'description': 'Expected minimum value for this '
                                                     'feature.',
                                      'name': 'range_min',
                                      'range': 'float'},
                        'unit': {'description': 'Unit of measurement (e.g., '
                                                "'micrometers', 'degrees', 'count').",
                                 'name': 'unit',
                                 'range': 'string'}}})

    id: str = Field(default=..., description="""Unique identifier within the class context.""", json_schema_extra = { "linkml_meta": {'alias': 'id',
         'aliases': ['identifier', 'structure_id', 'brain_region_id'],
         'domain_of': ['DataSet',
                       'DataItem',
                       'AlgorithmRun',
                       'ClusterHierarchy',
                       'Cluster',
                       'BrainRegion',
                       'ZarrArray',
                       'ZarrDataset',
                       'ParquetDataset',
                       'ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'CellFeatureMatrix',
                       'CellFeatureMeasurement',
                       'CellGeneData',
                       'SingleCellReconstruction',
                       'MappingSet',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    description: Optional[str] = Field(default=None, description="""Detailed description of what this feature measures.""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'MappingSet',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    unit: Optional[str] = Field(default=None, description="""Unit of measurement (e.g., 'micrometers', 'degrees', 'count').""", json_schema_extra = { "linkml_meta": {'alias': 'unit',
         'domain_of': ['ProjectionMeasurementMatrix',
                       'CellFeatureDefinition',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    data_type: Optional[str] = Field(default=None, description="""Data type as NumPy typestr (byteorder + code + bytes), e.g., '<i2', '<f4', '|u1'.""", json_schema_extra = { "linkml_meta": {'alias': 'data_type', 'domain_of': ['CellFeatureDefinition']} })
    range_min: Optional[float] = Field(default=None, description="""Expected minimum value for this feature.""", json_schema_extra = { "linkml_meta": {'alias': 'range_min', 'domain_of': ['CellFeatureDefinition']} })
    range_max: Optional[float] = Field(default=None, description="""Expected maximum value for this feature.""", json_schema_extra = { "linkml_meta": {'alias': 'range_max', 'domain_of': ['CellFeatureDefinition']} })

    @field_validator('data_type')
    def pattern_data_type(cls, v):
        pattern=re.compile(r"^([<>|=])[tbiufcmMOSUV]\d+$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid data_type format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid data_type format: {v}"
            raise ValueError(err_msg)
        return v


class CellFeatureMatrix(ProjectScoped):
    """
    Pointer to a Wide form measurement matrix of feature values for a particular FeatureSet in Parquet format.
    One column (cell_index_column) should be the DataItemId and the rest of columns of this matrix should be named according to the CellFeatureDefinition in the CellFeatureSet.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-morphology-features-schema',
         'mixins': ['ProjectScoped'],
         'slot_usage': {'cell_index_column': {'description': 'Column of the parquet '
                                                             'which corresponds to the '
                                                             'DataItem',
                                              'name': 'cell_index_column',
                                              'range': 'string'},
                        'feature_set_id': {'description': 'Reference to the '
                                                          'CellFeatureSet that defines '
                                                          'the features in this '
                                                          'matrix.',
                                           'name': 'feature_set_id',
                                           'range': 'CellFeatureSet',
                                           'required': True},
                        'parquet_path': {'description': 'Path to parquet dataset '
                                                        'containing wide-form data. '
                                                        'Columns should be named the '
                                                        'id of a CellFeatureDefinition '
                                                        'in the CellFeatureSet.',
                                         'name': 'parquet_path',
                                         'range': 'ParquetDataset'}}})

    id: str = Field(default=..., description="""Unique identifier within the class context.""", json_schema_extra = { "linkml_meta": {'alias': 'id',
         'aliases': ['identifier', 'structure_id', 'brain_region_id'],
         'domain_of': ['DataSet',
                       'DataItem',
                       'AlgorithmRun',
                       'ClusterHierarchy',
                       'Cluster',
                       'BrainRegion',
                       'ZarrArray',
                       'ZarrDataset',
                       'ParquetDataset',
                       'ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'CellFeatureMatrix',
                       'CellFeatureMeasurement',
                       'CellGeneData',
                       'SingleCellReconstruction',
                       'MappingSet',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    feature_set_id: str = Field(default=..., description="""Reference to the CellFeatureSet that defines the features in this matrix.""", json_schema_extra = { "linkml_meta": {'alias': 'feature_set_id', 'domain_of': ['CellFeatureMatrix']} })
    parquet_path: Optional[str] = Field(default=None, description="""Path to parquet dataset containing wide-form data. Columns should be named the id of a CellFeatureDefinition in the CellFeatureSet.""", json_schema_extra = { "linkml_meta": {'alias': 'parquet_path', 'domain_of': ['CellFeatureMatrix']} })
    cell_index_column: Optional[str] = Field(default=None, description="""Column of the parquet which corresponds to the DataItem""", json_schema_extra = { "linkml_meta": {'alias': 'cell_index_column', 'domain_of': ['CellFeatureMatrix']} })
    project_id: str = Field(default=..., description="""Identifier for the project or acquisition program context for this record.""", json_schema_extra = { "linkml_meta": {'alias': 'project_id',
         'aliases': ['project', 'program_id'],
         'domain_of': ['ProjectScoped']} })

    @field_validator('parquet_path')
    def pattern_parquet_path(cls, v):
        pattern=re.compile(r"^(s3://|gs://|https?://|file://).+")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid parquet_path format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid parquet_path format: {v}"
            raise ValueError(err_msg)
        return v


class CellFeatureMeasurement(ConfiguredBaseModel):
    """
    Long-form measurement row: one (cell, feature) value with strict dtype.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-morphology-features-schema',
         'slot_usage': {'dataitem_id': {'name': 'dataitem_id',
                                        'range': 'DataItem',
                                        'required': True},
                        'dtype': {'description': 'NumPy typestr of the stored value '
                                                 '(see arrays.interface).',
                                  'name': 'dtype',
                                  'pattern': '^([<>|=])[tbiufcmMOSUV]\\\\d+$',
                                  'range': 'string'},
                        'feature_id': {'name': 'feature_id',
                                       'range': 'CellFeatureDefinition',
                                       'required': True},
                        'feature_set_id': {'description': 'Denormalized reference to '
                                                          'the feature set (helps '
                                                          'partitioning and joins).',
                                           'name': 'feature_set_id',
                                           'range': 'CellFeatureSet'},
                        'unit': {'description': 'Unit of measurement for the values in '
                                                'this matrix.',
                                 'name': 'unit',
                                 'range': 'Unit'},
                        'value_bool': {'name': 'value_bool', 'range': 'boolean'},
                        'value_bytes': {'description': 'Base64-encoded bytes when '
                                                       'binary values are needed.',
                                        'name': 'value_bytes',
                                        'range': 'string'},
                        'value_datetime': {'description': 'ISO 8601 timestamp when '
                                                          'dtype corresponds to '
                                                          'datetime.',
                                           'name': 'value_datetime',
                                           'range': 'datetime'},
                        'value_float': {'name': 'value_float', 'range': 'float'},
                        'value_int': {'name': 'value_int', 'range': 'integer'},
                        'value_string': {'name': 'value_string', 'range': 'string'}}})

    id: str = Field(default=..., description="""Unique identifier within the class context.""", json_schema_extra = { "linkml_meta": {'alias': 'id',
         'aliases': ['identifier', 'structure_id', 'brain_region_id'],
         'domain_of': ['DataSet',
                       'DataItem',
                       'AlgorithmRun',
                       'ClusterHierarchy',
                       'Cluster',
                       'BrainRegion',
                       'ZarrArray',
                       'ZarrDataset',
                       'ParquetDataset',
                       'ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'CellFeatureMatrix',
                       'CellFeatureMeasurement',
                       'CellGeneData',
                       'SingleCellReconstruction',
                       'MappingSet',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    dataitem_id: str = Field(default=..., description="""The DataItem for which projection measurements are reported.""", json_schema_extra = { "linkml_meta": {'alias': 'dataitem_id',
         'domain_of': ['DataItemDataSetAssociation',
                       'CellFeatureMeasurement',
                       'CellGeneData',
                       'SingleCellReconstruction']} })
    feature_id: str = Field(default=..., description="""Reference to a feature definition used for a measurement.""", json_schema_extra = { "linkml_meta": {'alias': 'feature_id', 'domain_of': ['CellFeatureMeasurement']} })
    dtype: Optional[str] = Field(default=None, description="""NumPy typestr of the stored value (see arrays.interface).""", json_schema_extra = { "linkml_meta": {'alias': 'dtype', 'domain_of': ['CellFeatureMeasurement']} })
    value_float: Optional[float] = Field(default=None, description="""Floating point value for a (cell, feature) measurement.""", json_schema_extra = { "linkml_meta": {'alias': 'value_float', 'domain_of': ['CellFeatureMeasurement']} })
    value_int: Optional[int] = Field(default=None, description="""Integer value for a (cell, feature) measurement.""", json_schema_extra = { "linkml_meta": {'alias': 'value_int', 'domain_of': ['CellFeatureMeasurement']} })
    value_bool: Optional[bool] = Field(default=None, description="""Boolean value for a (cell, feature) measurement.""", json_schema_extra = { "linkml_meta": {'alias': 'value_bool', 'domain_of': ['CellFeatureMeasurement']} })
    value_string: Optional[str] = Field(default=None, description="""String value for a (cell, feature) measurement.""", json_schema_extra = { "linkml_meta": {'alias': 'value_string', 'domain_of': ['CellFeatureMeasurement']} })
    value_bytes: Optional[str] = Field(default=None, description="""Base64-encoded bytes when binary values are needed.""", json_schema_extra = { "linkml_meta": {'alias': 'value_bytes', 'domain_of': ['CellFeatureMeasurement']} })
    value_datetime: Optional[datetime ] = Field(default=None, description="""ISO 8601 timestamp when dtype corresponds to datetime.""", json_schema_extra = { "linkml_meta": {'alias': 'value_datetime', 'domain_of': ['CellFeatureMeasurement']} })

    @field_validator('dtype')
    def pattern_dtype(cls, v):
        pattern=re.compile(r"^([<>|=])[tbiufcmMOSUV]\\d+$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid dtype format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid dtype format: {v}"
            raise ValueError(err_msg)
        return v


class CellGeneData(ConfiguredBaseModel):
    """
    Cell x gene expression data from barcoding experiments.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-cell-gene-schema',
         'slot_usage': {'cell_gene_matrix': {'description': 'Zarr array containing the '
                                                            'cell x gene expression '
                                                            'matrix.',
                                             'name': 'cell_gene_matrix',
                                             'range': 'ZarrArray',
                                             'required': True},
                        'cell_index': {'description': 'index of cells.',
                                       'multivalued': True,
                                       'name': 'cell_index',
                                       'range': 'DataItem'},
                        'dataitem_id': {'description': 'Reference to the core DataItem '
                                                       'this expression data belongs '
                                                       'to.',
                                        'name': 'dataitem_id',
                                        'range': 'DataItem',
                                        'required': True},
                        'experiment_metadata': {'description': 'Metadata about the '
                                                               'barcoding experiment.',
                                                'name': 'experiment_metadata',
                                                'range': 'BarcodingExperimentMetadata'},
                        'gene_metadata': {'description': 'Zarr array containing '
                                                         'gene-level metadata (gene '
                                                         'symbols, types, etc.).',
                                          'name': 'gene_metadata',
                                          'range': 'ZarrArray'}}})

    id: str = Field(default=..., description="""Unique identifier within the class context.""", json_schema_extra = { "linkml_meta": {'alias': 'id',
         'aliases': ['identifier', 'structure_id', 'brain_region_id'],
         'domain_of': ['DataSet',
                       'DataItem',
                       'AlgorithmRun',
                       'ClusterHierarchy',
                       'Cluster',
                       'BrainRegion',
                       'ZarrArray',
                       'ZarrDataset',
                       'ParquetDataset',
                       'ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'CellFeatureMatrix',
                       'CellFeatureMeasurement',
                       'CellGeneData',
                       'SingleCellReconstruction',
                       'MappingSet',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    dataitem_id: str = Field(default=..., description="""Reference to the core DataItem this expression data belongs to.""", json_schema_extra = { "linkml_meta": {'alias': 'dataitem_id',
         'domain_of': ['DataItemDataSetAssociation',
                       'CellFeatureMeasurement',
                       'CellGeneData',
                       'SingleCellReconstruction']} })
    cell_index: Optional[list[str]] = Field(default=None, description="""index of cells.""", json_schema_extra = { "linkml_meta": {'alias': 'cell_index', 'domain_of': ['CellGeneData']} })
    cell_gene_matrix: str = Field(default=..., description="""Zarr array containing the cell x gene expression matrix.""", json_schema_extra = { "linkml_meta": {'alias': 'cell_gene_matrix', 'domain_of': ['CellGeneData']} })
    gene_metadata: Optional[str] = Field(default=None, description="""Zarr array containing gene-level metadata (gene symbols, types, etc.).""", json_schema_extra = { "linkml_meta": {'alias': 'gene_metadata', 'domain_of': ['CellGeneData']} })
    experiment_metadata: Optional[BarcodingExperimentMetadata] = Field(default=None, description="""Metadata about the barcoding experiment.""", json_schema_extra = { "linkml_meta": {'alias': 'experiment_metadata', 'domain_of': ['CellGeneData']} })


class BarcodingExperimentMetadata(ConfiguredBaseModel):
    """
    Metadata about the barcoding experiment and data processing.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-cell-gene-schema',
         'slot_usage': {'data_processing_pipeline': {'description': 'Pipeline used for '
                                                                    'data processing.',
                                                     'name': 'data_processing_pipeline',
                                                     'range': 'string'},
                        'experiment_type': {'description': 'Type of barcoding '
                                                           'experiment (e.g., '
                                                           "'BarSeq', 'PathSeq', "
                                                           "'MERFISH').",
                                            'name': 'experiment_type',
                                            'range': 'string',
                                            'required': True},
                        'sequencing_platform': {'description': 'Sequencing platform '
                                                               'used (e.g., '
                                                               "'Illumina', 'PacBio').",
                                                'name': 'sequencing_platform',
                                                'range': 'string'}}})

    experiment_type: str = Field(default=..., description="""Type of barcoding experiment (e.g., 'BarSeq', 'PathSeq', 'MERFISH').""", json_schema_extra = { "linkml_meta": {'alias': 'experiment_type', 'domain_of': ['BarcodingExperimentMetadata']} })
    sequencing_platform: Optional[str] = Field(default=None, description="""Sequencing platform used (e.g., 'Illumina', 'PacBio').""", json_schema_extra = { "linkml_meta": {'alias': 'sequencing_platform', 'domain_of': ['BarcodingExperimentMetadata']} })
    data_processing_pipeline: Optional[str] = Field(default=None, description="""Pipeline used for data processing.""", json_schema_extra = { "linkml_meta": {'alias': 'data_processing_pipeline',
         'domain_of': ['BarcodingExperimentMetadata']} })
    normalization_method: Optional[str] = Field(default=None, description="""Normalization method applied to the data.""", json_schema_extra = { "linkml_meta": {'alias': 'normalization_method', 'domain_of': ['BarcodingExperimentMetadata']} })


class GeneMetadata(ConfiguredBaseModel):
    """
    Metadata about genes in the expression matrix.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-cell-gene-schema',
         'slot_usage': {'gene_id': {'description': 'Unique gene identifier.',
                                    'name': 'gene_id',
                                    'range': 'string',
                                    'required': True},
                        'gene_symbol': {'description': 'Gene symbol or name.',
                                        'name': 'gene_symbol',
                                        'range': 'string'}}})

    gene_id: str = Field(default=..., description="""Unique gene identifier.""", json_schema_extra = { "linkml_meta": {'alias': 'gene_id', 'domain_of': ['GeneMetadata']} })
    gene_symbol: Optional[str] = Field(default=None, description="""Gene symbol or name.""", json_schema_extra = { "linkml_meta": {'alias': 'gene_symbol', 'domain_of': ['GeneMetadata']} })


class CellMetadata(ConfiguredBaseModel):
    """
    Metadata about individual cells in the expression matrix.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-cell-gene-schema',
         'slot_usage': {'cell_id': {'description': 'Unique cell identifier.',
                                    'name': 'cell_id',
                                    'range': 'string',
                                    'required': True},
                        'n_genes_detected': {'description': 'Number of genes detected '
                                                            'in this cell.',
                                             'name': 'n_genes_detected',
                                             'range': 'integer'},
                        'quality_score': {'description': 'Overall quality score for '
                                                         'the cell.',
                                          'name': 'quality_score',
                                          'range': 'float'},
                        'spatial_location': {'description': '3D spatial coordinates of '
                                                            'the cell.',
                                             'name': 'spatial_location',
                                             'range': 'SpatialLocation'},
                        'total_counts': {'description': 'Total number of counts (UMIs) '
                                                        'for this cell.',
                                         'name': 'total_counts',
                                         'range': 'integer'}}})

    cell_id: str = Field(default=..., description="""Unique cell identifier.""", json_schema_extra = { "linkml_meta": {'alias': 'cell_id', 'domain_of': ['CellMetadata']} })
    spatial_location: Optional[SpatialLocation] = Field(default=None, description="""3D spatial coordinates of the cell.""", json_schema_extra = { "linkml_meta": {'alias': 'spatial_location', 'domain_of': ['CellMetadata']} })
    quality_score: Optional[float] = Field(default=None, description="""Overall quality score for the cell.""", json_schema_extra = { "linkml_meta": {'alias': 'quality_score', 'domain_of': ['CellMetadata']} })
    total_counts: Optional[int] = Field(default=None, description="""Total number of counts (UMIs) for this cell.""", json_schema_extra = { "linkml_meta": {'alias': 'total_counts', 'domain_of': ['CellMetadata']} })
    n_genes_detected: Optional[int] = Field(default=None, description="""Number of genes detected in this cell.""", json_schema_extra = { "linkml_meta": {'alias': 'n_genes_detected', 'domain_of': ['CellMetadata']} })


class SingleCellReconstruction(ConfiguredBaseModel):
    """
    Single cell reconstruction data with CCF registration and morphology features.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-single-cell-schema',
         'slot_usage': {'ccf_registered_file': {'description': 'Zarr array containing '
                                                               'the CCF-registered '
                                                               'reconstruction data.',
                                                'name': 'ccf_registered_file',
                                                'range': 'ZarrArray'},
                        'dataitem_id': {'description': 'Reference to the core DataItem '
                                                       'this reconstruction belongs '
                                                       'to.',
                                        'name': 'dataitem_id',
                                        'range': 'DataItem',
                                        'required': True},
                        'soma_location': {'description': '3D coordinates of the soma '
                                                         'in CCF space.',
                                          'name': 'soma_location',
                                          'range': 'SpatialLocation'}}})

    id: str = Field(default=..., description="""Unique identifier within the class context.""", json_schema_extra = { "linkml_meta": {'alias': 'id',
         'aliases': ['identifier', 'structure_id', 'brain_region_id'],
         'domain_of': ['DataSet',
                       'DataItem',
                       'AlgorithmRun',
                       'ClusterHierarchy',
                       'Cluster',
                       'BrainRegion',
                       'ZarrArray',
                       'ZarrDataset',
                       'ParquetDataset',
                       'ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'CellFeatureMatrix',
                       'CellFeatureMeasurement',
                       'CellGeneData',
                       'SingleCellReconstruction',
                       'MappingSet',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    dataitem_id: str = Field(default=..., description="""Reference to the core DataItem this reconstruction belongs to.""", json_schema_extra = { "linkml_meta": {'alias': 'dataitem_id',
         'domain_of': ['DataItemDataSetAssociation',
                       'CellFeatureMeasurement',
                       'CellGeneData',
                       'SingleCellReconstruction']} })
    ccf_registered_file: Optional[str] = Field(default=None, description="""Zarr array containing the CCF-registered reconstruction data.""", json_schema_extra = { "linkml_meta": {'alias': 'ccf_registered_file', 'domain_of': ['SingleCellReconstruction']} })
    soma_location: Optional[SpatialLocation] = Field(default=None, description="""3D coordinates of the soma in CCF space.""", json_schema_extra = { "linkml_meta": {'alias': 'soma_location', 'domain_of': ['SingleCellReconstruction']} })


class MappingSet(ProjectScoped):
    """
    A coherent set of mappings produced by a specific method between datasets.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-mappings-schema',
         'mixins': ['ProjectScoped'],
         'slot_usage': {'author': {'name': 'author', 'range': 'string'},
                        'created_at': {'name': 'created_at', 'range': 'datetime'},
                        'description': {'name': 'description', 'range': 'string'},
                        'json_object': {'description': 'Arbitrary method parameters '
                                                       '(JSON string).',
                                        'name': 'json_object'},
                        'method_name': {'name': 'method_name',
                                        'range': 'string',
                                        'required': True},
                        'method_version': {'name': 'method_version', 'range': 'string'},
                        'name': {'name': 'name', 'range': 'string'},
                        'source_dataset': {'name': 'source_dataset',
                                           'range': 'DataSet',
                                           'required': True},
                        'target_dataset': {'name': 'target_dataset',
                                           'range': 'DataSet',
                                           'required': True}}})

    id: str = Field(default=..., description="""Unique identifier within the class context.""", json_schema_extra = { "linkml_meta": {'alias': 'id',
         'aliases': ['identifier', 'structure_id', 'brain_region_id'],
         'domain_of': ['DataSet',
                       'DataItem',
                       'AlgorithmRun',
                       'ClusterHierarchy',
                       'Cluster',
                       'BrainRegion',
                       'ZarrArray',
                       'ZarrDataset',
                       'ParquetDataset',
                       'ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'CellFeatureMatrix',
                       'CellFeatureMeasurement',
                       'CellGeneData',
                       'SingleCellReconstruction',
                       'MappingSet',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    name: str = Field(default=..., description="""A human-readable name or title.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'aliases': ['structure_name', 'region_name'],
         'domain_of': ['DataSet', 'DataItem', 'BrainRegion', 'MappingSet']} })
    description: Optional[str] = Field(default=None, description="""Free-text human-readable description.""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'MappingSet',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    method_name: str = Field(default=..., description="""Name of the mapping method.""", json_schema_extra = { "linkml_meta": {'alias': 'method_name', 'domain_of': ['MappingSet']} })
    method_version: Optional[str] = Field(default=None, description="""Version of the mapping method.""", json_schema_extra = { "linkml_meta": {'alias': 'method_version', 'domain_of': ['MappingSet']} })
    author: Optional[str] = Field(default=None, description="""Author or organization who produced this mapping.""", json_schema_extra = { "linkml_meta": {'alias': 'author', 'domain_of': ['MappingSet']} })
    created_at: Optional[datetime ] = Field(default=None, description="""Timestamp when the mapping set was created.""", json_schema_extra = { "linkml_meta": {'alias': 'created_at', 'domain_of': ['MappingSet']} })
    source_dataset: str = Field(default=..., description="""Dataset providing the source entities.""", json_schema_extra = { "linkml_meta": {'alias': 'source_dataset', 'domain_of': ['MappingSet']} })
    target_dataset: str = Field(default=..., description="""Dataset providing the target entities.""", json_schema_extra = { "linkml_meta": {'alias': 'target_dataset', 'domain_of': ['MappingSet']} })
    json_object: Optional[str] = Field(default=None, description="""Arbitrary method parameters (JSON string).""", json_schema_extra = { "linkml_meta": {'alias': 'json_object', 'domain_of': ['AlgorithmRun', 'MappingSet']} })
    project_id: str = Field(default=..., description="""Identifier for the project or acquisition program context for this record.""", json_schema_extra = { "linkml_meta": {'alias': 'project_id',
         'aliases': ['project', 'program_id'],
         'domain_of': ['ProjectScoped']} })

    @field_validator('json_object')
    def pattern_json_object(cls, v):
        pattern=re.compile(r"^\s*\{.*\}\s*$")
        if isinstance(v, list):
            for element in v:
                if isinstance(element, str) and not pattern.match(element):
                    err_msg = f"Invalid json_object format: {element}"
                    raise ValueError(err_msg)
        elif isinstance(v, str) and not pattern.match(v):
            err_msg = f"Invalid json_object format: {v}"
            raise ValueError(err_msg)
        return v


class CellToCellMapping(ProjectScoped):
    """
    Mapping between a source cell and a target cell, with scores/probabilities.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-mappings-schema',
         'mixins': ['ProjectScoped'],
         'slot_usage': {'mapping_set': {'name': 'mapping_set',
                                        'range': 'MappingSet',
                                        'required': True},
                        'notes': {'name': 'notes', 'range': 'string'},
                        'probability': {'description': 'Normalized probability (0..1) '
                                                       'when applicable.',
                                        'name': 'probability'},
                        'score': {'description': 'Confidence or similarity score '
                                                 '(algorithm-defined).',
                                  'name': 'score',
                                  'range': 'float'},
                        'source_cell': {'name': 'source_cell',
                                        'range': 'DataItem',
                                        'required': True},
                        'target_cell': {'name': 'target_cell',
                                        'range': 'DataItem',
                                        'required': True}}})

    id: str = Field(default=..., description="""Unique identifier within the class context.""", json_schema_extra = { "linkml_meta": {'alias': 'id',
         'aliases': ['identifier', 'structure_id', 'brain_region_id'],
         'domain_of': ['DataSet',
                       'DataItem',
                       'AlgorithmRun',
                       'ClusterHierarchy',
                       'Cluster',
                       'BrainRegion',
                       'ZarrArray',
                       'ZarrDataset',
                       'ParquetDataset',
                       'ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'CellFeatureMatrix',
                       'CellFeatureMeasurement',
                       'CellGeneData',
                       'SingleCellReconstruction',
                       'MappingSet',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    mapping_set: str = Field(default=..., description="""The mapping set this entry belongs to.""", json_schema_extra = { "linkml_meta": {'alias': 'mapping_set',
         'domain_of': ['CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping']} })
    source_cell: str = Field(default=..., description="""Source cell (DataItem) in the mapping.""", json_schema_extra = { "linkml_meta": {'alias': 'source_cell',
         'domain_of': ['CellToCellMapping', 'CellToClusterMapping']} })
    target_cell: str = Field(default=..., description="""Target cell (DataItem) in the mapping.""", json_schema_extra = { "linkml_meta": {'alias': 'target_cell', 'domain_of': ['CellToCellMapping']} })
    score: Optional[float] = Field(default=None, description="""Confidence or similarity score (algorithm-defined).""", json_schema_extra = { "linkml_meta": {'alias': 'score',
         'domain_of': ['Cluster',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping']} })
    probability: Optional[float] = Field(default=None, description="""Normalized probability (0..1) when applicable.""", ge=0.0, le=1.0, json_schema_extra = { "linkml_meta": {'alias': 'probability',
         'domain_of': ['ClusterMembership',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping']} })
    notes: Optional[str] = Field(default=None, description="""Free-text notes about this mapping entry.""", json_schema_extra = { "linkml_meta": {'alias': 'notes',
         'domain_of': ['CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping']} })
    project_id: str = Field(default=..., description="""Identifier for the project or acquisition program context for this record.""", json_schema_extra = { "linkml_meta": {'alias': 'project_id',
         'aliases': ['project', 'program_id'],
         'domain_of': ['ProjectScoped']} })


class CellToClusterMapping(ProjectScoped):
    """
    Mapping between a source cell and a target cluster, with scores/probabilities.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-mappings-schema',
         'mixins': ['ProjectScoped'],
         'slot_usage': {'mapping_set': {'name': 'mapping_set',
                                        'range': 'MappingSet',
                                        'required': True},
                        'notes': {'name': 'notes', 'range': 'string'},
                        'score': {'name': 'score', 'range': 'float'},
                        'source_cell': {'name': 'source_cell',
                                        'range': 'DataItem',
                                        'required': True},
                        'target_cluster': {'name': 'target_cluster',
                                           'range': 'Cluster',
                                           'required': True}}})

    id: str = Field(default=..., description="""Unique identifier within the class context.""", json_schema_extra = { "linkml_meta": {'alias': 'id',
         'aliases': ['identifier', 'structure_id', 'brain_region_id'],
         'domain_of': ['DataSet',
                       'DataItem',
                       'AlgorithmRun',
                       'ClusterHierarchy',
                       'Cluster',
                       'BrainRegion',
                       'ZarrArray',
                       'ZarrDataset',
                       'ParquetDataset',
                       'ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'CellFeatureMatrix',
                       'CellFeatureMeasurement',
                       'CellGeneData',
                       'SingleCellReconstruction',
                       'MappingSet',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    mapping_set: str = Field(default=..., description="""The mapping set this entry belongs to.""", json_schema_extra = { "linkml_meta": {'alias': 'mapping_set',
         'domain_of': ['CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping']} })
    source_cell: str = Field(default=..., description="""Source cell (DataItem) in the mapping.""", json_schema_extra = { "linkml_meta": {'alias': 'source_cell',
         'domain_of': ['CellToCellMapping', 'CellToClusterMapping']} })
    target_cluster: str = Field(default=..., description="""Target cluster in the mapping.""", json_schema_extra = { "linkml_meta": {'alias': 'target_cluster',
         'domain_of': ['CellToClusterMapping', 'ClusterToClusterMapping']} })
    score: Optional[float] = Field(default=None, description="""Confidence or similarity score.""", json_schema_extra = { "linkml_meta": {'alias': 'score',
         'domain_of': ['Cluster',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping']} })
    probability: Optional[float] = Field(default=None, description="""Probability value between 0 and 1.""", ge=0.0, le=1.0, json_schema_extra = { "linkml_meta": {'alias': 'probability',
         'domain_of': ['ClusterMembership',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping']} })
    notes: Optional[str] = Field(default=None, description="""Free-text notes about this mapping entry.""", json_schema_extra = { "linkml_meta": {'alias': 'notes',
         'domain_of': ['CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping']} })
    project_id: str = Field(default=..., description="""Identifier for the project or acquisition program context for this record.""", json_schema_extra = { "linkml_meta": {'alias': 'project_id',
         'aliases': ['project', 'program_id'],
         'domain_of': ['ProjectScoped']} })


class ClusterToClusterMapping(ProjectScoped):
    """
    Mapping between a source cluster and a target cluster, with scores/probabilities.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-mappings-schema',
         'mixins': ['ProjectScoped'],
         'slot_usage': {'mapping_set': {'name': 'mapping_set',
                                        'range': 'MappingSet',
                                        'required': True},
                        'notes': {'name': 'notes', 'range': 'string'},
                        'score': {'name': 'score', 'range': 'float'},
                        'source_cluster': {'name': 'source_cluster',
                                           'range': 'Cluster',
                                           'required': True},
                        'target_cluster': {'name': 'target_cluster',
                                           'range': 'Cluster',
                                           'required': True}}})

    id: str = Field(default=..., description="""Unique identifier within the class context.""", json_schema_extra = { "linkml_meta": {'alias': 'id',
         'aliases': ['identifier', 'structure_id', 'brain_region_id'],
         'domain_of': ['DataSet',
                       'DataItem',
                       'AlgorithmRun',
                       'ClusterHierarchy',
                       'Cluster',
                       'BrainRegion',
                       'ZarrArray',
                       'ZarrDataset',
                       'ParquetDataset',
                       'ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'CellFeatureMatrix',
                       'CellFeatureMeasurement',
                       'CellGeneData',
                       'SingleCellReconstruction',
                       'MappingSet',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    mapping_set: str = Field(default=..., description="""The mapping set this entry belongs to.""", json_schema_extra = { "linkml_meta": {'alias': 'mapping_set',
         'domain_of': ['CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping']} })
    source_cluster: str = Field(default=..., description="""Source cluster in the mapping.""", json_schema_extra = { "linkml_meta": {'alias': 'source_cluster', 'domain_of': ['ClusterToClusterMapping']} })
    target_cluster: str = Field(default=..., description="""Target cluster in the mapping.""", json_schema_extra = { "linkml_meta": {'alias': 'target_cluster',
         'domain_of': ['CellToClusterMapping', 'ClusterToClusterMapping']} })
    score: Optional[float] = Field(default=None, description="""Confidence or similarity score.""", json_schema_extra = { "linkml_meta": {'alias': 'score',
         'domain_of': ['Cluster',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping']} })
    probability: Optional[float] = Field(default=None, description="""Probability value between 0 and 1.""", ge=0.0, le=1.0, json_schema_extra = { "linkml_meta": {'alias': 'probability',
         'domain_of': ['ClusterMembership',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping']} })
    notes: Optional[str] = Field(default=None, description="""Free-text notes about this mapping entry.""", json_schema_extra = { "linkml_meta": {'alias': 'notes',
         'domain_of': ['CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping']} })
    project_id: str = Field(default=..., description="""Identifier for the project or acquisition program context for this record.""", json_schema_extra = { "linkml_meta": {'alias': 'project_id',
         'aliases': ['project', 'program_id'],
         'domain_of': ['ProjectScoped']} })


class CellCellConnectivityLong(ProjectScoped):
    """
    Long-form connectivity measurements between pairs of cells.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-cell-cell-schema',
         'mixins': ['ProjectScoped'],
         'slot_usage': {'measurement_type': {'name': 'measurement_type',
                                             'range': 'SynapticMeasurementType'},
                        'modality': {'name': 'modality', 'range': 'Modality'},
                        'postsynaptic_cell': {'description': 'The postsynaptic cell '
                                                             'for this measurement.',
                                              'inlined': False,
                                              'name': 'postsynaptic_cell',
                                              'range': 'DataItem'},
                        'presynaptic_cell': {'description': 'The presynaptic cell for '
                                                            'this measurement.',
                                             'inlined': False,
                                             'name': 'presynaptic_cell',
                                             'range': 'DataItem'},
                        'unit': {'name': 'unit', 'range': 'Unit', 'required': True},
                        'value': {'description': 'Numeric value quantifying '
                                                 'connectivity between the presynaptic '
                                                 'and postsynaptic cell.',
                                  'name': 'value',
                                  'range': 'float',
                                  'required': True}}})

    id: str = Field(default=..., description="""Unique identifier within the class context.""", json_schema_extra = { "linkml_meta": {'alias': 'id',
         'aliases': ['identifier', 'structure_id', 'brain_region_id'],
         'domain_of': ['DataSet',
                       'DataItem',
                       'AlgorithmRun',
                       'ClusterHierarchy',
                       'Cluster',
                       'BrainRegion',
                       'ZarrArray',
                       'ZarrDataset',
                       'ParquetDataset',
                       'ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'CellFeatureMatrix',
                       'CellFeatureMeasurement',
                       'CellGeneData',
                       'SingleCellReconstruction',
                       'MappingSet',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    description: Optional[str] = Field(default=None, description="""Free-text human-readable description.""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'MappingSet',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    presynaptic_cell: Optional[str] = Field(default=None, description="""The presynaptic cell for this measurement.""", json_schema_extra = { "linkml_meta": {'alias': 'presynaptic_cell', 'domain_of': ['CellCellConnectivityLong']} })
    postsynaptic_cell: Optional[str] = Field(default=None, description="""The postsynaptic cell for this measurement.""", json_schema_extra = { "linkml_meta": {'alias': 'postsynaptic_cell', 'domain_of': ['CellCellConnectivityLong']} })
    measurement_type: Optional[SynapticMeasurementType] = Field(default=None, description="""The specific projection measurement type (enum) for this set.""", json_schema_extra = { "linkml_meta": {'alias': 'measurement_type',
         'domain_of': ['ProjectionMeasurementMatrix',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    modality: Optional[Modality] = Field(default=None, description="""Source modality for the data item (if relevant).""", json_schema_extra = { "linkml_meta": {'alias': 'modality',
         'domain_of': ['DataSet',
                       'ProjectionMeasurementMatrix',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    value: float = Field(default=..., description="""Numeric value quantifying connectivity between the presynaptic and postsynaptic cell.""", json_schema_extra = { "linkml_meta": {'alias': 'value', 'domain_of': ['CellCellConnectivityLong']} })
    unit: Unit = Field(default=..., description="""Unit of measure for values.""", json_schema_extra = { "linkml_meta": {'alias': 'unit',
         'domain_of': ['ProjectionMeasurementMatrix',
                       'CellFeatureDefinition',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    project_id: str = Field(default=..., description="""Identifier for the project or acquisition program context for this record.""", json_schema_extra = { "linkml_meta": {'alias': 'project_id',
         'aliases': ['project', 'program_id'],
         'domain_of': ['ProjectScoped']} })


class CellCellMeasurementMatrix(ProjectScoped):
    """
    Aggregated projection measurements for a cohort (e.g., all cells) for a single measurement type.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://brain-connects.org/ic3-cell-cell-schema',
         'mixins': ['ProjectScoped'],
         'slot_usage': {'description': {'description': 'Free-text description of what '
                                                       'this measurement matrix '
                                                       'represents.',
                                        'name': 'description',
                                        'range': 'string'},
                        'measurement_type': {'name': 'measurement_type',
                                             'range': 'SynapticMeasurementType'},
                        'modality': {'name': 'modality', 'range': 'Modality'},
                        'postsynaptic_index': {'description': 'Ordered data items '
                                                              'defining columns of the '
                                                              'matrix, where each '
                                                              'column is a '
                                                              'postsynaptic data item '
                                                              '(cell, region, etc).',
                                               'inlined': False,
                                               'multivalued': True,
                                               'name': 'postsynaptic_index',
                                               'range': 'DataItem'},
                        'presynaptic_index': {'description': 'Ordered data items '
                                                             'defining rows of the '
                                                             'matrix, where each row '
                                                             'is a presynpatic data '
                                                             'item (cell, injection '
                                                             'location, etc).',
                                              'inlined': False,
                                              'multivalued': True,
                                              'name': 'presynaptic_index',
                                              'range': 'DataItem'},
                        'unit': {'name': 'unit', 'range': 'Unit', 'required': True},
                        'values': {'description': 'Zarr array containing matrix values '
                                                  'quantifying connectivity with shape '
                                                  '(presynaptic_index x '
                                                  'postsynaptic_index).\n'
                                                  "NaN values reflect 'unmeasured' "
                                                  'connectivity.',
                                   'name': 'values',
                                   'range': 'ZarrArray',
                                   'required': True}}})

    id: str = Field(default=..., description="""Unique identifier within the class context.""", json_schema_extra = { "linkml_meta": {'alias': 'id',
         'aliases': ['identifier', 'structure_id', 'brain_region_id'],
         'domain_of': ['DataSet',
                       'DataItem',
                       'AlgorithmRun',
                       'ClusterHierarchy',
                       'Cluster',
                       'BrainRegion',
                       'ZarrArray',
                       'ZarrDataset',
                       'ParquetDataset',
                       'ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'CellFeatureMatrix',
                       'CellFeatureMeasurement',
                       'CellGeneData',
                       'SingleCellReconstruction',
                       'MappingSet',
                       'CellToCellMapping',
                       'CellToClusterMapping',
                       'ClusterToClusterMapping',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    description: Optional[str] = Field(default=None, description="""Free-text description of what this measurement matrix represents.""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['ProjectionMeasurementMatrix',
                       'CellFeatureSet',
                       'CellFeatureDefinition',
                       'MappingSet',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    presynaptic_index: Optional[list[str]] = Field(default=None, description="""Ordered data items defining rows of the matrix, where each row is a presynpatic data item (cell, injection location, etc).""", json_schema_extra = { "linkml_meta": {'alias': 'presynaptic_index', 'domain_of': ['CellCellMeasurementMatrix']} })
    postsynaptic_index: Optional[list[str]] = Field(default=None, description="""Ordered data items defining columns of the matrix, where each column is a postsynaptic data item (cell, region, etc).""", json_schema_extra = { "linkml_meta": {'alias': 'postsynaptic_index', 'domain_of': ['CellCellMeasurementMatrix']} })
    measurement_type: Optional[SynapticMeasurementType] = Field(default=None, description="""The specific projection measurement type (enum) for this set.""", json_schema_extra = { "linkml_meta": {'alias': 'measurement_type',
         'domain_of': ['ProjectionMeasurementMatrix',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    modality: Optional[Modality] = Field(default=None, description="""Source modality for the data item (if relevant).""", json_schema_extra = { "linkml_meta": {'alias': 'modality',
         'domain_of': ['DataSet',
                       'ProjectionMeasurementMatrix',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    values: str = Field(default=..., description="""Zarr array containing matrix values quantifying connectivity with shape (presynaptic_index x postsynaptic_index).
NaN values reflect 'unmeasured' connectivity.""", json_schema_extra = { "linkml_meta": {'alias': 'values',
         'domain_of': ['ProjectionMeasurementMatrix', 'CellCellMeasurementMatrix']} })
    unit: Unit = Field(default=..., description="""Unit of measure for values.""", json_schema_extra = { "linkml_meta": {'alias': 'unit',
         'domain_of': ['ProjectionMeasurementMatrix',
                       'CellFeatureDefinition',
                       'CellCellConnectivityLong',
                       'CellCellMeasurementMatrix']} })
    project_id: str = Field(default=..., description="""Identifier for the project or acquisition program context for this record.""", json_schema_extra = { "linkml_meta": {'alias': 'project_id',
         'aliases': ['project', 'program_id'],
         'domain_of': ['ProjectScoped']} })


# Model rebuild
# see https://pydantic-docs.helpmanual.io/usage/models/#rebuilding-a-model
SpatialLocation.model_rebuild()
AlgorithmRun.model_rebuild()
ClusterHierarchy.model_rebuild()
BrainRegion.model_rebuild()
ProjectScoped.model_rebuild()
DataSet.model_rebuild()
DataItem.model_rebuild()
DataItemDataSetAssociation.model_rebuild()
Cluster.model_rebuild()
ClusterMembership.model_rebuild()
ZarrArray.model_rebuild()
ZarrDataset.model_rebuild()
ParquetDataset.model_rebuild()
ProjectionMeasurementMatrix.model_rebuild()
CellFeatureSet.model_rebuild()
CellFeatureDefinition.model_rebuild()
CellFeatureMatrix.model_rebuild()
CellFeatureMeasurement.model_rebuild()
CellGeneData.model_rebuild()
BarcodingExperimentMetadata.model_rebuild()
GeneMetadata.model_rebuild()
CellMetadata.model_rebuild()
SingleCellReconstruction.model_rebuild()
MappingSet.model_rebuild()
CellToCellMapping.model_rebuild()
CellToClusterMapping.model_rebuild()
ClusterToClusterMapping.model_rebuild()
CellCellConnectivityLong.model_rebuild()
CellCellMeasurementMatrix.model_rebuild()

