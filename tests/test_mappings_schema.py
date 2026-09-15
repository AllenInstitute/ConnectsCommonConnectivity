import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# MappingSet — source/target endpoints can be DataSet or ClusterHierarchy
# ---------------------------------------------------------------------------


def test_mapping_set_dataset_to_dataset(models):
    """Mapping sets must support dataset-to-dataset endpoints."""
    # Cell-to-cell shape: source_dataset + target_dataset (back-compat).
    MappingSet = models["MappingSet"]
    ms = MappingSet(
        id="ms_cell_cell",
        name="ms_cell_cell",
        method_name="nearest_neighbor",
        source_dataset="visp_inh_patchseq",
        target_dataset="visp_exc_patchseq",
        project_id="visp_patchseq",
    )
    assert ms.source_dataset == "visp_inh_patchseq"
    assert ms.target_dataset == "visp_exc_patchseq"
    assert ms.source_hierarchy is None
    assert ms.target_hierarchy is None


def test_mapping_set_dataset_to_hierarchy(models):
    """Mapping sets must support dataset-to-hierarchy endpoints."""
    # Cell-to-cluster shape: source_dataset + target_hierarchy.
    MappingSet = models["MappingSet"]
    ms = MappingSet(
        id="ms_cell_cluster",
        name="ms_cell_cluster",
        method_name="patchseqtools_v2",
        source_dataset="visp_inh_patchseq",
        target_hierarchy="tasic_2018",
        project_id="visp_patchseq",
    )
    assert ms.source_dataset == "visp_inh_patchseq"
    assert ms.target_dataset is None
    assert ms.target_hierarchy == "tasic_2018"
    assert ms.source_hierarchy is None


def test_mapping_set_hierarchy_to_hierarchy(models):
    """Mapping sets must support hierarchy-to-hierarchy endpoints."""
    # Cluster-to-cluster shape: source_hierarchy + target_hierarchy.
    MappingSet = models["MappingSet"]
    ms = MappingSet(
        id="ms_cluster_cluster",
        name="ms_cluster_cluster",
        method_name="ontology_align_v1",
        source_hierarchy="tasic_2018",
        target_hierarchy="visp_met_types_v1",
        project_id="visp_patchseq",
    )
    assert ms.source_dataset is None
    assert ms.target_dataset is None
    assert ms.source_hierarchy == "tasic_2018"
    assert ms.target_hierarchy == "visp_met_types_v1"


def test_mapping_set_endpoints_optional(models):
    """Mapping sets must allow all endpoint fields to be omitted."""
    # All four endpoint slots are optional at the schema level (LinkML can't enforce
    # "exactly one of"); convention is enforced per-mapping kind.
    MappingSet = models["MappingSet"]
    ms = MappingSet(
        id="ms_minimal",
        name="ms_minimal", method_name="m", project_id="p1",
    )
    assert ms.source_dataset is None
    assert ms.target_dataset is None
    assert ms.source_hierarchy is None
    assert ms.target_hierarchy is None


def test_mapping_set_method_name_still_required(models):
    """Mapping sets must require a method name."""
    MappingSet = models["MappingSet"]
    with pytest.raises(ValidationError, match=r"(?s)method_name.*Field required"):
        MappingSet(id="ms1", project_id="p1")


def test_mapping_set_project_id_still_required(models):
    """Mapping sets must require a project identifier."""
    MappingSet = models["MappingSet"]
    with pytest.raises(ValidationError, match=r"(?s)project_id.*Field required"):
        MappingSet(id="ms1", method_name="m")


def test_mapping_set_hierarchy_fields_must_be_strings(models):
    """Mapping-set hierarchy endpoints must be strings."""
    MappingSet = models["MappingSet"]
    with pytest.raises(ValidationError, match=r"(?s)target_hierarchy.*Input should be a valid string"):
        MappingSet(
            id="ms1", method_name="m", project_id="p1",
            source_dataset="ds1", target_hierarchy=123,
        )


# ---------------------------------------------------------------------------
# CellToClusterMapping — round-trip with the cell-to-cluster MappingSet shape
# ---------------------------------------------------------------------------


def test_cell_to_cluster_mapping_round_trip(models):
    """Cell-to-cluster mappings must retain endpoints and scores."""
    CellToClusterMapping = models["CellToClusterMapping"]
    m = CellToClusterMapping(
        id="map_001",
        mapping_set="ms_cell_cluster",
        source_cell="cell_1",
        target_cluster="cluster_42",
        score=0.87,
        probability=0.91,
        project_id="visp_patchseq",
    )
    assert m.mapping_set == "ms_cell_cluster"
    assert m.source_cell == "cell_1"
    assert m.target_cluster == "cluster_42"
    assert m.score == 0.87
    assert m.probability == 0.91


def test_cell_to_cluster_mapping_requires_target_cluster(models):
    """Cell-to-cluster mappings must require a target cluster."""
    CellToClusterMapping = models["CellToClusterMapping"]
    with pytest.raises(ValidationError, match=r"(?s)target_cluster.*Field required"):
        CellToClusterMapping(
            id="map_001",
            mapping_set="ms_cell_cluster",
            source_cell="cell_1",
            project_id="visp_patchseq",
        )
