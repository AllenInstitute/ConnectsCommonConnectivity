import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Cluster — no longer ProjectScoped (taxonomies are global reference artifacts)
# ---------------------------------------------------------------------------


def test_cluster_has_no_project_id_field(models):
    Cluster = models["Cluster"]
    assert "project_id" not in Cluster.model_fields


def test_cluster_constructs_without_project_id(models):
    Cluster = models["Cluster"]
    cluster = Cluster(id="c1")
    assert cluster.id == "c1"


def test_cluster_rejects_project_id(models):
    # Pydantic config is extra='forbid', so passing project_id raises rather than silently dropping.
    Cluster = models["Cluster"]
    with pytest.raises(ValidationError, match=r"(?s)project_id.*Extra inputs are not permitted"):
        Cluster(id="c1", project_id="visp_patchseq")


# ---------------------------------------------------------------------------
# ClusterMembership — still ProjectScoped, gains optional hierarchy_id
# ---------------------------------------------------------------------------


def test_cluster_membership_project_id_required(models):
    ClusterMembership = models["ClusterMembership"]
    with pytest.raises(ValidationError, match=r"(?s)project_id.*Field required"):
        ClusterMembership(item="cell_1", cluster="c1")


def test_cluster_membership_hierarchy_id_optional(models):
    ClusterMembership = models["ClusterMembership"]
    cm = ClusterMembership(item="cell_1", cluster="c1", project_id="visp_patchseq")
    assert cm.hierarchy_id is None


def test_cluster_membership_hierarchy_id_round_trip(models):
    ClusterMembership = models["ClusterMembership"]
    cm = ClusterMembership(
        item="cell_1",
        cluster="c1",
        project_id="visp_patchseq",
        hierarchy_id="visp_met_types_v1",
    )
    assert cm.hierarchy_id == "visp_met_types_v1"


def test_cluster_membership_hierarchy_id_must_be_string(models):
    ClusterMembership = models["ClusterMembership"]
    with pytest.raises(ValidationError, match=r"(?s)hierarchy_id.*Input should be a valid string"):
        ClusterMembership(
            item="cell_1",
            cluster="c1",
            project_id="visp_patchseq",
            hierarchy_id=123,
        )


# ---------------------------------------------------------------------------
# Cluster.hierarchy_id discriminator (taxonomy id, optional string)
# ---------------------------------------------------------------------------


def test_cluster_hierarchy_id_optional(models):
    Cluster = models["Cluster"]
    cluster = Cluster(id="c1")
    assert cluster.hierarchy_id is None


def test_cluster_hierarchy_id_round_trip(models):
    Cluster = models["Cluster"]
    cluster = Cluster(id="c1", hierarchy_id="visp_met_types_v1")
    assert cluster.hierarchy_id == "visp_met_types_v1"


def test_cluster_hierarchy_id_must_be_string(models):
    Cluster = models["Cluster"]
    with pytest.raises(ValidationError, match=r"(?s)hierarchy_id.*Input should be a valid string"):
        Cluster(id="c1", hierarchy_id=123)


def test_cluster_still_has_no_project_id_after_hierarchy_id_added(models):
    # Regression guard: hierarchy_id was added without re-introducing ProjectScoped on Cluster.
    Cluster = models["Cluster"]
    assert "project_id" not in Cluster.model_fields
    with pytest.raises(ValidationError, match=r"(?s)project_id.*Extra inputs are not permitted"):
        Cluster(id="c1", project_id="visp_patchseq")


# ---------------------------------------------------------------------------
# ClusterHierarchy / AlgorithmRun / HierarchyCategory (Tasic 01 notebook)
# ---------------------------------------------------------------------------


def test_cluster_hierarchy_constructs_with_id_run_root_clusters(models):
    ClusterHierarchy = models["ClusterHierarchy"]
    h = ClusterHierarchy(id="h1", run="run1", root="root", clusters=["root", "c1"])
    assert h.id == "h1"
    assert h.root == "root"
    assert h.clusters == ["root", "c1"]


def test_cluster_hierarchy_requires_id(models):
    ClusterHierarchy = models["ClusterHierarchy"]
    with pytest.raises(ValidationError, match=r"(?s)id.*Field required"):
        ClusterHierarchy(run="run1", root="root", clusters=["root"])


def test_algorithm_run_requires_algorithm_name(models):
    AlgorithmRun = models["AlgorithmRun"]
    with pytest.raises(ValidationError, match=r"(?s)algorithm_name.*Field required"):
        AlgorithmRun(id="run1")


def test_algorithm_run_constructs_without_input_dataset(models):
    AlgorithmRun = models["AlgorithmRun"]
    run = AlgorithmRun(id="run1", algorithm_name="hierarchical")
    assert run.input_dataset is None


def test_hierarchy_category_requires_id(models):
    HierarchyCategory = models["HierarchyCategory"]
    with pytest.raises(ValidationError, match=r"(?s)id.*Field required"):
        HierarchyCategory(description="leaf")


def test_hierarchy_category_level_optional(models):
    HierarchyCategory = models["HierarchyCategory"]
    cat = HierarchyCategory(id="cluster")
    assert cat.level is None
