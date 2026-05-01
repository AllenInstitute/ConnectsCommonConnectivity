import pytest
from pydantic import ValidationError

import connects_common_connectivity as ccc


def _models():
    return ccc.generate_pydantic_models()


# ---------------------------------------------------------------------------
# Cluster — no longer ProjectScoped (taxonomies are global reference artifacts)
# ---------------------------------------------------------------------------


def test_cluster_has_no_project_id_field():
    Cluster = _models()["Cluster"]
    assert "project_id" not in Cluster.model_fields


def test_cluster_constructs_without_project_id():
    Cluster = _models()["Cluster"]
    cluster = Cluster(id="c1")
    assert cluster.id == "c1"


def test_cluster_rejects_project_id():
    # Pydantic config is extra='forbid', so passing project_id raises rather than silently dropping.
    Cluster = _models()["Cluster"]
    with pytest.raises(ValidationError, match=r"(?s)project_id.*Extra inputs are not permitted"):
        Cluster(id="c1", project_id="visp_patchseq")


# ---------------------------------------------------------------------------
# ClusterMembership — still ProjectScoped, gains optional hierarchy_id
# ---------------------------------------------------------------------------


def test_cluster_membership_project_id_required():
    ClusterMembership = _models()["ClusterMembership"]
    with pytest.raises(ValidationError, match=r"(?s)project_id.*Field required"):
        ClusterMembership(item="cell_1", cluster="c1")


def test_cluster_membership_hierarchy_id_optional():
    ClusterMembership = _models()["ClusterMembership"]
    cm = ClusterMembership(item="cell_1", cluster="c1", project_id="visp_patchseq")
    assert cm.hierarchy_id is None


def test_cluster_membership_hierarchy_id_round_trip():
    ClusterMembership = _models()["ClusterMembership"]
    cm = ClusterMembership(
        item="cell_1",
        cluster="c1",
        project_id="visp_patchseq",
        hierarchy_id="visp_met_types_v1",
    )
    assert cm.hierarchy_id == "visp_met_types_v1"


def test_cluster_membership_hierarchy_id_must_be_string():
    ClusterMembership = _models()["ClusterMembership"]
    with pytest.raises(ValidationError, match=r"(?s)hierarchy_id.*Input should be a valid string"):
        ClusterMembership(
            item="cell_1",
            cluster="c1",
            project_id="visp_patchseq",
            hierarchy_id=123,
        )
