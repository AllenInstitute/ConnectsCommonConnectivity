"""Tests for the IO writer dispatch core.

Covers:

* The patchseq regression — overlapping ``project_id`` writes do not wipe
  each other (the original motivating bug).
* Idempotency, multi-scope-group dispatch, predicate construction.
* Append-new-by-id semantics.
* A per-class round-trip smoke test for every entry in ``WRITABLE_CLASSES``.
* ``write_projection_matrix`` enrichment + write.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow as pa
import pytest
from pydantic import BaseModel

from connects_common_connectivity.config import Settings
from connects_common_connectivity.io.write_spec import REGISTRY
from connects_common_connectivity.io.writers import (
    WRITABLE_CLASSES,
    WrittenResult,
    _build_predicate,
    _group_by_scope,
    write_models,
    write_projection_matrix,
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
    HierarchyCategory,
    Laterality,
    MappingSet,
    Modality,
    ProjectionMeasurementMatrix,
    ProjectionMeasurementType,
    Unit,
)

# ---------------------------------------------------------------------------
# Predicate construction
# ---------------------------------------------------------------------------


def test_build_predicate_format():
    assert (
        _build_predicate(["project_id"], ["minnie65"])
        == "project_id = 'minnie65'"
    )
    assert (
        _build_predicate(["project_id", "id"], ["minnie65", "ds_a"])
        == "project_id = 'minnie65' AND id = 'ds_a'"
    )


@pytest.mark.parametrize(
    "value,expected_literal",
    [
        ("O'Hara", "'O''Hara'"),
        ("", "''"),
        ("a\\b", "'a\\b'"),
        ("café", "'café'"),
    ],
)
def test_build_predicate_escapes(value, expected_literal):
    assert _build_predicate(["name"], [value]) == f"name = {expected_literal}"


# ---------------------------------------------------------------------------
# _group_by_scope
# ---------------------------------------------------------------------------


def test_group_by_scope_preserves_first_appearance_order():
    table = pa.table(
        {
            "project_id": ["p", "p", "p"],
            "id": ["b", "a", "b"],
            "value": [1, 2, 3],
        }
    )
    groups = _group_by_scope(table, ["project_id", "id"])
    keys = [k for k, _ in groups]
    assert keys == [("p", "b"), ("p", "a")]
    # The first 'b' group should hold rows 0 and 2 (preserved order).
    first_sub = groups[0][1]
    assert first_sub.column("value").to_pylist() == [1, 3]


# ---------------------------------------------------------------------------
# Patchseq regression: the headline test
# ---------------------------------------------------------------------------


def test_patchseq_regression_two_datasets_same_project(settings, read_delta):
    """Two DataSet rows with the same ``project_id`` but different ``id`` must coexist.

    Before W2/W3 the notebooks predicated on ``project_id`` only, so a
    second write wiped the first. The new ``scope_columns=[project_id, id]``
    keeps each row independent.
    """
    ds_a = DataSet(id="visp_exc_patchseq", name="exc", project_id="visp_patchseq")
    ds_b = DataSet(id="visp_inh_patchseq", name="inh", project_id="visp_patchseq")
    write_models(ds_a, settings=settings)
    write_models(ds_b, settings=settings)

    rows = read_delta(settings.output_root / "dataset")
    ids = sorted(rows["id"].to_list())
    assert ids == ["visp_exc_patchseq", "visp_inh_patchseq"], (
        f"patchseq regression: second write wiped first. "
        f"Expected both datasets, got {ids}"
    )


def test_overwrite_scoped_is_idempotent(settings, read_delta):
    ds = DataSet(id="d1", name="example", project_id="p1")
    write_models(ds, settings=settings)
    write_models(ds, settings=settings)
    rows = read_delta(settings.output_root / "dataset")
    assert rows.shape[0] == 1, f"idempotent rewrite produced {rows.shape[0]} rows"
    assert rows["id"].to_list() == ["d1"], "row identity changed across rewrites"
    assert rows["name"].to_list() == ["example"], "row content drifted across rewrites"


def test_dry_run_does_not_write(tmp_path):
    settings = Settings(output_root=tmp_path, dry_run=True)
    ds = DataSet(id="d1", name="d", project_id="p1")

    result = write_models(ds, settings=settings)

    assert result.rows_written == 0, "dry_run must report 0 rows written"
    assert not (tmp_path / "dataset").exists(), "dry_run must not create tables"


def test_multi_scope_group_dispatch_yields_one_predicate_per_group(settings, read_delta):
    rows_in = [
        DataSet(id="a", name="A", project_id="p1"),
        DataSet(id="b", name="B", project_id="p1"),
    ]
    result = write_models(rows_in, settings=settings)
    assert isinstance(result, WrittenResult)
    assert len(result.predicates) == 2
    assert result.rows_written == 2
    # Both end up in the table.
    rows = read_delta(settings.output_root / "dataset")
    assert sorted(rows["id"].to_list()) == ["a", "b"]


# ---------------------------------------------------------------------------
# append_new_by_id semantics (DataItem)
# ---------------------------------------------------------------------------


def test_append_new_by_id_only_appends_unseen(settings, read_delta):
    items_first = [
        DataItem(id="cell_1", name="cell_1", project_id="p1"),
        DataItem(id="cell_2", name="cell_2", project_id="p1"),
    ]
    r1 = write_models(items_first, settings=settings)
    assert r1.mode == "append_new_by_id"
    assert r1.predicates == ()
    assert r1.rows_written == 2

    items_second = [
        DataItem(id="cell_2", name="cell_2", project_id="p1"),  # already there
        DataItem(id="cell_3", name="cell_3", project_id="p1"),  # new
    ]
    r2 = write_models(items_second, settings=settings)
    assert r2.rows_written == 1

    rows = read_delta(settings.output_root / "dataitem")
    assert sorted(rows["id"].to_list()) == ["cell_1", "cell_2", "cell_3"]


def test_append_new_by_id_rejects_mixed_project_ids(settings):
    bad = [
        DataItem(id="x", name="x", project_id="p1"),
        DataItem(id="y", name="y", project_id="p2"),
    ]
    with pytest.raises(ValueError, match="single project_id"):
        write_models(bad, settings=settings)


# ---------------------------------------------------------------------------
# Per-class smoke (every entry in WRITABLE_CLASSES exercised)
# ---------------------------------------------------------------------------


INSTANCE_FACTORIES = {
    DataSet: lambda: DataSet(id="ds1", name="ds", project_id="p1"),
    DataItem: lambda: DataItem(id="di1", name="di1", project_id="p1"),
    DataItemDataSetAssociation: lambda: DataItemDataSetAssociation(
        dataitem_id="di1", dataset_id="ds1", project_id="p1"
    ),
    Cluster: lambda: Cluster(id="c1", hierarchy_id="h1", level=0),
    ClusterHierarchy: lambda: ClusterHierarchy(id="h1", root="c1", clusters=["c1"]),
    ClusterMembership: lambda: ClusterMembership(
        item="cell_1", cluster="c1", hierarchy_id="h1", project_id="p1"
    ),
    MappingSet: lambda: MappingSet(id="m1", project_id="p1", name="m", method_name="dummy"),
    CellToClusterMapping: lambda: CellToClusterMapping(
        id="ctc1",
        project_id="p1",
        mapping_set="m1",
        source_cell="cell_1",
        target_cluster="c1",
    ),
    CellFeatureSet: lambda: CellFeatureSet(id="fs1", project_id="p1"),
    CellFeatureDefinition: lambda: CellFeatureDefinition(
        id="feat_a",
        project_id="p1",
        feature_set_id="fs1",
        data_type="<f4",
        unit=Unit.MICRONS_LENGTH.value,
    ),
    CellFeatureMatrix: lambda: CellFeatureMatrix(
        id="cfm1",
        project_id="p1",
        feature_set_id="fs1",
        parquet_path="file:///tmp/wide.parquet",
        cell_index_column="id",
    ),
    ProjectionMeasurementMatrix: lambda: ProjectionMeasurementMatrix(
        id="pmm1",
        measurement_type=ProjectionMeasurementType.MICRONS_OF_AXON,
        modality=Modality.MORPHOLOGY,
        laterality=Laterality.IPSILATERAL,
        unit=Unit.MICRONS_LENGTH,
        data_item_index=["cell_1"],
        region_index=["VISp"],
        values="file:///tmp/pmm.delta",
    ),
    AlgorithmRun: lambda: AlgorithmRun(id="run1", algorithm_name="kmeans"),
    HierarchyCategory: lambda: HierarchyCategory(id="cluster", description="leaf", level="0"),
}


def _make_instance(cls):
    """Return a minimal valid instance of ``cls`` for the round-trip smoke test."""
    try:
        return INSTANCE_FACTORIES[cls]()
    except KeyError:
        pytest.fail(
            f"No fixture for {cls.__name__}. Add an entry to "
            "INSTANCE_FACTORIES in tests/test_writers.py."
        )


def test_every_writable_class_has_a_fixture():
    missing = set(WRITABLE_CLASSES) - set(INSTANCE_FACTORIES)
    assert not missing, (
        f"WRITABLE_CLASSES added entries without fixtures: "
        f"{sorted(c.__name__ for c in missing)}"
    )
    stale = set(INSTANCE_FACTORIES) - set(WRITABLE_CLASSES)
    assert not stale, (
        f"INSTANCE_FACTORIES has stale entries not in WRITABLE_CLASSES: "
        f"{sorted(c.__name__ for c in stale)}"
    )


@pytest.mark.parametrize("cls", WRITABLE_CLASSES, ids=[c.__name__ for c in WRITABLE_CLASSES])
def test_round_trip_each_writable_class(cls, settings, read_delta):
    instance = _make_instance(cls)
    result = write_models(instance, settings=settings)
    assert result.class_name == cls.__name__
    spec = REGISTRY[cls.__name__]
    assert result.path == settings.output_root / spec.subdir
    assert result.rows_written == 1
    rows = read_delta(result.path)
    assert rows.shape[0] >= 1


# ---------------------------------------------------------------------------
# write_projection_matrix
# ---------------------------------------------------------------------------


def test_write_projection_matrix_enriches_and_does_not_mutate_input(settings, read_delta):
    pmm = ProjectionMeasurementMatrix(
        id="pmm_test",
        measurement_type=ProjectionMeasurementType.MICRONS_OF_AXON,
        modality=Modality.MORPHOLOGY,
        laterality=Laterality.IPSILATERAL,
        unit=Unit.MICRONS_LENGTH,
        data_item_index=["c1", "c2"],
        region_index=["VISp", "ACA", "MOB"],
        values="file:///tmp/pmm.delta",
    )
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    assert pmm.region_coverage in (None, [])

    result = write_projection_matrix(pmm, matrix, settings=settings)
    assert result.class_name == "ProjectionMeasurementMatrix"
    assert pmm.region_coverage in (None, [])  # input not mutated

    rows = read_delta(settings.output_root / "projectionmeasurementmatrix")
    coverage = rows.filter(pl.col("id") == "pmm_test")["region_coverage"].to_list()[0]
    assert list(coverage) == ["VISp", "MOB"]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_write_models_rejects_empty(settings):
    with pytest.raises(ValueError, match="empty"):
        write_models([], settings=settings)


def test_write_models_rejects_heterogeneous(settings):
    with pytest.raises(TypeError, match="homogeneous"):
        write_models(
            [
                DataSet(id="d1", name="d", project_id="p1"),
                DataItem(id="x", name="x", project_id="p1"),
            ],
            settings=settings,
        )


def test_write_models_rejects_unregistered_class(settings):
    class NotInRegistry:
        pass

    with pytest.raises(TypeError, match="pydantic model or iterable"):
        write_models(NotInRegistry(), settings=settings)


def test_write_models_rejects_unregistered_pydantic_model(settings):
    class UnregisteredModel(BaseModel):
        id: str

    with pytest.raises(KeyError, match="UnregisteredModel"):
        write_models(UnregisteredModel(id="u1"), settings=settings)


# ---------------------------------------------------------------------------
# Per-call output_root override
# ---------------------------------------------------------------------------


def test_write_models_output_root_override_writes_to_given_path(tmp_path):
    """Passing output_root= writes under that root, bypassing get_settings()."""
    alt_root = tmp_path / "alt_dataset"
    ds = DataSet(id="d_alt", name="alt", project_id="p_alt")

    result = write_models(ds, output_root=alt_root)

    assert result.path == alt_root / "dataset"
    rows = pl.read_delta(str(alt_root / "dataset")).filter(
        pl.col("id") == "d_alt"
    )
    assert rows.shape[0] == 1


def test_write_models_output_root_accepts_string(tmp_path):
    """str and Path are both accepted for output_root."""
    alt_root = tmp_path / "string_root"
    ds = DataSet(id="d_str", name="s", project_id="p_str")

    result = write_models(ds, output_root=str(alt_root))

    assert result.path == alt_root / "dataset"


def test_write_models_rejects_both_settings_and_output_root(settings, tmp_path):
    """Passing both settings= and output_root= raises (no precedence to memorize)."""
    ds = DataSet(id="d_x", name="x", project_id="p_x")
    with pytest.raises(TypeError, match="either settings= or output_root="):
        write_models(ds, settings=settings, output_root=tmp_path / "other")


def test_write_projection_matrix_output_root_override(tmp_path):
    """write_projection_matrix forwards output_root through write_models."""
    alt_root = tmp_path / "pmm_alt"
    pmm = ProjectionMeasurementMatrix(
        id="pmm_alt",
        measurement_type=ProjectionMeasurementType.MICRONS_OF_AXON,
        modality=Modality.MORPHOLOGY,
        laterality=Laterality.IPSILATERAL,
        unit=Unit.MICRONS_LENGTH,
        data_item_index=["c1", "c2"],
        region_index=["r1", "r2"],
        values="file:///tmp/pmm_alt.delta",
    )
    matrix = np.array([[1.0, 0.0], [0.0, 2.0]])

    result = write_projection_matrix(pmm, matrix, output_root=alt_root)

    assert result.path == alt_root / "projectionmeasurementmatrix"


def test_write_projection_matrix_rejects_both_settings_and_output_root(
    settings, tmp_path
):
    pmm = ProjectionMeasurementMatrix(
        id="pmm_x",
        measurement_type=ProjectionMeasurementType.MICRONS_OF_AXON,
        modality=Modality.MORPHOLOGY,
        laterality=Laterality.IPSILATERAL,
        unit=Unit.MICRONS_LENGTH,
        data_item_index=["c1"],
        region_index=["r1"],
        values="file:///tmp/pmm_x.delta",
    )
    matrix = np.array([[1.0]])
    with pytest.raises(TypeError, match="either settings= or output_root="):
        write_projection_matrix(
            pmm, matrix, settings=settings, output_root=tmp_path / "other"
        )
