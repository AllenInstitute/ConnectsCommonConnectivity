"""Tests for the IO writer dispatch core.

Covers:

* The patchseq regression — overlapping ``project_id`` writes do not wipe
  each other (the original motivating bug).
* Idempotency, merge dispatch, predicate construction, and batch deduplication.
* A per-class round-trip smoke test for every entry in ``WRITABLE_CLASSES``.
* ``write_projection_matrix`` enrichment + write.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow as pa
import pytest
import yaml
from deltalake.exceptions import DeltaError
from deltalake.table import TableMerger
from pydantic import BaseModel

from connects_common_connectivity.config import ConfigNotFoundError, Settings
from connects_common_connectivity.io.write_spec import REGISTRY, WriteSpec
from connects_common_connectivity.io.writers import (
    _MAX_PRUNE_LITERALS,
    WRITABLE_CLASSES,
    WrittenResult,
    _build_merge_predicate,
    _build_merge_update_predicate,
    _build_partition_prune_predicate,
    _build_predicate,
    _deduplicate_on_keys,
    _dispatch_merge_scoped,
    _dispatch_overwrite_scoped,
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
    SynapseFeatureMatrix,
    Unit,
)

# ---------------------------------------------------------------------------
# Predicate construction
# ---------------------------------------------------------------------------


def test_build_predicate_format():
    """Predicates must join scoped equality clauses with SQL conjunctions."""
    assert (
        _build_predicate(["project_id"], ["minnie65"])
        == '"project_id" = \'minnie65\''
    )
    assert (
        _build_predicate(["project_id", "id"], ["minnie65", "ds_a"])
        == '"project_id" = \'minnie65\' AND "id" = \'ds_a\''
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
    """Predicate values must be escaped as valid SQL string literals."""
    assert _build_predicate(["name"], [value]) == f'"name" = {expected_literal}'


def test_build_merge_predicate_uses_all_declared_keys():
    """Merge predicates must compare every identity column through aliases."""
    assert _build_merge_predicate(["project_id", "id"]) == (
        'target."project_id" = source."project_id" '
        'AND target."id" = source."id"'
    )


def test_merge_predicates_quote_keyword_like_and_escaped_identifiers():
    """Merge expressions must quote identifiers and escape embedded quotes."""
    assert _build_merge_predicate(["item", 'cluster"name']) == (
        'target."item" = source."item" '
        'AND target."cluster""name" = source."cluster""name"'
    )
    assert _build_merge_update_predicate(["item", "order"], ["item"]) == (
        '(source."order" IS DISTINCT FROM target."order")'
    )


def test_partition_prune_predicate_restates_batch_partition_values():
    """Partitioned merge keys must be restated as literals for file skipping."""
    table = pa.table(
        {
            "project_id": ["p", "p"],
            "hierarchy_id": ["h1", "h0"],
            "item": ["a", "b"],
        }
    )

    assert _build_partition_prune_predicate(
        table, ["project_id", "hierarchy_id"], ["project_id", "hierarchy_id", "item"]
    ) == (
        'target."project_id" IN (\'p\') '
        'AND target."hierarchy_id" IN (\'h0\', \'h1\')'
    )


def test_partition_prune_predicate_skips_columns_outside_merge_keys():
    """Constraining a partition column that is not a merge key could drop matches."""
    table = pa.table({"project_id": ["p"], "id": ["a"]})

    assert (
        _build_partition_prune_predicate(table, ["project_id"], ["id"]) is None
    )


def test_partition_prune_predicate_skips_high_cardinality_columns():
    """Very wide literal lists cost more to plan than the skipping saves."""
    values = [f"p{index}" for index in range(_MAX_PRUNE_LITERALS + 1)]
    table = pa.table({"project_id": values})

    assert (
        _build_partition_prune_predicate(table, ["project_id"], ["project_id"]) is None
    )


def test_merge_scoped_predicate_prunes_untouched_partitions(
    settings, read_delta, monkeypatch
):
    """A merge must scan only the partitions its source batch touches."""
    merge_metrics = {}
    original_execute = TableMerger.execute

    def capture_metrics(merger):
        metrics = original_execute(merger)
        merge_metrics.update(metrics)
        return metrics

    monkeypatch.setattr(TableMerger, "execute", capture_metrics)

    write_models(
        [
            ClusterMembership(
                project_id=project_id,
                hierarchy_id=hierarchy_id,
                item=f"cell_{index}",
                cluster="c0",
            )
            for project_id in ("visp_patchseq", "minnie65")
            for hierarchy_id in ("met_types", "tx_types")
            for index in range(3)
        ],
        settings=settings,
    )

    result = write_models(
        [
            ClusterMembership(
                project_id="visp_patchseq",
                hierarchy_id="met_types",
                item="cell_99",
                cluster="c1",
            )
        ],
        settings=settings,
    )

    assert result.predicates == (
        'target."project_id" = source."project_id" '
        'AND target."hierarchy_id" = source."hierarchy_id" '
        'AND target."item" = source."item" '
        'AND target."cluster" = source."cluster" '
        'AND target."project_id" IN (\'visp_patchseq\') '
        'AND target."hierarchy_id" IN (\'met_types\')',
    )
    assert merge_metrics["num_target_files_scanned"] == 1
    assert merge_metrics["num_target_files_skipped_during_scan"] == 3
    rows = read_delta(settings.output_root / "clustermembership")
    assert rows.height == 13, "pruned merge must not drop rows in other partitions"


def test_deduplicate_on_keys_keeps_last_row_in_stable_order():
    """Later source rows must deterministically replace earlier duplicate keys."""
    table = pa.table(
        {
            "project_id": ["p", "p", "p"],
            "id": ["a", "b", "a"],
            "value": [1, 2, 3],
        }
    )

    deduplicated = _deduplicate_on_keys(table, ["project_id", "id"])

    assert deduplicated.to_pylist() == [
        {"project_id": "p", "id": "b", "value": 2},
        {"project_id": "p", "id": "a", "value": 3},
    ]


def test_deduplicate_on_keys_supports_chunked_keys_and_empty_tables():
    """Arrow-native deduplication must handle chunk boundaries and no rows."""
    chunked = pa.table(
        {
            "id": pa.chunked_array([["a", "b"], ["a"]]),
            "value": [1, 2, 3],
        }
    )
    empty = pa.table(
        {
            "id": pa.array([], type=pa.string()),
            "value": pa.array([], type=pa.int64()),
        }
    )

    assert _deduplicate_on_keys(chunked, ["id"]).to_pylist() == [
        {"id": "b", "value": 2},
        {"id": "a", "value": 3},
    ]
    assert _deduplicate_on_keys(empty, ["id"]).equals(empty)


def test_deduplicate_on_keys_rejects_earliest_null_key():
    """Null validation must report the first invalid row across all key columns."""
    table = pa.table(
        {
            "project_id": ["p", None, "p"],
            "id": [None, "a", "b"],
        }
    )

    with pytest.raises(
        ValueError,
        match=r"row 0 has key \('p', None\)",
    ):
        _deduplicate_on_keys(table, ["project_id", "id"])


def test_deduplicate_on_keys_avoids_temporary_column_collisions():
    """A source column resembling the internal row index must be preserved."""
    table = pa.table(
        {
            "id": ["a", "a"],
            "__ccc_row_index": [10, 20],
            "__ccc_row_index__max": [30, 40],
        }
    )

    assert _deduplicate_on_keys(table, ["id"]).to_pylist() == [
        {
            "id": "a",
            "__ccc_row_index": 20,
            "__ccc_row_index__max": 40,
        }
    ]


# ---------------------------------------------------------------------------
# _group_by_scope
# ---------------------------------------------------------------------------


def test_group_by_scope_preserves_first_appearance_order():
    """Scope grouping must preserve group and row appearance order."""
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


def test_overwrite_scoped_dispatch_remains_available_for_bulk_tables(
    tmp_path, monkeypatch
):
    """The retained bulk-table dispatcher must issue one overwrite per scope."""
    table = pa.table(
        {
            "project_id": ["p", "p", "p"],
            "dataset_id": ["a", "b", "a"],
            "value": [1, 2, 3],
        }
    )
    spec = WriteSpec(
        model_cls=DataSet,
        subdir="synapse",
        partition_by=["project_id"],
        scope_columns=["project_id", "dataset_id"],
        write_mode="overwrite_scoped",
    )
    calls = []

    def record_write(path, batch, **kwargs):
        calls.append((path, batch.to_pylist(), kwargs))

    monkeypatch.setattr(
        "connects_common_connectivity.io.writers.write_deltalake", record_write
    )

    path = tmp_path / "synapse"
    result = _dispatch_overwrite_scoped(table, spec, path)

    assert result.mode == "overwrite_scoped"
    assert result.predicates == (
        '"project_id" = \'p\' AND "dataset_id" = \'a\'',
        '"project_id" = \'p\' AND "dataset_id" = \'b\'',
    )
    assert result.rows_written == 3
    assert [call[1] for call in calls] == [
        [
            {"project_id": "p", "dataset_id": "a", "value": 1},
            {"project_id": "p", "dataset_id": "a", "value": 3},
        ],
        [{"project_id": "p", "dataset_id": "b", "value": 2}],
    ]
    assert all(call[2]["mode"] == "overwrite" for call in calls)
    assert all(call[2]["partition_by"] == ["project_id"] for call in calls)


def test_merge_scoped_recovers_from_concurrent_table_creation(tmp_path, monkeypatch):
    """A writer losing table creation must reopen and merge its batch."""
    table = pa.table({"project_id": ["p1"], "id": ["d1"], "name": ["one"]})
    spec = WriteSpec(
        model_cls=DataSet,
        subdir="dataset",
        partition_by=["project_id"],
        scope_columns=["project_id", "id"],
        write_mode="merge_scoped",
        merge_on=["project_id", "id"],
    )
    events = []

    class FakeMerger:
        def when_matched_update_all(self, **kwargs):
            return self

        def when_not_matched_insert_all(self):
            return self

        def execute(self):
            events.append("merge")
            return {"num_target_rows_inserted": 1, "num_target_rows_updated": 0}

    class FakeDeltaTable:
        @staticmethod
        def is_deltatable(path):
            return False

        def __init__(self, path):
            events.append("reopen")

        def merge(self, **kwargs):
            return FakeMerger()

    def lose_creation_race(*args, **kwargs):
        events.append("create")
        raise DeltaError("table already exists")

    monkeypatch.setattr(
        "connects_common_connectivity.io.writers.DeltaTable", FakeDeltaTable
    )
    monkeypatch.setattr(
        "connects_common_connectivity.io.writers.write_deltalake",
        lose_creation_race,
    )

    result = _dispatch_merge_scoped(table, spec, tmp_path / "dataset")

    assert events == ["create", "reopen", "merge"]
    assert result.rows_written == 1


# ---------------------------------------------------------------------------
# Patchseq regression: the headline test
# ---------------------------------------------------------------------------


def test_patchseq_regression_two_datasets_same_project(settings, read_delta):
    """Datasets sharing a project but not an ID must coexist."""
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


def test_merge_scoped_is_idempotent(settings, read_delta):
    """Repeated merges must preserve one unchanged row."""
    ds = DataSet(id="d1", name="example", project_id="p1")
    first = write_models(ds, settings=settings)
    second = write_models(ds, settings=settings)

    assert first.rows_written == 1
    assert second.rows_written == 0
    rows = read_delta(settings.output_root / "dataset")
    assert rows.shape[0] == 1, f"idempotent rewrite produced {rows.shape[0]} rows"
    assert rows["id"].to_list() == ["d1"], "row identity changed across rewrites"
    assert rows["name"].to_list() == ["example"], "row content drifted across rewrites"


def test_same_hierarchy_category_id_coexists_across_hierarchies(settings, read_delta):
    """Category IDs shared by taxonomies must retain hierarchy-local metadata."""
    categories = [
        HierarchyCategory(
            id="class",
            hierarchy_id="tasic_2018_visp_taxonomy",
            description="Top-level transcriptomic class.",
            level=2,
        ),
        HierarchyCategory(
            id="class",
            hierarchy_id="visp_met_types_taxonomy",
            description="Top-level MET-type class.",
            level=1,
        ),
    ]

    for category in categories:
        write_models(category, settings=settings)

    rows = read_delta(settings.output_root / "hierarchycategory").sort("hierarchy_id")
    assert rows.select("hierarchy_id", "id", "description", "level").to_dicts() == [
        {
            "hierarchy_id": "tasic_2018_visp_taxonomy",
            "id": "class",
            "description": "Top-level transcriptomic class.",
            "level": 2,
        },
        {
            "hierarchy_id": "visp_met_types_taxonomy",
            "id": "class",
            "description": "Top-level MET-type class.",
            "level": 1,
        },
    ]


def test_dry_run_does_not_write(tmp_path):
    """Dry runs must report no writes and create no tables."""
    settings = Settings(output_root=tmp_path, dry_run=True)
    ds = DataSet(id="d1", name="d", project_id="p1")

    result = write_models(ds, settings=settings)

    assert result.rows_written == 0, "dry_run must report 0 rows written"
    assert not (tmp_path / "dataset").exists(), "dry_run must not create tables"


def test_merge_batch_uses_one_identity_predicate(settings, read_delta):
    """One merge transaction must handle all identities in a batch."""
    rows_in = [
        DataSet(id="a", name="A", project_id="p1"),
        DataSet(id="b", name="B", project_id="p1"),
    ]
    result = write_models(rows_in, settings=settings)
    assert isinstance(result, WrittenResult)
    assert result.predicates == (
        'target."project_id" = source."project_id" '
        'AND target."id" = source."id" '
        'AND target."project_id" IN (\'p1\')',
    )
    assert result.rows_written == 2
    # Both end up in the table.
    rows = read_delta(settings.output_root / "dataset")
    assert sorted(rows["id"].to_list()) == ["a", "b"]


# ---------------------------------------------------------------------------
# merge_scoped semantics
# ---------------------------------------------------------------------------


def test_merge_scoped_inserts_and_updates_dataitems(settings, read_delta):
    """DataItem writes must insert unseen IDs and update existing metadata."""
    items_first = [
        DataItem(id="cell_1", name="cell_1", project_id="p1"),
        DataItem(id="cell_2", name="cell_2", project_id="p1"),
    ]
    r1 = write_models(items_first, settings=settings)
    assert r1.mode == "merge_scoped"
    assert r1.rows_written == 2

    items_second = [
        DataItem(id="cell_2", name="updated", project_id="p1"),
        DataItem(id="cell_3", name="cell_3", project_id="p1"),  # new
    ]
    r2 = write_models(items_second, settings=settings)
    assert r2.rows_written == 2

    rows = read_delta(settings.output_root / "dataitem").sort("id")
    assert sorted(rows["id"].to_list()) == ["cell_1", "cell_2", "cell_3"]
    assert rows.filter(pl.col("id") == "cell_2")["name"].item() == "updated"


def test_merge_scoped_counts_only_inserted_and_changed_rows(settings):
    """Unchanged matches must not contribute to rows_written."""
    initial = [
        DataItem(id="cell_1", name="one", project_id="p1"),
        DataItem(id="cell_2", name="two", project_id="p1"),
    ]
    write_models(initial, settings=settings)

    result = write_models(
        [
            DataItem(id="cell_1", name="one", project_id="p1"),
            DataItem(id="cell_2", name="updated", project_id="p1"),
            DataItem(id="cell_3", name="three", project_id="p1"),
        ],
        settings=settings,
    )

    assert result.rows_written == 2


def test_merge_scoped_preserves_shared_scope_contributions(settings, read_delta):
    """A later writer must not delete associations from an earlier writer."""
    first = [
        DataItemDataSetAssociation(
            project_id="p1", dataset_id="d1", dataitem_id="cell_1"
        ),
        DataItemDataSetAssociation(
            project_id="p1", dataset_id="d1", dataitem_id="cell_2"
        ),
    ]
    second = [
        DataItemDataSetAssociation(
            project_id="p1", dataset_id="d1", dataitem_id="cell_3"
        )
    ]

    write_models(first, settings=settings)
    write_models(second, settings=settings)

    rows = read_delta(settings.output_root / "dataitem_dataset_association")
    assert sorted(rows["dataitem_id"].to_list()) == ["cell_1", "cell_2", "cell_3"]


def test_merge_scoped_preserves_shared_hierarchy_memberships(settings, read_delta):
    """Excitatory and inhibitory writers must coexist in one hierarchy scope."""
    excitatory = [
        ClusterMembership(
            project_id="visp_patchseq",
            hierarchy_id="visp_met_types_taxonomy",
            item="exc_1",
            cluster="met_a",
        ),
        ClusterMembership(
            project_id="visp_patchseq",
            hierarchy_id="visp_met_types_taxonomy",
            item="exc_1",
            cluster="root",
        ),
    ]
    inhibitory = [
        ClusterMembership(
            project_id="visp_patchseq",
            hierarchy_id="visp_met_types_taxonomy",
            item="inh_1",
            cluster="met_b",
        ),
        ClusterMembership(
            project_id="visp_patchseq",
            hierarchy_id="visp_met_types_taxonomy",
            item="inh_1",
            cluster="root",
        ),
    ]

    write_models(excitatory, settings=settings)
    write_models(inhibitory, settings=settings)

    rows = read_delta(settings.output_root / "clustermembership")
    assert sorted(rows.select("item", "cluster").rows()) == [
        ("exc_1", "met_a"),
        ("exc_1", "root"),
        ("inh_1", "met_b"),
        ("inh_1", "root"),
    ]


def test_merge_scoped_deduplicates_incoming_batch(settings, read_delta):
    """Duplicate source identities must keep the final input row."""
    items = [
        DataItem(id="cell_1", name="first", project_id="p1"),
        DataItem(id="cell_2", name="other", project_id="p1"),
        DataItem(id="cell_1", name="last", project_id="p1"),
    ]

    result = write_models(items, settings=settings)

    assert result.rows_written == 2
    rows = read_delta(settings.output_root / "dataitem").sort("id")
    assert rows.select("id", "name").to_dicts() == [
        {"id": "cell_1", "name": "last"},
        {"id": "cell_2", "name": "other"},
    ]


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
        project_id="p1",
        measurement_type=ProjectionMeasurementType.MICRONS_OF_AXON,
        modality=Modality.MORPHOLOGY,
        laterality=Laterality.IPSILATERAL,
        unit=Unit.MICRONS_LENGTH,
        data_item_index=["cell_1"],
        region_index=["VISp"],
        values="file:///tmp/pmm.delta",
    ),
    AlgorithmRun: lambda: AlgorithmRun(id="run1", algorithm_name="kmeans"),
    HierarchyCategory: lambda: HierarchyCategory(
        id="cluster", hierarchy_id="h1", description="leaf", level=0
    ),
    SynapseFeatureMatrix: lambda: SynapseFeatureMatrix(
        id="sfm1",
        project_id="p1",
        dataset_id="ds1",
        parquet_path="file:///tmp/syn_wide.parquet",
        synapse_index_column="id",
    ),
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
    """Every writable class must have exactly one current smoke-test fixture."""
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
    """Every writable class must round-trip one row through its registered table."""
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
    """Projection writes must derive coverage without mutating their input."""
    pmm = ProjectionMeasurementMatrix(
        id="pmm_test",
        project_id="p1",
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


@pytest.mark.parametrize(
    "batch_factory",
    [
        pytest.param(lambda models: models[0], id="single-model"),
        pytest.param(list, id="list"),
        pytest.param(tuple, id="tuple"),
        pytest.param(lambda models: (model for model in models), id="generator"),
    ],
)
def test_write_models_accepts_supported_input_shapes(tmp_path, batch_factory):
    """One model, list, tuple, and one-shot generator inputs must all dispatch."""
    models = [
        DataSet(id="d1", name="one", project_id="p1"),
        DataSet(id="d2", name="two", project_id="p1"),
    ]
    settings = Settings(output_root=tmp_path, dry_run=True)

    result = write_models(batch_factory(models), settings=settings)

    assert result.class_name == "DataSet"
    assert result.rows_written == 0


def test_write_models_rejects_empty(settings):
    """The public writer must reject an empty batch before registry lookup or IO."""
    with pytest.raises(ValueError, match="empty"):
        write_models([], settings=settings)


def test_write_models_rejects_heterogeneous(settings):
    """A batch containing different Pydantic classes must fail with its index."""
    with pytest.raises(TypeError, match=r"index 1.*DataItem"):
        write_models(
            [
                DataSet(id="d1", name="d", project_id="p1"),
                DataItem(id="x", name="x", project_id="p1"),
            ],
            settings=settings,
        )


def test_write_models_rejects_homogeneous_non_models(settings):
    """An iterable containing only non-Pydantic objects must fail at its first item."""
    with pytest.raises(TypeError, match=r"index 0.*object"):
        write_models(iter([object(), object()]), settings=settings)  # type: ignore[arg-type]


def test_write_models_rejects_mixed_model_and_non_model(settings):
    """A non-Pydantic member later in an otherwise valid model batch must be named."""
    models = [
        DataSet(id="d1", name="d", project_id="p1"),
        object(),
    ]

    with pytest.raises(TypeError, match=r"index 1.*object"):
        write_models(models, settings=settings)  # type: ignore[arg-type]


def test_write_models_materializes_generator_before_rejecting_later_member(settings):
    """A one-shot generator must expose a later invalid member during normalization."""
    yielded: list[str] = []

    def models():
        yielded.append("DataSet")
        yield DataSet(id="d1", name="d", project_id="p1")
        yielded.append("DataItem")
        yield DataItem(id="i1", name="i", project_id="p1")

    with pytest.raises(TypeError, match=r"index 1.*DataItem"):
        write_models(models(), settings=settings)

    assert yielded == ["DataSet", "DataItem"]


def test_write_models_rejects_unregistered_class(settings):
    """The public writer must reject objects outside the model hierarchy."""
    class NotInRegistry:
        pass

    with pytest.raises(TypeError, match="pydantic model or iterable"):
        write_models(NotInRegistry(), settings=settings)


def test_write_models_rejects_unregistered_pydantic_model(settings):
    """The public writer must reject Pydantic models absent from the registry."""
    class UnregisteredModel(BaseModel):
        id: str

    with pytest.raises(KeyError, match="UnregisteredModel"):
        write_models(UnregisteredModel(id="u1"), settings=settings)


# ---------------------------------------------------------------------------
# Per-call output_root override
# ---------------------------------------------------------------------------


def test_write_models_output_root_works_without_discoverable_config(tmp_path):
    """An explicit root uses default controls when no config is discoverable."""
    output_root = tmp_path / "isolated_dataset"
    ds = DataSet(id="d_isolated", name="isolated", project_id="p_isolated")

    result = write_models(ds, output_root=output_root)

    assert result.path == output_root / "dataset"
    rows = pl.read_delta(str(output_root / "dataset")).filter(
        pl.col("id") == "d_isolated"
    )
    assert rows.shape[0] == 1


def test_write_models_without_settings_or_output_root_requires_config():
    """A write with no configuration source must retain the discovery error."""
    ds = DataSet(id="d_missing", name="missing", project_id="p_missing")

    with pytest.raises(ConfigNotFoundError, match="ccc_config.yaml"):
        write_models(ds)


def test_write_models_output_root_does_not_hide_malformed_config(tmp_path):
    """An explicit root must not suppress errors from a discovered config."""
    (tmp_path / "ccc_config.yaml").write_text("- invalid\n")
    ds = DataSet(id="d_invalid", name="invalid", project_id="p_invalid")

    with pytest.raises(RuntimeError, match="expected a YAML mapping"):
        write_models(ds, output_root=tmp_path / "isolated_dataset")


def test_write_models_output_root_override_writes_to_given_path(tmp_path):
    """Passing output_root= overrides the root from discovered settings."""
    config = {
        "output_root": str(tmp_path / "configured"),
        "dry_run": False,
    }
    (tmp_path / "ccc_config.yaml").write_text(yaml.safe_dump(config))
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
    settings = Settings(output_root=tmp_path / "configured")
    alt_root = tmp_path / "string_root"
    ds = DataSet(id="d_str", name="s", project_id="p_str")

    result = write_models(ds, settings=settings, output_root=str(alt_root))

    assert result.path == alt_root / "dataset"


def test_write_models_output_root_preserves_settings_dry_run(tmp_path):
    """A root override must not discard dry-run controls from settings."""
    settings = Settings(output_root=tmp_path / "configured", dry_run=True)
    alt_root = tmp_path / "other"
    ds = DataSet(id="d_x", name="x", project_id="p_x")

    result = write_models(ds, settings=settings, output_root=alt_root)

    assert result.path == alt_root / "dataset"
    assert result.rows_written == 0
    assert not (alt_root / "dataset").exists()


def test_write_models_output_root_preserves_discovered_dry_run(tmp_path):
    """A root override must retain dry-run controls from discovered settings."""
    config = {
        "output_root": str(tmp_path / "configured"),
        "dry_run": True,
    }
    (tmp_path / "ccc_config.yaml").write_text(yaml.safe_dump(config))
    alt_root = tmp_path / "other"
    ds = DataSet(id="d_discovered", name="x", project_id="p_x")

    result = write_models(ds, output_root=alt_root)

    assert result.path == alt_root / "dataset"
    assert result.rows_written == 0
    assert not (alt_root / "dataset").exists()


def test_write_projection_matrix_output_root_override(tmp_path):
    """write_projection_matrix forwards output_root through write_models."""
    settings = Settings(output_root=tmp_path / "configured")
    alt_root = tmp_path / "pmm_alt"
    pmm = ProjectionMeasurementMatrix(
        id="pmm_alt",
        project_id="p1",
        measurement_type=ProjectionMeasurementType.MICRONS_OF_AXON,
        modality=Modality.MORPHOLOGY,
        laterality=Laterality.IPSILATERAL,
        unit=Unit.MICRONS_LENGTH,
        data_item_index=["c1", "c2"],
        region_index=["r1", "r2"],
        values="file:///tmp/pmm_alt.delta",
    )
    matrix = np.array([[1.0, 0.0], [0.0, 2.0]])

    result = write_projection_matrix(
        pmm, matrix, settings=settings, output_root=alt_root
    )

    assert result.path == alt_root / "projectionmeasurementmatrix"


def test_write_projection_matrix_output_root_preserves_dry_run(tmp_path):
    """Projection root overrides must retain settings-based dry-run controls."""
    settings = Settings(output_root=tmp_path / "configured", dry_run=True)
    alt_root = tmp_path / "other"
    pmm = ProjectionMeasurementMatrix(
        id="pmm_x",
        project_id="p1",
        measurement_type=ProjectionMeasurementType.MICRONS_OF_AXON,
        modality=Modality.MORPHOLOGY,
        laterality=Laterality.IPSILATERAL,
        unit=Unit.MICRONS_LENGTH,
        data_item_index=["c1"],
        region_index=["r1"],
        values="file:///tmp/pmm_x.delta",
    )
    matrix = np.array([[1.0]])

    result = write_projection_matrix(
        pmm, matrix, settings=settings, output_root=alt_root
    )

    assert result.path == alt_root / "projectionmeasurementmatrix"
    assert result.rows_written == 0
    assert not (alt_root / "projectionmeasurementmatrix").exists()
