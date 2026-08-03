from __future__ import annotations

from pathlib import Path

import polars as pl
import pyarrow as pa
import pytest
from deltalake import write_deltalake

from connects_common_connectivity.io import DatasetReader


def _write_table(root: Path, subdir: str, data: dict) -> None:
    write_deltalake(str(root / subdir), pa.table(data), mode="overwrite")


def _build_reader_root(tmp_path: Path) -> Path:
    root = tmp_path / "reader-root"
    _write_table(
        root,
        "dataset",
        {
            "id": ["dataset_a", "dataset_b"],
            "name": ["Dataset A", "Dataset B"],
            "project_id": ["project_a", "project_b"],
            "modality": ["ELECTRON_MICROSCOPY", "ELECTRON_MICROSCOPY"],
        },
    )
    _write_table(
        root,
        "dataitem_dataset_association",
        {
            "dataitem_id": ["a", "b", "c", "a"],
            "dataset_id": ["dataset_a", "dataset_a", "dataset_a", "dataset_b"],
            "project_id": ["project_a", "project_a", "project_a", "project_b"],
        },
    )
    _write_table(
        root,
        "cellfeatureset",
        {
            "id": ["features_a", "features_b", "features_unrelated"],
            "description": ["First", "Second", "Unrelated"],
            "extraction_method": ["method-a", "method-b", "method-z"],
            "feature_definition_ids": [["alpha"], ["beta"], ["zeta"]],
            "project_id": ["project_a", "project_a", "project_a"],
        },
    )
    _write_table(
        root,
        "cellfeaturematrix",
        {
            "id": ["matrix_a", "matrix_b", "matrix_unrelated"],
            "feature_set_id": [
                "features_a",
                "features_b",
                "features_unrelated",
            ],
            "cell_index_column": ["id", "cell_key", "id"],
            "project_id": ["project_a", "project_a", "project_a"],
        },
    )
    _write_table(
        root,
        "cellfeatures/features_a",
        {
            "id": ["a", "b"],
            "alpha": [1.0, 2.0],
            "project_id": ["project_a", "project_a"],
            "feature_set_id": ["features_a", "features_a"],
        },
    )
    _write_table(
        root,
        "cellfeatures/features_b",
        {
            "cell_key": ["b", "c"],
            "beta": [20, 30],
            "project_id": ["project_a", "project_a"],
            "feature_set_id": ["features_b", "features_b"],
        },
    )
    _write_table(
        root,
        "cellfeatures/features_unrelated",
        {
            "id": ["z"],
            "zeta": [99],
            "project_id": ["project_a"],
            "feature_set_id": ["features_unrelated"],
        },
    )
    _write_table(
        root,
        "clusterhierarchy",
        {
            "id": ["hierarchy_a", "hierarchy_b"],
            "run": ["run-a", "run-b"],
            "root": ["root-a", "root-b"],
            "clusters": [
                ["root-a", "type-a", "type-b"],
                ["root-b", "other"],
            ],
        },
    )
    _write_table(
        root,
        "cluster",
        {
            "id": ["root-a", "type-a", "type-b", "root-b", "other"],
            "hierarchy_id": [
                "hierarchy_a",
                "hierarchy_a",
                "hierarchy_a",
                "hierarchy_b",
                "hierarchy_b",
            ],
            "level": [0, 1, 1, 0, 1],
        },
    )
    _write_table(
        root,
        "clustermembership",
        {
            "item": ["a", "a", "b", "b", "a", "a"],
            "cluster": [
                "root-a",
                "type-a",
                "root-a",
                "type-b",
                "root-b",
                "other",
            ],
            "project_id": [
                "project_a",
                "project_a",
                "project_a",
                "project_a",
                "project_b",
                "project_b",
            ],
            "hierarchy_id": [
                "hierarchy_a",
                "hierarchy_a",
                "hierarchy_a",
                "hierarchy_a",
                "hierarchy_b",
                "hierarchy_b",
            ],
        },
    )
    return root


@pytest.fixture
def reader_root(tmp_path: Path) -> Path:
    return _build_reader_root(tmp_path)


def test_displays_datasets_and_discovers_related_sets(reader_root: Path):
    reader = DatasetReader(reader_root)

    datasets = reader.display_dataset_names()
    assert datasets["id"].to_list() == ["dataset_a", "dataset_b"]

    features = reader.display_featuresets("dataset_a")
    assert features["feature_set_id"].to_list() == ["features_a", "features_b"]

    clusters = reader.display_clustersets("dataset_a")
    assert clusters["clusterset_id"].to_list() == ["hierarchy_a"]


def test_read_dataset_joins_features_and_cluster_levels(reader_root: Path):
    table = DatasetReader(reader_root).read_dataset("dataset_a")

    assert table["dataitem_id"].to_list() == ["a", "b", "c"]
    assert table.columns == [
        "dataitem_id",
        "alpha",
        "beta",
        "hierarchy_a_level_0",
        "hierarchy_a_level_1",
    ]
    assert table.filter(pl.col("dataitem_id") == "a").row(0, named=True) == {
        "dataitem_id": "a",
        "alpha": 1.0,
        "beta": None,
        "hierarchy_a_level_0": "root-a",
        "hierarchy_a_level_1": "type-a",
    }
    assert table.filter(pl.col("dataitem_id") == "c").row(0, named=True) == {
        "dataitem_id": "c",
        "alpha": None,
        "beta": 30,
        "hierarchy_a_level_0": None,
        "hierarchy_a_level_1": None,
    }


def test_read_dataset_accepts_subset_and_empty_selections(reader_root: Path):
    reader = DatasetReader(reader_root)

    subset = reader.read_dataset(
        "dataset_a",
        featuresets="features_b",
        clustersets=[],
    )
    assert subset.columns == ["dataitem_id", "beta"]

    ids_only = reader.read_dataset(
        "dataset_a",
        featuresets=[],
        clustersets=[],
    )
    assert ids_only.columns == ["dataitem_id"]
    assert ids_only.height == 3


def test_project_scoping_excludes_other_project_memberships(reader_root: Path):
    table = DatasetReader(reader_root).read_dataset(
        "dataset_b",
        featuresets=[],
    )

    assert table.columns == [
        "dataitem_id",
        "hierarchy_b_level_0",
        "hierarchy_b_level_1",
    ]
    assert table.row(0, named=True) == {
        "dataitem_id": "a",
        "hierarchy_b_level_0": "root-b",
        "hierarchy_b_level_1": "other",
    }


def test_unknown_dataset_and_selectors_raise_clear_errors(reader_root: Path):
    reader = DatasetReader(reader_root)

    with pytest.raises(KeyError, match="Unknown dataset"):
        reader.read_dataset("missing")
    with pytest.raises(KeyError, match="Unknown feature set"):
        reader.read_dataset("dataset_a", featuresets="missing")
    with pytest.raises(KeyError, match="Unknown cluster set"):
        reader.read_dataset("dataset_a", clustersets="missing")


def test_duplicate_feature_names_raise(reader_root: Path):
    _write_table(
        reader_root,
        "cellfeatureset",
        {
            "id": [
                "features_a",
                "features_b",
                "features_unrelated",
                "features_duplicate",
            ],
            "description": ["First", "Second", "Unrelated", "Duplicate"],
            "extraction_method": [
                "method-a",
                "method-b",
                "method-z",
                "method-d",
            ],
            "feature_definition_ids": [
                ["alpha"],
                ["beta"],
                ["zeta"],
                ["alpha"],
            ],
            "project_id": [
                "project_a",
                "project_a",
                "project_a",
                "project_a",
            ],
        },
    )
    _write_table(
        reader_root,
        "cellfeaturematrix",
        {
            "id": [
                "matrix_a",
                "matrix_b",
                "matrix_unrelated",
                "matrix_duplicate",
            ],
            "feature_set_id": [
                "features_a",
                "features_b",
                "features_unrelated",
                "features_duplicate",
            ],
            "cell_index_column": ["id", "cell_key", "id", "id"],
            "project_id": [
                "project_a",
                "project_a",
                "project_a",
                "project_a",
            ],
        },
    )
    _write_table(
        reader_root,
        "cellfeatures/features_duplicate",
        {
            "id": ["a"],
            "alpha": [100.0],
            "project_id": ["project_a"],
            "feature_set_id": ["features_duplicate"],
        },
    )

    with pytest.raises(ValueError, match="duplicate feature columns"):
        DatasetReader(reader_root).read_dataset("dataset_a")


def test_duplicate_same_level_cluster_assignments_raise(reader_root: Path):
    _write_table(
        reader_root,
        "clustermembership",
        {
            "item": ["a", "a", "a"],
            "cluster": ["root-a", "type-a", "type-b"],
            "project_id": ["project_a", "project_a", "project_a"],
            "hierarchy_id": ["hierarchy_a", "hierarchy_a", "hierarchy_a"],
        },
    )

    with pytest.raises(ValueError, match="multiple clusters at the same level"):
        DatasetReader(reader_root).read_dataset(
            "dataset_a",
            featuresets=[],
        )


def test_missing_required_tables_raise(tmp_path: Path):
    root = tmp_path / "empty-root"
    root.mkdir()

    with pytest.raises(FileNotFoundError, match="Required Delta table"):
        DatasetReader(root)
