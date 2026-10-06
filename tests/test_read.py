from __future__ import annotations

from pathlib import Path

import polars as pl
import pyarrow as pa
import pytest
from deltalake import write_deltalake

from connects_common_connectivity.config import Settings
from connects_common_connectivity.io import (
    DatasetReader,
    read_cell_cell_connectivity,
    read_reference_spaces,
    read_spatial_locations,
    read_synapse_table,
    write_models,
)
from connects_common_connectivity.models import (
    Default2DView,
    LocationType,
    ReferenceSpace,
    SignedAxis,
    SpatialLocation,
)


@pytest.fixture
def spatial_root(tmp_path):
    """Provide global and project-owned spaces with coordinates across cells and location types."""
    write_models([
        ReferenceSpace(id="CCF_v3", default_2d_view=Default2DView(
            left_to_right=SignedAxis.PLUS_Z, bottom_to_top=SignedAxis.MINUS_Y)),
        ReferenceSpace(id="CCF_v3", project_id="first"),
        ReferenceSpace(id="CCF_v3", project_id="second"),
        ReferenceSpace(id="first_original", project_id="first"),
        ReferenceSpace(id="second_original", project_id="second"),
    ], output_root=tmp_path)
    write_models([
        SpatialLocation(project_id=project, dataitem_id=cell, reference_space=space,
                        location_type=kind, x=1.0, y=2.0, z=3.0)
        for project in ("first", "second")
        for cell in ("a", "b")
        for space in ("CCF_v3", f"{project}_original")
        for kind in (LocationType.SOMA, LocationType.CENTROID)
    ], output_root=tmp_path)
    return tmp_path


def test_spatial_reader_filters_and_preserves_coordinates(spatial_root):
    """Spatial filters must compose within a project without transforming coordinate values."""
    result = read_spatial_locations(
        "first", reference_spaces="CCF_v3", dataitem_ids=["a"],
        location_types=LocationType.SOMA, output_root=spatial_root,
    )
    assert result.select(
        "project_id", "dataitem_id", "location_type", "x", "y", "z"
    ).rows() == [
        ("first", "a", "SOMA", 1.0, 2.0, 3.0)
    ]
    assert read_spatial_locations("first", output_root=spatial_root).height == 8
    assert read_spatial_locations(
        "second", location_types=[LocationType.CENTROID],
        settings=Settings(output_root=spatial_root),
    ).height == 4


@pytest.mark.parametrize("filters", [
    {"dataitem_ids": []}, {"dataitem_ids": "missing"},
    {"reference_spaces": []}, {"reference_spaces": "missing"},
    {"location_types": []}, {"location_types": "missing"},
])
def test_spatial_reader_empty_matches_keep_schema(spatial_root, filters):
    """Empty selections and unmatched IDs must return zero rows with the stored column types."""
    expected = read_spatial_locations("first", output_root=spatial_root)
    empty = read_spatial_locations("first", output_root=spatial_root, **filters)
    assert empty.is_empty()
    assert empty.schema == expected.schema


@pytest.mark.parametrize("filters,expected_filters", [
    ({"dataitem_ids": ["a", "missing"]}, {"dataitem_ids": "a"}),
    ({"reference_spaces": ["CCF_v3", "missing"]}, {"reference_spaces": "CCF_v3"}),
    ({"location_types": [LocationType.SOMA, "missing"]}, {"location_types": "SOMA"}),
])
def test_spatial_reader_mixed_values_keep_matches(spatial_root, filters, expected_filters):
    """Unknown filter values must not discard rows matching known values."""
    expected = read_spatial_locations("first", output_root=spatial_root, **expected_filters)
    result = read_spatial_locations("first", output_root=spatial_root, **filters)
    assert not result.is_empty()
    assert result.equals(expected)


def test_reference_space_reader_selects_exact_scope(spatial_root):
    """Exact scope reads preserve view structs, nulls, and empty schemas."""
    assert read_reference_spaces(output_root=spatial_root).height == 5
    visible = read_reference_spaces(project_id="first", output_root=spatial_root)
    assert set(visible["id"]) == {"CCF_v3", "first_original"}
    assert visible["project_id"].to_list() == ["first", "first"]
    assert isinstance(visible.schema["default_2d_view"], pl.Struct)
    global_spaces = read_reference_spaces(project_id=None, output_root=spatial_root)
    assert global_spaces["id"].to_list() == ["CCF_v3"]
    assert global_spaces["project_id"].to_list() == [None]
    assert global_spaces["default_2d_view"].to_list() == [
        {"left_to_right": "PLUS_Z", "bottom_to_top": "MINUS_Y"}
    ]
    original = read_reference_spaces(reference_space_ids="first_original", output_root=spatial_root)
    assert original["default_2d_view"].to_list() == [None]
    assert read_reference_spaces(
        project_id="missing", settings=Settings(output_root=spatial_root)
    ).is_empty()
    empty = read_reference_spaces(reference_space_ids=[], output_root=spatial_root)
    assert empty.is_empty()
    assert empty.schema == visible.schema


@pytest.mark.parametrize("filters,expected_projects", [
    ({}, {None, "first", "second"}),
    ({"project_id": None}, {None}),
    ({"project_id": "first"}, {"first"}),
    ({"project_id": "second"}, {"second"}),
    ({"project_id": "missing"}, set()),
])
def test_reference_space_reader_disambiguates_shared_ids(
    spatial_root, filters, expected_projects
):
    """Scope filters must distinguish reference spaces that share an ID without global fallback."""
    result = read_reference_spaces(
        reference_space_ids="CCF_v3", output_root=spatial_root, **filters
    )
    assert set(result["project_id"]) == expected_projects
    assert result.height == len(expected_projects)


@pytest.mark.parametrize("reader,args", [
    (read_spatial_locations, ("first",)), (read_reference_spaces, ()),
])
def test_spatial_readers_missing_tables_and_root_conflict(reader, args, tmp_path):
    """Both spatial readers must reject missing storage and conflicting root overrides."""
    with pytest.raises(FileNotFoundError):
        reader(*args, output_root=tmp_path)
    with pytest.raises(TypeError, match="either settings=.*output_root"):
        reader(*args, output_root=tmp_path, settings=Settings(output_root=tmp_path))


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
    """Provide dataset-centric tables for DatasetReader tests."""
    return _build_reader_root(tmp_path)


@pytest.fixture
def cell_cell_root(tmp_path: Path) -> Path:
    """Provide canonical cell-cell rows spanning source and connectome scopes."""
    root = tmp_path / "cell-cell-root"
    _write_table(
        root,
        "cellcellconnectivitylong",
        {
            "id": [
                "c1-count",
                "c1-size",
                "c1-other-pre",
                "c2-count",
                "other-source",
                "p2-count",
            ],
            "connectome_id": [
                "connectome-1",
                "connectome-1",
                "connectome-1",
                "connectome-2",
                "connectome-1",
                "connectome-1",
            ],
            "synapse_table_id": [
                "synapses-1",
                "synapses-1",
                "synapses-1",
                "synapses-1",
                "synapses-2",
                "synapses-1",
            ],
            "presynaptic_cell": [
                "pre-1",
                "pre-1",
                "pre-2",
                "pre-1",
                "pre-1",
                "pre-1",
            ],
            "postsynaptic_cell": ["post-1"] * 6,
            "measurement_type": [
                "SYNAPSE_COUNT",
                "SUM_ANATOMICAL_SIZE",
                "SYNAPSE_COUNT",
                "SYNAPSE_COUNT",
                "SYNAPSE_COUNT",
                "SYNAPSE_COUNT",
            ],
            "modality": ["ELECTRON_MICROSCOPY"] * 6,
            "value": [2.0, 5.0, 1.0, 4.0, 9.0, 8.0],
            "unit": [
                "COUNT",
                "MICRONS_SQUARE",
                "COUNT",
                "COUNT",
                "COUNT",
                "COUNT",
            ],
            "project_id": [
                "project-1",
                "project-1",
                "project-1",
                "project-1",
                "project-1",
                "project-2",
            ],
        },
    )
    return root


@pytest.fixture
def synapse_root(tmp_path: Path) -> Path:
    """Provide long synapses and one matching wide feature table."""
    root = tmp_path / "synapse-root"
    _write_table(
        root,
        "synapse",
        {
            "id": ["s1", "s2", "s3", "s4"],
            "presynaptic_cell": ["pre-1", "pre-1", "pre-2", "pre-1"],
            "postsynaptic_cell": ["post-1", "post-2", "post-1", "post-1"],
            "synapse_table_id": [
                "synapses-1",
                "synapses-1",
                "synapses-2",
                "synapses-1",
            ],
            "project_id": ["project-1", "project-1", "project-1", "project-2"],
        },
    )
    _write_table(
        root,
        "synapsefeatures/features-1",
        {
            "id": ["s1", "s2", "s3", "s4"],
            "size": [1.0, 2.0, 3.0, 4.0],
            "synapse_table_id": [
                "synapses-1",
                "synapses-1",
                "synapses-2",
                "synapses-1",
            ],
            "project_id": ["project-1", "project-1", "project-1", "project-2"],
        },
    )
    return root


def test_read_cell_cell_connectivity_requires_connectome_scope(cell_cell_root: Path):
    """Cell-cell reads must identify both project and connectome context."""
    with pytest.raises(TypeError):
        read_cell_cell_connectivity(output_root=cell_cell_root)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        read_cell_cell_connectivity(  # type: ignore[call-arg]
            "project-1",
            output_root=cell_cell_root,
        )


def test_read_cell_cell_connectivity_selects_project_and_connectome(
    cell_cell_root: Path,
):
    """An optional connectome filter must isolate derived contexts."""
    first = read_cell_cell_connectivity(
        "project-1",
        "connectome-1",
        output_root=cell_cell_root,
    )
    second = read_cell_cell_connectivity(
        "project-1",
        "connectome-2",
        output_root=cell_cell_root,
    )

    assert first["id"].to_list() == [
        "c1-count",
        "c1-size",
        "c1-other-pre",
        "other-source",
    ]
    assert second["id"].to_list() == ["c2-count"]


def test_read_cell_cell_connectivity_can_filter_optional_source_provenance(
    cell_cell_root: Path,
):
    """Source-table provenance may narrow an already selected connectome."""
    result = read_cell_cell_connectivity(
        "project-1",
        "connectome-1",
        synapse_table_id="synapses-1",
        output_root=cell_cell_root,
    )

    assert result["id"].to_list() == [
        "c1-count",
        "c1-size",
        "c1-other-pre",
    ]


def test_read_cell_cell_connectivity_without_source_provenance(tmp_path: Path):
    """Cell-cell tables need not identify a source synapse table."""
    root = tmp_path / "cell-cell-without-provenance"
    _write_table(
        root,
        "cellcellconnectivitylong",
        {
            "id": ["measurement-1"],
            "connectome_id": ["connectome-1"],
            "presynaptic_cell": ["pre-1"],
            "postsynaptic_cell": ["post-1"],
            "measurement_type": ["SYNAPSE_COUNT"],
            "modality": ["ELECTRON_MICROSCOPY"],
            "value": [2.0],
            "unit": ["COUNT"],
            "project_id": ["project-1"],
        },
    )

    result = read_cell_cell_connectivity(
        "project-1",
        "connectome-1",
        output_root=root,
    )

    assert result["id"].to_list() == ["measurement-1"]


def test_read_cell_cell_connectivity_rejects_unavailable_provenance_filter(
    tmp_path: Path,
):
    """An explicit provenance filter requires provenance in storage."""
    root = tmp_path / "cell-cell-without-provenance"
    _write_table(
        root,
        "cellcellconnectivitylong",
        {
            "id": ["measurement-1"],
            "connectome_id": ["connectome-1"],
            "presynaptic_cell": ["pre-1"],
            "postsynaptic_cell": ["post-1"],
            "measurement_type": ["SYNAPSE_COUNT"],
            "modality": ["ELECTRON_MICROSCOPY"],
            "value": [2.0],
            "unit": ["COUNT"],
            "project_id": ["project-1"],
        },
    )

    with pytest.raises(
        ValueError,
        match="stored table does not contain source-synapse provenance",
    ):
        read_cell_cell_connectivity(
            "project-1",
            "connectome-1",
            synapse_table_id="synapses-1",
            output_root=root,
        )


@pytest.mark.parametrize(
    ("filters", "expected_ids"),
    [
        ({"presynaptic_cells": "pre-1"}, ["c1-count", "c1-size"]),
        ({"presynaptic_cells": ["pre-2"]}, ["c1-other-pre"]),
        ({"postsynaptic_cells": ["post-1"]}, ["c1-count", "c1-size", "c1-other-pre"]),
        ({"measurement_types": ["SUM_ANATOMICAL_SIZE"]}, ["c1-size"]),
        (
            {
                "presynaptic_cells": ["pre-1"],
                "postsynaptic_cells": ["post-1"],
                "measurement_types": ["SYNAPSE_COUNT"],
            },
            ["c1-count"],
        ),
    ],
)
def test_read_cell_cell_connectivity_applies_explicit_filters(
    cell_cell_root: Path,
    filters: dict,
    expected_ids: list[str],
):
    """Cell-cell reads must compose explicit endpoint and measurement filters."""
    result = read_cell_cell_connectivity(
        "project-1",
        "connectome-1",
        synapse_table_id="synapses-1",
        output_root=cell_cell_root,
        **filters,
    )

    assert result["id"].to_list() == expected_ids


@pytest.mark.parametrize(
    "filters",
    [
        {"presynaptic_cells": []},
        {"postsynaptic_cells": ["missing"]},
        {"measurement_types": ["EXISTENCE"]},
    ],
)
def test_read_cell_cell_connectivity_returns_typed_empty_results(
    cell_cell_root: Path,
    filters: dict,
):
    """Valid filters with no matches must retain the stored table schema."""
    result = read_cell_cell_connectivity(
        "project-1",
        "connectome-1",
        synapse_table_id="synapses-1",
        output_root=cell_cell_root,
        **filters,
    )

    assert result.is_empty()
    assert result.schema["id"] == pl.String


def test_read_cell_cell_connectivity_resolves_settings(cell_cell_root: Path):
    """Cell-cell reads must resolve storage from explicit settings."""
    result = read_cell_cell_connectivity(
        "project-1",
        "connectome-2",
        synapse_table_id="synapses-1",
        settings=Settings(output_root=cell_cell_root),
    )

    assert result["id"].to_list() == ["c2-count"]


def test_read_cell_cell_connectivity_requires_canonical_table(tmp_path: Path):
    """Missing canonical cell-cell storage must fail with its expected path."""
    root = tmp_path / "missing-cell-cell"
    root.mkdir()

    with pytest.raises(FileNotFoundError, match="cellcellconnectivitylong"):
        read_cell_cell_connectivity(
            "project-1",
            "connectome-1",
            output_root=root,
        )


def test_read_synapse_table_scopes_and_filters_endpoints(synapse_root: Path):
    """Synapse reads must share project, table, and endpoint selectors."""
    result = read_synapse_table(
        "project-1",
        synapse_table_id="synapses-1",
        presynaptic_cells="pre-1",
        postsynaptic_cells=["post-2"],
        output_root=synapse_root,
    )

    assert result["id"].to_list() == ["s2"]


def test_read_synapse_table_requires_source_scope(synapse_root: Path):
    """Synapse reads must identify the logical table within a project."""
    with pytest.raises(TypeError):
        read_synapse_table(  # type: ignore[call-arg]
            "project-1",
            output_root=synapse_root,
        )


def test_read_synapse_table_joins_requested_features(synapse_root: Path):
    """Synapse reads must add requested features after applying source scope."""
    result = read_synapse_table(
        "project-1",
        synapse_table_id="synapses-1",
        features=["size"],
        feature_matrix_id="features-1",
        output_root=synapse_root,
    )

    assert result.select("id", "size").rows() == [("s1", 1.0), ("s2", 2.0)]


def test_displays_datasets_and_discovers_related_sets(reader_root: Path):
    """Dataset displays must expose only sets related to each dataset."""
    reader = DatasetReader(reader_root)

    datasets = reader.display_dataset_names()
    assert datasets["id"].to_list() == ["dataset_a", "dataset_b"]

    features = reader.display_featuresets("dataset_a")
    assert features["feature_set_id"].to_list() == ["features_a", "features_b"]

    clusters = reader.display_clustersets("dataset_a")
    assert clusters["clusterset_id"].to_list() == ["hierarchy_a"]


def test_read_dataset_joins_features_and_cluster_levels(reader_root: Path):
    """Dataset reads must left-join selected features and cluster levels."""
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
    """Dataset reads must honor explicit and empty related-set selections."""
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
    """Cluster discovery must not leak memberships across projects."""
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
    """Unknown dataset and related-set names must fail clearly."""
    reader = DatasetReader(reader_root)

    with pytest.raises(KeyError, match="Unknown dataset"):
        reader.read_dataset("missing")
    with pytest.raises(KeyError, match="Unknown feature set"):
        reader.read_dataset("dataset_a", featuresets="missing")
    with pytest.raises(KeyError, match="Unknown cluster set"):
        reader.read_dataset("dataset_a", clustersets="missing")


def test_duplicate_feature_names_raise(reader_root: Path):
    """Feature sets with colliding output columns must be rejected."""
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
    """Multiple cluster assignments at one hierarchy level must fail."""
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
    """DatasetReader must reject roots missing its required model tables."""
    root = tmp_path / "empty-root"
    root.mkdir()

    with pytest.raises(FileNotFoundError, match="Required Delta table"):
        DatasetReader(root)
