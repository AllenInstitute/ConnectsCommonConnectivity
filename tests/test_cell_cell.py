from __future__ import annotations

import polars as pl
import pytest
from pydantic import ValidationError

from connects_common_connectivity.io import (
    cell_cell_connectivity_to_arrow,
    derive_cell_cell_connectivity,
)
from connects_common_connectivity.io.arrow_utils import build_arrow_schema
from connects_common_connectivity.models import (
    CellCellConnectivityLong,
    Modality,
    SynapseConnectivityLong,
    SynapseFeatureMatrix,
    SynapticMeasurementType,
    Unit,
)


def test_cell_cell_connectivity_requires_connectome_id():
    """Cell-cell rows must identify their measurement context."""
    with pytest.raises(
        ValidationError,
        match=r"(?s)connectome_id.*Field required",
    ):
        CellCellConnectivityLong(
            id="measurement-1",
            synapse_table_id="synapses-1",
            presynaptic_cell="pre-1",
            postsynaptic_cell="post-1",
            measurement_type=SynapticMeasurementType.SYNAPSE_COUNT,
            modality=Modality.ELECTRON_MICROSCOPY,
            project_id="project-1",
            value=1.0,
            unit=Unit.COUNT,
        )


def test_cell_cell_connectivity_accepts_connectome_id():
    """A complete cell-cell row need not have synapse-table provenance."""
    measurement = CellCellConnectivityLong(
        id="measurement-1",
        connectome_id="connectome-1",
        presynaptic_cell="pre-1",
        postsynaptic_cell="post-1",
        measurement_type=SynapticMeasurementType.SYNAPSE_COUNT,
        modality=Modality.ELECTRON_MICROSCOPY,
        project_id="project-1",
        value=1.0,
        unit=Unit.COUNT,
    )

    assert measurement.connectome_id == "connectome-1"
    assert measurement.synapse_table_id is None


@pytest.mark.parametrize(
    "model_cls, kwargs",
    [
        (
            SynapseConnectivityLong,
            {
                "id": "synapse-1",
                "presynaptic_cell": "pre-1",
                "postsynaptic_cell": "post-1",
                "project_id": "project-1",
            },
        ),
        (
            SynapseFeatureMatrix,
            {
                "id": "features-1",
                "parquet_path": "file:///tmp/features.parquet",
                "project_id": "project-1",
            },
        ),
    ],
)
def test_synapse_contracts_require_synapse_table_id(model_cls, kwargs):
    """Synapse rows and feature pointers must name their logical table."""
    with pytest.raises(
        ValidationError,
        match=r"(?s)synapse_table_id.*Field required",
    ):
        model_cls(**kwargs)


def test_cell_cell_contract_requires_measurement_fields_not_provenance():
    """Derived rows require measurement identity but not source provenance."""
    required = {
        "presynaptic_cell",
        "postsynaptic_cell",
        "measurement_type",
        "modality",
    }

    assert required <= {
        name
        for name, field in CellCellConnectivityLong.model_fields.items()
        if field.is_required()
    }
    assert not CellCellConnectivityLong.model_fields[
        "synapse_table_id"
    ].is_required()


def test_derives_counts_for_endpoint_pairs_in_project():
    """Counts must aggregate independently by endpoint pair."""
    synapses = pl.DataFrame(
        {
            "project_id": ["project-1"] * 3,
            "synapse_table_id": ["synapses-1"] * 3,
            "presynaptic_cell": ["pre-1", "pre-1", "pre-1"],
            "postsynaptic_cell": ["post-1", "post-1", "post-2"],
        }
    )

    result = derive_cell_cell_connectivity(
        synapses,
        project_id="project-1",
        connectome_id="connectome-1",
        modality=Modality.ELECTRON_MICROSCOPY,
    ).sort("project_id", "presynaptic_cell", "postsynaptic_cell")

    assert result["value"].to_list() == [2.0, 1.0]
    assert result["measurement_type"].unique().to_list() == [
        SynapticMeasurementType.SYNAPSE_COUNT.value
    ]
    assert result["unit"].unique().to_list() == [Unit.COUNT.value]
    assert result["connectome_id"].unique().to_list() == ["connectome-1"]


@pytest.mark.parametrize(
    "input_projects",
    [["project-1", "project-2"], ["project-2"]],
)
def test_project_scope_must_match_all_input_rows(input_projects):
    """A connectome derivation must not span or silently select projects."""
    synapses = pl.DataFrame(
        {
            "project_id": input_projects,
            "presynaptic_cell": ["pre-1"] * len(input_projects),
            "postsynaptic_cell": ["post-1"] * len(input_projects),
        }
    )

    with pytest.raises(ValueError, match="must all match 'project-1'"):
        derive_cell_cell_connectivity(
            synapses,
            project_id="project-1",
            connectome_id="connectome-1",
            modality=Modality.ELECTRON_MICROSCOPY,
        )


def test_derives_optional_size_sum_with_explicit_unit():
    """Requested size totals must use the caller's explicit unit."""
    synapses = pl.DataFrame(
        {
            "project_id": ["project-1", "project-1"],
            "synapse_table_id": ["synapses-1", "synapses-1"],
            "presynaptic_cell": ["pre-1", "pre-1"],
            "postsynaptic_cell": ["post-1", "post-1"],
            "size": [2, 3],
        }
    )

    result = derive_cell_cell_connectivity(
        synapses,
        project_id="project-1",
        connectome_id="connectome-1",
        modality=Modality.ELECTRON_MICROSCOPY,
        size_column="size",
        size_unit=Unit.MICRONS_SQUARE,
    ).sort("measurement_type")

    assert result.select("measurement_type", "value", "unit").rows() == [
        (SynapticMeasurementType.SUM_ANATOMICAL_SIZE.value, 5.0, "MICRONS_SQUARE"),
        (SynapticMeasurementType.SYNAPSE_COUNT.value, 2.0, "COUNT"),
    ]


@pytest.mark.parametrize(
    ("size_column", "size_unit"),
    [("size", None), (None, Unit.MICRONS_SQUARE)],
)
def test_size_arguments_must_be_supplied_together(size_column, size_unit):
    """Size aggregation must not infer either its source column or unit."""
    synapses = _synapse_frame()

    with pytest.raises(ValueError, match="supplied together"):
        derive_cell_cell_connectivity(
            synapses,
            project_id="project-1",
            connectome_id="connectome-1",
            modality=Modality.ELECTRON_MICROSCOPY,
            size_column=size_column,
            size_unit=size_unit,
        )


def test_size_column_must_exist_and_be_numeric_and_non_null():
    """Size totals must reject absent, nonnumeric, or incomplete data."""
    base = _synapse_frame()

    with pytest.raises(ValueError, match="not found"):
        derive_cell_cell_connectivity(
            base,
            project_id="project-1",
            connectome_id="connectome-1",
            modality=Modality.ELECTRON_MICROSCOPY,
            size_column="missing",
            size_unit=Unit.MICRONS_SQUARE,
        )
    with pytest.raises(TypeError, match="must be numeric"):
        derive_cell_cell_connectivity(
            base.with_columns(pl.lit("large").alias("size")),
            project_id="project-1",
            connectome_id="connectome-1",
            modality=Modality.ELECTRON_MICROSCOPY,
            size_column="size",
            size_unit=Unit.MICRONS_SQUARE,
        )
    with pytest.raises(ValueError, match="contains null"):
        derive_cell_cell_connectivity(
            base.with_columns(pl.lit(None, dtype=pl.Float64).alias("size")),
            project_id="project-1",
            connectome_id="connectome-1",
            modality=Modality.ELECTRON_MICROSCOPY,
            size_column="size",
            size_unit=Unit.MICRONS_SQUARE,
        )


@pytest.mark.parametrize(
    "missing_column",
    ["project_id", "presynaptic_cell", "postsynaptic_cell"],
)
def test_identity_columns_must_exist(missing_column):
    """Every aggregation identity column must be present in the source."""
    synapses = _synapse_frame().drop(missing_column)

    with pytest.raises(ValueError, match="Missing required synapse columns"):
        derive_cell_cell_connectivity(
            synapses,
            project_id="project-1",
            connectome_id="connectome-1",
            modality=Modality.ELECTRON_MICROSCOPY,
        )


@pytest.mark.parametrize(
    "null_column",
    ["project_id", "presynaptic_cell", "postsynaptic_cell"],
)
def test_identity_columns_must_not_contain_nulls(null_column):
    """Aggregation identity columns must never contain null values."""
    synapses = _synapse_frame().with_columns(pl.lit(None).alias(null_column))

    with pytest.raises(ValueError, match="Identity columns contain null"):
        derive_cell_cell_connectivity(
            synapses,
            project_id="project-1",
            connectome_id="connectome-1",
            modality=Modality.ELECTRON_MICROSCOPY,
        )


def test_empty_input_returns_typed_schema_shaped_frame():
    """An empty transform must retain the generated model's table schema."""
    empty = pl.DataFrame(
        schema={
            "project_id": pl.String,
            "synapse_table_id": pl.String,
            "presynaptic_cell": pl.String,
            "postsynaptic_cell": pl.String,
        }
    )

    result = derive_cell_cell_connectivity(
        empty,
        project_id="project-1",
        connectome_id="connectome-1",
        modality=Modality.ELECTRON_MICROSCOPY,
    )

    assert result.is_empty()
    assert result.columns == list(CellCellConnectivityLong.model_fields)
    assert result.schema["value"] == pl.Float64
    assert all(
        result.schema[column] == pl.String
        for column in result.columns
        if column != "value"
    )


@pytest.mark.parametrize("empty", [False, True])
def test_arrow_conversion_uses_model_schema(empty):
    """Arrow conversion must preserve model types and required-field nullability."""
    synapses = _synapse_frame()
    if empty:
        synapses = synapses.clear()
    result = derive_cell_cell_connectivity(
        synapses,
        project_id="project-1",
        connectome_id="connectome-1",
        modality=Modality.ELECTRON_MICROSCOPY,
    )

    table = cell_cell_connectivity_to_arrow(result)

    assert table.schema == build_arrow_schema(CellCellConnectivityLong)


def test_ids_are_deterministic_and_separate_connectome_contexts():
    """Stable IDs must separate connectomes derived from identical endpoints."""
    synapses = _synapse_frame()

    first = derive_cell_cell_connectivity(
        synapses,
        project_id="project-1",
        connectome_id="connectome-1",
        modality=Modality.ELECTRON_MICROSCOPY,
    )
    repeated = derive_cell_cell_connectivity(
        synapses,
        project_id="project-1",
        connectome_id="connectome-1",
        modality=Modality.ELECTRON_MICROSCOPY,
    )
    other_context = derive_cell_cell_connectivity(
        synapses,
        project_id="project-1",
        connectome_id="connectome-2",
        modality=Modality.ELECTRON_MICROSCOPY,
    )

    assert first["id"].to_list() == repeated["id"].to_list()
    assert first["id"].to_list() != other_context["id"].to_list()
    assert first["id"].to_list() == [
        "project-1_connectome-1_pre-1_post-1_SYNAPSE_COUNT"
    ]


def test_source_synapse_table_does_not_change_identity():
    """Optional source provenance must not change connectome row identity."""
    first = derive_cell_cell_connectivity(
        _synapse_frame(),
        project_id="project-1",
        connectome_id="connectome-1",
        modality=Modality.ELECTRON_MICROSCOPY,
    )
    second = derive_cell_cell_connectivity(
        _synapse_frame().with_columns(
            pl.lit("synapses-2").alias("synapse_table_id")
        ),
        project_id="project-1",
        connectome_id="connectome-1",
        modality=Modality.ELECTRON_MICROSCOPY,
    )

    assert first["id"].to_list() == second["id"].to_list()
    assert first["synapse_table_id"].to_list() == ["synapses-1"]
    assert second["synapse_table_id"].to_list() == ["synapses-2"]


def test_transform_allows_rows_without_synapse_table_provenance():
    """Derivation must work when connectivity did not come from a table."""
    result = derive_cell_cell_connectivity(
        _synapse_frame().drop("synapse_table_id"),
        project_id="project-1",
        connectome_id="connectome-1",
        modality=Modality.ELECTRON_MICROSCOPY,
    )

    assert result["synapse_table_id"].to_list() == [None]


def _synapse_frame() -> pl.DataFrame:
    """Return one valid source synapse for focused transform tests."""
    return pl.DataFrame(
        {
            "project_id": ["project-1"],
            "synapse_table_id": ["synapses-1"],
            "presynaptic_cell": ["pre-1"],
            "postsynaptic_cell": ["post-1"],
        }
    )