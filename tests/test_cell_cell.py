from __future__ import annotations

import polars as pl
import pytest
from pydantic import ValidationError

from connects_common_connectivity.io import (
    WRITABLE_CLASSES,
    derive_cell_cell_connectivity,
)
from connects_common_connectivity.io.write_spec import REGISTRY
from connects_common_connectivity.models import (
    CellCellConnectivityLong,
    Modality,
    SynapticMeasurementType,
    Unit,
)


def test_cell_cell_connectivity_requires_connectome_id():
    with pytest.raises(
        ValidationError,
        match=r"(?s)connectome_id.*Field required",
    ):
        CellCellConnectivityLong(
            id="measurement-1",
            project_id="project-1",
            value=1.0,
            unit=Unit.COUNT,
        )


def test_cell_cell_connectivity_accepts_connectome_id():
    measurement = CellCellConnectivityLong(
        id="measurement-1",
        connectome_id="connectome-1",
        project_id="project-1",
        value=1.0,
        unit=Unit.COUNT,
    )

    assert measurement.connectome_id == "connectome-1"


def test_derives_counts_for_pairs_and_projects():
    synapses = pl.DataFrame(
        {
            "project_id": ["project-1", "project-1", "project-1", "project-2"],
            "presynaptic_cell": ["pre-1", "pre-1", "pre-1", "pre-1"],
            "postsynaptic_cell": ["post-1", "post-1", "post-2", "post-1"],
        }
    )

    result = derive_cell_cell_connectivity(
        synapses,
        connectome_id="connectome-1",
        modality=Modality.ELECTRON_MICROSCOPY,
    ).sort("project_id", "presynaptic_cell", "postsynaptic_cell")

    assert result["value"].to_list() == [2.0, 1.0, 1.0]
    assert result["measurement_type"].unique().to_list() == [
        SynapticMeasurementType.SYNAPSE_COUNT.value
    ]
    assert result["unit"].unique().to_list() == [Unit.COUNT.value]
    assert result["connectome_id"].unique().to_list() == ["connectome-1"]


def test_derives_optional_size_sum_with_explicit_unit():
    synapses = pl.DataFrame(
        {
            "project_id": ["project-1", "project-1"],
            "presynaptic_cell": ["pre-1", "pre-1"],
            "postsynaptic_cell": ["post-1", "post-1"],
            "size": [2, 3],
        }
    )

    result = derive_cell_cell_connectivity(
        synapses,
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
    synapses = _synapse_frame()

    with pytest.raises(ValueError, match="supplied together"):
        derive_cell_cell_connectivity(
            synapses,
            connectome_id="connectome-1",
            modality=Modality.ELECTRON_MICROSCOPY,
            size_column=size_column,
            size_unit=size_unit,
        )


def test_size_column_must_exist_and_be_numeric_and_non_null():
    base = _synapse_frame()

    with pytest.raises(ValueError, match="not found"):
        derive_cell_cell_connectivity(
            base,
            connectome_id="connectome-1",
            modality=Modality.ELECTRON_MICROSCOPY,
            size_column="missing",
            size_unit=Unit.MICRONS_SQUARE,
        )
    with pytest.raises(TypeError, match="must be numeric"):
        derive_cell_cell_connectivity(
            base.with_columns(pl.lit("large").alias("size")),
            connectome_id="connectome-1",
            modality=Modality.ELECTRON_MICROSCOPY,
            size_column="size",
            size_unit=Unit.MICRONS_SQUARE,
        )
    with pytest.raises(ValueError, match="contains null"):
        derive_cell_cell_connectivity(
            base.with_columns(pl.lit(None, dtype=pl.Float64).alias("size")),
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
    synapses = _synapse_frame().drop(missing_column)

    with pytest.raises(ValueError, match="Missing required synapse columns"):
        derive_cell_cell_connectivity(
            synapses,
            connectome_id="connectome-1",
            modality=Modality.ELECTRON_MICROSCOPY,
        )


@pytest.mark.parametrize(
    "null_column",
    ["project_id", "presynaptic_cell", "postsynaptic_cell"],
)
def test_identity_columns_must_not_contain_nulls(null_column):
    synapses = _synapse_frame().with_columns(pl.lit(None).alias(null_column))

    with pytest.raises(ValueError, match="Identity columns contain null"):
        derive_cell_cell_connectivity(
            synapses,
            connectome_id="connectome-1",
            modality=Modality.ELECTRON_MICROSCOPY,
        )


def test_empty_input_returns_typed_schema_shaped_frame():
    empty = pl.DataFrame(
        schema={
            "project_id": pl.String,
            "presynaptic_cell": pl.String,
            "postsynaptic_cell": pl.String,
        }
    )

    result = derive_cell_cell_connectivity(
        empty,
        connectome_id="connectome-1",
        modality=Modality.ELECTRON_MICROSCOPY,
    )

    assert result.is_empty()
    assert result.schema == {
        "id": pl.String,
        "description": pl.String,
        "connectome_id": pl.String,
        "presynaptic_cell": pl.String,
        "postsynaptic_cell": pl.String,
        "measurement_type": pl.String,
        "modality": pl.String,
        "value": pl.Float64,
        "unit": pl.String,
        "project_id": pl.String,
    }


def test_ids_are_deterministic_and_separate_connectome_contexts():
    synapses = _synapse_frame()

    first = derive_cell_cell_connectivity(
        synapses,
        connectome_id="connectome-1",
        modality=Modality.ELECTRON_MICROSCOPY,
    )
    repeated = derive_cell_cell_connectivity(
        synapses,
        connectome_id="connectome-1",
        modality=Modality.ELECTRON_MICROSCOPY,
    )
    other_context = derive_cell_cell_connectivity(
        synapses,
        connectome_id="connectome-2",
        modality=Modality.ELECTRON_MICROSCOPY,
    )

    assert first["id"].to_list() == repeated["id"].to_list()
    assert first["id"].to_list() != other_context["id"].to_list()
    assert first["id"].str.contains(r"^sha256:[0-9a-f]{64}$").all()


def _synapse_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "project_id": ["project-1"],
            "presynaptic_cell": ["pre-1"],
            "postsynaptic_cell": ["post-1"],
        }
    )