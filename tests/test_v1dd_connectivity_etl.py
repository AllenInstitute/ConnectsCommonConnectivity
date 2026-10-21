"""Exercise the V1DD connectivity cell against small, multi-batch Feather inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.feather as feather
import pytest
from deltalake.exceptions import DeltaError
from polars.testing import assert_frame_equal


@pytest.fixture
def connectivity_cell(tmp_path):
    """Load notebook imports and connectivity code without executing other ETL cells."""
    path = Path(__file__).resolve().parents[1] / "code/EM/etl_v1dd1196_01.ipynb"
    notebook = json.loads(path.read_text())
    namespace = {}
    exec(compile("".join(notebook["cells"][1]["source"]), str(path), "exec"), namespace)
    namespace.update(
        SYNAPSE_PATH=tmp_path / "synapses.feather",
        OUTPUT_ROOT=tmp_path / "output",
        PROJECT_ID="v1dd_test",
        CONNECTOME_ID="connectome_test",
        SYNAPSE_TABLE_ID="synapses_test",
        axon_proof_root_ids=np.array([10, 11], dtype=np.int64),
        unique_root_ids=np.array([20, 21], dtype=np.int64),
    )
    cell = next(cell for cell in notebook["cells"] if cell["id"] == "3e77e310")
    return namespace, compile("".join(cell["source"]), str(path), "exec")


@pytest.mark.parametrize("empty", [False, True])
def test_connectivity_cell_aggregates_across_source_batches_and_reruns(connectivity_cell, empty):
    """Empty inputs retain schema; nonempty batch totals survive scoped reruns."""
    namespace, code = connectivity_cell
    source = pa.table({
        "id": [1, 2, 3, 4, 5, 6],
        "pre_pt_root_id": [10, 10, 11, 12, 10, 11],
        "post_pt_root_id": [20, 20, 21, 20, 99, 21],
        "size": [2, 3, 7, 11, 13, 17],
        "unused": ["not needed"] * 6,
    })
    if empty:
        source = source.slice(0, 0)
    feather.write_feather(source, namespace["SYNAPSE_PATH"], chunksize=1)
    namespace["synapse_count"] = 0 if empty else 4
    exec(code, namespace)
    path = namespace["OUTPUT_ROOT"] / namespace["MODEL_TABLE_PATHS"]["CellCellConnectivityLong"]
    first = pl.read_delta(path).sort("id")
    assert first.height == (0 if empty else 4)
    assert namespace["pair_count"] == (0 if empty else 2)
    assert namespace["selected_size_sum"] == (0 if empty else 29)
    if empty:
        assert first.schema["value"] == pl.Float64
        assert first.schema["presynaptic_cell"] == pl.String
        return
    if not empty:
        assert first.select(
            "presynaptic_cell", "postsynaptic_cell", "measurement_type", "value",
        ).rows() == [
            ("10", "20", "SUM_ANATOMICAL_SIZE", 5.0),
            ("10", "20", "SYNAPSE_COUNT", 2.0),
            ("11", "21", "SUM_ANATOMICAL_SIZE", 24.0),
            ("11", "21", "SYNAPSE_COUNT", 2.0),
        ]
        other = first.with_columns(pl.lit("other_project").alias("project_id"))
        namespace["write_deltalake"](path, other.to_arrow(), mode="append")
    exec(code, namespace)
    stored = pl.read_delta(path)
    assert_frame_equal(
        first, stored.filter(pl.col("project_id") == namespace["PROJECT_ID"]).sort("id"),
    )
    if not empty:
        assert_frame_equal(
            other, stored.filter(pl.col("project_id") == "other_project").sort("id"),
        )


def test_connectivity_cell_rejects_duplicates_across_source_batches(connectivity_cell):
    """Repeated synapse IDs in separate Feather batches fail without committing connectivity."""
    namespace, code = connectivity_cell
    source = pa.table({
        "id": [1, 1],
        "pre_pt_root_id": [10, 11],
        "post_pt_root_id": [20, 21],
        "size": [2, 3],
    })
    feather.write_feather(source, namespace["SYNAPSE_PATH"], chunksize=1)
    namespace["synapse_count"] = 2
    with pytest.raises(DeltaError, match="Duplicate synapse ids"):
        exec(code, namespace)
    path = namespace["OUTPUT_ROOT"] / namespace["MODEL_TABLE_PATHS"]["CellCellConnectivityLong"]
    assert not namespace["DeltaTable"].is_deltatable(str(path))
