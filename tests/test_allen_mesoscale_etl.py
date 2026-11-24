"""Exercise the mesoscale notebook without downloading data or touching real releases."""

from __future__ import annotations

from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import polars as pl
import pytest
from deltalake import write_deltalake
from polars.testing import assert_frame_equal

from connects_common_connectivity.config import Settings
from connects_common_connectivity.io import write_models
from connects_common_connectivity.io.arrow_utils import models_to_table
from connects_common_connectivity.models import BrainRegion, DataItem, DataSet

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "examples/ETL/Allen_Mesoscale/etl_allen_mesoscale_01.ipynb"
)
PREPARATION = [
    "constants", "load", "source-projections", "relative-projections",
    "prepare-regions", "prepare-features",
]
WRITES = ["prepare-release", "write-features", "write-projections", "verify"]


def run_cells(namespace, cells, identifiers):
    """Execute selected notebook cells in their shared namespace."""
    for identifier in identifiers:
        exec(compile(cells[identifier], f"{NOTEBOOK}#{identifier}", "exec"), namespace)


@pytest.fixture
def mesoscale(tmp_path, monkeypatch):
    """Supply both source sides and a midline experiment beside an unrelated v2 dataset."""
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    cells = {cell.id: cell.source for cell in notebook.cells if cell.cell_type == "code"}
    namespace = {}
    run_cells(namespace, cells, ["imports"])

    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    source = tmp_path / "data/Allen Mesoscale"
    source.mkdir(parents=True)
    unionize_dir = source / "structure_unionizes"
    unionize_dir.mkdir()
    base = tmp_path / "scratch/ccm_v1.0.1_data_202611_v2"
    output = tmp_path / "scratch/ccm_v1.0.1_data_202611_v3"
    settings = Settings(output_root=base)
    write_models(DataSet(id="old", name="Existing data", project_id="other"), settings=settings)
    write_models(DataItem(id="10", name="Existing item", project_id="other"), settings=settings)
    write_deltalake(
        base / "brainregion",
        models_to_table([BrainRegion(id="900", name="Unrelated atlas region")]),
    )
    monkeypatch.chdir(tmp_path)
    namespace["get_settings"] = lambda: settings

    experiments = pd.DataFrame({
        "id": [30, 10, 20],
        "gender": ["F", "M", "F"],
        "specimen_name": ["midline", "left", "right"],
        "strain": [None, "C57BL/6J", "C57BL/6J"],
        "transgenic_line": [None, "Example-Cre", "Other-Cre"],
        "transgenic_line_id": [None, 12.0, 13.0],
        "product_id": [5, 5, 5],
        "structure_id": [202, 201, 201],
        "primary_injection_structure": [202, 201, 201],
        "structure_abbrev": ["B", "A", "A"],
        "structure_name": ["Region B", "Region A", "Region A"],
        "injection_structures": [[202], [201], [201, 202]],
        "injection_volume": [0.2, 0.1, 0.3],
        "injection_x": [5000, 4000, 4500],
        "injection_y": [3000, 2000, 2500],
        "injection_z": [5700, 1000, 9000],
    })
    experiments.to_parquet(source / "experiments.parquet", index=False)
    structures = pd.DataFrame({
        "id": [997, 201, 202],
        "name": ["Root", "Region A", "Region B"],
        "acronym": ["root", "A", "B"],
        "graph_id": [1, 1, 1],
        "graph_order": [0, 1, 2],
        "structure_id_path": [[997], [997, 201], [997, 202]],
        "structure_set_ids": [[], [167587189], [167587189]],
        "rgb_triplet": [[255, 255, 255], [10, 20, 30], [40, 50, 60]],
    })
    structures.to_parquet(source / "structures.parquet", index=False)
    structures.iloc[1:].to_parquet(source / "summary_structures.parquet", index=False)
    downloaded = pd.DataFrame({
        "experiment_id": [10, 20, 30],
        "A_right": [0.11, 0.21, 0.31],
        "B_right": [0.12, 0.22, 0.32],
        "A_left": [0.01, 0.02, 0.03],
        "B_left": [np.nan, 0.0, 0.04],
    })
    downloaded.to_parquet(source / "projection_matrix_summary_structures.parquet", index=False)
    for row in downloaded.to_dict("records"):
        eid = int(row["experiment_id"])
        records = [
            {
                "experiment_id": eid, "structure_id": region, "hemisphere_id": hemisphere,
                "is_injection": False, "projection_density": row[f"{acronym}_{side}"],
            }
            for region, acronym in [(201, "A"), (202, "B")]
            for hemisphere, side in [(1, "left"), (2, "right")]
        ]
        records.extend([
            {
                "experiment_id": eid, "structure_id": 201, "hemisphere_id": 1,
                "is_injection": True, "projection_density": 0.99,
            },
            {
                "experiment_id": eid, "structure_id": 201, "hemisphere_id": 3,
                "is_injection": False, "projection_density": 0.88,
            },
        ])
        pd.DataFrame(records).to_parquet(unionize_dir / f"experiment_{eid}.parquet", index=False)
    return namespace, cells, source, base, output


def test_mesoscale_round_trip_and_scoped_rerun(mesoscale):
    """V3 retains v2 and other projects while reruns preserve exact source values and nulls."""
    namespace, cells, _, base, output = mesoscale
    baseline = {path.relative_to(base): path.read_bytes() for path in base.rglob("*")
                if path.is_file()}
    run_cells(namespace, cells, PREPARATION + WRITES)
    assert namespace["OUTPUT_ROOT"] == output
    assert namespace["SETTINGS"].output_root == output
    assert namespace["get_settings"]().output_root == base
    ipsi = namespace["projections"]["allen_mesoscale_projection_density_summary_ipsi"]
    contra = namespace["projections"]["allen_mesoscale_projection_density_summary_contra"]
    np.testing.assert_allclose(ipsi.loc["10"], [0.01, np.nan], equal_nan=True)
    np.testing.assert_allclose(ipsi.loc["20"], [0.21, 0.22])
    np.testing.assert_allclose(contra.loc["10"], [0.11, 0.12])
    np.testing.assert_allclose(contra.loc["20"], [0.02, 0.0])
    assert namespace["midline_ids"] == ["30"]
    matrices = pl.read_delta(output / "projectionmeasurementmatrix")
    for side, expected in [("left", [0.03, 0.04]), ("right", [0.31, 0.32])]:
        matrix_id = f"allen_mesoscale_projection_density_summary_midline_{side}"
        metadata = matrices.filter(pl.col("id") == matrix_id).row(0, named=True)
        assert metadata["laterality"] == "UNKNOWN"
        assert f"Absolute {side} hemisphere" in metadata["description"]
        assert "/issues/91" in metadata["description"]
        assert metadata["data_item_index"] == ["30"]
        assert metadata["region_index"] == ["201", "202"]
        np.testing.assert_allclose(namespace["projections"][matrix_id].loc["30"], expected)
    feature_path = namespace["feature_path"]
    features = pl.read_delta(feature_path)
    assert features.filter(pl.col("id") == "30")["transgenic_line"].item() is None
    assert features.filter(pl.col("id") == "20")["injection_structures"].item() == "[201, 202]"

    other_features = features.with_columns(pl.lit("other").alias("project_id"))
    write_deltalake(feature_path, other_features.to_arrow(), mode="append")
    association_path = namespace["association_path"]
    other_associations = pl.read_delta(association_path).with_columns(
        pl.lit("other").alias("project_id"),
    )
    write_deltalake(association_path, other_associations.to_arrow(), mode="append")
    run_cells(namespace, cells, WRITES)
    assert_frame_equal(
        pl.read_delta(feature_path).filter(pl.col("project_id") == "other").sort("id"),
        other_features.sort("id"),
    )
    assert_frame_equal(
        pl.read_delta(association_path).filter(pl.col("project_id") == "other")
        .sort("dataitem_id", "brainregion_id"),
        other_associations.sort("dataitem_id", "brainregion_id"),
    )
    assert pl.read_delta(output / "brainregion").height == 4
    assert {path.relative_to(base): path.read_bytes() for path in base.rglob("*")
            if path.is_file()} == baseline
    for model in ("dataset", "dataitem"):
        assert_frame_equal(
            pl.read_delta(output / model).filter(pl.col("project_id") == "other"),
            pl.read_delta(base / model),
        )
    for relative, content in baseline.items():
        assert (output / relative).read_bytes() == content


@pytest.mark.parametrize("case", ["missing_file", "duplicate", "changed_value", "unknown_region"])
def test_mesoscale_invalid_sources_fail_before_release_copy(mesoscale, case):
    """Incomplete, duplicate, inconsistent or unknown-region sources fail before creating v3."""
    namespace, cells, source, _, output = mesoscale
    path = source / "structure_unionizes/experiment_10.parquet"
    if case == "missing_file":
        path.unlink()
    elif case == "duplicate":
        frame = pd.read_parquet(path)
        pd.concat([frame, frame.iloc[:1]]).to_parquet(path, index=False)
    elif case == "changed_value":
        frame = pd.read_parquet(path)
        frame.loc[0, "projection_density"] = 0.75
        frame.to_parquet(path, index=False)
    else:
        path = source / "experiments.parquet"
        frame = pd.read_parquet(path)
        frame["injection_structures"] = frame["injection_structures"].map(lambda _: [123456])
        frame.to_parquet(path, index=False)
    with pytest.raises(AssertionError):
        run_cells(namespace, cells, PREPARATION)
    assert not output.exists()


@pytest.mark.parametrize("midline_only", [False, True])
def test_mesoscale_single_side_cohort_kind(mesoscale, midline_only):
    """All-midline or all-lateral cohorts produce only their applicable pair of matrices."""
    namespace, cells, source, _, _ = mesoscale
    path = source / "experiments.parquet"
    frame = pd.read_parquet(path)
    frame["injection_z"] = 5700 if midline_only else 9000
    frame.to_parquet(path, index=False)
    run_cells(namespace, cells, PREPARATION + WRITES)
    assert len(namespace["projection_specs"]) == 2
    assert all(len(frame) == 3 for frame in namespace["projections"].values())
    expected_laterality = {"UNKNOWN"} if midline_only else {"IPSILATERAL", "CONTRALATERAL"}
    assert {spec[1].value for spec in namespace["projection_specs"]} == expected_laterality


def test_mesoscale_rejects_global_region_collision(mesoscale):
    """A conflicting global region ID aborts rather than overwriting an existing atlas row."""
    namespace, cells, _, base, output = mesoscale
    write_deltalake(
        base / "brainregion",
        models_to_table([BrainRegion(id="201", name="Different atlas region")]),
        mode="append",
    )
    run_cells(namespace, cells, PREPARATION)
    with pytest.raises(AssertionError):
        run_cells(namespace, cells, ["prepare-release"])
    assert_frame_equal(pl.read_delta(output / "brainregion"), pl.read_delta(base / "brainregion"))
    assert not (output / "projectionmeasurementmatrix").exists()
