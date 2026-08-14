"""Tests for IO write utilities."""
import polars as pl
import pyarrow as pa

from connects_common_connectivity.io.write_utils import (
    append_new_dataitems,
    populate_region_coverage,
)
from connects_common_connectivity.models import (
    Laterality,
    Modality,
    ProjectionMeasurementMatrix,
    ProjectionMeasurementType,
    Unit,
)


def _make_table(ids: list[str], project_id: str = "proj_a") -> pa.Table:
    return pa.table(
        {
            "id": pa.array(ids, type=pa.string()),
            "name": pa.array(ids, type=pa.string()),
            "neuroglancer_link": pa.array([None] * len(ids), type=pa.string()),
            "project_id": pa.array([project_id] * len(ids), type=pa.string()),
        }
    )


def test_populate_region_coverage_accepts_nested_list():
    pmm = ProjectionMeasurementMatrix(
        id="pmm_list",
        measurement_type=ProjectionMeasurementType.MICRONS_OF_AXON,
        modality=Modality.MORPHOLOGY,
        laterality=Laterality.IPSILATERAL,
        unit=Unit.MICRONS_LENGTH,
        data_item_index=["c1", "c2"],
        region_index=["VISp", "ACA", "MOB"],
        values="file:///tmp/pmm.delta",
    )

    enriched = populate_region_coverage(
        pmm,
        [[1.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
    )

    assert enriched.region_coverage == ["VISp", "MOB"]
    assert pmm.region_coverage in (None, [])


# ---------------------------------------------------------------------------
# First write (table does not exist yet)
# ---------------------------------------------------------------------------


def test_first_write_appends_all(tmp_path):
    table = _make_table(["a", "b", "c"])
    n = append_new_dataitems(str(tmp_path / "dataitem"), table, project_id="proj_a")
    assert n == 3


def test_first_write_empty_table(tmp_path):
    table = _make_table([])
    n = append_new_dataitems(str(tmp_path / "dataitem"), table, project_id="proj_a")
    assert n == 0


# ---------------------------------------------------------------------------
# Idempotency (re-run same rows)
# ---------------------------------------------------------------------------


def test_idempotent_rerun(tmp_path):
    path = str(tmp_path / "dataitem")
    table = _make_table(["a", "b", "c"])
    first = append_new_dataitems(path, table, project_id="proj_a")
    second = append_new_dataitems(path, table, project_id="proj_a")
    assert first == 3
    assert second == 0  # all already present


def test_idempotent_partial_rerun(tmp_path):
    path = str(tmp_path / "dataitem")
    append_new_dataitems(path, _make_table(["a", "b"]), project_id="proj_a")
    n = append_new_dataitems(path, _make_table(["a", "b", "c"]), project_id="proj_a")
    assert n == 1, f"expected only 'c' to be new; appended {n} rows"


# ---------------------------------------------------------------------------
# Multi-project isolation
# ---------------------------------------------------------------------------


def test_different_projects_do_not_interfere(tmp_path):
    path = str(tmp_path / "dataitem")
    append_new_dataitems(path, _make_table(["x", "y"], project_id="proj_a"), project_id="proj_a")
    # proj_b has the same ids — they are independent rows
    n = append_new_dataitems(
        path, _make_table(["x", "y"], project_id="proj_b"), project_id="proj_b"
    )
    assert n == 2  # treated as new because different project

    # Re-run proj_b — still idempotent
    n2 = append_new_dataitems(
        path, _make_table(["x", "y"], project_id="proj_b"), project_id="proj_b"
    )
    assert n2 == 0


def test_shared_project_two_sources(tmp_path):
    """Simulates inh_01 and exc_01 both writing to the same project partition."""
    path = str(tmp_path / "dataitem")
    inh_ids = ["100", "200", "300"]
    exc_ids = ["400", "500"]

    n_inh = append_new_dataitems(
        path, _make_table(inh_ids, project_id="visp_patchseq"), project_id="visp_patchseq"
    )
    n_exc = append_new_dataitems(
        path, _make_table(exc_ids, project_id="visp_patchseq"), project_id="visp_patchseq"
    )

    assert n_inh == 3
    assert n_exc == 2  # not wiped by second call

    # Re-run inh — should append 0
    n_inh2 = append_new_dataitems(
        path, _make_table(inh_ids, project_id="visp_patchseq"), project_id="visp_patchseq"
    )
    assert n_inh2 == 0

    # Total rows for the shared project
    total = pl.read_delta(path).filter(pl.col("project_id") == "visp_patchseq").shape[0]
    assert total == 5
