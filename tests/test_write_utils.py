"""Tests for write_utils.append_new_dataitems."""
import pyarrow as pa
import pytest

from connects_common_connectivity.io.write_utils import append_new_dataitems


def _make_table(ids: list[str], project_id: str = "proj_a") -> pa.Table:
    return pa.table(
        {
            "id": pa.array(ids, type=pa.string()),
            "name": pa.array(ids, type=pa.string()),
            "neuroglancer_link": pa.array([None] * len(ids), type=pa.string()),
            "project_id": pa.array([project_id] * len(ids), type=pa.string()),
        }
    )


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
    assert n == 1  # only "c" is new


# ---------------------------------------------------------------------------
# Multi-project isolation
# ---------------------------------------------------------------------------


def test_different_projects_do_not_interfere(tmp_path):
    path = str(tmp_path / "dataitem")
    append_new_dataitems(path, _make_table(["x", "y"], project_id="proj_a"), project_id="proj_a")
    # proj_b has the same ids — they are independent rows
    n = append_new_dataitems(path, _make_table(["x", "y"], project_id="proj_b"), project_id="proj_b")
    assert n == 2  # treated as new because different project

    # Re-run proj_b — still idempotent
    n2 = append_new_dataitems(path, _make_table(["x", "y"], project_id="proj_b"), project_id="proj_b")
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
    import polars as pl

    total = pl.read_delta(path).filter(pl.col("project_id") == "visp_patchseq").shape[0]
    assert total == 5
