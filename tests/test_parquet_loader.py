from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from connects_common_connectivity.parquet_loader import load_parquet_to_models


def _write_parquet(path, columns: dict[str, list[str]]) -> None:
    table = pa.table({name: pa.array(values) for name, values in columns.items()})
    pq.write_table(table, path)


def test_load_parquet_to_models_happy_path_dataitem(tmp_path):
    parquet_path = tmp_path / "dataitems.parquet"
    _write_parquet(
        parquet_path,
        {
            "id": ["d1", "d2"],
            "name": ["item-1", "item-2"],
            "project_id": ["p1", "p1"],
        },
    )

    instances, report = load_parquet_to_models(
        "connectivity_schema.yaml",
        "DataItem",
        str(parquet_path),
    )

    assert len(instances) == 2
    assert [item.id for item in instances] == ["d1", "d2"]
    assert [item.project_id for item in instances] == ["p1", "p1"]
    assert report["mapping"]["id"] == "id"
    assert report["mapping"]["project_id"] == "project_id"
    assert report["counts"]["rows"] == 2
    assert report["counts"]["instances"] == 2
    assert report["counts"]["errors"] == 0


def test_load_parquet_to_models_reports_missing_required_slot(tmp_path):
    parquet_path = tmp_path / "missing_project_id.parquet"
    _write_parquet(
        parquet_path,
        {
            "id": ["d1"],
            "name": ["item-1"],
        },
    )

    instances, report = load_parquet_to_models(
        "connectivity_schema.yaml",
        "DataItem",
        str(parquet_path),
    )

    assert instances == []
    assert report["counts"]["errors"] == 1
    assert any("project_id" in err["message"] for err in report["errors"])
