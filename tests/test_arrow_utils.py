from enum import Enum
from typing import Optional

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from connects_common_connectivity.io.arrow_utils import (
    build_arrow_schema,
    flatten_refs,
    model_to_row,
    models_to_table,
)


class Axis(str, Enum):
    PLUS_X = "PLUS_X"
    MINUS_Y = "MINUS_Y"


class View(BaseModel):
    id: str
    horizontal: Axis
    vertical: Axis


class Frame(BaseModel):
    view: View | None = None
    views: list[View]
    reference: str


class NestedFrame(BaseModel):
    frame: Optional[Frame] = None
    frames: list[Frame]


def test_nested_models_round_trip_as_structs(tmp_path):
    """Embedded models, lists, and null views must survive Parquet without losing their IDs."""
    view = View(id="embedded", horizontal=Axis.PLUS_X, vertical=Axis.MINUS_Y)
    rows = [
        Frame(view=view, views=[view], reference="ref"),
        Frame(views=[], reference="ref"),
    ]
    table = models_to_table(rows)

    assert pa.types.is_struct(table.schema.field("view").type)
    assert pa.types.is_struct(table.schema.field("views").type.value_type)
    assert model_to_row(rows[0])["view"]["id"] == "embedded"
    path = tmp_path / "frames.parquet"
    pq.write_table(table, path)
    result = pl.read_parquet(path)
    assert isinstance(result.schema["view"], pl.Struct)
    assert [Frame.model_validate(row) for row in result.to_dicts()] == rows


def test_schema_controls_reference_flattening():
    """Only schema-declared references collapse to IDs; embedded structs retain their fields."""
    schema = pa.schema([
        pa.field("reference", pa.string()),
        pa.field("references", pa.list_(pa.string())),
        pa.field("view", pa.struct(build_arrow_schema(View))),
    ])
    row = {
        "reference": {"id": "ref"},
        "references": [{"identifier": "other"}],
        "view": {"id": "embedded", "horizontal": "PLUS_X", "vertical": "MINUS_Y"},
    }
    result = flatten_refs(row, schema=schema)
    assert result["reference"] == "ref"
    assert result["references"] == ["other"]
    assert result["view"]["id"] == "embedded"


def test_explicit_schema_and_flatten_false():
    """Disabling flattening preserves structs, while an explicit string schema selects IDs."""
    view = View(id="embedded", horizontal=Axis.PLUS_X, vertical=Axis.MINUS_Y)
    frame = Frame(view=view, views=[view], reference="ref")
    assert models_to_table([frame], flatten=False).to_pylist()[0]["view"]["id"] == "embedded"
    schema = pa.schema([
        pa.field("view", pa.string()),
        pa.field("views", pa.list_(pa.string())),
    ])
    assert models_to_table([frame], schema=schema).to_pylist() == [
        {"view": "embedded", "views": ["embedded"]}
    ]


def test_empty_batch_retains_struct_schema():
    """An empty model batch must retain the supplied schema, including its struct fields."""
    schema = build_arrow_schema(Frame)
    table = models_to_table([], schema=schema)
    assert table.num_rows == 0
    assert table.schema == schema


def test_recursive_structs_and_optional_annotations(tmp_path):
    """Nested structs must preserve required-field nullability and round-trip through Parquet."""
    frame = Frame(views=[], reference="ref")
    rows = [NestedFrame(frame=frame, frames=[frame]), NestedFrame(frames=[])]
    table = models_to_table(rows)
    assert table.schema.field("frame").nullable
    assert table.schema.field("frame").type.field("view").nullable
    assert not table.schema.field("frame").type.field("reference").nullable
    path = tmp_path / "nested.parquet"
    pq.write_table(table, path)
    assert [NestedFrame.model_validate(row) for row in pl.read_parquet(path).to_dicts()] == rows


def test_reference_flattening_recurses_inside_structs():
    """References inside structs and lists flatten to IDs without breaking legacy calls."""
    nested_type = pa.struct([pa.field("reference", pa.string())])
    schema = pa.schema([
        pa.field("nested", nested_type),
        pa.field("items", pa.list_(nested_type)),
    ])
    row = {"nested": {"reference": {"id": "first"}},
           "items": [{"reference": {"identifier": "second"}}]}
    assert flatten_refs(row, schema=schema) == {
        "nested": {"reference": "first"}, "items": [{"reference": "second"}]
    }
    assert flatten_refs({"reference": {"id": "legacy"}}) == {"reference": "legacy"}