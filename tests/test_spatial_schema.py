from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from connects_common_connectivity.models import (
    CellGeneData,
    CellMetadata,
    DataItem,
    Default2DView,
    LocationType,
    ReferenceSpace,
    SignedAxis,
    SingleCellReconstruction,
    SpatialLocation,
    Unit,
    ZarrArray,
)


def test_reference_space_can_be_global():
    """Reference spaces may omit ownership and optional metadata; DataItems still need a project."""
    space = ReferenceSpace(id="CCF_v3")
    assert space.project_id is None
    assert space.name is None
    assert space.default_2d_view is None
    assert space.unit is None
    assert not space.voxel_size
    assert space.voxel_size_unit is None
    assert ReferenceSpace(id="local", project_id="project").project_id == "project"
    with pytest.raises(ValidationError):
        DataItem(id="cell", name="cell")


@pytest.mark.parametrize("unit", [
    Unit.NANOMETERS_LENGTH, Unit.MICRONS_LENGTH,
    Unit.MILLIMETERS_LENGTH, Unit.CENTIMETERS_LENGTH,
])
def test_reference_space_physical_units(unit):
    """Reference spaces must accept each supported physical length unit without changing it."""
    space = ReferenceSpace(id="physical", unit=unit)
    assert space.unit == unit


def test_reference_space_anisotropic_voxel_size():
    """Unequal per-axis voxel dimensions and their physical unit must survive JSON round-trips."""
    space = ReferenceSpace(
        id="voxel_frame", unit=Unit.VOXELS,
        voxel_size=(4.0, 4.0, 40.0), voxel_size_unit=Unit.NANOMETERS_LENGTH,
    )
    assert list(space.voxel_size) == [4.0, 4.0, 40.0]
    assert space.voxel_size_unit == Unit.NANOMETERS_LENGTH
    assert ReferenceSpace.model_validate_json(space.model_dump_json()) == space


@pytest.mark.parametrize("dimensions", [(4.0, 4.0), (4.0, 4.0, 40.0, 40.0)])
def test_voxel_size_requires_three_dimensions(dimensions):
    """Voxel dimensions must contain exactly three values, one for each spatial axis."""
    with pytest.raises(ValidationError):
        ReferenceSpace(
            id="voxel_frame", unit=Unit.VOXELS,
            voxel_size=dimensions, voxel_size_unit=Unit.NANOMETERS_LENGTH,
        )


@pytest.mark.parametrize("missing", ["project_id", "dataitem_id", "reference_space",
                                     "location_type", "x", "y", "z"])
def test_coordinates_require_all_identity_and_position_fields(missing):
    """Coordinates store reference IDs and reject omission of any key or x/y/z component."""
    values = dict(project_id="p", dataitem_id="cell", reference_space="CCF_v3",
                  location_type=LocationType.SOMA, x=1.0, y=2.0, z=3.0)
    location = SpatialLocation(**values)
    assert location.reference_space == "CCF_v3"
    assert location.dataitem_id == "cell"
    del values[missing]
    with pytest.raises(ValidationError):
        SpatialLocation(**values)


@pytest.mark.parametrize("description", [None, "Axon initial segment origin."])
def test_other_location_description_round_trip(description):
    """OTHER locations may omit details or preserve them through JSON serialization."""
    values = dict(project_id="p", dataitem_id="cell", reference_space="CCF_v3",
                  location_type=LocationType.OTHER, x=1.0, y=2.0, z=3.0)
    if description is not None:
        values["description"] = description
    location = SpatialLocation(**values)
    assert location.description == description
    assert SpatialLocation.model_validate_json(location.model_dump_json()) == location


def test_view_requires_two_valid_signed_axes():
    """Views require two recognized signed axes, and location types retain the agreed vocabulary."""
    view = Default2DView(left_to_right=SignedAxis.PLUS_X, bottom_to_top=SignedAxis.MINUS_Y)
    assert ReferenceSpace(id="frame", default_2d_view=view).default_2d_view == view
    with pytest.raises(ValidationError):
        Default2DView(left_to_right=SignedAxis.PLUS_X)
    with pytest.raises(ValidationError):
        Default2DView(left_to_right="X", bottom_to_top=SignedAxis.MINUS_Y)
    assert {value.value for value in LocationType} == {
        "SOMA", "CENTROID", "INJECTION_SITE", "OTHER"
    }


def test_spatial_schema_identity_and_removed_slots():
    """Schemas retain the coordinate key and DataItem index contract, not old embedded slots."""
    schemas = Path(__file__).resolve().parents[1] / "schemas"
    spatial = yaml.safe_load((schemas / "spatial_schema.yaml").read_text())
    for enum_name in ["SignedAxis", "LocationType"]:
        for value in spatial["enums"][enum_name]["permissible_values"].values():
            assert value and value.get("description", "").strip()
    identity = spatial["classes"]["SpatialLocation"]["unique_keys"]["location_identity"]
    assert identity["unique_key_slots"] == [
        "project_id", "dataitem_id", "reference_space", "location_type"
    ]
    assert "soma_location" not in SingleCellReconstruction.model_fields
    assert "spatial_location" not in CellMetadata.model_fields
    for filename, slot in [("single_cell_schema.yaml", "soma_location"),
                           ("cell_gene_schema.yaml", "spatial_location")]:
        assert slot not in yaml.safe_load((schemas / filename).read_text())["slots"]
    cell_gene = yaml.safe_load((schemas / "cell_gene_schema.yaml").read_text())
    cell_index = cell_gene["classes"]["CellGeneData"]["slot_usage"]["cell_index"]
    assert cell_index["range"] == "DataItem"
    assert cell_index["multivalued"] is True


def test_cell_gene_index_uses_same_dataitem_ids_as_coordinates():
    """Expression row indices and coordinate references can share the same ordered DataItem IDs."""
    cells = [DataItem(id=identifier, name=identifier, project_id="project")
             for identifier in ["cell_a", "cell_b"]]
    matrix = ZarrArray(id="expression", path="file:///synthetic/expression.zarr")
    expression = CellGeneData(
        id="experiment", dataitem_id="experiment_item", cell_gene_matrix=matrix.id,
        cell_index=[cell.id for cell in cells],
    )
    coordinates = [SpatialLocation(
        project_id=cell.project_id, dataitem_id=cell.id, reference_space="original",
        location_type=LocationType.SOMA, x=1, y=2, z=3,
    ) for cell in cells]
    assert expression.cell_index == [location.dataitem_id for location in coordinates]