from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import nbformat
import numpy as np
import pandas as pd

from connects_common_connectivity.io import read_spatial_locations, write_models
from connects_common_connectivity.models import (
    AnatomicalOrientation,
    Default2DView,
    LocationType,
    ReferenceSpace,
    SignedAxis,
    SpatialLocation,
    Unit,
)

NOTEBOOK_ROOT = Path(__file__).resolve().parents[1] / "code/EM"
CAVE_SPACE = "minnie65_v1822_cave_nm"
STANDARD_SPACE = "minnie65_v1822_standard_transform_um"


def test_minnie_preparation_preserves_transform_microns():
    """Micron-valued transform output is cached unchanged, without another division by 1,000."""
    notebook = nbformat.read(NOTEBOOK_ROOT / "EM_minnie1822_cave_query.ipynb", as_version=4)
    source = next(cell.source for cell in notebook.cells if cell.id == "3b366fe4")
    positions = pd.DataFrame({
        "pt_position_x": [915264],
        "pt_position_y": [959104],
        "pt_position_z": [783720],
    })
    expected = np.array([[828.19, 638.554, 783.72]])
    transform = Mock()
    transform.apply.return_value = expected
    namespace = {
        "nuc_df": positions,
        "standard_transform": SimpleNamespace(minnie_transform_nm=lambda: transform),
    }

    exec(compile(source, "<minnie transform cell>", "exec"), namespace)

    np.testing.assert_array_equal(
        transform.apply.call_args.args[0],
        [[915264, 959104, 783720]],
    )
    np.testing.assert_array_equal(
        positions[["x_medial-lateral", "y_dorsal-ventral", "z_caudal-rostral"]],
        expected,
    )


def test_minnie_spatial_write_repairs_existing_rows_without_duplicates(settings):
    """Repeated micron writes update matching keys without duplicates or changes to other rows."""
    notebook = nbformat.read(NOTEBOOK_ROOT / "etl_minnie1822_01.ipynb", as_version=4)
    source = next(cell.source for cell in notebook.cells if cell.id == "646a9272")
    common = dict(project_id="minnie65", dataitem_id="101", location_type=LocationType.SOMA)
    old_rows = [
        SpatialLocation(reference_space=STANDARD_SPACE, x=0.82819, y=0.638554, z=0.78372,
                        **common),
        SpatialLocation(reference_space=CAVE_SPACE, x=915264, y=959104, z=783720,
                        **common),
        SpatialLocation(reference_space="other_space", x=1, y=2, z=3, **common),
        SpatialLocation(project_id="other_project", dataitem_id="101",
                        reference_space=STANDARD_SPACE, location_type=LocationType.SOMA,
                        x=4, y=5, z=6),
    ]
    write_models(old_rows, output_root=settings.output_root)
    namespace = {
        "AnatomicalOrientation": AnatomicalOrientation,
        "Default2DView": Default2DView,
        "LocationType": LocationType,
        "ReferenceSpace": ReferenceSpace,
        "SignedAxis": SignedAxis,
        "SpatialLocation": SpatialLocation,
        "Unit": Unit,
        "write_models": write_models,
        "CAVE_REFERENCE_SPACE_ID": CAVE_SPACE,
        "STANDARD_REFERENCE_SPACE_ID": STANDARD_SPACE,
        "PROJECT_ID": "minnie65",
        "OUTPUT_ROOT": settings.output_root,
        "positions": pd.DataFrame([{
            "dataitem_id": "101",
            "pt_position_x": 915264,
            "pt_position_y": 959104,
            "pt_position_z": 783720,
            "x_medial-lateral": 828.19,
            "y_dorsal-ventral": 638.554,
            "z_caudal-rostral": 783.72,
        }]),
    }

    for _ in range(2):
        exec(compile(source, "<minnie spatial write cell>", "exec"), namespace)
        rows = read_spatial_locations("minnie65", output_root=settings.output_root)
        assert rows.height == 3
        by_space = {row["reference_space"]: row for row in rows.to_dicts()}
        assert [by_space[STANDARD_SPACE][axis] for axis in ("x", "y", "z")] == [
            828.19, 638.554, 783.72,
        ]
        assert [by_space[CAVE_SPACE][axis] for axis in ("x", "y", "z")] == [
            915264, 959104, 783720,
        ]
        assert [by_space["other_space"][axis] for axis in ("x", "y", "z")] == [1, 2, 3]
        other = read_spatial_locations("other_project", output_root=settings.output_root)
        assert other.height == 1
        assert other.select("x", "y", "z").row(0) == (4, 5, 6)
