"""Tests for IO write utilities."""

from connects_common_connectivity.io.write_utils import populate_region_coverage
from connects_common_connectivity.models import (
    Laterality,
    Modality,
    ProjectionMeasurementMatrix,
    ProjectionMeasurementType,
    Unit,
)


def test_populate_region_coverage_accepts_nested_list():
    """Region coverage must derive populated columns without mutating input."""
    pmm = ProjectionMeasurementMatrix(
        id="pmm_list",
        project_id="proj_a",
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
