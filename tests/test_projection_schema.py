import pytest
from pydantic import ValidationError

import connects_common_connectivity as ccc


def test_laterality_enum():
    models = ccc.generate_pydantic_models()
    Laterality = models["Laterality"]
    assert Laterality.IPSILATERAL.name == "IPSILATERAL"
    assert Laterality.CONTRALATERAL.name == "CONTRALATERAL"
    assert Laterality.BILATERAL.name == "BILATERAL"
    assert Laterality.UNKNOWN.name == "UNKNOWN"


def test_projection_measurement_matrix_laterality():
    models = ccc.generate_pydantic_models()
    PMM = models["ProjectionMeasurementMatrix"]
    Laterality = models["Laterality"]
    Modality = models["Modality"]
    PMType = models["ProjectionMeasurementType"]
    # laterality is required; omitting it should raise
    with pytest.raises(ValidationError, match=r"(?s)laterality.*Field required"):
        PMM(id="P1", measurement_type=PMType.NUMBER_OF_TIPS, modality=Modality.MORPHOLOGY)
    # Valid laterality values accepted
    pmm = PMM(id="P1", measurement_type=PMType.NUMBER_OF_TIPS,
              modality=Modality.MORPHOLOGY, laterality=Laterality.IPSILATERAL)
    assert str(pmm.laterality) in {Laterality.IPSILATERAL.value, Laterality.IPSILATERAL.name, str(Laterality.IPSILATERAL)}
    # Invalid laterality should raise
    with pytest.raises(ValidationError, match=r"(?s)laterality.*Input should be"):
        PMM(id="P2", measurement_type=PMType.NUMBER_OF_TIPS,
            modality=Modality.MORPHOLOGY, laterality="NOT_VALID")


def test_region_coverage_on_pmm():
    models = ccc.generate_pydantic_models()
    PMM = models["ProjectionMeasurementMatrix"]
    Laterality = models["Laterality"]
    Modality = models["Modality"]
    PMType = models["ProjectionMeasurementType"]
    # region_coverage is optional — PMM without it should validate
    pmm_no_cov = PMM(id="P1", measurement_type=PMType.NUMBER_OF_TIPS,
                      modality=Modality.MORPHOLOGY, laterality=Laterality.IPSILATERAL,
                      region_index=["R1", "R2", "R3"])
    assert pmm_no_cov.region_coverage is None
    # PMM with region_coverage as subset of region_index
    pmm_with_cov = PMM(id="P2", measurement_type=PMType.NUMBER_OF_TIPS,
                        modality=Modality.MORPHOLOGY, laterality=Laterality.CONTRALATERAL,
                        region_index=["R1", "R2", "R3"],
                        region_coverage=["R1", "R2"])
    assert isinstance(pmm_with_cov.region_coverage, list)
    assert len(pmm_with_cov.region_coverage) == 2
    assert set(pmm_with_cov.region_coverage).issubset(set(pmm_with_cov.region_index))
