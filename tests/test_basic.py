def test_import():
    import connects_common_connectivity as ccc
    assert ccc.__version__


def test_model_generation():
    import connects_common_connectivity as ccc
    models = ccc.generate_pydantic_models()
    assert "BrainRegion" in models
    BrainRegion = models["BrainRegion"]
    # Root region without parent (parent_identifier now optional)
    br = BrainRegion(id="X", name="Some Region")
    assert br.id == "X"


def test_required_field_enforcement():
    import pytest
    import connects_common_connectivity as ccc
    models = ccc.generate_pydantic_models()
    DataItem = models["DataItem"]
    # dataset is required; omitting should raise a validation error
    with pytest.raises(Exception):
        DataItem(id="D1", name="Item 1")


def test_enum_validation():
    import pytest
    import connects_common_connectivity as ccc
    models = ccc.generate_pydantic_models()
    Modality = models["Modality"]  # Enum
    assert Modality.TRACER.name == "TRACER"
    DataSet = models["DataSet"]
    ds = DataSet(id="DS1", name="Dataset 1", modality=Modality.TRACER)
    # Depending on dynamic generation, modality may be stored as enum value or raw string
    assert str(ds.modality) in {Modality.TRACER.value, Modality.TRACER.name, str(Modality.TRACER)}
    # Invalid modality should raise error now that slot has enum range
    with pytest.raises(Exception):
        DataSet(id="DS2", name="Dataset 2", modality="NOT_A_VALID_MODALITY")


def test_multivalued_slot_list_type():
    import connects_common_connectivity as ccc
    models = ccc.generate_pydantic_models()
    BrainRegion = models["BrainRegion"]
    # child_identifiers multivalued; we can pass list
    parent = BrainRegion(id="BRP", name="Parent")
    child = BrainRegion(id="BRC", name="Child", parent_identifier=parent)
    # Provide descendants as list (may be forward ref objects)
    parent2 = BrainRegion(id="BRX", name="Parent2", parent_identifier=parent, descendants=[child])
    assert isinstance(parent2.descendants, list)


def test_probability_bounds_and_pattern():
    import pytest
    import connects_common_connectivity as ccc
    models = ccc.generate_pydantic_models()
    MappingSet = models["MappingSet"]
    CellToCellMapping = models["CellToCellMapping"]
    DataSet = models["DataSet"]
    DataItem = models["DataItem"]
    ds = DataSet(id="DS1", name="Dataset", modality=models["Modality"].TRACER)
    cell1 = DataItem(id="C1", name="Cell1", dataset=ds)
    cell2 = DataItem(id="C2", name="Cell2", dataset=ds)
    ms = MappingSet(id="MS1", name="Map1", method_name="Method", source_dataset=ds, target_dataset=ds)
    # Valid probability
    mapping = CellToCellMapping(id="M1", mapping_set=ms, source_cell=cell1, target_cell=cell2, probability=0.5)
    assert 0 <= mapping.probability <= 1
    # Invalid probability > 1
    with pytest.raises(Exception):
        CellToCellMapping(id="M2", mapping_set=ms, source_cell=cell1, target_cell=cell2, probability=1.5)

