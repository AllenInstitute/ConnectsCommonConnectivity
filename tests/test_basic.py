def test_import():
    import connects_common_connectivity as ccc
    assert ccc.__version__


def test_model_generation():
    import connects_common_connectivity as ccc
    models = ccc.generate_pydantic_models()
    assert "BrainRegion" in models
    BrainRegion = models["BrainRegion"]
    br = BrainRegion(id="X", label="Some Region", species="mouse")
    assert br.id == "X"
