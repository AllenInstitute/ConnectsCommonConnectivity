"""Example showing dynamic generation of Pydantic models from the LinkML schema."""
from connects_common_connectivity import generate_pydantic_models

models = generate_pydantic_models()
BrainRegion = models["BrainRegion"]
Connection = models["Connection"]

br1 = BrainRegion(id="BR1", label="Region 1", species="mouse")
br2 = BrainRegion(id="BR2", label="Region 2", species="mouse")
conn = Connection(source=br1, target=br2, weight=0.87, modality="tracer")
print(conn)
