# tests/test_model.py

import torch
from src.model import get_model

def test_all_models_forward():
    model_names = ["gcn", "gin", "gat", "sage"]
    in_channels = 10
    x = torch.rand((100, in_channels))
    edge_index = torch.randint(0, 100, (2, 200))
    batch = torch.zeros(100, dtype=torch.long)

    for name in model_names:
        model = get_model(name, in_channels=in_channels, hidden_dim=16, out_channels=1)
        out = model(x, edge_index, batch)
        assert out.ndim in (1, 2), f"Model {name} produced unexpected shape {out.shape}"
