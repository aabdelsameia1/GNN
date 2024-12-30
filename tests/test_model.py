# tests/test_model.py

import torch
from src.model import GNNModel

def test_gnn_forward():
    """Basic test to ensure the GNN forward pass doesn’t crash and outputs correct shape."""
    model = GNNModel(input_dim=16, hidden_dim=32, output_dim=1)
    dummy_x = torch.randn(8, 16)  # 8 nodes, 16 features
    dummy_edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # Some dummy connections
    dummy_batch = torch.zeros(8, dtype=torch.long)  # All nodes in the same graph

    output = model(dummy_x, dummy_edge_index, dummy_batch)
    assert output.shape == (1, 1), "Output shape should be [batch_size, output_dim]."
