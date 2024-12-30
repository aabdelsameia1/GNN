# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        A simple GNN model with two GCN layers and a global mean pool.
        
        Args:
            input_dim (int):  Number of input features per node.
            hidden_dim (int): Hidden dimensionality of the GCN layers.
            output_dim (int): Output dimension (e.g., 1 for regression).
        """
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Global mean pooling (aggregates node embeddings into a single graph embedding)
        x = global_mean_pool(x, batch)

        # Linear layer to get final predictions
        x = self.lin(x)
        return x


if __name__ == "__main__":
    # Quick debug/demo
    dummy_x = torch.randn((8, 16))    # 8 nodes, each with 16 features
    dummy_edge_index = torch.tensor([[0,1,2],[1,2,0]])  # Example edges, shape [2, E]
    dummy_batch = torch.zeros(8, dtype=torch.long)       # All nodes in the same graph (batch=0)
    
    model = GNNModel(input_dim=16, hidden_dim=32, output_dim=1)
    output = model(dummy_x, dummy_edge_index, dummy_batch)
    print("Model output shape:", output.shape)
