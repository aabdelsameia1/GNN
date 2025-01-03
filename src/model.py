# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, 
    GINConv, 
    GATConv, 
    SAGEConv,
    GINEConv,
    global_mean_pool,
    global_max_pool,
    GlobalAttention
)
from data_utils import get_activation_fn, get_activation_module

###############################################################################
# Pooling Helpers
###############################################################################
def get_pooling_fn(pool_type, hidden_dim):
    pool_type = pool_type.lower()
    if pool_type == 'mean':
        return global_mean_pool
    elif pool_type == 'max':
        return global_max_pool
    elif pool_type == 'attention':
        gate_nn = nn.Sequential(nn.Linear(hidden_dim, 1))
        return GlobalAttention(gate_nn)
    else:
        raise ValueError(f"Unsupported pooling type: {pool_type}")


###############################################################################
# 1. GCN Model (no edge features)
###############################################################################
class GCNModel(nn.Module):
    """
    A flexible 2-layer GCN supporting dropout, batch norm, residual, etc.
    This model ignores edge_attr entirely.
    """
    def __init__(
        self, 
        in_channels, 
        hidden_dim=64, 
        out_channels=1,
        dropout=0.0,
        activation='relu',
        pool='mean',
        residual=False,
        batch_norm=False
    ):
        super(GCNModel, self).__init__()
        
        self.residual = residual
        self.batch_norm = batch_norm

        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation_fn = get_activation_fn(activation)
        self.pool = get_pooling_fn(pool, hidden_dim)
        self.lin = nn.Linear(hidden_dim, out_channels)

    def forward(self, x, edge_index, batch, edge_attr=None):
        # We ignore edge_attr for standard GCN
        x_in = x
        x = self.conv1(x, edge_index)
        x = self.activation_fn(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.dropout_layer(x)
        if self.residual and x.shape == x_in.shape:
            x = x + x_in

        x_in = x
        x = self.conv2(x, edge_index)
        x = self.activation_fn(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.dropout_layer(x)
        if self.residual and x.shape == x_in.shape:
            x = x + x_in
        
        x = self.pool(x, batch)
        x = self.lin(x)
        return x


###############################################################################
# 2. GIN Model (no edge features)
###############################################################################
class GINModel(nn.Module):
    """
    Standard 2-layer GIN ignoring edge_attr.
    """
    def __init__(
        self,
        in_channels,
        hidden_dim=64,
        out_channels=1,
        dropout=0.0,
        activation='relu',
        pool='mean',
        residual=False,
        batch_norm=False
    ):
        super(GINModel, self).__init__()
        self.residual = residual
        self.batch_norm = batch_norm
        self.activation_fn = get_activation_fn(activation)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            get_activation_module(activation),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = GINConv(self.mlp1)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            get_activation_module(activation),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv2 = GINConv(self.mlp2)

        if self.batch_norm:
            self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.pool = get_pooling_fn(pool, hidden_dim)
        self.lin = nn.Linear(hidden_dim, out_channels)

    def forward(self, x, edge_index, batch, edge_attr=None):
        # We ignore edge_attr for standard GIN
        x_in = x
        x = self.conv1(x, edge_index)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.dropout_layer(x)
        if self.residual and x.shape == x_in.shape:
            x = x + x_in

        x_in = x
        x = self.conv2(x, edge_index)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.dropout_layer(x)
        if self.residual and x.shape == x_in.shape:
            x = x + x_in

        x = self.pool(x, batch)
        x = self.lin(x)
        return x


###############################################################################
# 3. GINE Model (USES edge features)
###############################################################################
class GINEModel(nn.Module):
    """
    GINE variant that can incorporate edge_attr (e.g., bond types).
    """
    def __init__(
        self,
        in_channels,
        hidden_dim=64,
        out_channels=1,
        dropout=0.0,
        activation='relu',
        pool='mean',
        residual=False,
        batch_norm=False,
        edge_dim=None  # dimension of edge_attr, if known
    ):
        super(GINEModel, self).__init__()
        self.residual = residual
        self.batch_norm = batch_norm
        self.edge_dim = edge_dim

        self.activation_fn = get_activation_fn(activation)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # MLP for first GINEConv
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            get_activation_module(activation),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = GINEConv(nn=self.mlp1, edge_dim=edge_dim if edge_dim else 0)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # MLP for second GINEConv
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            get_activation_module(activation),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv2 = GINEConv(nn=self.mlp2, edge_dim=edge_dim if edge_dim else 0)

        if self.batch_norm:
            self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.pool = get_pooling_fn(pool, hidden_dim)
        self.lin = nn.Linear(hidden_dim, out_channels)

    def forward(self, x, edge_index, batch, edge_attr=None):
        # Fix dimension if edge_attr is 1D
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        
        # 1st GINEConv
        x_in = x
        x = self.conv1(x, edge_index, edge_attr)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.dropout_layer(x)
        if self.residual and x.shape == x_in.shape:
            x = x + x_in

        # 2nd GINEConv
        x_in = x
        x = self.conv2(x, edge_index, edge_attr)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.dropout_layer(x)
        if self.residual and x.shape == x_in.shape:
            x = x + x_in

        x = self.pool(x, batch)
        x = self.lin(x)
        return x


###############################################################################
# 4. GAT Model (no edge features)
###############################################################################
class GATModel(nn.Module):
    """
    A flexible 2-layer GAT supporting dropout, batch norm, residual, etc.
    Ignores edge_attr unless you implement a custom attention mechanism.
    """
    def __init__(
        self, 
        in_channels, 
        hidden_dim=64, 
        out_channels=1, 
        heads=4,
        dropout=0.0, 
        activation='relu', 
        pool='mean', 
        residual=False, 
        batch_norm=False
    ):
        super(GATModel, self).__init__()
        
        self.residual = residual
        self.batch_norm = batch_norm
        
        self.activation_fn = get_activation_fn(activation)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.conv1 = GATConv(in_channels, hidden_dim, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=True)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
            self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.pool = get_pooling_fn(pool, hidden_dim)
        self.lin = nn.Linear(hidden_dim, out_channels)

    def forward(self, x, edge_index, batch, edge_attr=None):
        x_in = x
        x = self.conv1(x, edge_index)
        x = self.activation_fn(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.dropout_layer(x)
        if self.residual and x.shape == x_in.shape:
            x = x + x_in

        x_in = x
        x = self.conv2(x, edge_index)
        x = self.activation_fn(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.dropout_layer(x)
        if self.residual and x.shape == x_in.shape:
            x = x + x_in

        x = self.pool(x, batch)
        x = self.lin(x)
        return x


###############################################################################
# 5. GraphSAGE Model (no edge features)
###############################################################################
class SAGEModel(nn.Module):
    """
    A flexible 2-layer GraphSAGE supporting dropout, batch norm, residual, etc.
    Ignores edge_attr unless you implement a custom aggregator.
    """
    def __init__(
        self,
        in_channels,
        hidden_dim=64,
        out_channels=1,
        dropout=0.0,
        activation='relu',
        pool='mean',
        residual=False,
        batch_norm=False
    ):
        super(SAGEModel, self).__init__()
        
        self.residual = residual
        self.batch_norm = batch_norm
        
        self.activation_fn = get_activation_fn(activation)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.conv1 = SAGEConv(in_channels, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.pool = get_pooling_fn(pool, hidden_dim)
        self.lin = nn.Linear(hidden_dim, out_channels)

    def forward(self, x, edge_index, batch, edge_attr=None):
        x_in = x
        x = self.conv1(x, edge_index)
        x = self.activation_fn(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.dropout_layer(x)
        if self.residual and x.shape == x_in.shape:
            x = x + x_in

        x_in = x
        x = self.conv2(x, edge_index)
        x = self.activation_fn(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.dropout_layer(x)
        if self.residual and x.shape == x_in.shape:
            x = x + x_in

        x = self.pool(x, batch)
        x = self.lin(x)
        return x


###############################################################################
# 6. Graph Transformer (no edge features by default)
###############################################################################
class GraphTransformer(nn.Module):
    """
    Placeholder for a Graph Transformer approach.
    Currently does not incorporate edge_attr in attention.
    """
    def __init__(self, in_channels, hidden_dim=64, out_channels=1, num_heads=4):
        super(GraphTransformer, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=2
        )
        self.lin_in = nn.Linear(in_channels, hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, out_channels)

    def forward(self, x, edge_index, batch, edge_attr=None):
        # We ignore edge_index and edge_attr in this placeholder
        x = self.lin_in(x)
        x = x.unsqueeze(1)           # [num_nodes, 1, hidden_dim]
        x = x.permute(0, 1, 2)       # [num_nodes, 1, hidden_dim]
        x = self.encoder(x)          # [num_nodes, 1, hidden_dim]
        x = x.squeeze(1)             # [num_nodes, hidden_dim]
        x = global_mean_pool(x, batch)
        x = self.lin_out(x)
        return x


###############################################################################
# 7. Factory Method
###############################################################################
def get_model(
    model_name, 
    in_channels, 
    hidden_dim=64, 
    out_channels=1,
    dropout=0.0,
    activation='relu',
    pool='mean',
    residual=False,
    batch_norm=False,
    heads=4,    # used for GAT/Transformer
    edge_dim=None  # used for GINE
):
    """
    Returns an instance of the requested model by name.
    - 'gcn', 'gin', 'gat', 'sage', 'transformer' ignore edge_attr
    - 'gine' uses edge_attr
    """
    model_name = model_name.lower()

    if model_name == 'gcn':
        return GCNModel(
            in_channels, 
            hidden_dim, 
            out_channels,
            dropout=dropout,
            activation=activation,
            pool=pool,
            residual=residual,
            batch_norm=batch_norm
        )
    elif model_name == 'gin':
        return GINModel(
            in_channels,
            hidden_dim,
            out_channels,
            dropout=dropout,
            activation=activation,
            pool=pool,
            residual=residual,
            batch_norm=batch_norm
        )
    elif model_name == 'gine':
        return GINEModel(
            in_channels,
            hidden_dim,
            out_channels,
            dropout=dropout,
            activation=activation,
            pool=pool,
            residual=residual,
            batch_norm=batch_norm,
            edge_dim=edge_dim
        )
    elif model_name == 'gat':
        return GATModel(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            activation=activation,
            pool=pool,
            residual=residual,
            batch_norm=batch_norm
        )
    elif model_name == 'sage':
        return SAGEModel(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            dropout=dropout,
            activation=activation,
            pool=pool,
            residual=residual,
            batch_norm=batch_norm
        )
    elif model_name == 'transformer':
        return GraphTransformer(
            in_channels,
            hidden_dim,
            out_channels,
            num_heads=heads
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}.")
