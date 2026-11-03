"""
GNN architectures: AttentiveFP (primary), GCN, GAT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    AttentiveFP,
    GCNConv, GATConv,
    global_mean_pool, global_add_pool,
)


NODE_DIM = 40   # atom feature dimension
EDGE_DIM = 6    # bond feature dimension


class AttentiveFPModel(nn.Module):
    """
    AttentiveFP: Pushing the Boundaries of Molecular Representation for Drug Discovery
    Xiong et al., J. Med. Chem. 2020.
    Specifically designed for molecular property prediction using graph attention.
    """
    def __init__(self, hidden=200, num_layers=2, num_timesteps=2, dropout=0.2):
        super().__init__()
        self.gnn = AttentiveFP(
            in_channels=NODE_DIM,
            hidden_channels=hidden,
            out_channels=1,
            edge_dim=EDGE_DIM,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )

    def forward(self, x, edge_index=None, edge_attr=None, batch=None, **kwargs):
        if hasattr(x, 'edge_index'):
            # Called with a PyG Data/DataBatch object (training path)
            data = x
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        return self.gnn(x, edge_index, edge_attr, batch).squeeze(-1)


class GCNModel(nn.Module):
    """3-layer Graph Convolutional Network baseline."""
    def __init__(self, hidden=256, dropout=0.2):
        super().__init__()
        self.lin_in = nn.Linear(NODE_DIM, hidden)
        self.conv1  = GCNConv(hidden, hidden)
        self.conv2  = GCNConv(hidden, hidden)
        self.conv3  = GCNConv(hidden, hidden)
        self.head   = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.dropout = dropout

    def forward(self, data):
        x = F.relu(self.lin_in(data.x))
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, data.edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.head(x).squeeze(-1)


class GATModel(nn.Module):
    """3-layer Graph Attention Network with 4 attention heads."""
    def __init__(self, hidden=64, heads=4, dropout=0.2):
        super().__init__()
        self.lin_in = nn.Linear(NODE_DIM, hidden * heads)
        self.conv1  = GATConv(hidden * heads, hidden, heads=heads, dropout=dropout)
        self.conv2  = GATConv(hidden * heads, hidden, heads=heads, dropout=dropout)
        self.conv3  = GATConv(hidden * heads, hidden, heads=1,     dropout=dropout)
        self.head   = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.dropout = dropout

    def forward(self, data):
        x = F.elu(self.lin_in(data.x))
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.elu(self.conv2(x, data.edge_index))
        x = F.elu(self.conv3(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.head(x).squeeze(-1)


def get_model(name, **kwargs):
    name = name.lower()
    if name == 'attentivefp':
        return AttentiveFPModel(**kwargs)
    elif name == 'gcn':
        return GCNModel(**kwargs)
    elif name == 'gat':
        return GATModel(**kwargs)
    raise ValueError(f"Unknown model: {name}")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
