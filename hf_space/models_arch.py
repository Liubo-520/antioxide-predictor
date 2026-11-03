"""
GNN architectures for AntioxFP: AttentiveFP (primary model).
Copied from the research codebase and trimmed to inference-only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import AttentiveFP


NODE_DIM = 40  # atom feature dimension
EDGE_DIM = 6   # bond feature dimension


class AttentiveFPModel(nn.Module):
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
            data = x
            x, edge_index, edge_attr, batch = (
                data.x, data.edge_index, data.edge_attr, data.batch)
        return self.gnn(x, edge_index, edge_attr, batch).squeeze(-1)
