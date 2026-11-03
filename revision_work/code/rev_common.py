"""
Shared utilities for the revision analyses (CHERD-D-26-01950).

Generalised GCN / GAT / AttentiveFP builders whose default settings reproduce
the architectures in code/models.py exactly, plus the training loop used in the
original study (Adam, ReduceLROnPlateau on validation R2, gradient clipping,
early stopping with patience 30, max 300 epochs).
"""

import os, sys, pickle, json, time, warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentiveFP, GCNConv, GATConv, global_mean_pool
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

REV_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ_DIR = os.path.dirname(REV_DIR)
ORIG_RESULT_DIR = os.path.join(PROJ_DIR, "results")
REV_RESULT_DIR = os.path.join(REV_DIR, "results")
REV_FIG_DIR = os.path.join(REV_DIR, "figures")
os.makedirs(REV_RESULT_DIR, exist_ok=True)
os.makedirs(REV_FIG_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NODE_DIM, EDGE_DIM = 40, 6
MAX_EPOCHS, PATIENCE = 300, 30


class AFPNet(nn.Module):
    def __init__(self, hidden=200, num_layers=2, num_timesteps=2, dropout=0.2):
        super().__init__()
        self.gnn = AttentiveFP(in_channels=NODE_DIM, hidden_channels=hidden,
                               out_channels=1, edge_dim=EDGE_DIM,
                               num_layers=num_layers, num_timesteps=num_timesteps,
                               dropout=dropout)

    def forward(self, data):
        return self.gnn(data.x, data.edge_index, data.edge_attr, data.batch).squeeze(-1)


class GCNNet(nn.Module):
    """n_layers=3, hidden=256, dropout=0.2 reproduces code/models.py::GCNModel."""
    def __init__(self, hidden=256, n_layers=3, dropout=0.2):
        super().__init__()
        self.lin_in = nn.Linear(NODE_DIM, hidden)
        self.convs = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden // 2, 1))
        self.dropout = dropout

    def forward(self, data):
        x = F.relu(self.lin_in(data.x))
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, data.edge_index))
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, data.batch)
        return self.head(x).squeeze(-1)


class GATNet(nn.Module):
    """n_layers=3, hidden=64, heads=4, dropout=0.2 reproduces code/models.py::GATModel
    (n_layers-1 multi-head layers with concatenation, then one single-head layer)."""
    def __init__(self, hidden=64, heads=4, n_layers=3, dropout=0.2):
        super().__init__()
        self.lin_in = nn.Linear(NODE_DIM, hidden * heads)
        convs = [GATConv(hidden * heads, hidden, heads=heads, dropout=dropout)
                 for _ in range(n_layers - 1)]
        convs.append(GATConv(hidden * heads, hidden, heads=1, dropout=dropout))
        self.convs = nn.ModuleList(convs)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, data):
        x = F.elu(self.lin_in(data.x))
        for conv in self.convs:
            x = F.elu(conv(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.head(x).squeeze(-1)


ARCH_KEYS = {
    'attentivefp': ('hidden', 'num_layers', 'num_timesteps', 'dropout'),
    'gcn': ('hidden', 'n_layers', 'dropout'),
    'gat': ('hidden', 'heads', 'n_layers', 'dropout'),
}


def build_model(name, cfg):
    kw = {k: cfg[k] for k in ARCH_KEYS[name] if k in cfg}
    if name == 'attentivefp':
        return AFPNet(**kw)
    if name == 'gcn':
        return GCNNet(**kw)
    if name == 'gat':
        return GATNet(**kw)
    raise ValueError(name)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_data():
    with open(os.path.join(ORIG_RESULT_DIR, "processed_data.pkl"), 'rb') as f:
        return pickle.load(f)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    out = []
    for batch in loader:
        out.append(model(batch.to(DEVICE)).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def targets_of(graphs):
    return np.array([float(g.y.view(-1)[0]) for g in graphs])


def train_one(name, cfg, tr_graphs, val_graphs, seed):
    """Train one model with early stopping on validation R2.
    Returns (best_state, best_val_r2, best_epoch)."""
    set_seed(seed)
    bs = cfg.get('batch_size', 64)
    tr_loader = DataLoader(tr_graphs, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=256)
    y_val = targets_of(val_graphs)
    model = build_model(name, cfg).to(DEVICE)
    opt = Adam(model.parameters(), lr=cfg.get('lr', 1e-3),
               weight_decay=cfg.get('weight_decay', 1e-5))
    sched = ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-5)
    best_r2, best_state, best_epoch, wait = -np.inf, None, 0, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch in tr_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            loss = F.mse_loss(model(batch), batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        r2 = r2_score(y_val, predict(model, val_loader))
        sched.step(-r2)
        if r2 > best_r2:
            best_r2, best_epoch, wait = r2, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    return best_state, float(best_r2), best_epoch


def regression_metrics(y, p):
    return dict(r2=float(r2_score(y, p)),
                rmse=float(np.sqrt(mean_squared_error(y, p))),
                mae=float(mean_absolute_error(y, p)))
