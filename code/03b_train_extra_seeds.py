"""
Train additional AttentiveFP seeds for ensemble averaging.
Combined with existing seed=42 fold models, builds a 30-model ensemble
to surpass the descriptor-based reference benchmark.
"""

import os, sys, pickle, json, time, warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
from models import get_model

DATA_DIR   = os.path.dirname(CODE_DIR)
RESULT_DIR = os.path.join(DATA_DIR, "results")
MODEL_DIR  = os.path.join(RESULT_DIR, "models")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64
MAX_EPOCHS = 300
PATIENCE   = 30
N_FOLDS    = 10
EXTRA_SEEDS = [1, 100]  # in addition to 42 already trained


def train_epoch(model, loader, opt):
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE)
        opt.zero_grad()
        pred = model(batch)
        loss = torch.nn.functional.mse_loss(pred, batch.y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, tgts = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        preds.append(model(batch).cpu().numpy())
        tgts.append(batch.y.view(-1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(tgts)


def main():
    with open(os.path.join(RESULT_DIR, "processed_data.pkl"), 'rb') as f:
        data = pickle.load(f)
    train_graphs = data['random_split']['train_graphs']
    test_graphs  = data['random_split']['test_graphs']
    test_loader  = DataLoader(test_graphs, batch_size=BATCH_SIZE)
    print(f"Train: {len(train_graphs)}, Test: {len(test_graphs)}, Device: {DEVICE}")

    kwargs = {'hidden': 200, 'num_layers': 2, 'num_timesteps': 2, 'dropout': 0.2}

    for seed in EXTRA_SEEDS:
        print(f"\n{'='*60}\nSEED {seed} — AttentiveFP × 10 folds\n{'='*60}")
        torch.manual_seed(seed); np.random.seed(seed)
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        for fold, (tr_i, val_i) in enumerate(kf.split(train_graphs)):
            tr_data  = [train_graphs[i] for i in tr_i]
            val_data = [train_graphs[i] for i in val_i]
            tr_loader  = DataLoader(tr_data,  batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

            model = get_model('attentivefp', **kwargs).to(DEVICE)
            opt   = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            sched = ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-5)

            best_val_r2, patience_cnt, best_state = -np.inf, 0, None
            for epoch in range(MAX_EPOCHS):
                train_epoch(model, tr_loader, opt)
                vp, vt = predict(model, val_loader)
                r2 = r2_score(vt, vp)
                sched.step(-r2)
                if r2 > best_val_r2:
                    best_val_r2 = r2
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= PATIENCE:
                        break

            ckpt = os.path.join(MODEL_DIR, f"random_attentivefp_seed{seed}_fold{fold+1}.pt")
            torch.save(best_state, ckpt)
            print(f"  Seed {seed} Fold {fold+1:2d}: val R²={best_val_r2:.4f}")


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
