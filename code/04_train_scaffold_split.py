"""
Scaffold split: flavonoid compounds → test set.
5-fold CV on non-flavonoid training set. Ensemble of fold models for test prediction.
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
from scipy import stats

warnings.filterwarnings('ignore')

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
from models import get_model

DATA_DIR   = os.path.dirname(CODE_DIR)
RESULT_DIR = os.path.join(DATA_DIR, "results")
MODEL_DIR  = os.path.join(RESULT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42
BATCH_SIZE  = 64
MAX_EPOCHS  = 300
PATIENCE    = 30
N_FOLDS     = 5


def train_epoch(model, loader, optimizer):
    model.train()
    total_loss, total_n = 0, 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(batch)
        y    = batch.y.view(-1)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_n    += batch.num_graphs
    return total_loss / total_n


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, targets = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        preds.append(model(batch).cpu().numpy())
        targets.append(batch.y.view(-1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


def metrics(preds, targets):
    r2   = r2_score(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae  = mean_absolute_error(targets, preds)
    return r2, rmse, mae


def train_one(model_name, kwargs, train_graphs, test_graphs, tag):
    print(f"\n  [{tag}] {model_name.upper()}")
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    cv_r2s, fold_test_preds = [], []

    for fold, (tr_i, val_i) in enumerate(kf.split(train_graphs)):
        tr_data    = [train_graphs[i] for i in tr_i]
        val_data   = [train_graphs[i] for i in val_i]
        tr_loader  = DataLoader(tr_data,  batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

        model = get_model(model_name, **kwargs).to(DEVICE)
        opt   = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = ReduceLROnPlateau(opt, patience=10, factor=0.5, min_lr=1e-5)

        best_val_r2, patience_cnt, best_state = -np.inf, 0, None
        for epoch in range(MAX_EPOCHS):
            train_epoch(model, tr_loader, opt)
            val_preds, val_tgts = predict(model, val_loader)
            val_r2, _, _ = metrics(val_preds, val_tgts)
            sched.step(-val_r2)
            if val_r2 > best_val_r2:
                best_val_r2  = val_r2
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    break

        model.load_state_dict(best_state)
        test_preds_fold, _ = predict(model, test_loader)
        fold_test_preds.append(test_preds_fold)
        cv_r2s.append(best_val_r2)
        print(f"    Fold {fold+1}: val R²={best_val_r2:.4f}")

        torch.save(best_state, os.path.join(MODEL_DIR, f"scaffold_{model_name}_fold{fold+1}.pt"))

    cv_r2s = np.array(cv_r2s)
    n = len(cv_r2s)
    ci = stats.t.ppf(0.975, df=n-1) * cv_r2s.std(ddof=1) / np.sqrt(n)

    # Get test targets
    test_tgts = []
    for batch in test_loader:
        test_tgts.append(batch.y.view(-1).numpy())
    test_tgts = np.concatenate(test_tgts)

    # Ensemble test prediction
    ensemble_preds = np.mean(fold_test_preds, axis=0)
    test_r2, test_rmse, test_mae = metrics(ensemble_preds, test_tgts)

    print(f"    CV R²: {cv_r2s.mean():.4f}±{ci:.4f}  "
          f"| Flavonoid test → R²={test_r2:.4f} RMSE={test_rmse:.4f} MAE={test_mae:.4f}")

    # Save best fold model as canonical
    best_fold = int(np.argmax(cv_r2s))
    best_ckpt = os.path.join(MODEL_DIR, f"scaffold_{model_name}_fold{best_fold+1}.pt")
    import shutil
    shutil.copy(best_ckpt, os.path.join(MODEL_DIR, f"scaffold_{model_name}.pt"))

    return {
        'model':         model_name,
        'tag':           tag,
        'cv_r2_mean':    float(cv_r2s.mean()),
        'cv_r2_ci':      float(ci),
        'cv_r2_median':  float(np.median(cv_r2s)),
        'cv_r2_all':     cv_r2s.tolist(),
        'test_r2':       float(test_r2),
        'test_rmse':     float(test_rmse),
        'test_mae':      float(test_mae),
        'predictions':   ensemble_preds.tolist(),
        'targets':       test_tgts.tolist(),
    }


def main():
    with open(os.path.join(RESULT_DIR, "processed_data.pkl"), 'rb') as f:
        data = pickle.load(f)

    ss = data['scaffold_split']
    train_graphs = ss['train_graphs']
    test_graphs  = ss['test_graphs']
    print(f"Scaffold split — Train: {len(train_graphs)}, "
          f"Test (flavonoids): {len(test_graphs)}")

    models_config = {
        'attentivefp': {'hidden': 200, 'num_layers': 2, 'num_timesteps': 2, 'dropout': 0.2},
        'gcn':         {'hidden': 256, 'dropout': 0.2},
        'gat':         {'hidden': 64,  'heads': 4,     'dropout': 0.2},
    }

    all_results = []
    t0 = time.time()
    for mname, kwargs in models_config.items():
        r = train_one(mname, kwargs, train_graphs, test_graphs, 'scaffold')
        all_results.append(r)

    print("\n" + "="*65)
    print("SCAFFOLD SPLIT — Flavonoid Generalization Test")
    print("="*65)
    print(f"{'Model':<14} {'CV R²':>8} {'±CI':>8} {'Flav R²':>9} {'RMSE':>8} {'MAE':>8}")
    print("-"*65)
    for r in all_results:
        print(f"{r['model']:<14} {r['cv_r2_mean']:>8.4f} {r['cv_r2_ci']:>8.4f} "
              f"{r['test_r2']:>9.4f} {r['test_rmse']:>8.4f} {r['test_mae']:>8.4f}")
    print(f"\nTime: {(time.time()-t0)/60:.1f} min")

    out_path = os.path.join(RESULT_DIR, "scaffold_split_results.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved → {out_path}")


if __name__ == '__main__':
    main()
