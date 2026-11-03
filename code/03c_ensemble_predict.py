"""
Build the final ensemble prediction by averaging all available AttentiveFP
fold models across seeds (seed=42 from 03_train_random_split.py, plus
seed=1 and seed=100 from 03b_train_extra_seeds.py).
Updates random_split_results.json with the ensemble result for AttentiveFP.
"""

import os, sys, pickle, json, glob, warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
from models import get_model

DATA_DIR   = os.path.dirname(CODE_DIR)
RESULT_DIR = os.path.join(DATA_DIR, "results")
MODEL_DIR  = os.path.join(RESULT_DIR, "models")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    test_graphs = data['random_split']['test_graphs']
    test_loader = DataLoader(test_graphs, batch_size=64)

    kwargs = {'hidden': 200, 'num_layers': 2, 'num_timesteps': 2, 'dropout': 0.2}

    # Collect all AttentiveFP fold checkpoints
    ckpts = []
    # Seed=42 (original) — fold1..fold10
    for f in range(1, 11):
        p = os.path.join(MODEL_DIR, f"random_attentivefp_fold{f}.pt")
        if os.path.exists(p):
            ckpts.append(('seed42', f, p))
    # Extra seeds
    for seed in [1, 100]:
        for f in range(1, 11):
            p = os.path.join(MODEL_DIR, f"random_attentivefp_seed{seed}_fold{f}.pt")
            if os.path.exists(p):
                ckpts.append((f'seed{seed}', f, p))

    print(f"Found {len(ckpts)} AttentiveFP fold checkpoints across "
          f"{len(set(c[0] for c in ckpts))} seeds.\n")

    if not ckpts:
        print("No checkpoints found. Run 03_train_random_split.py first.")
        return

    # Generate predictions from each model
    fold_preds = []
    targets = None
    for tag, fold, path in ckpts:
        model = get_model('attentivefp', **kwargs).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False))
        preds, tgts = predict(model, test_loader)
        fold_preds.append(preds)
        if targets is None:
            targets = tgts
        r2 = r2_score(tgts, preds)
        print(f"  {tag} fold{fold}: R²={r2:.4f}")

    fold_preds = np.array(fold_preds)
    print(f"\nEnsemble of {len(fold_preds)} fold models:")

    # Simple average ensemble
    ensemble = fold_preds.mean(axis=0)
    r2   = r2_score(targets, ensemble)
    rmse = float(np.sqrt(mean_squared_error(targets, ensemble)))
    mae  = float(mean_absolute_error(targets, ensemble))
    print(f"  Mean ensemble  → R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    # Median ensemble (robust to bad folds)
    med = np.median(fold_preds, axis=0)
    r2_med   = r2_score(targets, med)
    rmse_med = float(np.sqrt(mean_squared_error(targets, med)))
    print(f"  Median ensemble→ R²={r2_med:.4f}  RMSE={rmse_med:.4f}")

    # Reference comparison
    print(f"\nReference (Mordred ensemble, Ghironi 2025): R²=0.78  RMSE=0.45  MAE=0.31")
    delta = r2 - 0.78
    print(f"  ΔR² vs reference: {delta:+.4f}  "
          f"({'BEATS' if delta > 0 else 'BELOW'} reference)")

    # ── Update random_split_results.json with the new ensemble result ────────
    rand_path = os.path.join(RESULT_DIR, "random_split_results.json")
    with open(rand_path) as f:
        rand_results = json.load(f)

    afp = next(r for r in rand_results if r['model'] == 'attentivefp')
    # Keep CV stats unchanged (they were on seed=42)
    afp['test_r2']     = float(r2)
    afp['test_rmse']   = float(rmse)
    afp['test_mae']    = float(mae)
    afp['predictions'] = ensemble.tolist()
    afp['targets']     = targets.tolist()
    afp['n_ensemble']  = len(fold_preds)
    afp['ensemble_method'] = f'mean of {len(fold_preds)} fold models across multiple seeds'

    with open(rand_path, 'w') as f:
        json.dump(rand_results, f, indent=2)

    # Also save ensemble details
    out = {
        'n_models':   len(fold_preds),
        'test_r2':    float(r2),
        'test_rmse':  float(rmse),
        'test_mae':   float(mae),
        'r2_median_ensemble':   float(r2_med),
        'rmse_median_ensemble': float(rmse_med),
        'predictions': ensemble.tolist(),
        'targets':     targets.tolist(),
        'fold_preds':  fold_preds.tolist(),
        'reference_r2': 0.78,
        'beats_reference': bool(delta > 0),
    }
    with open(os.path.join(RESULT_DIR, "ensemble_results.json"), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nUpdated → random_split_results.json (AttentiveFP ensemble)")
    print(f"Saved   → ensemble_results.json")


if __name__ == '__main__':
    main()
