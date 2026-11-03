"""
Retrain the tuned AttentiveFP ensemble and keep the fold weights.

r1_tuned_retrain.py stored only each fold's test predictions, which is enough
for the performance tables but not for predicting new molecules. The weights
are needed so that the independent literature set (Reviewer 3, comment 2) and
the deployed predictor can use the same model the manuscript reports.

Identical protocol and seeds as r1_tuned_retrain.py, so the fold models are the
same ones; each fold is skipped when its checkpoint already exists.

Outputs: revision_work/results/tuned_models/attentivefp_tuned_seed<seed>_fold<k>.pt
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev_common import *
from sklearn.model_selection import KFold

CFG = dict(hidden=300, num_layers=3, num_timesteps=2, dropout=0.2,
           lr=1e-3, weight_decay=1e-5, batch_size=64)
SEEDS = [42, 1, 100]
N_FOLDS = 10
WDIR = os.path.join(REV_RESULT_DIR, "tuned_models")
os.makedirs(WDIR, exist_ok=True)


def main():
    data = load_data()
    train_graphs = data['random_split']['train_graphs']
    test_graphs = data['random_split']['test_graphs']
    y_test = targets_of(test_graphs)
    test_loader = DataLoader(test_graphs, batch_size=256)
    preds = []
    for seed in SEEDS:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_i, va_i) in enumerate(kf.split(train_graphs)):
            ck = os.path.join(WDIR, "attentivefp_tuned_seed%d_fold%d.pt" % (seed, fold + 1))
            if os.path.exists(ck):
                model = build_model('attentivefp', CFG).to(DEVICE)
                model.load_state_dict(torch.load(ck, map_location=DEVICE))
                preds.append(predict(model, test_loader))
                continue
            tr = [train_graphs[i] for i in tr_i]
            va = [train_graphs[i] for i in va_i]
            t0 = time.time()
            state, val_r2, ep = train_one('attentivefp', CFG, tr, va, seed=seed + fold)
            torch.save(state, ck)
            model = build_model('attentivefp', CFG).to(DEVICE)
            model.load_state_dict(state)
            preds.append(predict(model, test_loader))
            print("  seed %d fold %2d  val R2 %.4f  (%.1f min)"
                  % (seed, fold + 1, val_r2, (time.time() - t0) / 60), flush=True)
    ens = np.mean(np.stack(preds), axis=0)
    m = regression_metrics(y_test, ens)
    print("\ntuned ensemble from saved weights: test R2 %.4f RMSE %.4f MAE %.4f"
          % (m['r2'], m['rmse'], m['mae']), flush=True)
    np.save(os.path.join(REV_RESULT_DIR, "tuned_ensemble_test_preds.npy"), ens)
    print("wrote %d checkpoints to %s" % (len(preds), WDIR), flush=True)


if __name__ == '__main__':
    main()
