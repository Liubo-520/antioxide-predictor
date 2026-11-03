"""
Reviewer 1, comment 2: retrain each architecture at its tuned configuration
using exactly the training protocol of the submitted study, so the effect of
hyperparameter optimisation on external-test accuracy can be reported.

Configurations come from the 90-config search (stage 1 + stage 2) ranked by
5-fold CV R2 on the 1,528-compound training set. The test set is never used
for selection.

  AttentiveFP: 3 seeds x 10 folds = 30 fold models (the submitted protocol)
  GCN, GAT:    1 seed  x 10 folds = 10 fold models (the submitted protocol)

Every fold writes its test prediction to its own .npy immediately, so an
interrupted run resumes without repeating finished folds.

Outputs: revision_work/results/tuned/<tag>.npy  (per-fold test predictions)
         revision_work/results/tuned_retrain.json
"""

import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev_common import *
from sklearn.model_selection import KFold

TUNED_DIR = os.path.join(REV_RESULT_DIR, "tuned")
os.makedirs(TUNED_DIR, exist_ok=True)
N_FOLDS = 10

# tuned configurations (best 5-fold CV R2 over the 90-config search)
TUNED = {
    'attentivefp': (dict(hidden=300, num_layers=3, num_timesteps=2, dropout=0.2,
                         lr=1e-3, weight_decay=1e-5, batch_size=64), [42, 1, 100]),
    'gat': (dict(hidden=128, heads=8, n_layers=3, dropout=0.2,
                 lr=1e-3, weight_decay=1e-5, batch_size=64), [42]),
    'gcn': (dict(hidden=512, n_layers=4, dropout=0.2,
                 lr=1e-3, weight_decay=1e-5, batch_size=32), [42]),
}
# as-submitted configurations, for the reference row
ORIGINAL = {
    'attentivefp': dict(hidden=200, num_layers=2, num_timesteps=2, dropout=0.2),
    'gat': dict(hidden=64, heads=4, n_layers=3, dropout=0.2),
    'gcn': dict(hidden=256, n_layers=3, dropout=0.2),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='attentivefp,gat,gcn')
    a = ap.parse_args()

    data = load_data()
    train_graphs = data['random_split']['train_graphs']
    test_graphs = data['random_split']['test_graphs']
    y_test = targets_of(test_graphs)
    test_loader = DataLoader(test_graphs, batch_size=256)

    for name in a.models.split(','):
        cfg, seeds = TUNED[name]
        for seed in seeds:
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            for fold, (tr_i, va_i) in enumerate(kf.split(train_graphs)):
                tag = "%s_tuned_seed%d_fold%d" % (name, seed, fold + 1)
                out = os.path.join(TUNED_DIR, tag + ".npy")
                if os.path.exists(out):
                    continue
                tr = [train_graphs[i] for i in tr_i]
                va = [train_graphs[i] for i in va_i]
                t0 = time.time()
                state, val_r2, ep = train_one(name, cfg, tr, va, seed=seed + fold)
                model = build_model(name, cfg).to(DEVICE)
                model.load_state_dict(state)
                np.save(out, predict(model, test_loader))
                np.save(os.path.join(TUNED_DIR, tag + "_val.npy"),
                        np.array([val_r2, ep, (time.time() - t0) / 60]))
                print("  %-34s val R2 %.4f (epoch %d, %.1f min)"
                      % (tag, val_r2, ep, (time.time() - t0) / 60), flush=True)

    # ------------------------------------------------ assemble the ensembles
    res = {}
    for name in TUNED:
        cfg, seeds = TUNED[name]
        preds, vals = [], []
        for seed in seeds:
            for fold in range(1, N_FOLDS + 1):
                p = os.path.join(TUNED_DIR, "%s_tuned_seed%d_fold%d.npy" % (name, seed, fold))
                v = os.path.join(TUNED_DIR, "%s_tuned_seed%d_fold%d_val.npy" % (name, seed, fold))
                if os.path.exists(p):
                    preds.append(np.load(p))
                if os.path.exists(v):
                    vals.append(float(np.load(v)[0]))
        if not preds:
            continue
        ens = np.mean(np.stack(preds), axis=0)
        m = regression_metrics(y_test, ens)
        res[name] = dict(config=cfg, seeds=seeds, n_fold_models=len(preds),
                         n_params=count_params(build_model(name, cfg)),
                         cv_val_r2_mean=float(np.mean(vals)) if vals else None,
                         cv_val_r2_sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                         test=m, predictions=ens.tolist())
        print("\n%s tuned ensemble (%d fold models): test R2 %.4f RMSE %.4f MAE %.4f"
              % (name, len(preds), m['r2'], m['rmse'], m['mae']), flush=True)

    # hybrid with the tuned graph component
    mc_path = os.path.join(ORIG_RESULT_DIR, "mordred_combo_preds.npy")
    if 'attentivefp' in res and os.path.exists(mc_path):
        mc = np.load(mc_path)
        hyb = (np.array(res['attentivefp']['predictions']) + mc) / 2.0
        m = regression_metrics(y_test, hyb)
        res['hybrid_tuned'] = dict(test=m, predictions=hyb.tolist(),
                                   note='mean of tuned AttentiveFP ensemble and Mordred-Combo')
        print("\nhybrid (tuned AFP + Mordred-Combo): test R2 %.4f RMSE %.4f MAE %.4f"
              % (m['r2'], m['rmse'], m['mae']), flush=True)

    res['_targets'] = y_test.tolist()
    res['_original_configs'] = ORIGINAL
    with open(os.path.join(REV_RESULT_DIR, "tuned_retrain.json"), 'w') as f:
        json.dump(res, f, indent=1)
    print("\nwrote tuned_retrain.json", flush=True)


if __name__ == '__main__':
    main()
