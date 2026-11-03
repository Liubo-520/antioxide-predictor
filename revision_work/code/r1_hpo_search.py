"""
Reviewer 1, comments 1-2: systematic hyperparameter search for GCN, GAT and
AttentiveFP.

Selection uses 5-fold cross-validation on the 1,528-compound training set only
(KFold, shuffle=True, random_state=42). The 383-compound external test set is
never used for selection.

Stage 1: architecture grid at the original training settings
         (dropout 0.2, Adam lr 1e-3, weight decay 1e-5, batch 64).
Stage 2: one-factor-at-a-time scan of training hyperparameters around the
         best Stage-1 architecture of each model.

Usage:
  python r1_hpo_search.py --stage 1 --worker 0 --nworkers 6
  python r1_hpo_search.py --stage 2 --worker 0 --nworkers 6
"""

import os, sys, json, time, argparse, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev_common import *
from sklearn.model_selection import KFold

N_FOLDS = 5
HPO_DIR = os.path.join(REV_RESULT_DIR, "hpo")
os.makedirs(HPO_DIR, exist_ok=True)

ORIGINAL = {
    'attentivefp': dict(hidden=200, num_layers=2, num_timesteps=2, dropout=0.2),
    'gcn': dict(hidden=256, n_layers=3, dropout=0.2),
    'gat': dict(hidden=64, heads=4, n_layers=3, dropout=0.2),
}
TRAIN_DEFAULT = dict(lr=1e-3, weight_decay=1e-5, batch_size=64)


def stage1_configs():
    cfgs = []
    for h, l, t in itertools.product([100, 200, 300], [2, 3, 4], [1, 2, 3]):
        cfgs.append(('attentivefp', dict(hidden=h, num_layers=l, num_timesteps=t, dropout=0.2)))
    for h, l in itertools.product([64, 128, 256, 512], [2, 3, 4]):
        cfgs.append(('gcn', dict(hidden=h, n_layers=l, dropout=0.2)))
    for h, hd, l in itertools.product([32, 64, 128], [2, 4, 8], [2, 3, 4]):
        cfgs.append(('gat', dict(hidden=h, heads=hd, n_layers=l, dropout=0.2)))
    return [(m, {**c, **TRAIN_DEFAULT}) for m, c in cfgs]


def best_stage1(model):
    rows = read_results(1)
    rows = [r for r in rows if r['model'] == model]
    return max(rows, key=lambda r: r['cv_r2_mean'])['config']


def stage2_configs():
    cfgs = []
    scans = [('dropout', [0.1, 0.3]), ('lr', [5e-4, 2e-3]),
             ('batch_size', [32, 128]), ('weight_decay', [0.0, 1e-4])]
    for model in ['attentivefp', 'gcn', 'gat']:
        base = best_stage1(model)
        for key, vals in scans:
            for v in vals:
                cfgs.append((model, {**base, key: v}))
    return cfgs


def read_results(stage):
    rows = []
    for fn in sorted(os.listdir(HPO_DIR)):
        if fn.startswith(f"stage{stage}_") and fn.endswith(".jsonl"):
            with open(os.path.join(HPO_DIR, fn)) as f:
                rows += [json.loads(l) for l in f if l.strip()]
    return rows


def cfg_key(model, cfg):
    return model + '|' + json.dumps(cfg, sort_keys=True)


def run_cv(model, cfg, train_graphs):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_r2, fold_rmse, epochs = [], [], []
    for fold, (tr_i, va_i) in enumerate(kf.split(train_graphs)):
        tr = [train_graphs[i] for i in tr_i]
        va = [train_graphs[i] for i in va_i]
        state, r2, ep = train_one(model, cfg, tr, va, seed=42 + fold)
        m = build_model(model, cfg).to(DEVICE)
        m.load_state_dict(state)
        p = predict(m, DataLoader(va, batch_size=256))
        met = regression_metrics(targets_of(va), p)
        fold_r2.append(met['r2']); fold_rmse.append(met['rmse']); epochs.append(ep)
    return fold_r2, fold_rmse, epochs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, required=True)
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--nworkers', type=int, default=1)
    a = ap.parse_args()

    data = load_data()
    train_graphs = data['random_split']['train_graphs']
    cfgs = stage1_configs() if a.stage == 1 else stage2_configs()
    # Spread the slow AttentiveFP jobs evenly across workers
    mine = cfgs[a.worker::a.nworkers]
    out_path = os.path.join(HPO_DIR, f"stage{a.stage}_w{a.worker}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {cfg_key(r['model'], r['config']) for r in map(json.loads, f) if r}
    print(f"[stage {a.stage} worker {a.worker}] {len(mine)} configs, {len(done)} done", flush=True)

    for model, cfg in mine:
        if cfg_key(model, cfg) in done:
            continue
        t0 = time.time()
        fold_r2, fold_rmse, epochs = run_cv(model, cfg, train_graphs)
        n_par = count_params(build_model(model, cfg))
        row = dict(stage=a.stage, model=model, config=cfg, n_params=n_par,
                   cv_r2_folds=fold_r2, cv_r2_mean=float(np.mean(fold_r2)),
                   cv_r2_sd=float(np.std(fold_r2, ddof=1)),
                   cv_rmse_mean=float(np.mean(fold_rmse)),
                   best_epochs=epochs, minutes=(time.time() - t0) / 60)
        with open(out_path, 'a') as f:
            f.write(json.dumps(row) + '\n')
        print(f"  {model:11s} {cfg} -> CV R2 {row['cv_r2_mean']:.4f} "
              f"({row['minutes']:.1f} min)", flush=True)


if __name__ == '__main__':
    main()
