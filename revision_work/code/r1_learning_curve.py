"""
Reviewer 1, comment 3: is 1,911 compounds enough for a graph neural network,
given the model's capacity?

Protocol
  For each training fraction f in {0.2, 0.4, 0.6, 0.8, 1.0}, draw f x 1,528
  training compounds at random, hold back 10 per cent of that subsample as the
  early-stopping validation split, train AttentiveFP from scratch, and evaluate
  on the fixed 383-compound external test set. Three independent subsamples per
  fraction. The test set is identical in every run, so the curve isolates the
  effect of training-set size.

Two capacities are compared at every fraction:
  'tuned'    hidden 300, L = 3, T = 2  (2.64 M parameters, the CV-selected model)
  'compact'  hidden 100, L = 2, T = 2  (0.23 M parameters)
A curve that is still rising at f = 1 indicates the endpoint is data-limited
rather than capacity-limited; a gap that closes as f grows indicates the larger
model only pays off once enough data is available.

Each run writes its own .json, so an interrupted run resumes.

Outputs: revision_work/results/learning_curve/*.json
         revision_work/results/learning_curve.json
"""

import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev_common import *

LC_DIR = os.path.join(REV_RESULT_DIR, "learning_curve")
os.makedirs(LC_DIR, exist_ok=True)

FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
REPEATS = [0, 1, 2]
CAPACITY = {
    'tuned': dict(hidden=300, num_layers=3, num_timesteps=2, dropout=0.2,
                  lr=1e-3, weight_decay=1e-5, batch_size=64),
    'compact': dict(hidden=100, num_layers=2, num_timesteps=2, dropout=0.2,
                    lr=1e-3, weight_decay=1e-5, batch_size=64),
}


def main():
    data = load_data()
    train_graphs = data['random_split']['train_graphs']
    test_graphs = data['random_split']['test_graphs']
    y_test = targets_of(test_graphs)
    test_loader = DataLoader(test_graphs, batch_size=256)
    n_full = len(train_graphs)

    for cap_name, cfg in CAPACITY.items():
        for frac in FRACTIONS:
            for rep in REPEATS:
                tag = "%s_f%03d_r%d" % (cap_name, int(frac * 100), rep)
                out = os.path.join(LC_DIR, tag + ".json")
                if os.path.exists(out):
                    continue
                rs = np.random.RandomState(1000 * rep + int(frac * 100))
                n_sub = int(round(frac * n_full))
                idx = rs.permutation(n_full)[:n_sub]
                n_val = max(30, int(round(0.1 * n_sub)))
                va = [train_graphs[i] for i in idx[:n_val]]
                tr = [train_graphs[i] for i in idx[n_val:]]
                t0 = time.time()
                state, val_r2, ep = train_one('attentivefp', cfg, tr, va, seed=42 + rep)
                model = build_model('attentivefp', cfg).to(DEVICE)
                model.load_state_dict(state)
                m = regression_metrics(y_test, predict(model, test_loader))
                rec = dict(capacity=cap_name, config=cfg, fraction=frac, repeat=rep,
                           n_train=len(tr), n_val=len(va), val_r2=val_r2,
                           best_epoch=ep, test=m,
                           n_params=count_params(build_model('attentivefp', cfg)),
                           minutes=(time.time() - t0) / 60)
                json.dump(rec, open(out, 'w'), indent=1)
                print("  %-18s n_train=%4d  test R2 %.4f  RMSE %.4f  (%.1f min)"
                      % (tag, len(tr), m['r2'], m['rmse'], rec['minutes']), flush=True)

    rows = [json.load(open(os.path.join(LC_DIR, f)))
            for f in sorted(os.listdir(LC_DIR)) if f.endswith('.json')]
    summary = {}
    for cap_name in CAPACITY:
        summary[cap_name] = []
        for frac in FRACTIONS:
            rs = [r for r in rows if r['capacity'] == cap_name and r['fraction'] == frac]
            if not rs:
                continue
            r2s = [r['test']['r2'] for r in rs]
            rmses = [r['test']['rmse'] for r in rs]
            summary[cap_name].append(dict(
                fraction=frac, n_train=int(np.mean([r['n_train'] for r in rs])),
                n_runs=len(rs),
                test_r2_mean=float(np.mean(r2s)),
                test_r2_sd=float(np.std(r2s, ddof=1)) if len(r2s) > 1 else 0.0,
                test_rmse_mean=float(np.mean(rmses)),
                test_rmse_sd=float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
                n_params=rs[0]['n_params']))
    print("\n=== learning curve (mean +- sd over %d subsamples) ===" % len(REPEATS), flush=True)
    for cap_name, rs in summary.items():
        print(" %s (%s parameters)" % (cap_name, format(rs[0]['n_params'], ',') if rs else '?'), flush=True)
        for r in rs:
            print("   f=%.1f  n_train=%4d   test R2 %.4f +- %.4f   RMSE %.4f"
                  % (r['fraction'], r['n_train'], r['test_r2_mean'],
                     r['test_r2_sd'], r['test_rmse_mean']), flush=True)
    json.dump(dict(summary=summary, runs=rows, fractions=FRACTIONS,
                   capacities=CAPACITY),
              open(os.path.join(REV_RESULT_DIR, "learning_curve.json"), 'w'), indent=1)
    print("\nwrote learning_curve.json", flush=True)


if __name__ == '__main__':
    main()
