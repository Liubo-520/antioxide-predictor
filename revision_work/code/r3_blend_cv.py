"""
Reviewer 3, comment 1 (methodology): choose the hybrid blend on TRAINING data
only, then report its external-test performance once.

In the submitted version the two-component blend was picked from several
candidate stacks by comparing their external-test R2. That makes the reported
margin over the reference model optimistically biased. Here the same candidate
set is ranked by out-of-fold (OOF) cross-validation R2 on the 1,528-compound
training set, with no access to the 383-compound test set. The winner is then
evaluated once on the test set.

Out-of-fold predictions
  AttentiveFP: the 30 saved fold models (3 seeds x 10 folds). For each seed the
    KFold partition of the training set is reproduced exactly
    (KFold(10, shuffle=True, random_state=seed)), and every fold model predicts
    only its own held-out fold. Averaging the three seeds gives one OOF value
    per training compound.
  Mordred RF / GBM / ET: refitted inside the same folds, so the descriptor
    selection and scaling are fitted on training folds only.

Also persists the full-training-set Mordred pipelines so that the independent
external set (Reviewer 3, comment 2) can be predicted with the descriptor
component.

Outputs: revision_work/results/blend_cv.json
         revision_work/results/mordred_models.joblib
"""

import os, sys, json, pickle, time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE); PROJ = os.path.dirname(REV)
sys.path.insert(0, os.path.join(PROJ, "code"))
from models import get_model

RES = os.path.join(PROJ, "results")
MODELS = os.path.join(RES, "models")
OUT = os.path.join(REV, "results"); os.makedirs(OUT, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 1, 100]
N_FOLDS = 10
K_SEL = 152


@torch.no_grad()
def predict_graphs(model, graphs):
    model.eval()
    out = []
    for batch in DataLoader(graphs, batch_size=256):
        out.append(model(batch.to(DEVICE)).cpu().numpy())
    return np.concatenate(out)


def afp_ckpt(seed, fold):
    if seed == 42:
        return os.path.join(MODELS, "random_attentivefp_fold%d.pt" % fold)
    return os.path.join(MODELS, "random_attentivefp_seed%d_fold%d.pt" % (seed, fold))


def make_pipe(model):
    return Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('var', VarianceThreshold(threshold=0.0)),
                     ('sel', SelectKBest(f_regression, k=K_SEL)),
                     ('std', StandardScaler()),
                     ('mdl', model)])


def fresh_models():
    return {
        'RF': make_pipe(RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)),
        'GBM': make_pipe(GradientBoostingRegressor(n_estimators=500, random_state=42)),
        'ET': make_pipe(ExtraTreesRegressor(n_estimators=500, random_state=42, n_jobs=-1)),
    }


def metrics(y, p):
    return dict(r2=float(r2_score(y, p)),
                rmse=float(np.sqrt(mean_squared_error(y, p))),
                mae=float(mean_absolute_error(y, p)))


def main():
    with open(os.path.join(RES, "processed_data.pkl"), 'rb') as f:
        data = pickle.load(f)
    df = data['df']
    pIC50 = df['pIC50'].values
    train_idx = np.array(data['random_split']['train_idx'])
    test_idx = np.array(data['random_split']['test_idx'])
    train_graphs = data['random_split']['train_graphs']
    test_graphs = data['random_split']['test_graphs']
    y_train = pIC50[train_idx]
    y_test = pIC50[test_idx]
    n_tr = len(train_graphs)
    assert n_tr == len(train_idx), (n_tr, len(train_idx))

    z = np.load(os.path.join(RES, "mordred_cache.npz"), allow_pickle=True)
    X_full, feat = z['X'], z['feat']
    X_train, X_test = X_full[train_idx], X_full[test_idx]
    print("Mordred matrix %s, training %s" % (X_full.shape, X_train.shape), flush=True)

    cache_path = os.path.join(OUT, "blend_oof_cache.npz")
    cached = np.load(cache_path) if os.path.exists(cache_path) else None

    # ---------------- OOF: AttentiveFP ------------------------------------
    oof_afp = np.zeros((len(SEEDS), n_tr))
    if cached is not None and 'oof_afp' in cached:
        oof_afp = cached['oof_afp']
        print("  AFP OOF loaded from cache", flush=True)
    else:
      for si, seed in enumerate(SEEDS):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_i, va_i) in enumerate(kf.split(train_graphs)):
            ck = afp_ckpt(seed, fold + 1)
            if not os.path.exists(ck):
                raise FileNotFoundError(ck)
            model = get_model('attentivefp').to(DEVICE)
            model.load_state_dict(torch.load(ck, map_location=DEVICE))
            va = [train_graphs[i] for i in va_i]
            oof_afp[si, va_i] = predict_graphs(model, va)
        print("  AFP OOF seed %d done (R2 %.4f)" % (seed, r2_score(y_train, oof_afp[si])), flush=True)
      np.savez(cache_path, oof_afp=oof_afp)
    oof_afp_mean = oof_afp.mean(axis=0)
    print("AFP OOF (3-seed mean) R2 %.4f" % r2_score(y_train, oof_afp_mean), flush=True)

    # ---------------- OOF: Mordred models ---------------------------------
    oof_mord = {k: np.zeros((len(SEEDS), n_tr)) for k in ('RF', 'GBM', 'ET')}
    if cached is not None and 'oof_RF' in cached:
        for k in oof_mord:
            oof_mord[k] = cached['oof_' + k]
        print("  Mordred OOF loaded from cache", flush=True)
        SEED_ITER = []
    else:
        SEED_ITER = list(enumerate(SEEDS))
    for si, seed in SEED_ITER:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        t0 = time.time()
        for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_tr))):
            for name, pipe in fresh_models().items():
                pipe.fit(X_train[tr_i], y_train[tr_i])
                oof_mord[name][si, va_i] = pipe.predict(X_train[va_i])
        print("  Mordred OOF seed %d done (%.1f min)" % (seed, (time.time() - t0) / 60), flush=True)
        for name in oof_mord:
            print("     %-4s R2 %.4f" % (name, r2_score(y_train, oof_mord[name][si])), flush=True)
    np.savez(cache_path, oof_afp=oof_afp, oof_RF=oof_mord['RF'],
             oof_GBM=oof_mord['GBM'], oof_ET=oof_mord['ET'])
    oof = {name: oof_mord[name].mean(axis=0) for name in oof_mord}
    oof['AFP'] = oof_afp_mean
    oof['Mordred-Combo'] = np.mean([oof['RF'], oof['GBM'], oof['ET']], axis=0)

    # ---------------- candidate blends, ranked on OOF ----------------------
    candidates = {
        'AFP alone': ['AFP'],
        'Mordred-Combo alone': ['Mordred-Combo'],
        'AFP + Mordred-Combo': ['AFP', 'Mordred-Combo'],
        'AFP + Mordred-ET': ['AFP', 'ET'],
        'AFP + Mordred-RF': ['AFP', 'RF'],
        'AFP + Mordred-GBM': ['AFP', 'GBM'],
        'AFP + RF + GBM': ['AFP', 'RF', 'GBM'],
        'AFP + RF + GBM + ET': ['AFP', 'RF', 'GBM', 'ET'],
    }
    cv_rank = []
    for name, comps in candidates.items():
        p = np.mean([oof[c] for c in comps], axis=0)
        m = metrics(y_train, p)
        cv_rank.append(dict(name=name, components=comps, **{'cv_' + k: v for k, v in m.items()}))
    cv_rank.sort(key=lambda r: -r['cv_r2'])
    print("\n=== candidate blends ranked by TRAINING OOF R2 ===", flush=True)
    for r in cv_rank:
        print("  %-24s OOF R2 %.4f  RMSE %.4f  MAE %.4f"
              % (r['name'], r['cv_r2'], r['cv_rmse'], r['cv_mae']), flush=True)
    selected = cv_rank[0]
    print("\nselected by CV: %s" % selected['name'], flush=True)

    # ---------------- fit on full training set, evaluate on test ----------
    full_mord = {}
    test_pred = {}
    for name, pipe in fresh_models().items():
        pipe.fit(X_train, y_train)
        full_mord[name] = pipe
        test_pred[name] = pipe.predict(X_test)
        print("  test %-4s R2 %.4f" % (name, r2_score(y_test, test_pred[name])), flush=True)
    joblib.dump(dict(models=full_mord, feat=feat.tolist(), k_sel=K_SEL),
                os.path.join(OUT, "mordred_models.joblib"))
    test_pred['Mordred-Combo'] = np.mean([test_pred['RF'], test_pred['GBM'], test_pred['ET']], axis=0)

    afp_test = np.array(json.load(open(os.path.join(RES, "ensemble_results.json")))['predictions'])
    test_pred['AFP'] = afp_test

    rows = []
    for r in cv_rank:
        p = np.mean([test_pred[c] for c in r['components']], axis=0)
        m = metrics(y_test, p)
        rows.append(dict(r, **{'test_' + k: v for k, v in m.items()}))
    print("\n=== test performance (reported once, after CV selection) ===", flush=True)
    for r in rows:
        star = " <- CV-selected" if r['name'] == selected['name'] else ""
        print("  %-24s OOF R2 %.4f | test R2 %.4f RMSE %.4f MAE %.4f%s"
              % (r['name'], r['cv_r2'], r['test_r2'], r['test_rmse'], r['test_mae'], star), flush=True)

    sel_row = next(r for r in rows if r['name'] == selected['name'])
    sel_pred = np.mean([test_pred[c] for c in sel_row['components']], axis=0)

    # bootstrap CI for the test R2 of the selected blend and for the margin
    rng = np.random.default_rng(0)
    n = len(y_test)
    bs = np.array([r2_score(y_test[i], sel_pred[i])
                   for i in (rng.integers(0, n, n) for _ in range(10000))])
    ci = np.percentile(bs, [2.5, 97.5])
    print("\nselected blend test R2 %.4f, bootstrap 95%% CI [%.4f, %.4f]"
          % (sel_row['test_r2'], ci[0], ci[1]), flush=True)
    print("P(test R2 > 0.78 reference) = %.3f" % float((bs > 0.78).mean()), flush=True)

    json.dump(dict(ranking=rows, selected=selected['name'],
                   selected_test=dict(r2=sel_row['test_r2'], rmse=sel_row['test_rmse'],
                                      mae=sel_row['test_mae']),
                   bootstrap_ci=[float(ci[0]), float(ci[1])],
                   p_beats_reference=float((bs > 0.78).mean()),
                   oof_r2={k: float(r2_score(y_train, v)) for k, v in oof.items()},
                   seeds=SEEDS, n_folds=N_FOLDS, k_sel=K_SEL,
                   selected_test_predictions=sel_pred.tolist(),
                   test_targets=y_test.tolist()),
              open(os.path.join(OUT, "blend_cv.json"), 'w'), indent=1)
    print("\nwrote blend_cv.json and mordred_models.joblib", flush=True)


if __name__ == '__main__':
    main()
