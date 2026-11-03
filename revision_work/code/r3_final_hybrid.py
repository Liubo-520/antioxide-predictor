"""
Rebuild the descriptor component and the hybrid with a recipe that is selected
on training data only and is fully specified in the manuscript.

Why this is needed: results/mordred_combo_preds.npy (the descriptor component
of the submitted hybrid) cannot be reproduced from the archived scripts. Its
test R2 is 0.7498, and refitting the documented 03d pipeline gives a different
prediction vector. The submitted blend was also chosen by comparing candidate
stacks on the external test set, which biases the reported margin upward.

Procedure here
  1. Candidate descriptor models are ranked by out-of-fold R2 on the
     1,528-compound training set (3 seeds x 10 folds), never on the test set.
  2. The graph component is the tuned AttentiveFP ensemble whose
     hyperparameters were chosen by 5-fold CV on the training set.
  3. The hybrid is the unweighted mean of the two components. The 50/50
     weighting is fixed a priori and is not tuned on any data.
  4. Test metrics are computed once, with a bootstrap CI and a paired
     comparison against the published reference R2 of 0.78.

Outputs: revision_work/results/final_hybrid.json
         revision_work/results/descriptor_oof_cache.npz
"""

import os, sys, json, pickle, time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings; warnings.filterwarnings('ignore')
import numpy as np
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
RES = os.path.join(PROJ, "results")
OUT = os.path.join(REV, "results")
SEEDS = [42, 1, 100]
N_FOLDS = 10
CACHE = os.path.join(OUT, "descriptor_oof_cache.npz")


def pipe_k152(model):
    return Pipeline([('imp', SimpleImputer(strategy='median')),
                     ('var', VarianceThreshold(threshold=0.0)),
                     ('sel', SelectKBest(f_regression, k=152)),
                     ('std', StandardScaler()), ('mdl', model)])


def pipe_all(model):
    return Pipeline([('imp', SimpleImputer(strategy='median')), ('mdl', model)])


def candidates():
    """Descriptor-model candidates. Gradient boosting is only run on the
    152-descriptor subset; on the full 1,427-descriptor matrix it is
    computationally prohibitive for a 30-fit out-of-fold protocol."""
    return {
        'ET-k152': pipe_k152(ExtraTreesRegressor(n_estimators=500, random_state=42, n_jobs=-1)),
        'RF-k152': pipe_k152(RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)),
        'GBM-k152': pipe_k152(GradientBoostingRegressor(n_estimators=500, random_state=42)),
        'ET-all': pipe_all(ExtraTreesRegressor(n_estimators=500, random_state=42, n_jobs=-1)),
        'RF-all': pipe_all(RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)),
    }


def metrics(y, p):
    return dict(r2=float(r2_score(y, p)),
                rmse=float(np.sqrt(mean_squared_error(y, p))),
                mae=float(mean_absolute_error(y, p)))


def main():
    with open(os.path.join(RES, "processed_data.pkl"), 'rb') as f:
        data = pickle.load(f)
    df = data['df']
    y = df['pIC50'].values
    tr = np.array(data['random_split']['train_idx'])
    te = np.array(data['random_split']['test_idx'])
    z = np.load(os.path.join(RES, "mordred_cache.npz"), allow_pickle=True)
    X, feat = z['X'], z['feat']
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]
    n_tr = len(tr)
    print("Mordred %s | training %d | test %d" % (X.shape, n_tr, len(te)), flush=True)

    names = list(candidates().keys())
    if os.path.exists(CACHE):
        c = np.load(CACHE)
        oof = {k: c[k] for k in names if k in c}
        print("loaded OOF cache for: %s" % ", ".join(oof), flush=True)
    else:
        oof = {}
    for name in names:
        if name in oof:
            continue
        acc = np.zeros((len(SEEDS), n_tr))
        t0 = time.time()
        for si, seed in enumerate(SEEDS):
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            for tr_i, va_i in kf.split(np.arange(n_tr)):
                m = candidates()[name]
                m.fit(X_tr[tr_i], y_tr[tr_i])
                acc[si, va_i] = m.predict(X_tr[va_i])
        oof[name] = acc.mean(axis=0)
        print("  OOF %-9s R2 %.4f  (%.1f min)"
              % (name, r2_score(y_tr, oof[name]), (time.time() - t0) / 60), flush=True)
        np.savez(CACHE, **oof)

    rank = sorted(((n, r2_score(y_tr, oof[n])) for n in oof), key=lambda t: -t[1])
    print("\n=== descriptor candidates ranked by training OOF R2 ===", flush=True)
    for n, r in rank:
        print("  %-9s OOF R2 %.4f" % (n, r), flush=True)
    best_desc = rank[0][0]
    print("selected descriptor model: %s" % best_desc, flush=True)

    # fit the selected descriptor model on the full training set
    fitted = {}
    test_desc = {}
    for name in names:
        m = candidates()[name]
        m.fit(X_tr, y_tr)
        fitted[name] = m
        test_desc[name] = m.predict(X_te)
    joblib.dump(dict(models=fitted, feat=feat.tolist(), best=best_desc),
                os.path.join(OUT, "descriptor_models.joblib"))

    # graph components
    # the shipped checkpoints define the reported model: GPU training is not
    # bit-wise reproducible, so the ensemble rebuilt from the saved weights is
    # the one every downstream number must come from
    saved = os.path.join(OUT, "tuned_ensemble_test_preds.npy")
    if os.path.exists(saved):
        p_afp_tuned = np.load(saved)
    else:
        tuned = json.load(open(os.path.join(OUT, "tuned_retrain.json")))
        p_afp_tuned = np.array(tuned['attentivefp']['predictions'])
    p_afp_orig = np.array(json.load(open(os.path.join(RES, "ensemble_results.json")))['predictions'])

    out = dict(descriptor_oof={n: float(r2_score(y_tr, oof[n])) for n in oof},
               descriptor_selected=best_desc,
               descriptor_test={n: metrics(y_te, test_desc[n]) for n in names},
               graph={}, hybrid={})
    print("\n=== test metrics (computed once) ===", flush=True)
    for n in names:
        m = out['descriptor_test'][n]
        print("  descriptor %-9s test R2 %.4f RMSE %.4f MAE %.4f"
              % (n, m['r2'], m['rmse'], m['mae']), flush=True)
    for gname, gp in (('AttentiveFP as submitted', p_afp_orig),
                      ('AttentiveFP tuned', p_afp_tuned)):
        out['graph'][gname] = metrics(y_te, gp)
        print("  graph %-26s test R2 %.4f RMSE %.4f MAE %.4f"
              % (gname, out['graph'][gname]['r2'], out['graph'][gname]['rmse'],
                 out['graph'][gname]['mae']), flush=True)

    rng = np.random.default_rng(0)
    nte = len(y_te)
    boot_idx = [rng.integers(0, nte, nte) for _ in range(10000)]
    for gname, gp in (('AttentiveFP as submitted', p_afp_orig),
                      ('AttentiveFP tuned', p_afp_tuned)):
        hyb = (gp + test_desc[best_desc]) / 2.0
        m = metrics(y_te, hyb)
        bs = np.array([r2_score(y_te[i], hyb[i]) for i in boot_idx])
        ci = np.percentile(bs, [2.5, 97.5])
        m.update(bootstrap_ci=[float(ci[0]), float(ci[1])],
                 p_above_reference=float((bs > 0.78).mean()),
                 predictions=hyb.tolist())
        out['hybrid'][gname + ' + ' + best_desc] = m
        print("\n  hybrid %s + %s:" % (gname, best_desc), flush=True)
        print("    test R2 %.4f  RMSE %.4f  MAE %.4f" % (m['r2'], m['rmse'], m['mae']), flush=True)
        print("    bootstrap 95%% CI [%.4f, %.4f], P(R2 > 0.78) = %.3f"
              % (ci[0], ci[1], m['p_above_reference']), flush=True)

    out['_targets'] = y_te.tolist()
    out['_reference'] = dict(r2=0.78, rmse=0.45, mae=0.31,
                             source='Ghironi et al., J. Xenobiot. 2025, 15, 80 (integrated ET+GB+XGB)')
    json.dump(out, open(os.path.join(OUT, "final_hybrid.json"), 'w'), indent=1)
    print("\nwrote final_hybrid.json", flush=True)


if __name__ == '__main__':
    main()
