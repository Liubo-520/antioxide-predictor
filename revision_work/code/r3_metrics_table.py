"""
Reviewer 3, comment 1: Roy and Tropsha validation metrics for every model that
the revised manuscript reports, plus a quantitative treatment of the
compression of predictions toward the mean that the reviewer noticed in
Figure 3.

The compression is measured three ways: the ordinary-least-squares slope of
predicted on observed, the ratio of standard deviations, and the asymmetry
between r2m and r2m'. A variance-rescaling calibration fitted on out-of-fold
training predictions (never on the test set) is then applied to show that the
compression is a calibration property rather than a structural failure: it can
be removed, at a small cost in RMSE, without retraining.

Outputs: revision_work/results/validation_metrics_table.json
"""

import os, sys, json, pickle, importlib.util
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings; warnings.filterwarnings('ignore')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE); PROJ = os.path.dirname(REV)
RES = os.path.join(PROJ, "results"); OUT = os.path.join(REV, "results")

spec = importlib.util.spec_from_file_location("vm", os.path.join(HERE, "r3_validation_metrics.py"))
vm = importlib.util.module_from_spec(spec); spec.loader.exec_module(vm)


def main():
    with open(os.path.join(RES, "processed_data.pkl"), 'rb') as f:
        data = pickle.load(f)
    y_all = data['df']['pIC50'].values
    tr_idx = np.array(data['random_split']['train_idx'])
    y_tr = y_all[tr_idx]
    y_tr_mean = float(y_tr.mean())
    ss_tr = float(np.sum((y_tr - y_tr_mean) ** 2))

    fh = json.load(open(os.path.join(OUT, "final_hybrid.json")))
    y = np.array(fh['_targets'])

    afp_orig = np.array(json.load(open(os.path.join(RES, "ensemble_results.json")))['predictions'])
    saved = os.path.join(OUT, "tuned_ensemble_test_preds.npy")
    if os.path.exists(saved):
        afp_tuned = np.load(saved)
    else:
        tuned = json.load(open(os.path.join(OUT, "tuned_retrain.json")))
        afp_tuned = np.array(tuned['attentivefp']['predictions'])
    hyb_tuned = np.array(fh['hybrid']['AttentiveFP tuned + ET-all']['predictions'])
    hyb_orig = np.array(fh['hybrid']['AttentiveFP as submitted + ET-all']['predictions'])

    # descriptor-only test predictions, refitted here for the table
    import joblib
    z = np.load(os.path.join(RES, "mordred_cache.npz"), allow_pickle=True)
    X = z['X']
    te_idx = np.array(data['random_split']['test_idx'])
    bundle = joblib.load(os.path.join(OUT, "descriptor_models.joblib"))
    et_all = bundle['models']['ET-all'].predict(X[te_idx])

    models = [
        ('AttentiveFP ensemble (as submitted)', afp_orig),
        ('AttentiveFP ensemble (tuned)', afp_tuned),
        ('Mordred Extra-Trees (ET-all)', et_all),
        ('Hybrid (as-submitted AttentiveFP + ET-all)', hyb_orig),
        ('Hybrid (tuned AttentiveFP + ET-all)', hyb_tuned),
    ]

    table = {}
    for name, p in models:
        m = vm.roy_tropsha(y, p, y_tr_mean)
        sse = float(np.sum((y - p) ** 2))
        m['q2f3'] = 1.0 - (sse / len(y)) / (ss_tr / len(y_tr))
        crit = vm.criteria_table(m)
        m['criteria'] = [(c[0], float(c[1]), bool(c[2])) for c in crit]
        m['n_passed'] = int(sum(c[2] for c in crit))
        m['n_criteria'] = len(crit)
        table[name] = m
        print("%-44s Q2F2 %.4f r2m %.4f d_r2m %.4f CCC %.4f slope %.3f SDratio %.3f  %d/%d"
              % (name, m['q2f2'], m['r2m_mean'], m['r2m_delta'], m['ccc'],
                 m['ols_slope'], m['sd_ratio'], m['n_passed'], m['n_criteria']), flush=True)
        for nm, val, ok in crit:
            if not ok:
                print("      FAILS %-26s = %.4f" % (nm, val), flush=True)

    # ---- compression: variance rescaling fitted on out-of-fold training ----
    print("\n=== compression diagnostics and out-of-fold calibration ===", flush=True)
    oof_g = np.load(os.path.join(OUT, "blend_oof_cache.npz"))
    oof_d = np.load(os.path.join(OUT, "descriptor_oof_cache.npz"))
    oof_afp = oof_g['oof_afp'].mean(axis=0)
    oof_et = oof_d['ET-all']
    oof_hyb = (oof_afp + oof_et) / 2.0
    a, b = np.polyfit(y_tr, oof_hyb, 1)          # oof_pred = a*y + b
    print("  OOF regression of prediction on observation: slope %.4f intercept %.4f"
          % (a, b), flush=True)
    calib = (hyb_tuned - b) / a                   # invert, parameters from training only
    mc = vm.roy_tropsha(y, calib, y_tr_mean)
    sse = float(np.sum((y - calib) ** 2))
    mc['q2f3'] = 1.0 - (sse / len(y)) / (ss_tr / len(y_tr))
    crit = vm.criteria_table(mc)
    mc['criteria'] = [(c[0], float(c[1]), bool(c[2])) for c in crit]
    mc['n_passed'] = int(sum(c[2] for c in crit))
    mc['n_criteria'] = len(crit)
    table['Hybrid (tuned) after out-of-fold variance rescaling'] = mc
    bt = table['Hybrid (tuned AttentiveFP + ET-all)']
    print("  before calibration: slope %.3f SDratio %.3f RMSE %.4f r2m %.4f d_r2m %.4f (%d/%d)"
          % (bt['ols_slope'], bt['sd_ratio'], bt['rmse'], bt['r2m_mean'],
             bt['r2m_delta'], bt['n_passed'], 10), flush=True)
    print("  after  calibration: slope %.3f SDratio %.3f RMSE %.4f r2m %.4f d_r2m %.4f (%d/%d)"
          % (mc['ols_slope'], mc['sd_ratio'], mc['rmse'], mc['r2m_mean'],
             mc['r2m_delta'], mc['n_passed'], 10), flush=True)

    json.dump(dict(models=table, oof_calibration=dict(slope=float(a), intercept=float(b))),
              open(os.path.join(OUT, "validation_metrics_table.json"), 'w'), indent=1)
    print("\nwrote validation_metrics_table.json", flush=True)


if __name__ == '__main__':
    main()
