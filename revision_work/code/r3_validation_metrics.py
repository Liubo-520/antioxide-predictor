"""
Reviewer 3, comment 1: Roy's modified r^2m metrics and the Tropsha / Golbraikh
external-validation parameters, plus a quantitative description of the
regression-to-the-mean compression visible in Figure 3.

All metrics are computed on the 383-compound external test set from the
predictions stored by the original pipeline. No model is retrained here.

Definitions follow:
  Golbraikh & Tropsha, J. Comput. Aided Mol. Des. 2002, 16, 357-369.
  Roy et al., Chemom. Intell. Lab. Syst. 2008, 90, 31-42 (r^2m).
  Roy et al., J. Comput. Chem. 2013, 34, 1071-1082 (Delta r^2m, scaled form).
  Consonni et al., J. Chemom. 2010, 24, 194-201 (Q^2F1, Q^2F2, Q^2F3).
  Lin, Biometrics 1989, 45, 255-268 (CCC).
"""

import os, sys, json
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings
warnings.filterwarnings('ignore')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
PROJ = os.path.dirname(REV)
OUT = os.path.join(REV, "results")
os.makedirs(OUT, exist_ok=True)


def r2_through_origin(y, p):
    """R^2_0: coefficient of determination for the regression p = k*y forced
    through the origin, evaluated against the observed values y."""
    k = float(np.sum(y * p) / np.sum(y * y))
    resid = p - k * y
    return 1.0 - float(np.sum(resid ** 2) / np.sum((p - np.mean(p)) ** 2)), k


def roy_tropsha(y, p, y_train_mean=None):
    y = np.asarray(y, float); p = np.asarray(p, float)
    n = len(y)
    r = float(np.corrcoef(y, p)[0, 1])
    r2 = r ** 2
    # conventional R^2 (predictive, 1 - SSE/SST about the observed mean)
    R2_pred = 1.0 - float(np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))

    # Tropsha: regressions through the origin, both directions
    r20, k = r2_through_origin(y, p)       # p vs y
    r20p, kp = r2_through_origin(p, y)     # y vs p

    # Roy's r^2m, both directions, and the scaled Delta
    r2m = r2 * (1.0 - np.sqrt(abs(r2 - r20)))
    r2mp = r2 * (1.0 - np.sqrt(abs(r2 - r20p)))
    r2m_mean = float((r2m + r2mp) / 2.0)
    r2m_delta = float(abs(r2m - r2mp))

    # Consonni Q^2F1/F2/F3 (need training mean for F1/F3)
    sse = float(np.sum((y - p) ** 2))
    q2f1 = q2f2 = q2f3 = None
    q2f2 = 1.0 - sse / float(np.sum((y - np.mean(y)) ** 2))
    if y_train_mean is not None:
        q2f1 = 1.0 - sse / float(np.sum((y - y_train_mean) ** 2))
        q2f3 = None  # requires training SS; filled by caller when available

    # Lin's concordance correlation coefficient
    sy, sp = np.std(y, ddof=0), np.std(p, ddof=0)
    cov = float(np.mean((y - np.mean(y)) * (p - np.mean(p))))
    ccc = 2 * cov / (sy ** 2 + sp ** 2 + (np.mean(y) - np.mean(p)) ** 2)

    # regression-to-the-mean / compression diagnostics
    slope, intercept = np.polyfit(y, p, 1)
    out = dict(
        n=n, r=r, r2=r2, R2_pred=R2_pred, rmse=rmse, mae=mae,
        r2_0=r20, k=k, r2_0_prime=r20p, k_prime=kp,
        rel_diff=float(abs(r2 - r20) / r2), rel_diff_prime=float(abs(r2 - r20p) / r2),
        abs_diff_r20=float(abs(r20 - r20p)),
        r2m=float(r2m), r2m_prime=float(r2mp), r2m_mean=r2m_mean, r2m_delta=r2m_delta,
        q2f1=q2f1, q2f2=q2f2, ccc=float(ccc),
        ols_slope=float(slope), ols_intercept=float(intercept),
        sd_obs=float(np.std(y, ddof=1)), sd_pred=float(np.std(p, ddof=1)),
        sd_ratio=float(np.std(p, ddof=1) / np.std(y, ddof=1)),
        mean_obs=float(np.mean(y)), mean_pred=float(np.mean(p)),
    )
    return out


def criteria_table(m):
    """Golbraikh-Tropsha acceptance criteria plus Roy's r^2m criteria."""
    c = []
    c.append(("Q2F2 (= R2pred) > 0.5", m['q2f2'], m['q2f2'] > 0.5))
    c.append(("r2 > 0.6", m['r2'], m['r2'] > 0.6))
    ok_k = (0.85 <= m['k'] <= 1.15)
    ok_kp = (0.85 <= m['k_prime'] <= 1.15)
    c.append(("0.85 <= k <= 1.15", m['k'], ok_k))
    c.append(("0.85 <= k' <= 1.15", m['k_prime'], ok_kp))
    c.append(("(r2 - r2_0)/r2 < 0.1", m['rel_diff'], m['rel_diff'] < 0.1))
    c.append(("(r2 - r2_0')/r2 < 0.1", m['rel_diff_prime'], m['rel_diff_prime'] < 0.1))
    c.append(("|r2_0 - r2_0'| < 0.3", m['abs_diff_r20'], m['abs_diff_r20'] < 0.3))
    c.append(("r2m (mean) > 0.5", m['r2m_mean'], m['r2m_mean'] > 0.5))
    c.append(("delta r2m < 0.2", m['r2m_delta'], m['r2m_delta'] < 0.2))
    c.append(("CCC > 0.85", m['ccc'], m['ccc'] > 0.85))
    return c


def main():
    res = {}
    # training-set mean for Q2F1
    import pickle
    with open(os.path.join(PROJ, "results", "processed_data.pkl"), 'rb') as f:
        data = pickle.load(f)
    tr = data['random_split']['train_graphs']
    y_tr = np.array([float(g.y.view(-1)[0]) for g in tr])
    y_train_mean = float(np.mean(y_tr))

    models = {}
    ens = json.load(open(os.path.join(PROJ, "results", "ensemble_results.json")))
    models['AttentiveFP 30-model ensemble'] = (ens['targets'], ens['predictions'])
    fin = json.load(open(os.path.join(PROJ, "results", "final_ensemble.json")))
    models['Hybrid (AttentiveFP + Mordred-Combo)'] = (fin['targets'], fin['predictions'])

    for name, (y, p) in models.items():
        m = roy_tropsha(y, p, y_train_mean)
        # Q2F3 needs the training sum of squares
        sse = float(np.sum((np.asarray(y) - np.asarray(p)) ** 2))
        m['q2f3'] = 1.0 - (sse / len(y)) / (float(np.sum((y_tr - y_train_mean) ** 2)) / len(y_tr))
        m['criteria'] = [(c[0], float(c[1]), bool(c[2])) for c in criteria_table(m)]
        m['all_criteria_passed'] = all(c[2] for c in m['criteria'])
        res[name] = m
        print(f"\n=== {name} (n={m['n']}) ===")
        print(f"  R2pred/Q2F2 {m['q2f2']:.4f}   Q2F1 {m['q2f1']:.4f}   Q2F3 {m['q2f3']:.4f}")
        print(f"  r2 {m['r2']:.4f}  r2_0 {m['r2_0']:.4f} (k {m['k']:.4f})   "
              f"r2_0' {m['r2_0_prime']:.4f} (k' {m['k_prime']:.4f})")
        print(f"  r2m {m['r2m']:.4f}  r2m' {m['r2m_prime']:.4f}  "
              f"mean {m['r2m_mean']:.4f}  delta {m['r2m_delta']:.4f}")
        print(f"  CCC {m['ccc']:.4f}   RMSE {m['rmse']:.4f}  MAE {m['mae']:.4f}")
        print(f"  OLS slope {m['ols_slope']:.4f} intercept {m['ols_intercept']:.4f}  "
              f"SD(pred)/SD(obs) {m['sd_ratio']:.4f}")
        for nm, val, ok in m['criteria']:
            print(f"    [{'PASS' if ok else 'FAIL'}] {nm:28s} = {val:.4f}")

    res['_meta'] = dict(y_train_mean=y_train_mean, n_train=len(y_tr))
    with open(os.path.join(OUT, "r3_validation_metrics.json"), 'w') as f:
        json.dump(res, f, indent=2)
    print("\nwrote", os.path.join(OUT, "r3_validation_metrics.json"))


if __name__ == '__main__':
    main()
