"""
Applicability domain analysis using Williams plot.
Uses PCA-reduced Morgan fingerprints (50 PCs) to obtain a tighter, more
informative leverage threshold (vs raw 2048-dim FP where p >> n inflates h*).
Also reports a Tanimoto similarity-based AD as a cross-check.
"""

import os, sys, pickle, json, warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error

CODE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.dirname(CODE_DIR)
RESULT_DIR = os.path.join(DATA_DIR, "results")

PCA_N_COMPONENTS = 50  # tight, interpretable AD


def compute_leverage(X_train, X_query):
    """h_i = x_i^T (X^T X)^{-1} x_i for each query point."""
    XtX_inv = np.linalg.pinv(X_train.T @ X_train)
    return np.einsum('ij,jk,ik->i', X_query, XtX_inv, X_query)


def williams_plot_data(train_X, test_X, preds, targets):
    n, p = train_X.shape
    h = compute_leverage(train_X, test_X)
    h_star = 3.0 * (p + 1) / n
    residuals = preds - targets
    s = np.sqrt(np.sum(residuals**2) / max(len(residuals) - 1, 1))
    e_std = residuals / s if s > 0 else residuals
    return h, e_std, h_star, s


def ad_statistics(h, e_std, h_star, sigma=3.0):
    inside  = (h <= h_star) & (np.abs(e_std) <= sigma)
    hi_lev  = (h >  h_star) & (np.abs(e_std) <= sigma)
    hi_res  = (h <= h_star) & (np.abs(e_std) >  sigma)
    outlier = (h >  h_star) & (np.abs(e_std) >  sigma)
    return {
        'in_AD':         int(inside.sum()),
        'high_leverage': int(hi_lev.sum()),
        'high_residual': int(hi_res.sum()),
        'outlier':       int(outlier.sum()),
        'in_AD_pct':     float(100 * inside.mean()),
        'mask_in':       inside.tolist(),
        'mask_hilo':     hi_lev.tolist(),
        'mask_hire':     hi_res.tolist(),
        'mask_out':      outlier.tolist(),
    }


def tanimoto_ad(train_fps, test_fps):
    """
    Fingerprint-similarity AD (cross-check).
    Threshold = mean(D_train) - 3*std(D_train), where D_train_i = mean Tanimoto
    of compound i to all other training compounds.  Test compound is in AD if
    its mean Tanimoto similarity to the training set exceeds this threshold.
    """
    def tanimoto_matrix(A, B):
        A = A.astype(np.float32)
        B = B.astype(np.float32)
        inter = A @ B.T
        a = A.sum(axis=1, keepdims=True)
        b = B.sum(axis=1, keepdims=True)
        return inter / (a + b.T - inter + 1e-12)

    sim_train = tanimoto_matrix(train_fps, train_fps)
    np.fill_diagonal(sim_train, 0.0)
    train_mean_sim = sim_train.sum(axis=1) / (sim_train.shape[1] - 1)
    threshold = train_mean_sim.mean() - 3 * train_mean_sim.std()

    sim_test = tanimoto_matrix(test_fps, train_fps)
    test_mean_sim = sim_test.mean(axis=1)
    in_ad = test_mean_sim >= threshold

    return {
        'threshold':       float(threshold),
        'train_mean_sim':  float(train_mean_sim.mean()),
        'train_std_sim':   float(train_mean_sim.std()),
        'test_mean_sim':   test_mean_sim.tolist(),
        'in_AD':           int(in_ad.sum()),
        'in_AD_pct':       float(100 * in_ad.mean()),
        'mask':            in_ad.tolist(),
    }


def main():
    with open(os.path.join(RESULT_DIR, "processed_data.pkl"), 'rb') as f:
        data = pickle.load(f)
    with open(os.path.join(RESULT_DIR, "random_split_results.json")) as f:
        rand_results = json.load(f)

    best = next(r for r in rand_results if r['model'] == 'attentivefp')
    preds   = np.array(best['predictions'])
    targets = np.array(best['targets'])

    train_fps = data['random_split']['train_fps'].astype(np.float32)
    test_fps  = data['random_split']['test_fps'].astype(np.float32)
    print(f"Train FPs: {train_fps.shape},  Test FPs: {test_fps.shape}")

    # ── Williams plot in PCA(50)-reduced space ────────────────────────────────
    pca = PCA(n_components=PCA_N_COMPONENTS, random_state=42)
    train_pcs = pca.fit_transform(train_fps)
    test_pcs  = pca.transform(test_fps)
    explained = float(pca.explained_variance_ratio_.sum())
    print(f"PCA: {PCA_N_COMPONENTS} components, "
          f"cumulative variance explained = {explained:.3f}")

    h, e_std, h_star, s = williams_plot_data(train_pcs, test_pcs, preds, targets)
    ad_stats = ad_statistics(h, e_std, h_star, sigma=3.0)

    print(f"\nWilliams plot (PCA-reduced FP):")
    print(f"  h* = 3(p+1)/n = {h_star:.4f}")
    print(f"  Residual std s = {s:.4f}")
    print(f"  In AD:            {ad_stats['in_AD']:4d} ({ad_stats['in_AD_pct']:.1f}%)")
    print(f"  High leverage:    {ad_stats['high_leverage']:4d}")
    print(f"  High residual:    {ad_stats['high_residual']:4d}")
    print(f"  Outlier (both):   {ad_stats['outlier']:4d}")

    mask_in = np.array(ad_stats['mask_in'])
    r2_in   = r2_score(targets[mask_in], preds[mask_in]) if mask_in.sum() else None
    rmse_in = float(np.sqrt(mean_squared_error(targets[mask_in], preds[mask_in]))) \
              if mask_in.sum() else None
    print(f"  R²  within AD: {r2_in:.4f}")
    print(f"  RMSE within AD: {rmse_in:.4f}")

    # ── Tanimoto-similarity AD (cross-check) ──────────────────────────────────
    tani = tanimoto_ad(train_fps, test_fps)
    print(f"\nTanimoto similarity AD:")
    print(f"  Threshold (μ−3σ of train mean Tanimoto) = {tani['threshold']:.4f}")
    print(f"  Train mean Tanimoto: {tani['train_mean_sim']:.4f} ± "
          f"{tani['train_std_sim']:.4f}")
    print(f"  In AD:    {tani['in_AD']} ({tani['in_AD_pct']:.1f}%)")

    mask_t = np.array(tani['mask'])
    if mask_t.sum():
        r2_tani = r2_score(targets[mask_t], preds[mask_t])
        rmse_tani = float(np.sqrt(mean_squared_error(targets[mask_t], preds[mask_t])))
        print(f"  R²  within (Tanimoto) AD: {r2_tani:.4f}")
        print(f"  RMSE within (Tanimoto) AD: {rmse_tani:.4f}")
    else:
        r2_tani, rmse_tani = None, None

    williams_data = {
        'method':        f'Williams plot on PCA({PCA_N_COMPONENTS}) of Morgan FP',
        'pca_n_components':     PCA_N_COMPONENTS,
        'pca_var_explained':    explained,
        'h':             h.tolist(),
        'e_std':         e_std.tolist(),
        'preds':         preds.tolist(),
        'targets':       targets.tolist(),
        'h_star':        float(h_star),
        's':             float(s),
        'ad_stats':      ad_stats,
        'r2_in_ad':      float(r2_in)   if r2_in   is not None else None,
        'rmse_in_ad':    rmse_in,
        'tanimoto_ad':   tani,
        'r2_in_tani':    float(r2_tani) if r2_tani is not None else None,
        'rmse_in_tani':  rmse_tani,
    }
    out_path = os.path.join(RESULT_DIR, "williams_plot_data.json")
    with open(out_path, 'w') as f:
        json.dump(williams_data, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
