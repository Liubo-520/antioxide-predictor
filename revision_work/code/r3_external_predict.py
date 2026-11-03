"""
Reviewer 3, comment 2: evaluate the published model on the independent
literature set harvested by r3_harvest_external.py.

The compounds come from primary publications indexed in ChEMBL, were measured
with a 30 min DPPH incubation, and none of them appears in the training or test
set (exclusion on the InChIKey connectivity block).

Predictions
  graph component      : the published 30-model AttentiveFP ensemble
                         (3 seeds x 10 folds, weights in results/models)
  descriptor component : the Mordred Extra-Trees pipeline refitted on the
                         1,528-compound training set (mordred_models.joblib)
  hybrid               : mean of the two

Reliability
  applicability domain : leverage in the PCA(50)-reduced Morgan fingerprint
                         space with the same h* rule as the manuscript, plus
                         the Tanimoto-similarity criterion
  experimental noise   : for compounds measured in more than one publication,
                         the between-publication spread bounds how well any
                         model can agree with a single literature value

Outputs: revision_work/results/external_validation.json
         revision_work/results/external_predictions.csv
"""

import os, sys, json, pickle, importlib.util, csv
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import joblib
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE); PROJ = os.path.dirname(REV)
sys.path.insert(0, os.path.join(PROJ, "code"))
from models import get_model

RES = os.path.join(PROJ, "results")
MODELS = os.path.join(RES, "models")
OUT = os.path.join(REV, "results")
FIG = os.path.join(REV, "figures"); os.makedirs(FIG, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 1, 100]
N_FOLDS = 10
PCA_N = 50
TUNED_CFG = dict(hidden=300, num_layers=3, num_timesteps=2, dropout=0.2)
TUNED_DIR = os.path.join(OUT, "tuned_models")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def afp_ckpt(seed, fold):
    return os.path.join(TUNED_DIR, "attentivefp_tuned_seed%d_fold%d.pt" % (seed, fold))


@torch.no_grad()
def predict_graphs(model, graphs):
    model.eval()
    out = []
    for batch in DataLoader(graphs, batch_size=256):
        out.append(model(batch.to(DEVICE)).cpu().numpy())
    return np.concatenate(out)


def morgan(smiles_list, radius=2, nbits=2048):
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=nbits)
    arrs, bvs = [], []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        bv = gen.GetFingerprint(m)
        a = np.zeros((nbits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(bv, a)
        arrs.append(a); bvs.append(bv)
    return np.array(arrs, dtype=np.float64), bvs


def metrics(y, p):
    return dict(n=int(len(y)), r2=float(r2_score(y, p)),
                rmse=float(np.sqrt(mean_squared_error(y, p))),
                mae=float(mean_absolute_error(y, p)),
                pearson=float(stats.pearsonr(y, p)[0]),
                spearman=float(stats.spearmanr(y, p)[0]),
                kendall=float(stats.kendalltau(y, p)[0]))


def main():
    dp = load_module(os.path.join(PROJ, "code", "01_data_processing.py"), "dp")

    ext = json.load(open(os.path.join(OUT, "external_set_raw.json")))['compounds']
    smiles = [c['smiles'] for c in ext]
    y_ext = np.array([c['pIC50_exp'] for c in ext])
    print("external compounds: %d, pIC50 %.2f to %.2f (mean %.2f)"
          % (len(ext), y_ext.min(), y_ext.max(), y_ext.mean()), flush=True)

    graphs, keep = [], []
    for i, s in enumerate(smiles):
        g = dp.mol_to_graph(s, float(y_ext[i]))
        if g is not None:
            graphs.append(g); keep.append(i)
    keep = np.array(keep)
    if len(keep) != len(ext):
        print("  %d compounds dropped by the graph featuriser" % (len(ext) - len(keep)), flush=True)
    ext = [ext[i] for i in keep]
    smiles = [smiles[i] for i in keep]
    y_ext = y_ext[keep]

    # ---------------- graph component ------------------------------------
    fold_preds = []
    for seed in SEEDS:
        for fold in range(1, N_FOLDS + 1):
            ck = afp_ckpt(seed, fold)
            model = get_model('attentivefp', **TUNED_CFG).to(DEVICE)
            model.load_state_dict(torch.load(ck, map_location=DEVICE))
            fold_preds.append(predict_graphs(model, graphs))
    fold_preds = np.stack(fold_preds)
    p_afp = fold_preds.mean(axis=0)
    p_afp_sd = fold_preds.std(axis=0)
    print("AFP ensemble (%d fold models) done" % len(fold_preds), flush=True)

    # ---------------- descriptor component --------------------------------
    bundle = joblib.load(os.path.join(OUT, "descriptor_models.joblib"))
    feat = bundle['feat']
    from mordred import Calculator, descriptors
    import pandas as pd
    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    dfd = calc.pandas(mols, quiet=True, nproc=1)
    dfd = dfd.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    X_ext = dfd.reindex(columns=feat).values.astype(np.float64)
    print("Mordred matrix for external set: %s" % (X_ext.shape,), flush=True)
    p_et = bundle['models']['ET-all'].predict(X_ext)
    p_hyb = (p_afp + p_et) / 2.0

    # ---------------- applicability domain --------------------------------
    with open(os.path.join(RES, "processed_data.pkl"), 'rb') as f:
        data = pickle.load(f)
    df = data['df']
    smi_col = 'SMILES' if 'SMILES' in df.columns else df.columns[0]
    train_idx = np.array(data['random_split']['train_idx'])
    fp_train, bv_train = morgan([df[smi_col].values[i] for i in train_idx])
    fp_ext, bv_ext = morgan(smiles)

    pca = PCA(n_components=PCA_N, random_state=42).fit(fp_train)
    Ptr, Pex = pca.transform(fp_train), pca.transform(fp_ext)
    XtX_inv = np.linalg.pinv(Ptr.T @ Ptr)
    h_ext = np.einsum('ij,jk,ik->i', Pex, XtX_inv, Pex)
    n, p = Ptr.shape
    h_star = 3.0 * (p + 1) / n
    resid = y_ext - p_hyb
    e_std = resid / resid.std(ddof=1)
    in_ad = (h_ext <= h_star) & (np.abs(e_std) <= 3.0)
    print("Williams AD: h* = %.4f, inside %d/%d (%.1f%%)"
          % (h_star, int(in_ad.sum()), len(in_ad), 100.0 * in_ad.mean()), flush=True)

    sim_mat = np.array([DataStructs.BulkTanimotoSimilarity(b, bv_train) for b in bv_ext])
    mean_sim = sim_mat.mean(axis=1)
    max_sim = sim_mat.max(axis=1)
    d_train = np.array([np.mean(DataStructs.BulkTanimotoSimilarity(b, bv_train)) for b in bv_train])
    sim_thr = d_train.mean() - 3.0 * d_train.std(ddof=1)
    in_sim_ad = mean_sim > sim_thr
    print("Tanimoto AD: threshold %.4f, inside %d (%.1f%%); median max-similarity %.3f"
          % (sim_thr, int(in_sim_ad.sum()), 100.0 * in_sim_ad.mean(), float(np.median(max_sim))), flush=True)

    # ---------------- metrics ---------------------------------------------
    res = {}
    for name, p in (('AttentiveFP ensemble (tuned)', p_afp),
                    ('Mordred Extra-Trees (ET-all)', p_et), ('Hybrid', p_hyb)):
        res[name] = metrics(y_ext, p)
        print("\n%-22s n=%d R2 %.4f RMSE %.4f MAE %.4f | Pearson %.3f Spearman %.3f"
              % (name, res[name]['n'], res[name]['r2'], res[name]['rmse'],
                 res[name]['mae'], res[name]['pearson'], res[name]['spearman']), flush=True)

    y_tr = df['pIC50'].values[train_idx]
    sse = float(np.sum((y_ext - p_hyb) ** 2))
    res['Hybrid']['q2f1'] = 1.0 - sse / float(np.sum((y_ext - y_tr.mean()) ** 2))

    subsets = {
        'inside Williams AD': in_ad,
        'inside Tanimoto AD': in_sim_ad,
        'max Tanimoto >= 0.40': max_sim >= 0.40,
        'max Tanimoto >= 0.50': max_sim >= 0.50,
        'max Tanimoto >= 0.60': max_sim >= 0.60,
    }
    res['_subsets'] = {}
    print()
    for label, mask in subsets.items():
        if mask.sum() >= 10:
            m = metrics(y_ext[mask], p_hyb[mask])
            res['_subsets'][label] = m
            print("  hybrid, %-24s n=%3d R2 %.4f RMSE %.4f MAE %.4f"
                  % (label, m['n'], m['r2'], m['rmse'], m['mae']), flush=True)

    rep = [c for c in ext if c['n_measurements'] > 1]
    spreads = [c['pIC50_spread'] for c in rep]
    res['_experimental_noise'] = dict(
        n_replicated=len(rep),
        median_spread=float(np.median(spreads)) if spreads else None,
        mean_spread=float(np.mean(spreads)) if spreads else None)
    if spreads:
        print("\nbetween-publication spread for the %d replicated compounds: "
              "median %.2f, mean %.2f log units"
              % (len(rep), float(np.median(spreads)), float(np.mean(spreads))), flush=True)

    res['_meta'] = dict(h_star=float(h_star), pca_components=PCA_N,
                        tanimoto_threshold=float(sim_thr),
                        n_inside_williams=int(in_ad.sum()),
                        n_inside_tanimoto=int(in_sim_ad.sum()),
                        n_publications=len(set(d for c in ext for d in c['dois'])),
                        years=sorted(set(y for c in ext for y in c['years'])))
    json.dump(res, open(os.path.join(OUT, "external_validation.json"), 'w'), indent=1)

    with open(os.path.join(OUT, "external_predictions.csv"), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['chembl_id', 'smiles', 'pIC50_exp', 'pIC50_pred_hybrid',
                    'pIC50_pred_afp', 'pIC50_pred_mordred_et', 'afp_fold_sd',
                    'error', 'leverage', 'in_williams_ad', 'max_tanimoto',
                    'n_measurements', 'pIC50_spread', 'dois', 'years'])
        for i, c in enumerate(ext):
            w.writerow([c['chembl_id'], c['smiles'], "%.4f" % y_ext[i],
                        "%.4f" % p_hyb[i], "%.4f" % p_afp[i], "%.4f" % p_et[i],
                        "%.4f" % p_afp_sd[i], "%.4f" % (y_ext[i] - p_hyb[i]),
                        "%.5f" % h_ext[i], int(in_ad[i]), "%.3f" % max_sim[i],
                        c['n_measurements'], "%.3f" % c['pIC50_spread'],
                        '; '.join(c['dois']), '; '.join(str(v) for v in c['years'])])
    np.savez(os.path.join(OUT, "external_arrays.npz"), y=y_ext, hybrid=p_hyb,
             afp=p_afp, et=p_et, afp_sd=p_afp_sd, leverage=h_ext,
             in_ad=in_ad, max_sim=max_sim, mean_sim=mean_sim, h_star=h_star)
    print("\nwrote external_validation.json / external_predictions.csv", flush=True)


if __name__ == '__main__':
    main()
