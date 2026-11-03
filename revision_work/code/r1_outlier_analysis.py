"""
Reviewer 1, comment 7: outlier analysis. Which compounds have the largest
prediction errors, what do they look like, what does GNNExplainer highlight on
them, and what would make the model better for such molecules.

For every compound in the 383-compound external test set the script records the
error of the reported model, the Williams leverage, the maximum Tanimoto
similarity to the training set, a SMARTS-based chemical class, and the DOI of
the primary publication the measurement came from. The five largest absolute
errors are drawn with GNNExplainer atom-importance colouring, and ChEMBL is
queried for independent DPPH IC50 measurements of the same compound so that a
modelling error can be told apart from a disputed experimental value.

Atom maps use the single canonical AttentiveFP checkpoint, as in the submitted
interpretability figures; GNNExplainer explains one model, not an average.

Outputs: revision_work/results/outlier_analysis.json
         revision_work/figures/fig_outliers.png
"""

import os, sys, json, pickle, importlib.util, time, urllib.request, urllib.parse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Descriptors
from torch_geometric.explain import Explainer, GNNExplainer
RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE); PROJ = os.path.dirname(REV)
RES = os.path.join(PROJ, "results")
OUT = os.path.join(REV, "results")
FIG = os.path.join(REV, "figures"); os.makedirs(FIG, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_TOP = 5
PCA_N = 50

CLASSES = [
    ('Flavonoid', 'c1cc2c(cc1)oc(cc2=O)'),
    ('Catechol', '[c]([OX2H])[c][OX2H]'),
    ('Phenolic acid', '[c][CX3](=O)[OX2H1]'),
    ('Phenol', '[OX2H][c]'),
    ('Coumarin', 'O=c1ccc2ccccc2o1'),
    ('Quinone', 'O=C1C=CC(=O)C=C1'),
    ('Aromatic amine', '[c][NX3;H2,H1]'),
    ('Hydrazone/azomethine', '[CX3]=[NX2]'),
]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def classify(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return 'unparsed'
    for name, sma in CLASSES:
        q = Chem.MolFromSmarts(sma)
        if q is not None and m.HasSubstructMatch(q):
            return name
    return 'Other'


def morgan_bits(smiles_list, radius=2, nbits=2048):
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=nbits)
    arrs, bvs = [], []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        bv = gen.GetFingerprint(m)
        a = np.zeros((nbits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(bv, a)
        arrs.append(a); bvs.append(bv)
    return np.array(arrs, dtype=np.float64), bvs


def chembl_other_measurements(chembl_id):
    """Independent DPPH IC50 records for the same molecule, for provenance."""
    url = ("https://www.ebi.ac.uk/chembl/api/data/activity.json?"
           + urllib.parse.urlencode(dict(molecule_chembl_id=chembl_id,
                                         standard_type="IC50", limit=100)))
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            acts = json.load(r)['activities']
    except Exception as e:
        return dict(error=str(e))
    rows = []
    for a in acts:
        desc = (a.get('assay_description') or '')
        if 'DPPH' not in desc.upper():
            continue
        v, u = a.get('standard_value'), a.get('standard_units')
        if v is None or u != 'nM':
            continue
        rows.append(dict(pIC50=float(-np.log10(float(v) * 1e-9)),
                         assay=desc[:140],
                         document=a.get('document_chembl_id')))
    out = dict(n_dpph_ic50_records=len(rows), records=rows[:8])
    if len(rows) > 1:
        vals = [r['pIC50'] for r in rows]
        out['spread_log_units'] = float(max(vals) - min(vals))
    return out


def main():
    interp = load_module(os.path.join(PROJ, "code", "06_interpretability.py"), "interp")

    with open(os.path.join(RES, "processed_data.pkl"), 'rb') as f:
        data = pickle.load(f)
    df = data['df']
    y_all = df['pIC50'].values
    smiles_all = df['smiles'].values
    train_idx = np.array(data['random_split']['train_idx'])
    test_idx = np.array(data['random_split']['test_idx'])
    y_test = y_all[test_idx]
    smi_test = smiles_all[test_idx]

    # provenance from the source spreadsheet (same row order as df)
    import pandas as pd
    src = pd.read_excel(os.path.join(PROJ, "DPPH_30min_Dataset.xlsx"))
    assert len(src) == len(df), (len(src), len(df))
    assert np.allclose(src['minusLogIC50 (M)'].values, y_all, atol=1e-6), \
        "spreadsheet row order does not match the processed dataframe"
    doi_all = src['DOI'].values
    chembl_all = src['ChEMBL ID'].values

    # reported model predictions
    fh = os.path.join(OUT, "final_hybrid.json")
    if os.path.exists(fh):
        d = json.load(open(fh))
        key = [k for k in d['hybrid'] if k.startswith('AttentiveFP tuned')]
        key = key[0] if key else list(d['hybrid'])[0]
        preds = np.array(d['hybrid'][key]['predictions'])
        model_label = 'hybrid: ' + key
    else:
        preds = np.array(json.load(open(os.path.join(RES, "ensemble_results.json")))['predictions'])
        model_label = 'AttentiveFP 30-model ensemble'
    print("reported model: %s" % model_label, flush=True)

    err = y_test - preds
    abs_err = np.abs(err)

    # leverage and similarity
    fp_train, bv_train = morgan_bits([smiles_all[i] for i in train_idx])
    fp_test, bv_test = morgan_bits(list(smi_test))
    pca = PCA(n_components=PCA_N, random_state=42).fit(fp_train)
    Ptr, Pte = pca.transform(fp_train), pca.transform(fp_test)
    XtX_inv = np.linalg.pinv(Ptr.T @ Ptr)
    h = np.einsum('ij,jk,ik->i', Pte, XtX_inv, Pte)
    h_star = 3.0 * (Ptr.shape[1] + 1) / Ptr.shape[0]
    max_sim = np.array([max(DataStructs.BulkTanimotoSimilarity(b, bv_train)) for b in bv_test])

    order = np.argsort(-abs_err)
    top = order[:N_TOP]

    # ---------------- group diagnostics ---------------------------------
    rest = order[N_TOP:]
    diag = dict(
        model=model_label, h_star=float(h_star),
        rmse_all=float(np.sqrt(np.mean(err ** 2))),
        rmse_without_top=float(np.sqrt(np.mean(err[rest] ** 2))),
        mean_abs_err=float(abs_err.mean()),
        top_mean_abs_err=float(abs_err[top].mean()),
        exp_pIC50_top=[float(v) for v in y_test[top]],
        exp_pIC50_mean_all=float(y_test.mean()),
        exp_pIC50_sd_all=float(y_test.std(ddof=1)),
        max_sim_top_mean=float(max_sim[top].mean()),
        max_sim_all_mean=float(max_sim.mean()),
        leverage_top_mean=float(h[top].mean()),
        leverage_all_mean=float(h.mean()),
        n_top_above_h_star=int((h[top] > h_star).sum()),
    )
    # are large errors concentrated at the extremes of the activity range?
    q = np.quantile(y_test, [0.05, 0.95])
    extreme = (y_test <= q[0]) | (y_test >= q[1])
    diag['rmse_extreme_5pct_tails'] = float(np.sqrt(np.mean(err[extreme] ** 2)))
    diag['rmse_central_90pct'] = float(np.sqrt(np.mean(err[~extreme] ** 2)))
    diag['corr_abs_err_vs_max_sim'] = float(np.corrcoef(abs_err, max_sim)[0, 1])
    diag['corr_abs_err_vs_leverage'] = float(np.corrcoef(abs_err, h)[0, 1])
    print("RMSE all %.4f | excluding the %d largest errors %.4f"
          % (diag['rmse_all'], N_TOP, diag['rmse_without_top']), flush=True)
    print("RMSE in the 5%% activity tails %.4f vs central 90%% %.4f"
          % (diag['rmse_extreme_5pct_tails'], diag['rmse_central_90pct']), flush=True)
    print("corr(|error|, max Tanimoto) %.3f ; corr(|error|, leverage) %.3f"
          % (diag['corr_abs_err_vs_max_sim'], diag['corr_abs_err_vs_leverage']), flush=True)

    # ---------------- per-outlier detail + atom maps ----------------------
    model = interp.load_best_model('attentivefp')
    explainer = Explainer(model=model, algorithm=GNNExplainer(epochs=200),
                          explanation_type='model',
                          node_mask_type='attributes', edge_mask_type='object',
                          model_config=dict(mode='regression',
                                            task_level='graph',
                                            return_type='raw'))
    entries, records = [], []
    for rank, i in enumerate(top, 1):
        gi = int(test_idx[i])
        smi = smi_test[i]
        mol = Chem.MolFromSmiles(smi)
        pred_val, w, img, _ = interp.compute_atom_importance(smi, model, explainer)
        rec = dict(rank=rank, smiles=smi, chembl_id=str(chembl_all[gi]),
                   doi=str(doi_all[gi]), experimental_pIC50=float(y_test[i]),
                   predicted_pIC50=float(preds[i]), error=float(err[i]),
                   single_model_prediction=float(pred_val) if pred_val is not None else None,
                   leverage=float(h[i]), inside_leverage_threshold=bool(h[i] <= h_star),
                   max_tanimoto_to_training=float(max_sim[i]),
                   chemical_class=classify(smi),
                   mol_weight=float(Descriptors.MolWt(mol)) if mol else None,
                   n_heavy_atoms=int(mol.GetNumHeavyAtoms()) if mol else None)
        top_atoms = []
        if w is not None and mol is not None:
            for j in np.argsort(-np.asarray(w))[:5]:
                top_atoms.append(dict(atom_index=int(j),
                                      symbol=mol.GetAtomWithIdx(int(j)).GetSymbol(),
                                      aromatic=bool(mol.GetAtomWithIdx(int(j)).GetIsAromatic()),
                                      importance=float(w[j])))
        rec['top_importance_atoms'] = top_atoms
        rec['chembl_cross_check'] = chembl_other_measurements(str(chembl_all[gi]))
        records.append(rec)
        label = ("#%d  %s\nexp %.2f  pred %.2f  error %+.2f\n"
                 "max Tanimoto %.2f   leverage %.3f%s"
                 % (rank, rec['chemical_class'], rec['experimental_pIC50'],
                    rec['predicted_pIC50'], rec['error'],
                    rec['max_tanimoto_to_training'], rec['leverage'],
                    '' if rec['inside_leverage_threshold'] else '  (> h*)'))
        entries.append((img, label))
        print("  #%d err %+.2f  %s  simmax %.2f  %s"
              % (rank, rec['error'], rec['chemical_class'],
                 rec['max_tanimoto_to_training'], rec['chembl_id']), flush=True)
        time.sleep(0.3)

    # ---------------- figure ---------------------------------------------
    ncols = min(3, len(entries))
    nrows = int(np.ceil(len(entries) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (img, label) in zip(axes, entries):
        if img is not None:
            ax.imshow(img)
        ax.set_title(label, fontsize=8.2, linespacing=1.35)
        ax.axis('off')
    for ax in axes[len(entries):]:
        ax.axis('off')
    fig.suptitle('Largest prediction errors on the external test set, with '
                 'GNNExplainer atom importance\n(red = high importance, '
                 'blue = low importance)', fontsize=10.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ('png', 'pdf'):
        p = os.path.join(FIG, 'fig_outliers.' + ext)
        fig.savefig(p, dpi=330 if ext == 'png' else None, bbox_inches='tight',
                    facecolor='white')
        print('wrote', p, flush=True)

    json.dump(dict(diagnostics=diag, outliers=records,
                   all_abs_error_quantiles={
                       str(q): float(np.quantile(abs_err, q))
                       for q in (0.5, 0.75, 0.9, 0.95, 0.99)}),
              open(os.path.join(OUT, "outlier_analysis.json"), 'w'), indent=1)
    print("wrote outlier_analysis.json", flush=True)


if __name__ == '__main__':
    main()
