"""
Chemical space analysis: UMAP of Morgan fingerprints.
Three views: (a) activity colored, (b) train/test split, (c) major chemical
categories (flavonoid, phenolic acid, stilbene, coumarin, terpenoid,
vitamin/tocopherol, other).
"""

import os, sys, pickle, json, warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from rdkit import Chem

DATA_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(DATA_DIR, "results")
FIG_DIR    = os.path.join(RESULT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# Chemical category SMARTS — first match wins (priority order)
CATEGORY_PATTERNS = [
    ('Flavonoid',         ['c1cc2oc(-c3ccccc3)cc(=O)c2cc1',
                           'O=C1CC(c2ccccc2)Oc2ccccc12',
                           'OC1c2ccccc2OC(c2ccccc2)C1',
                           'O=c1c([OH])c(-c2ccccc2)oc2ccccc12']),
    ('Phenolic acid',     ['OC(=O)c1ccccc1[OH]',
                           'OC(=O)c1cc([OH])c([OH])cc1',
                           'OC(=O)/C=C/c1ccc([OH])cc1',
                           'OC(=O)/C=C/c1ccc([OH])c([OH])c1',
                           'OC(=O)Cc1ccc([OH])cc1']),
    ('Stilbene',          ['c1ccc(/C=C/c2ccccc2)cc1']),
    ('Coumarin',          ['O=c1ccc2ccccc2o1',
                           'O=c1cc(*)oc2ccccc12']),
    ('Lignan',            ['OCc1ccc(C(C)C(C)Cc2ccc([OH])cc2)cc1[OH]']),
    ('Terpenoid',         ['CC(=CCCC(C)=CCC)C',
                           'C1(C)CCCCC1',
                           'C12CCC(C)(CC1)C(C)CC2']),
    ('Vitamin/Tocopherol',['Cc1c(C)c2OC(C)(CCCC(C)CCCC(C)CCCC(C)C)CCc2c(C)c1[OH]',
                           'OC1=C(C)C(=O)OC1[CH](O)CO']),
    ('Quinone',           ['O=C1C=CC(=O)C=C1',
                           'O=C1c2ccccc2C(=O)c2ccccc21']),
    ('Aromatic amine',    ['Nc1ccccc1',
                           'NC(=O)c1ccccc1']),
]


def classify_compound(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 'Other'
    for name, smarts_list in CATEGORY_PATTERNS:
        for sm in smarts_list:
            patt = Chem.MolFromSmarts(sm)
            if patt is not None and mol.HasSubstructMatch(patt):
                return name
    return 'Other'


def run_umap(fps, seed=42):
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                       metric='jaccard', random_state=seed, n_jobs=1)
    return reducer.fit_transform(fps.astype(np.float32))


def main():
    with open(os.path.join(RESULT_DIR, "processed_data.pkl"), 'rb') as f:
        data = pickle.load(f)

    fps   = data['fps']
    df    = data['df']
    pIC50 = df['pIC50'].values

    train_idx = data['random_split']['train_idx']
    split = np.full(len(fps), 'Test', dtype=object)
    split[train_idx] = 'Train'

    print(f"Classifying {len(df)} compounds into chemical categories...")
    smiles_col = 'SMILES' if 'SMILES' in df.columns else df.columns[0]
    categories = np.array([classify_compound(s) for s in df[smiles_col].values],
                          dtype=object)

    cat_counts = pd.Series(categories).value_counts()
    print("\nCategory distribution:")
    for k, v in cat_counts.items():
        print(f"  {k:20s}: {v:4d}  ({100*v/len(categories):.1f}%)")

    print(f"\nRunning UMAP (Jaccard, n_neighbors=15)...")
    emb = run_umap(fps)
    print(f"  Done. Range: X=[{emb[:,0].min():.2f}, {emb[:,0].max():.2f}], "
          f"Y=[{emb[:,1].min():.2f}, {emb[:,1].max():.2f}]")

    # ── Figure: 3 panels ─────────────────────────────────────────────────────
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    plt.subplots_adjust(wspace=0.35, top=0.88, bottom=0.12)

    # (a) activity-colored
    ax = axes[0]
    sc = ax.scatter(emb[:,0], emb[:,1], c=pIC50, cmap='plasma',
                    s=12, alpha=0.75, linewidths=0,
                    norm=Normalize(vmin=pIC50.min(), vmax=pIC50.max()))
    cbar = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('pIC$_{50}$', fontsize=14)
    cbar.ax.tick_params(labelsize=11)
    ax.set_title('(a) Activity Distribution',
                 fontsize=15, fontweight='bold', pad=10)
    ax.set_xlabel('UMAP-1', fontsize=13)
    ax.set_ylabel('UMAP-2', fontsize=13)
    ax.tick_params(labelsize=10)

    # (b) train/test
    ax = axes[1]
    train_mask = (split == 'Train')
    ax.scatter(emb[train_mask,0], emb[train_mask,1], c='#2196F3',
               s=8, alpha=0.45, linewidths=0,
               label=f'Train (n={train_mask.sum()})')
    ax.scatter(emb[~train_mask,0], emb[~train_mask,1], c='#FF5722',
               s=18, alpha=0.85, linewidths=0,
               label=f'Test (n={(~train_mask).sum()})')
    ax.legend(fontsize=12, markerscale=2, framealpha=0.9, loc='best')
    ax.set_title('(b) Train / Test Distribution',
                 fontsize=15, fontweight='bold', pad=10)
    ax.set_xlabel('UMAP-1', fontsize=13)
    ax.set_ylabel('UMAP-2', fontsize=13)
    ax.tick_params(labelsize=10)

    # (c) chemical categories
    ax = axes[2]
    cat_order = cat_counts.index.tolist()
    palette = {
        'Flavonoid':           '#E91E63',
        'Phenolic acid':       '#1565C0',
        'Stilbene':            '#FF8F00',
        'Coumarin':            '#7B1FA2',
        'Lignan':              '#00838F',
        'Terpenoid':           '#558B2F',
        'Vitamin/Tocopherol':  '#C62828',
        'Quinone':             '#F4511E',
        'Aromatic amine':      '#4E342E',
        'Other':               '#BDBDBD',
    }
    for cat in ['Other'] + [c for c in cat_order if c != 'Other']:
        m = (categories == cat)
        if not m.any():
            continue
        col = palette.get(cat, '#9E9E9E')
        s = 6 if cat == 'Other' else 22
        a = 0.30 if cat == 'Other' else 0.88
        z = 1 if cat == 'Other' else 3
        ax.scatter(emb[m,0], emb[m,1], c=col, s=s, alpha=a, linewidths=0,
                   label=f'{cat} (n={m.sum()})', zorder=z)
    ax.legend(fontsize=10, markerscale=1.5, framealpha=0.92,
              loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.set_title('(c) Chemical Categories',
                 fontsize=15, fontweight='bold', pad=10)
    ax.set_xlabel('UMAP-1', fontsize=13)
    ax.set_ylabel('UMAP-2', fontsize=13)
    ax.tick_params(labelsize=10)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Chemical Space of the 1,911 Antioxidants (UMAP of Morgan Fingerprints)',
                 fontsize=16, fontweight='bold', y=0.97)
    out_path = os.path.join(FIG_DIR, "chemical_space_umap.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out_path}")

    emb_data = {
        'embedding':       emb.tolist(),
        'pIC50':           pIC50.tolist(),
        'split_labels':    split.tolist(),
        'categories':      categories.tolist(),
        'category_counts': cat_counts.to_dict(),
    }
    with open(os.path.join(RESULT_DIR, "umap_embedding.json"), 'w') as f:
        json.dump(emb_data, f)

    print("\nMean pIC50 per category:")
    by_cat = {}
    for cat in cat_order:
        m = (categories == cat)
        if m.any():
            print(f"  {cat:20s}: {pIC50[m].mean():.3f} ± {pIC50[m].std():.3f}  "
                  f"(n={m.sum()})")
            by_cat[cat] = {'mean': float(pIC50[m].mean()),
                           'std':  float(pIC50[m].std()),
                           'n':    int(m.sum())}
    with open(os.path.join(RESULT_DIR, "category_stats.json"), 'w') as f:
        json.dump(by_cat, f, indent=2)


if __name__ == '__main__':
    main()
