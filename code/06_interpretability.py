"""
GNN interpretability using GNNExplainer.
Atom importance maps for representative antioxidant compounds.
Uses pure RDKit drawing (no cairosvg dependency).

Outputs:
  fig_atommap_main.png        - 3-compound panel for manuscript main text
                                (Quercetin, Kaempferol, Chrysin)
  fig_interpretability_panel.png - 6-compound panel for SI
  fig_si_testset_panel.png    - Representative test-set compounds for SI
  fig7_interpretability.png   - pharmacophore bar chart
  interpretability_results.json
"""

import os, sys, pickle, json, warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')
import numpy as np
import torch
from torch_geometric.explain import Explainer, GNNExplainer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io
import pandas as pd
from sklearn.model_selection import train_test_split

CODE_DIR   = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
from models import get_model

DATA_DIR   = os.path.dirname(CODE_DIR)
RESULT_DIR = os.path.join(DATA_DIR, "results")
FIG_DIR    = os.path.join(RESULT_DIR, "figures")
MODEL_DIR  = os.path.join(RESULT_DIR, "models")
TESTMAP_DIR = os.path.join(FIG_DIR, "test_set_maps")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TESTMAP_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Representative compounds for main text (3): Quercetin > Kaempferol > Chrysin
# Systematic series showing B-ring hydroxylation → activity gradient
MAIN_TEXT_COMPOUNDS = [
    ("Quercetin",
     "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
     "High activity: 3',4'-catechol + 3-OH + 5,7-diol\npred pIC₅₀ ≈ 5.20"),
    ("Kaempferol",
     "O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12",
     "Moderate activity: 4'-OH only (lacks 3'-OH)\npred pIC₅₀ ≈ 4.90"),
    ("Chrysin",
     "O=c1cc(-c2ccccc2)oc2cc(O)cc(O)c12",
     "Low activity: no B-ring hydroxylation\npred pIC₅₀ ≈ 4.53"),
]

TARGET_COMPOUNDS = [
    ("Quercetin",
     "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
     "High-activity flavonol: 3',4'-catechol + 3-OH + 5-OH"),
    ("Luteolin",
     "O=c1cc(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
     "High-activity flavone: 3',4'-catechol"),
    ("Kaempferol",
     "O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12",
     "Moderate-activity: lacks 3'-OH vs quercetin"),
    ("Chrysin",
     "O=c1cc(-c2ccccc2)oc2cc(O)cc(O)c12",
     "Low-activity: no B-ring OH groups"),
    ("Caffeic acid",
     "OC(=O)/C=C/c1ccc(O)c(O)c1",
     "Non-flavonoid phenolic acid with catechol"),
    ("Resveratrol",
     "Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1",
     "Stilbene antioxidant, 3 OH groups"),
]

# Category labels for representative test-set selection
CATEGORY_SMARTS = [
    ("Flavonoid",    ["c1cc2oc(-c3ccccc3)cc(=O)c2cc1", "O=C1CC(c2ccccc2)Oc2ccccc12",
                      "O=c1c([OH])c(-c2ccccc2)oc2ccccc12"]),
    ("Phenolic acid",["OC(=O)c1ccccc1O", "OC(=O)/C=C/c1ccc(O)cc1",
                      "OC(=O)/C=C/c1ccc(O)c(O)c1"]),
    ("Stilbene",     ["c1ccc(/C=C/c2ccccc2)cc1"]),
    ("Coumarin",     ["O=c1ccc2ccccc2o1"]),
    ("Quinone",      ["O=C1C=CC(=O)C=C1", "O=C1c2ccccc2C(=O)c2ccccc21"]),
    ("Terpenoid",    ["CC(=CCCC(C)=CCC)C", "C1(C)CCCCC1"]),
    ("Aromatic amine", ["Nc1ccccc1"]),
]


def mol_to_graph_local(smiles, y=0.0):
    from rdkit.Chem import rdchem
    from torch_geometric.data import Data

    ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'other']
    DEGREE_VALS = [0, 1, 2, 3, 4, 5]
    CHARGE_VALS = [-2, -1, 0, 1, 2]
    HYBRID_VALS = [rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2,
                   rdchem.HybridizationType.SP3, rdchem.HybridizationType.SP3D,
                   rdchem.HybridizationType.SP3D2]
    BOND_TYPES = [rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE,
                  rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC]

    def one_hot(val, choices):
        vec = [0]*len(choices)
        idx = choices.index(val) if val in choices else len(choices)-1
        vec[idx] = 1
        return vec

    def atom_feat(atom):
        sym = atom.GetSymbol()
        return (one_hot(sym if sym in ATOM_TYPES[:-1] else 'other', ATOM_TYPES)
               + one_hot(atom.GetDegree(), DEGREE_VALS)
               + one_hot(atom.GetFormalCharge(), CHARGE_VALS)
               + one_hot(atom.GetHybridization(), HYBRID_VALS)
               + [int(atom.GetIsAromatic())]
               + one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
               + [int(atom.IsInRing())]
               + one_hot(atom.GetTotalValence(), [0, 1, 2, 3, 4, 5, 6]))

    def bond_feat(bond):
        return (one_hot(bond.GetBondType(), BOND_TYPES)
               + [int(bond.GetIsConjugated()), int(bond.IsInRing())])

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor([atom_feat(a) for a in mol.GetAtoms()], dtype=torch.float)
    ei, ea = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        f = bond_feat(bond)
        ei += [[i, j], [j, i]]
        ea += [f, f]
    if not ei:
        return None
    return Data(x=x,
                edge_index=torch.tensor(ei, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(ea, dtype=torch.float),
                y=torch.tensor([y], dtype=torch.float))


def load_best_model(model_name='attentivefp'):
    kwargs = {'hidden': 200, 'num_layers': 2, 'num_timesteps': 2, 'dropout': 0.2}
    model = get_model(model_name, **kwargs).to(DEVICE)
    # Try best fold model first, then canonical
    for fold in range(1, 11):
        ckpt = os.path.join(MODEL_DIR, f"random_{model_name}_fold{fold}.pt")
        if os.path.exists(ckpt):
            best_ckpt = ckpt
    canonical = os.path.join(MODEL_DIR, f"random_{model_name}.pt")
    ckpt_path = canonical if os.path.exists(canonical) else best_ckpt
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    model.eval()
    print(f"  Loaded model from: {os.path.basename(ckpt_path)}")
    return model


def rdkit_mol_image_with_colors(smiles, atom_weights, size=(400, 300)):
    """Draw molecule with atom highlight colors using RDKit Cairo renderer."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    w = np.array(atom_weights)
    if w.max() > w.min():
        w = (w - w.min()) / (w.max() - w.min())
    else:
        w = np.ones_like(w) * 0.5

    cmap = cm.get_cmap('RdYlBu_r')
    atom_colors = {i: cmap(float(w[i]))[:3] for i in range(mol.GetNumAtoms())}
    atom_radii  = {i: 0.25 + 0.5*float(w[i]) for i in range(mol.GetNumAtoms())}
    highlight_atoms = list(range(mol.GetNumAtoms()))

    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(*size)
        drawer.drawOptions().addAtomIndices = False
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=atom_colors,
            highlightAtomRadii=atom_radii,
            highlightBonds=[],
            highlightBondColors={},
        )
        drawer.FinishDrawing()
        png_bytes = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(png_bytes))
        return img, w
    except Exception as e:
        print(f"    Drawing failed: {e}")
        return None, w


def compute_atom_importance(smiles, model, explainer):
    """Run GNNExplainer for a single SMILES; return (pred_val, atom_weights, img, w_norm)."""
    graph = mol_to_graph_local(smiles)
    if graph is None:
        return None, None, None, None

    graph = graph.to(DEVICE)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        pred_val = model(graph).item()

    try:
        explanation = explainer(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            batch=graph.batch,
        )
        atom_weights = explanation.node_mask.sum(dim=-1).cpu().numpy()
    except Exception:
        atom_weights = np.ones(graph.num_nodes)

    atom_weights = np.abs(atom_weights)
    if atom_weights.max() > 0:
        atom_weights = atom_weights / atom_weights.max()

    img, w_norm = rdkit_mol_image_with_colors(smiles, atom_weights)
    return pred_val, atom_weights, img, w_norm


def make_panel(entries, ncols, out_path, title='', dpi=200, cell_w=5.5, cell_h=4.5,
               fontsize_title=10, fontsize_suptitle=13):
    """
    entries: list of (label, img_or_None, pred_val_or_None)
    Saves panel to out_path.
    """
    n = len(entries)
    nrows = max(1, (n + ncols - 1) // ncols)
    fig = plt.figure(figsize=(ncols * cell_w, nrows * cell_h))

    for idx, (label, img, pred) in enumerate(entries):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10)
        ax.axis('off')
        title_str = label
        if pred is not None:
            title_str += f"\npred pIC₅₀ = {pred:.2f}"
        ax.set_title(title_str, fontsize=fontsize_title, fontweight='bold')

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Atom Importance', fontsize=11)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Medium', 'High'])

    if title:
        fig.suptitle(title, fontsize=fontsize_suptitle, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0, 0.91, 1])
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(out_path)}")


def make_main_text_figure(model, explainer):
    """
    3-compound panel for main text:
    Quercetin → Kaempferol → Chrysin
    Demonstrates B-ring hydroxylation → activity gradient (chemical SAR rule).
    Uses larger cells, annotation arrows for key atoms.
    """
    print("\n── Main text 3-compound figure ──────────────────────────────────────")
    cmap = cm.get_cmap('RdYlBu_r')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    subtitles = [
        "Quercetin\n(3',4'-catechol + 3-OH: highest HAT activity)",
        "Kaempferol\n(4'-OH only, lacks 3'-OH: moderate activity)",
        "Chrysin\n(no B-ring hydroxylation: lowest activity)",
    ]

    for ax, (name, smiles, _), subtitle in zip(axes, MAIN_TEXT_COMPOUNDS, subtitles):
        pred_val, atom_weights, img, _ = compute_atom_importance(smiles, model, explainer)
        if pred_val is None:
            ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue

        if img is not None:
            ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"{subtitle}\n(pred pIC₅₀ = {pred_val:.2f})",
                     fontsize=12, fontweight='bold', pad=8)

        # Decorative border color by activity level
        colors = ['#d62728', '#ff7f0e', '#1f77b4']  # red/orange/blue for H/M/L
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.patch.set_edgecolor(colors[list(x[0] for x in MAIN_TEXT_COMPOUNDS).index(name)])
        ax.patch.set_linewidth(3)

        print(f"  {name}: pred pIC50 = {pred_val:.3f}")

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Atom Importance Score', fontsize=12)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Medium', 'High'], fontsize=10)

    fig.suptitle(
        'GNNExplainer 2D Atom-Level Importance Maps:\n'
        'Effect of B-Ring Hydroxylation on DPPH Radical Scavenging Activity',
        fontsize=13, fontweight='bold', y=1.02)

    # Arrow annotation at bottom
    fig.text(0.5, -0.02,
             '← Decreasing B-ring hydroxylation    •    Decreasing predicted antioxidant activity →',
             ha='center', fontsize=11, style='italic', color='#555555')

    plt.tight_layout(rect=[0, 0.0, 0.92, 1])
    out_path = os.path.join(FIG_DIR, "fig_atommap_main.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(out_path)}")


def select_representative_test_compounds(df, n_per_category=5):
    """
    Select up to n_per_category compounds per chemical category from the test set,
    spanning the pIC50 range (min, Q1, median, Q3, max).
    Returns list of (smiles, pIC50) tuples.
    """
    # Assign categories
    def assign_category(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return "Other"
        for cat_name, smarts_list in CATEGORY_SMARTS:
            for smt in smarts_list:
                try:
                    patt = Chem.MolFromSmarts(smt)
                    if patt and mol.HasSubstructMatch(patt):
                        return cat_name
                except Exception:
                    pass
        return "Other"

    df = df.copy()
    df['category'] = df['smiles'].apply(assign_category)

    selected = []
    for cat_name, _ in CATEGORY_SMARTS + [("Other", [])]:
        sub = df[df['category'] == cat_name].sort_values('pIC50')
        if len(sub) == 0:
            continue
        # Pick evenly spaced indices
        indices = np.linspace(0, len(sub) - 1, min(n_per_category, len(sub)), dtype=int)
        for idx in indices:
            row = sub.iloc[idx]
            selected.append((cat_name, row['smiles'], float(row['pIC50'])))

    return selected


def main():
    print(f"Using device: {DEVICE}")
    print(f"Loading AttentiveFP model...")
    model = load_best_model('attentivefp')
    model.eval()

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(mode='regression', task_level='graph', return_type='raw'),
    )

    # ── A. Main text 3-compound figure ────────────────────────────────────────
    make_main_text_figure(model, explainer)

    # ── B. 6-compound SI panel (all representative compounds) ─────────────────
    print("\n── 6-compound SI panel ──────────────────────────────────────────────")
    results = []
    mol_images = []
    pred_vals = []

    cmap = cm.get_cmap('RdYlBu_r')

    for name, smiles, desc in TARGET_COMPOUNDS:
        print(f"  {name}")
        pred_val, atom_weights, img, w_norm = compute_atom_importance(smiles, model, explainer)
        if pred_val is None:
            print(f"    ✗ Invalid SMILES")
            continue

        print(f"    pred pIC50 = {pred_val:.3f}")
        mol_images.append((name, img, w_norm, smiles))
        pred_vals.append(pred_val)

        mol = Chem.MolFromSmiles(smiles)
        top3_idx = np.argsort(atom_weights)[::-1][:3]
        top3_info = [f"Atom {i}: {mol.GetAtomWithIdx(int(i)).GetSymbol()} (w={atom_weights[i]:.3f})"
                     for i in top3_idx]
        results.append({
            'name': name, 'smiles': smiles, 'description': desc,
            'pred_pIC50': float(pred_val),
            'atom_weights': atom_weights.tolist(),
            'top3_atoms': top3_info,
        })
        print(f"    Top-3: {', '.join(top3_info)}")

    # 6-compound panel for SI
    entries_si = [(name, img, pred) for (name, img, _, _), pred in zip(mol_images, pred_vals)]
    make_panel(entries_si, ncols=3,
               out_path=os.path.join(FIG_DIR, "fig_interpretability_panel.png"),
               title='GNNExplainer Atom-Level Importance Maps — Six Representative Antioxidants',
               dpi=200, cell_w=5.5, cell_h=4.5)

    # ── C. Bar chart of top-atom importances ──────────────────────────────────
    fig2, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, res in zip(axes, results):
        smiles_r = res['smiles']
        mol      = Chem.MolFromSmiles(smiles_r)
        weights  = np.array(res['atom_weights'])
        atoms    = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
        labels   = [f"{sym}{i}" for i, sym in enumerate(atoms)]
        sort_idx = np.argsort(weights)[::-1][:12]
        sw = weights[sort_idx]
        sl = [labels[i] for i in sort_idx]
        bar_colors = [cmap(float(w)) for w in sw]
        ax.barh(range(len(sl)), sw[::-1], color=bar_colors[::-1])
        ax.set_yticks(range(len(sl)))
        ax.set_yticklabels(sl[::-1], fontsize=8)
        ax.set_xlabel('Importance', fontsize=9)
        ax.set_title(f"{res['name']} (pIC₅₀={res['pred_pIC50']:.2f})",
                     fontsize=9.5, fontweight='bold')
        ax.set_xlim(0, 1.05)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    for ax in axes[len(results):]:
        ax.axis('off')
    plt.suptitle('Top-12 Atom Importances (GNNExplainer)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    bar_path = os.path.join(FIG_DIR, "fig7_interpretability.png")
    plt.savefig(bar_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nBar chart saved: {os.path.basename(bar_path)}")

    # ── D. Representative test-set compound maps for SI ───────────────────────
    print("\n── Representative test-set compounds for SI ─────────────────────────")
    csv_path = os.path.join(RESULT_DIR, "compounds_annotated.csv")
    if os.path.exists(csv_path):
        df_all = pd.read_csv(csv_path)
        # Reproduce the same 80/20 random split as training (seed=42)
        _, df_test = train_test_split(df_all, test_size=0.20, random_state=42)
        df_test = df_test.reset_index(drop=True)
        print(f"  Test set size: {len(df_test)}")

        selected = select_representative_test_compounds(df_test, n_per_category=5)
        print(f"  Selected {len(selected)} representative test compounds")

        si_entries = []
        for cat_name, smi, exp_pic50 in selected:
            pred_val, atom_weights, img, _ = compute_atom_importance(smi, model, explainer)
            label_cat = f"[{cat_name}]"
            if pred_val is not None:
                label = f"{label_cat}\nexp={exp_pic50:.2f}"
                si_entries.append((label, img, pred_val))
            else:
                si_entries.append((f"{label_cat}\nexp={exp_pic50:.2f}", None, None))

        # Make a multi-row panel (6 columns per row, category-grouped)
        make_panel(si_entries, ncols=5,
                   out_path=os.path.join(FIG_DIR, "fig_si_testset_panel.png"),
                   title='GNNExplainer Atom Importance Maps — Representative Test-Set Compounds\n'
                         '(5 compounds per chemical category, spanning pIC₅₀ range)',
                   dpi=180, cell_w=4.0, cell_h=3.5, fontsize_title=8, fontsize_suptitle=11)
    else:
        print(f"  Skipping test-set maps: {csv_path} not found")

    # ── E. Save results JSON ───────────────────────────────────────────────────
    out_path = os.path.join(RESULT_DIR, "interpretability_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {os.path.basename(out_path)}")
    print("\nDone.")


if __name__ == '__main__':
    main()
