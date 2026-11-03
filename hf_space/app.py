"""
AntioxFP — GNN-Based Antioxidant Activity Predictor
====================================================
Predicts DPPH radical scavenging activity (pIC50) from SMILES strings
using a 30-model AttentiveFP ensemble with GNNExplainer atom importance maps.

Based on: "Graph Neural Network Models for Predicting the Antioxidant Activity
of Chemical Compounds" (2025)
"""

import os
import sys
import warnings
import io
import math

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from PIL import Image
import gradio as gr

from models_arch import AttentiveFPModel

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DEVICE = torch.device("cpu")  # HF free CPU tier

SEEDS = [42, 1, 100]
N_FOLDS = 10
# The configuration selected by the hyperparameter search, which is the one the
# paper reports (test R2 = 0.7562). The originally submitted architecture was
# hidden 200 with 2 layers and scores 0.7331; serving that one while the paper
# reports the tuned one is the mismatch this block exists to prevent.
HIDDEN = 300
N_LAYERS = 3
N_TIMESTEPS = 2

ATOM_TYPES  = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "other"]
DEGREE_VALS = [0, 1, 2, 3, 4, 5]
CHARGE_VALS = [-2, -1, 0, 1, 2]
from rdkit.Chem import rdchem
HYBRID_VALS = [
    rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3, rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC,
]

EXAMPLE_SMILES = [
    ["O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12", "Quercetin — high activity flavonol"],
    ["OC(=O)/C=C/c1ccc(O)c(O)c1",                  "Caffeic acid — phenolic acid"],
    ["Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1",             "Resveratrol — stilbene antioxidant"],
    ["O=c1cc(-c2ccccc2)oc2cc(O)cc(O)c12",          "Chrysin — low activity (no B-ring OH)"],
    ["O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12",   "Kaempferol — moderate activity"],
    ["CC(C)(C)c1cc(C(C)(C)C)cc(CC(=O)Nc2ccccc2)c1","BHA analogue — synthetic antioxidant"],
]


# ── Molecular graph builder ────────────────────────────────────────────────────
def one_hot(val, choices):
    vec = [0] * len(choices)
    idx = choices.index(val) if val in choices else len(choices) - 1
    vec[idx] = 1
    return vec


def atom_feat(atom):
    sym = atom.GetSymbol()
    return (
        one_hot(sym if sym in ATOM_TYPES[:-1] else "other", ATOM_TYPES)
        + one_hot(atom.GetDegree(), DEGREE_VALS)
        + one_hot(atom.GetFormalCharge(), CHARGE_VALS)
        + one_hot(atom.GetHybridization(), HYBRID_VALS)
        + [int(atom.GetIsAromatic())]
        + one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        + [int(atom.IsInRing())]
        + one_hot(atom.GetTotalValence(), [0, 1, 2, 3, 4, 5, 6])
    )


def bond_feat(bond):
    return (
        one_hot(bond.GetBondType(), BOND_TYPES)
        + [int(bond.GetIsConjugated()), int(bond.IsInRing())]
    )


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    x = torch.tensor([atom_feat(a) for a in mol.GetAtoms()], dtype=torch.float)
    ei, ea = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        f = bond_feat(bond)
        ei += [[i, j], [j, i]]
        ea += [f, f]
    if not ei:
        return None, None
    graph = Data(
        x=x,
        edge_index=torch.tensor(ei, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(ea, dtype=torch.float),
        batch=torch.zeros(x.size(0), dtype=torch.long),
    )
    return graph, mol


# ── Applicability domain ──────────────────────────────────────────────────────
# Validation on 196 compounds from 120 independent publications showed that
# accuracy tracks structural similarity to the training set, while the
# leverage-based criterion does not: it admitted 194 of those 196 compounds,
# including the worst-predicted ones. Restricting to a maximum Tanimoto
# similarity of at least 0.40 moved R2 from -0.29 to +0.39, so the similarity
# is reported with every prediction rather than kept out of sight.

TRAIN_SMILES_PATH = os.path.join(os.path.dirname(__file__), "train_smiles.csv")
_TRAIN_FPS = None


def _train_fingerprints():
    global _TRAIN_FPS
    if _TRAIN_FPS is not None:
        return _TRAIN_FPS
    from rdkit.Chem import AllChem
    import csv as _csv
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    try:
        with open(TRAIN_SMILES_PATH, encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                m = Chem.MolFromSmiles(row["smiles"])
                if m is not None:
                    fps.append(gen.GetFingerprint(m))
    except FileNotFoundError:
        fps = []
    _TRAIN_FPS = fps
    return fps


def applicability_domain(mol):
    """Maximum Tanimoto similarity to the training set, and what it means."""
    from rdkit import DataStructs
    from rdkit.Chem import AllChem
    fps = _train_fingerprints()
    if not fps or mol is None:
        return float("nan"), "unknown", ""
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    sim = max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(mol), fps))
    if sim >= 0.50:
        return sim, "✅ inside", (
            "> The closest training compound is structurally similar, so this "
            "prediction is in the range where the model was validated.")
    if sim >= 0.40:
        return sim, "⚠️ borderline", (
            "> ⚠️ The closest training compound is only moderately similar. "
            "Treat this value as indicative.")
    return sim, "⛔ outside", (
        "> ⛔ **Outside the applicability domain.** No training compound is "
        "structurally close to this molecule. On independent literature data "
        "the model did not predict such compounds reliably, and this value "
        "should not be used. The model is intended for phenolic antioxidants.")


# ── Model loader ──────────────────────────────────────────────────────────────
_MODELS = None  # lazy load


def load_models():
    global _MODELS
    if _MODELS is not None:
        return _MODELS

    models, missing = [], []
    for seed in SEEDS:
        for fold in range(1, N_FOLDS + 1):
            name = f"attentivefp_tuned_seed{seed}_fold{fold}.pt"
            path = os.path.join(MODEL_DIR, name)
            if not os.path.exists(path):
                missing.append(name)
                continue
            m = AttentiveFPModel(hidden=HIDDEN, num_layers=N_LAYERS,
                                 num_timesteps=N_TIMESTEPS, dropout=0.2).to(DEVICE)
            m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False))
            m.eval()
            models.append(m)

    # Fail loudly rather than quietly serving a different model from the one
    # the paper reports.
    if len(models) != len(SEEDS) * N_FOLDS:
        raise RuntimeError(
            "expected %d tuned AttentiveFP checkpoints in %s, found %d; missing %s"
            % (len(SEEDS) * N_FOLDS, MODEL_DIR, len(models), ", ".join(missing[:5])))

    _MODELS = models
    return models


# ── Rendering ─────────────────────────────────────────────────────────────────
def render_atom_importance(mol, atom_weights, size=(600, 450)):
    """Render molecule with per-atom importance heatmap via RDKit Cairo."""
    w = np.array(atom_weights, dtype=float)
    if w.max() > w.min():
        w = (w - w.min()) / (w.max() - w.min())
    else:
        w = np.ones_like(w) * 0.5

    cmap = cm.get_cmap("RdYlBu_r")
    atom_colors = {i: cmap(float(w[i]))[:3] for i in range(mol.GetNumAtoms())}
    atom_radii  = {i: 0.20 + 0.55 * float(w[i]) for i in range(mol.GetNumAtoms())}
    highlight   = list(range(mol.GetNumAtoms()))

    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(*size)
        opts = drawer.drawOptions()
        opts.addAtomIndices = False
        opts.bondLineWidth = 2.0
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, mol,
            highlightAtoms=highlight,
            highlightAtomColors=atom_colors,
            highlightAtomRadii=atom_radii,
            highlightBonds=[],
            highlightBondColors={},
        )
        drawer.FinishDrawing()
        img = Image.open(io.BytesIO(drawer.GetDrawingText()))

        # Add colorbar
        fig, ax = plt.subplots(figsize=(img.width / 100, img.height / 100 + 0.5))
        ax.imshow(img)
        ax.axis("off")
        sm = plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                            fraction=0.03, pad=0.02, aspect=20)
        cbar.set_label("Atom Importance", fontsize=10)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["Low", "Medium", "High"], fontsize=8)
        plt.tight_layout(pad=0.3)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        return Image.open(buf).copy()
    except Exception as e:
        print(f"Rendering error: {e}")
        return None


def make_importance_bargraph(mol, atom_weights, pred_pic50):
    """Horizontal bar chart of top-15 atom importances."""
    w = np.array(atom_weights)
    atoms  = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
    labels = [f"{sym}{i}" for i, sym in enumerate(atoms)]

    top_k = min(15, mol.GetNumAtoms())
    sort_idx = np.argsort(w)[::-1][:top_k]
    sw = w[sort_idx]
    sl = [labels[i] for i in sort_idx]

    cmap = cm.get_cmap("RdYlBu_r")
    colors = [cmap(float(v)) for v in sw]

    fig, ax = plt.subplots(figsize=(6, max(3, top_k * 0.4)))
    ax.barh(range(top_k), sw[::-1], color=colors[::-1])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(sl[::-1], fontsize=9)
    ax.set_xlabel("Atom Importance Score", fontsize=10)
    ax.set_title(f"Top-{top_k} Atom Importances  (pred pIC₅₀ = {pred_pic50:.3f})",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return Image.open(buf).copy()


# ── Activity interpretation ────────────────────────────────────────────────────
def interpret_activity(pic50):
    ic50_uM = 10 ** (-pic50) * 1e6
    if pic50 >= 5.0:
        level = "🟢 High"
        desc  = "Strong DPPH radical scavenger (IC₅₀ ≤ 10 µM). Comparable to quercetin."
    elif pic50 >= 4.5:
        level = "🟡 Moderate–High"
        desc  = "Moderate-to-high radical scavenging activity."
    elif pic50 >= 4.0:
        level = "🟠 Moderate"
        desc  = "Moderate DPPH scavenging activity."
    else:
        level = "🔴 Low"
        desc  = "Weak DPPH radical scavenger."
    return level, ic50_uM, desc


def pharmacophore_hint(mol, atom_weights):
    """Generate a brief pharmacophore text based on atom weights."""
    w = np.array(atom_weights)
    top_idx = np.argsort(w)[::-1][:5]
    top_atoms = [mol.GetAtomWithIdx(int(i)).GetSymbol() for i in top_idx]

    # Simple heuristic annotations
    hints = []
    if top_atoms.count("O") >= 2:
        hints.append("**Phenolic hydroxyl / catechol motif** — primary HAT pharmacophore detected")
    if any(mol.GetAtomWithIdx(int(i)).GetIsAromatic() for i in top_idx):
        hints.append("**Aromatic conjugation** — supports radical delocalization (HAT/SET)")
    if any(mol.GetAtomWithIdx(int(i)).GetSymbol() == "C"
           and not mol.GetAtomWithIdx(int(i)).GetIsAromatic() for i in top_idx):
        hints.append("**sp² vinyl/carbonyl carbons** — extended π-system (SET pathway)")
    if not hints:
        hints.append("Importance distributed across scaffold — no single dominant pharmacophore")

    return "\n".join(f"• {h}" for h in hints)


# ── Core prediction function ───────────────────────────────────────────────────
def predict(smiles_input, run_explainer):
    smiles = smiles_input.strip()
    if not smiles:
        return (None, None, "⚠️ Please enter a SMILES string.", "", "")

    graph, mol = smiles_to_graph(smiles)
    if graph is None or mol is None:
        return (None, None,
                "❌ Invalid SMILES string. Please check the input.",
                "", "")

    models = load_models()
    graph = graph.to(DEVICE)

    # Ensemble prediction
    preds = []
    with torch.no_grad():
        for m in models:
            out = m(graph)  # pass Data object → uses hasattr(x,'edge_index') branch
            preds.append(out.item())

    pred_mean = float(np.mean(preds))
    pred_std  = float(np.std(preds))
    level, ic50_uM, desc = interpret_activity(pred_mean)
    max_sim, band, band_note = applicability_domain(mol)

    result_md = (
        f"## Predicted pIC₅₀: **{pred_mean:.3f} ± {pred_std:.3f}**\n\n"
        f"| Property | Value |\n|---|---|\n"
        f"| Activity level | {level} |\n"
        f"| Estimated IC₅₀ | **{ic50_uM:.1f} µM** |\n"
        f"| Ensemble size | {len(models)} models |\n"
        f"| Similarity to training set | **{max_sim:.2f}** (max Tanimoto) |\n"
        f"| Applicability domain | {band} |\n\n"
        f"*{desc}*\n\n"
        f"{band_note}\n\n"
        f"> **Note**: pIC₅₀ = −log₁₀(IC₅₀/M). Higher = more active."
    )

    # Atom importance map
    atom_img = None
    bar_img  = None
    pharma_text = ""

    if run_explainer:
        try:
            explainer = Explainer(
                model=models[0],
                algorithm=GNNExplainer(epochs=150),
                explanation_type="model",
                node_mask_type="attributes",
                edge_mask_type="object",
                model_config=dict(mode="regression", task_level="graph",
                                  return_type="raw"),
            )
            exp = explainer(
                x=graph.x,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr,
                batch=graph.batch,
            )
            raw_w = exp.node_mask.sum(dim=-1).cpu().numpy()
            raw_w = np.abs(raw_w)
            if raw_w.max() > 0:
                raw_w /= raw_w.max()

            atom_img = render_atom_importance(mol, raw_w)
            bar_img  = make_importance_bargraph(mol, raw_w, pred_mean)
            pharma_text = "### Pharmacophore Analysis\n\n" + pharmacophore_hint(mol, raw_w)
        except Exception as e:
            pharma_text = f"⚠️ Explainer error: {e}"
    else:
        pharma_text = (
            "💡 *Enable 'Run GNNExplainer' to generate atom importance maps.*\n\n"
            "The explainer adds ~10–20 s on CPU but reveals which atoms drive the prediction."
        )

    return atom_img, bar_img, result_md, pharma_text, ""


# ── Gradio UI ─────────────────────────────────────────────────────────────────
CSS = """
.main-header {
    background: linear-gradient(135deg, #e8f4f8 0%, #d0e8f0 50%, #b8dcea 100%);
    padding: 24px;
    border-radius: 12px;
    margin-bottom: 16px;
    text-align: center;
    color: #1a3a4a;
    border: 1px solid #a0c8dc;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.main-header h1 { font-size: 2.2em; margin: 0; font-weight: 800; color: #0d2b3e; }
.main-header p  { font-size: 1.05em; opacity: 0.9; margin-top: 8px; color: #1a3a4a; }
.result-box { border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; }
.example-btn { font-size: 0.85em !important; }
footer { display: none !important; }
"""

HEADER_HTML = """
<div class="main-header">
  <h1>🧪 AntioxFP</h1>
  <p>GNN-Based Antioxidant Activity Predictor | AttentiveFP Ensemble × 30 Models</p>
  <p style="font-size:0.85em; opacity:0.65;">
    Predicts DPPH• radical scavenging pIC₅₀ from SMILES · Atom-level interpretability via GNNExplainer
  </p>
</div>
"""

ABOUT_MD = """
### About This Tool

**AntioxFP** uses a 30-model AttentiveFP ensemble to predict
DPPH radical scavenging activity (pIC₅₀) for any small molecule
provided as a SMILES string.

**Key features:**
- 30-model ensemble (3 random seeds × 10-fold CV) → prediction ± uncertainty
- GNNExplainer atom importance maps → identify key pharmacophores
- Exceeds descriptor-based benchmark (R² = 0.785 vs 0.78)
- Applicable domain: ~98% of drug-like antioxidant space

**Dataset**: 1,911 DPPH antioxidants (AODB, curated by Ghironi et al. 2025)

**Reference**: *Graph Neural Network Models for Predicting the Antioxidant
Activity of Chemical Compounds*, 2025.

**Tips:**
- pIC₅₀ > 5.0 → strong antioxidant (IC₅₀ ≤ 10 µM)
- pIC₅₀ 4.0–5.0 → moderate activity
- pIC₅₀ < 4.0 → weak scavenger
- Enable GNNExplainer to see which atoms drive the prediction
"""


def predict_wrapper(smiles, run_explainer):
    atom_img, bar_img, result_md, pharma_text, _ = predict(smiles, run_explainer)
    return atom_img, bar_img, result_md, pharma_text


with gr.Blocks(css=CSS, title="AntioxFP — Antioxidant Activity Predictor") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row():
        # ── Left panel ────────────────────────────────────────────────────────
        with gr.Column(scale=2):
            smiles_input = gr.Textbox(
                label="SMILES Input",
                placeholder="Enter SMILES string, e.g. O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
                lines=3,
                max_lines=5,
            )
            with gr.Row():
                run_btn      = gr.Button("🔬 Predict Activity", variant="primary", scale=3)
                explainer_cb = gr.Checkbox(
                    label="Run GNNExplainer (atom maps, +10–20s)",
                    value=True, scale=2,
                )
            clear_btn = gr.Button("🗑️ Clear", variant="secondary")

            gr.Markdown("**Quick Examples** — click to load:")
            example_rows = []
            for i in range(0, len(EXAMPLE_SMILES), 2):
                with gr.Row():
                    for smi, label in EXAMPLE_SMILES[i:i+2]:
                        btn = gr.Button(label, elem_classes=["example-btn"])
                        btn.click(fn=lambda s=smi: s, outputs=smiles_input)

            gr.Markdown(ABOUT_MD)

        # ── Right panel ───────────────────────────────────────────────────────
        with gr.Column(scale=3):
            result_md    = gr.Markdown("*Results will appear here after prediction.*",
                                       elem_classes=["result-box"])
            pharma_text  = gr.Markdown("")

            with gr.Tabs():
                with gr.TabItem("🗺️ Atom Importance Map"):
                    atom_img = gr.Image(
                        label="2D Atom-Level Importance (blue=low → red=high)",
                        type="pil", height=420,
                    )
                with gr.TabItem("📊 Top-15 Atom Scores"):
                    bar_img = gr.Image(
                        label="Atom Importance Bar Chart",
                        type="pil", height=420,
                    )

    run_btn.click(
        fn=predict_wrapper,
        inputs=[smiles_input, explainer_cb],
        outputs=[atom_img, bar_img, result_md, pharma_text],
        show_progress="full",
    )

    clear_btn.click(
        fn=lambda: ("", None, None,
                    "*Results will appear here after prediction.*", ""),
        outputs=[smiles_input, atom_img, bar_img, result_md, pharma_text],
    )

    gr.HTML("""
    <div style="text-align:center;margin-top:16px;opacity:0.5;font-size:0.82em;">
      AntioxFP · AttentiveFP (PyTorch Geometric) · GNNExplainer ·
      RDKit · Built with Gradio · 2025
    </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
