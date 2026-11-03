---
title: AntioxFP — GNN Antioxidant Activity Predictor
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Predict antioxidant activity (DPPH pIC50) with GNN
---

# AntioxFP 🧪

**GNN-Based Antioxidant Activity Predictor**

Predict DPPH• radical scavenging activity (pIC₅₀) from molecular SMILES strings
using a 30-model AttentiveFP ensemble, with GNNExplainer atom-level importance maps
to reveal which structural features drive the prediction.

## Features

- **30-model ensemble** (3 random seeds × 10-fold CV): AttentiveFP + Mordred hybrid
- **Prediction uncertainty**: mean ± std across all 30 fold models
- **Atom importance maps**: GNNExplainer 2D heatmaps, pharmacophore-level analysis
- **Exceeds benchmark**: R² = 0.785 vs. descriptor-based reference (R² = 0.78)
- **Applicability domain**: ~98% coverage of DPPH antioxidant chemical space

## How to Use

1. Enter a SMILES string in the input box (or click an example)
2. Optionally enable **GNNExplainer** (adds ~10–20 s) for atom importance maps
3. Click **Predict Activity**

## Interpretation

| pIC₅₀ | Activity Level | Estimated IC₅₀ |
|--------|---------------|----------------|
| ≥ 5.0  | 🟢 High        | ≤ 10 µM        |
| 4.5–5.0 | 🟡 Moderate–High | 10–32 µM |
| 4.0–4.5 | 🟠 Moderate  | 32–100 µM      |
| < 4.0  | 🔴 Low         | > 100 µM       |

## Scientific Background

The model was trained on 1,911 DPPH antioxidant compounds from the
[Antioxidant Database (AODB)](https://food.shujucloud.com/) curated by
Ghironi et al. (2025), using the same 80:20 random split for direct
head-to-head comparison.

**Architecture**: AttentiveFP (Xiong et al., J. Med. Chem. 2020) with
- 2 graph attention layers, hidden dim = 200, 2 readout timesteps
- Trained with Adam optimizer, ReduceLROnPlateau scheduler
- Early stopping on validation R²

**Reference paper**: *Graph Neural Network Models for Predicting the Antioxidant
Activity of Chemical Compounds: Scaffold-Based Evaluation and Interpretability Analysis*, 2025.

## Pharmacophore Insights

The GNNExplainer atom maps confirm that the model has learned chemically meaningful SAR:

- **Catechol B ring** (3',4'-OH): primary HAT pharmacophore → highest atom importance
- **C-ring 3-OH** (flavonols like quercetin): secondary importance
- **Conjugated chromone/carbonyl**: supports radical delocalization
- **Stilbene vinyl bridge**: SET pathway marker in resveratrol-type compounds

## Citation

If you use AntioxFP in your research, please cite:

```
Graph Neural Network Models for Predicting the Antioxidant Activity of Chemical Compounds, 2025.
```

## License

MIT License. Models and code available upon paper acceptance.
