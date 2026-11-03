"""
Data processing pipeline for GNN antioxidant activity prediction.
Replicates the 80:20 random split from jox-15-00080, and adds flavonoid scaffold split.
"""

import os
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XLSX_PATH = os.path.join(DATA_DIR, "DPPH_30min_Dataset.xlsx")
OUT_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42

# ─── Atom / Bond feature helpers ────────────────────────────────────────────

ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'other']
DEGREE_VALS = [0, 1, 2, 3, 4, 5]
CHARGE_VALS = [-2, -1, 0, 1, 2]
HYBRID_VALS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
H_COUNT_VALS = [0, 1, 2, 3, 4]
VALENCE_VALS = [0, 1, 2, 3, 4, 5, 6]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def one_hot(val, choices):
    vec = [0] * len(choices)
    idx = choices.index(val) if val in choices else len(choices) - 1
    vec[idx] = 1
    return vec


def atom_features(atom):
    symbol = atom.GetSymbol()
    symbol_oh = one_hot(symbol if symbol in ATOM_TYPES[:-1] else 'other', ATOM_TYPES)
    degree_oh = one_hot(atom.GetDegree(), DEGREE_VALS)
    charge_oh = one_hot(atom.GetFormalCharge(), CHARGE_VALS)
    hybrid_oh = one_hot(atom.GetHybridization(), HYBRID_VALS)
    aromatic = [int(atom.GetIsAromatic())]
    hcount_oh = one_hot(atom.GetTotalNumHs(), H_COUNT_VALS)
    in_ring = [int(atom.IsInRing())]
    valence_oh = one_hot(atom.GetTotalValence(), VALENCE_VALS)
    return symbol_oh + degree_oh + charge_oh + hybrid_oh + aromatic + hcount_oh + in_ring + valence_oh


def bond_features(bond):
    btype_oh = one_hot(bond.GetBondType(), BOND_TYPES)
    conj = [int(bond.GetIsConjugated())]
    in_ring = [int(bond.IsInRing())]
    return btype_oh + conj + in_ring


def mol_to_graph(smiles, y):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr += [feat, feat]

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([y], dtype=torch.float),
        smiles=smiles,
    )


# ─── Morgan fingerprint (for AD / chemical space) ───────────────────────────

def mol_to_fp(smiles, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp)


# ─── Flavonoid scaffold detection ───────────────────────────────────────────

FLAVONOID_SMARTS = [
    'O=C1CC(c2ccccc2)Oc2ccccc21',       # flavanone
    'O=C1C=C(c2ccccc2)Oc2ccccc21',      # flavone
    'O=C1c2ccccc2OC(c2ccccc2)C1',       # isoflavanone
    'O=C1c2cc(O)ccc2OC(c2ccccc2)C1',    # hydroxy flavanone
    'O=c1cc(-c2ccccc2)oc2ccccc12',      # flavone aromatic
    'O=c1cc(-c2ccccc2)oc2cc(O)ccc12',   # luteolin-like
    'Oc1ccc2c(c1)OC(c1ccccc1)CC2=O',    # naringenin-like
    'O=C1CCc2cc(O)ccc2O1',              # chromanone core
    'c1coc2ccccc2c(=O)1',               # chromone
    'O=c1ccoc2ccccc12',                 # coumarin
]


def is_flavonoid(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    for sma in FLAVONOID_SMARTS:
        patt = Chem.MolFromSmarts(sma)
        if patt and mol.HasSubstructMatch(patt):
            return True
    # Also check Murcko scaffold contains chromenone/chromanone ring system
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        sca_smi = Chem.MolToSmiles(scaffold)
        mol_sca = Chem.MolFromSmiles(sca_smi)
        for sma in FLAVONOID_SMARTS:
            patt = Chem.MolFromSmarts(sma)
            if patt and mol_sca and mol_sca.HasSubstructMatch(patt):
                return True
    except Exception:
        pass
    return False


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("Loading dataset...")
    df = pd.read_excel(XLSX_PATH)
    print(f"  Raw rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Identify SMILES and target columns (flexible naming)
    smiles_col = [c for c in df.columns if 'smiles' in c.lower() or 'SMILES' in c][0]
    target_col = [c for c in df.columns if 'pic50' in c.lower() or 'pIC50' in c or 'minusLog' in c][0]
    print(f"  SMILES column: '{smiles_col}', Target column: '{target_col}'")

    df = df[[smiles_col, target_col]].dropna()
    df.columns = ['smiles', 'pIC50']
    print(f"  After dropna: {len(df)} compounds")

    # Build molecular graphs and filter invalid SMILES
    print("Building molecular graphs...")
    graphs, fps, valid_idx = [], [], []
    for i, row in df.iterrows():
        g = mol_to_graph(row['smiles'], row['pIC50'])
        fp = mol_to_fp(row['smiles'])
        if g is not None and fp is not None:
            graphs.append(g)
            fps.append(fp)
            valid_idx.append(i)

    df_valid = df.loc[valid_idx].reset_index(drop=True)
    fps_array = np.array(fps)
    print(f"  Valid molecules: {len(graphs)}")
    print(f"  pIC50 stats: mean={df_valid.pIC50.mean():.3f}, "
          f"std={df_valid.pIC50.std():.3f}, "
          f"min={df_valid.pIC50.min():.3f}, max={df_valid.pIC50.max():.3f}")

    # ── Split A: 80:20 random split (matching jox-15-00080) ──────────────────
    all_idx = np.arange(len(graphs))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=0.2, random_state=RANDOM_SEED
    )
    print(f"\nRandom split (80:20):")
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

    train_graphs = [graphs[i] for i in train_idx]
    test_graphs  = [graphs[i] for i in test_idx]
    train_fps    = fps_array[train_idx]
    test_fps     = fps_array[test_idx]

    # ── Split B: Scaffold split — flavonoids → test ───────────────────────────
    print("\nIdentifying flavonoid compounds...")
    flav_mask = df_valid['smiles'].apply(is_flavonoid).values
    flav_idx  = np.where(flav_mask)[0]
    nonflav_idx = np.where(~flav_mask)[0]
    print(f"  Flavonoids: {len(flav_idx)} ({100*len(flav_idx)/len(graphs):.1f}%)")
    print(f"  Non-flavonoids: {len(nonflav_idx)}")

    scaf_train_graphs = [graphs[i] for i in nonflav_idx]
    scaf_test_graphs  = [graphs[i] for i in flav_idx]
    scaf_train_fps    = fps_array[nonflav_idx]
    scaf_test_fps     = fps_array[flav_idx]

    # ── Save everything ──────────────────────────────────────────────────────
    save = {
        'graphs': graphs,
        'fps': fps_array,
        'df': df_valid,
        'random_split': {
            'train_idx': train_idx,
            'test_idx': test_idx,
            'train_graphs': train_graphs,
            'test_graphs': test_graphs,
            'train_fps': train_fps,
            'test_fps': test_fps,
        },
        'scaffold_split': {
            'flav_idx': flav_idx,
            'nonflav_idx': nonflav_idx,
            'train_graphs': scaf_train_graphs,
            'test_graphs': scaf_test_graphs,
            'train_fps': scaf_train_fps,
            'test_fps': scaf_test_fps,
            'flav_mask': flav_mask,
        },
    }

    out_path = os.path.join(OUT_DIR, "processed_data.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(save, f)
    print(f"\nSaved processed data → {out_path}")

    # Summary statistics for paper
    print("\n=== Summary for paper ===")
    print(f"Total valid compounds: {len(graphs)}")
    print(f"Random split: {len(train_idx)} train / {len(test_idx)} test")
    print(f"Scaffold split: {len(nonflav_idx)} train / {len(flav_idx)} test (flavonoids)")
    print(f"Node feature dim: {graphs[0].x.shape[1]}")
    print(f"Edge feature dim: {graphs[0].edge_attr.shape[1] if graphs[0].edge_attr.numel() > 0 else 6}")

    df_valid['is_flavonoid'] = flav_mask
    df_valid.to_csv(os.path.join(OUT_DIR, "compounds_annotated.csv"), index=False)
    print(f"Annotated CSV saved.")


if __name__ == '__main__':
    main()
