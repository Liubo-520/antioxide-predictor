"""
Generate all publication-quality figures for the paper.
Requires results from scripts 01-07 to be completed first.
"""

import os, sys, json, pickle, warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

CODE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.dirname(CODE_DIR)
RESULT_DIR = os.path.join(DATA_DIR, "results")
FIG_DIR    = os.path.join(RESULT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Publication style — large, clear, journal-ready
plt.rcParams.update({
    'font.family':       'Arial',
    'font.size':         13,
    'axes.labelsize':    14,
    'axes.titlesize':    15,
    'xtick.labelsize':   12,
    'ytick.labelsize':   12,
    'legend.fontsize':   11,
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'lines.linewidth':   1.6,
})

MODEL_DISPLAY = {
    'attentivefp': 'AttentiveFP',
    'gcn':         'GCN',
    'gat':         'GAT',
}
MODEL_COLORS = {
    'attentivefp': '#1565C0',
    'gcn':         '#2E7D32',
    'gat':         '#C62828',
}


# ── Figure 1: pIC50 distribution ────────────────────────────────────────────

def fig_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ic50_um = 10**(-df['pIC50']) * 1e6

    ax = axes[0]
    ax.hist(ic50_um, bins=50, color='#78909C', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('IC$_{50}$ ($\\mu$M)', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title('(a) Before Log Transformation', fontsize=15, fontweight='bold', pad=10)
    skew_raw = float(stats.skew(ic50_um))
    ax.text(0.97, 0.97, f'Skewness = {skew_raw:.2f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=13,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF8E1', alpha=0.9,
                      edgecolor='#BDBDBD'))

    ax = axes[1]
    ax.hist(df['pIC50'], bins=50, color='#1565C0', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('pIC$_{50}$ ($-\\log_{10}$ M)', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title('(b) After Log Transformation', fontsize=15, fontweight='bold', pad=10)
    skew_log = float(stats.skew(df['pIC50']))
    ax.text(0.97, 0.97, f'Skewness = {skew_log:.2f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=13,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF8E1', alpha=0.9,
                      edgecolor='#BDBDBD'))

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig1_distribution.png'))
    plt.close()
    print("  Fig 1 saved")


# ── Figure 2: CV boxplot ─────────────────────────────────────────────────────

def fig_cv_boxplot(rand_results):
    fig, ax = plt.subplots(figsize=(8.5, 6))

    positions = list(range(1, len(rand_results) + 1))
    colors_list = [MODEL_COLORS[r['model']] for r in rand_results]
    labels = [MODEL_DISPLAY[r['model']] for r in rand_results]
    all_cv = [r['cv_r2_all'] for r in rand_results]

    bplots = ax.boxplot(all_cv, positions=positions, patch_artist=True,
                        widths=0.55, showfliers=True,
                        medianprops=dict(color='white', linewidth=2.5),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        flierprops=dict(marker='o', markersize=6, alpha=0.7))

    for patch, col in zip(bplots['boxes'], colors_list):
        patch.set_facecolor(col)
        patch.set_alpha(0.85)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)

    # Overlay individual fold points
    for pos, vals, col in zip(positions, all_cv, colors_list):
        x = np.random.normal(pos, 0.06, size=len(vals))
        ax.scatter(x, vals, color=col, s=28, alpha=0.8, edgecolor='black',
                   linewidth=0.5, zorder=3)

    ax.axhline(0.74, color='#FF6F00', linestyle='--', linewidth=1.8,
               label='Reference Extra Trees CV (R$^2$=0.74)', zorder=0)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=13, fontweight='bold')
    ax.set_ylabel('R$^2$ (10-fold cross-validation)', fontsize=14)
    ax.set_title('Internal Validation Performance',
                 fontsize=15, fontweight='bold', pad=10)
    ax.legend(fontsize=12, loc='lower right', framealpha=0.92)
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig2_cv_boxplot.png'))
    plt.close()
    print("  Fig 2 saved")


# ── Figure 3: Predicted vs actual scatter ───────────────────────────────────

def fig_scatter(rand_results):
    best = max(rand_results, key=lambda r: r['test_r2'])
    preds   = np.array(best['predictions'])
    targets = np.array(best['targets'])
    model_name = MODEL_DISPLAY[best['model']]

    fig, ax = plt.subplots(figsize=(7, 7))

    from scipy.stats import gaussian_kde
    try:
        xy  = np.vstack([targets, preds])
        kde = gaussian_kde(xy)
        colors_arr = kde(xy)
        sc = ax.scatter(targets, preds, c=colors_arr, cmap='viridis',
                        s=22, alpha=0.85, linewidths=0)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label('Point Density', fontsize=12)
        cbar.ax.tick_params(labelsize=11)
    except Exception:
        ax.scatter(targets, preds, color=MODEL_COLORS[best['model']],
                   s=22, alpha=0.7, linewidths=0)

    lo = min(targets.min(), preds.min()) - 0.3
    hi = max(targets.max(), preds.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.5, label='y = x', zorder=5)

    slope, intercept, r_value, p_value, _ = stats.linregress(targets, preds)
    x_fit = np.linspace(lo, hi, 100)
    ax.plot(x_fit, slope*x_fit + intercept, color='#E53935',
            linewidth=2.0, label=f'Linear fit (r={r_value:.3f})', zorder=5)

    r2   = r2_score(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae  = mean_absolute_error(targets, preds)

    # Stats box: bottom-right (legend will go top-left, no overlap)
    stats_text = (f'R$^2$ = {r2:.4f}\n'
                  f'RMSE = {rmse:.4f}\n'
                  f'MAE = {mae:.4f}\n'
                  f'n = {len(targets)}')
    ax.text(0.97, 0.03, stats_text, transform=ax.transAxes,
            va='bottom', ha='right', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      alpha=0.92, edgecolor='#9E9E9E', linewidth=1.0))

    ax.set_xlabel('Experimental pIC$_{50}$', fontsize=14)
    ax.set_ylabel('Predicted pIC$_{50}$',     fontsize=14)
    ax.set_title(f'{model_name}: Predicted vs. Experimental\n(External Test Set, n={len(targets)})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left', framealpha=0.92)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig3_scatter.png'))
    plt.close()
    print("  Fig 3 saved")


# ── Figure 4: Williams plot ──────────────────────────────────────────────────

def fig_williams(williams_data):
    h      = np.array(williams_data['h'])
    e_std  = np.array(williams_data['e_std'])
    h_star = williams_data['h_star']
    ad     = williams_data['ad_stats']

    mask_in   = np.array(ad['mask_in'])
    mask_hilo = np.array(ad['mask_hilo'])
    mask_hire = np.array(ad['mask_hire'])
    mask_out  = np.array(ad['mask_out'])

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    scatter_kw = dict(s=32, linewidths=0.4, edgecolor='white', alpha=0.85)
    if mask_in.sum():
        ax.scatter(h[mask_in],   e_std[mask_in],   color='#1565C0',
                   label=f'In AD (n={mask_in.sum()})', **scatter_kw)
    if mask_hilo.sum():
        ax.scatter(h[mask_hilo], e_std[mask_hilo], color='#FF8F00',
                   label=f'High leverage (n={mask_hilo.sum()})', **scatter_kw)
    if mask_hire.sum():
        ax.scatter(h[mask_hire], e_std[mask_hire], color='#2E7D32',
                   label=f'High residual (n={mask_hire.sum()})', **scatter_kw)
    if mask_out.sum():
        ax.scatter(h[mask_out],  e_std[mask_out],  color='#C62828',
                   label=f'Outlier (n={mask_out.sum()})', **scatter_kw)

    ax.axvline(h_star, color='k', linestyle='--', linewidth=1.8,
               label=f'h* = {h_star:.3f}')
    ax.axhline( 3.0, color='#9E9E9E', linestyle=':', linewidth=1.5,
               label='$\\pm$3$\\sigma$')
    ax.axhline(-3.0, color='#9E9E9E', linestyle=':', linewidth=1.5)
    ax.axhline( 0.0, color='k', linestyle='-', linewidth=0.6, alpha=0.4)

    ax.fill_betweenx([-3, 3], 0, h_star, alpha=0.08, color='#1565C0')

    ax.set_xlabel('Leverage ($h$)', fontsize=14)
    ax.set_ylabel('Standardized Residuals', fontsize=14)
    ax.set_title('Applicability Domain: Williams Plot\n(PCA-reduced Morgan FP, 50 components)',
                 fontsize=14, fontweight='bold', pad=10)
    # Legend OUTSIDE the plot
    ax.legend(fontsize=11, loc='center left', bbox_to_anchor=(1.02, 0.5),
              framealpha=0.95)

    in_ad_pct = ad['in_AD_pct']
    ax.text(0.03, 0.04,
            f"Compounds in AD: {ad['in_AD']}/{len(h)} ({in_ad_pct:.1f}%)",
            transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF8E1',
                      alpha=0.92, edgecolor='#BDBDBD'))

    ax.set_xlim(left=0)
    ylim = max(4.5, np.percentile(np.abs(e_std), 99) + 0.5)
    ax.set_ylim(-ylim, ylim)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig4_williams.png'))
    plt.close()
    print("  Fig 4 saved")


# ── Figure 5: GNN model performance bar chart (random split only) ────────────

def fig_comparison(rand_results):
    metrics = ['test_r2', 'test_rmse', 'test_mae']
    metric_labels = ['R²', 'RMSE', 'MAE']
    ref_lines = {'test_r2': 0.78, 'test_rmse': 0.45, 'test_mae': 0.31}
    ref_labels = {'test_r2': 'Ref. Ensemble (0.78)',
                  'test_rmse': 'Ref. Ensemble (0.45)',
                  'test_mae': 'Ref. Ensemble (0.31)'}
    n_models = len(rand_results)
    x = np.arange(n_models)
    width = 0.55

    fig, axes = plt.subplots(1, 3, figsize=(12, 5.2))

    for ax, met, mlbl in zip(axes, metrics, metric_labels):
        rand_vals = [r[met] for r in rand_results]
        model_lbls = [MODEL_DISPLAY[r['model']] for r in rand_results]
        colors = [MODEL_COLORS[r['model']] for r in rand_results]

        bars = ax.bar(x, rand_vals, width,
                      color=colors, alpha=0.88, edgecolor='white', linewidth=0.8)

        # Reference paper dashed line
        ref_val = ref_lines[met]
        ax.axhline(ref_val, color='#FF6F00', linestyle='--', linewidth=1.5,
                   label=ref_labels[met], zorder=0)

        # Value labels above bars
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=9,
                    fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(model_lbls, fontsize=12)
        ax.set_ylabel(mlbl, fontsize=12)
        ax.set_title(mlbl, fontsize=13, fontweight='bold')
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        # Legend above the subplot, never overlapping bars
        ax.legend(fontsize=9, loc='upper center',
                  bbox_to_anchor=(0.5, 1.18), framealpha=0.92,
                  edgecolor='#BDBDBD')

    plt.suptitle('GNN Model Performance (80:20 Random Split)',
                 fontsize=13, fontweight='bold', y=1.06)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig5_comparison.png'))
    plt.close()
    print("  Fig 5 saved")


# ── Figure 6: Comprehensive model comparison table (visual) ─────────────────

def fig_model_table(rand_results, scaf_results):
    """Create a visual performance comparison table including reference paper."""
    rows = []
    for r in rand_results:
        rows.append({
            'Model': MODEL_DISPLAY[r['model']] + ' (Ours)',
            'Split': 'Random',
            'CV R²': f"{r['cv_r2_mean']:.3f} ± {r['cv_r2_ci']:.3f}",
            'Test R²': f"{r['test_r2']:.3f}",
            'RMSE': f"{r['test_rmse']:.3f}",
            'MAE': f"{r['test_mae']:.3f}",
        })
    for r in scaf_results:
        rows.append({
            'Model': MODEL_DISPLAY[r['model']] + ' (Ours)',
            'Split': 'Scaffold',
            'CV R²': f"{r['cv_r2_mean']:.3f} ± {r['cv_r2_ci']:.3f}",
            'Test R²': f"{r['test_r2']:.3f}",
            'RMSE': f"{r['test_rmse']:.3f}",
            'MAE': f"{r['test_mae']:.3f}",
        })
    # Reference paper
    ref_models = [
        ('Extra Trees (Ref)', '0.74 ± 0.06', '0.77', '0.45', '0.31'),
        ('Ensemble ET+GB+XGB (Ref)', '—', '0.78', '0.45', '0.31'),
    ]
    for (name, cv, tr2, rmse, mae) in ref_models:
        rows.append({'Model': name, 'Split': 'Random', 'CV R²': cv,
                     'Test R²': tr2, 'RMSE': rmse, 'MAE': mae})

    df_table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 0.5 * len(rows) + 1.5))
    ax.axis('off')
    tbl = ax.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        cellLoc='center', loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.6)

    # Style header
    for col in range(len(df_table.columns)):
        tbl[0, col].set_facecolor('#1565C0')
        tbl[0, col].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for row in range(1, len(rows)+1):
        fc = '#E3F2FD' if row % 2 == 0 else 'white'
        for col in range(len(df_table.columns)):
            tbl[row, col].set_facecolor(fc)

    ax.set_title('Model Performance Comparison', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig6_model_table.png'), bbox_inches='tight')
    plt.close()
    print("  Fig 6 saved")


# ── Figure 7: Functional group importance heatmap ───────────────────────────

# Priority-ordered SMARTS. For each atom, the FIRST matching group wins.
_FG_DEFS = [
    # name,                   smarts,                                 color
    ('Catechol\n(ortho-diol)', '[OH]c1c([OH])cccc1',                 '#C62828'),
    ('Phenolic\n-OH (mono)',   '[OH]c',                              '#E53935'),
    ('Carboxyl\n-COOH',        'C(=O)[OH]',                          '#F4511E'),
    ('Carbonyl\nC=O',          '[C;!c]=O',                           '#FF8F00'),
    ('Vinyl\nC=C',             '[C;!c]=[C;!c]',                      '#2E7D32'),
    ('Ring O\n(ether)',        '[o]',                                 '#7B1FA2'),
    ('Aromatic C\n(ring)',     'c',                                   '#1565C0'),
]


def _assign_fg(smiles, atom_weights):
    """Return dict: fg_name → mean importance (NaN if group absent)."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    n = mol.GetNumAtoms()
    w = np.array(atom_weights)
    assigned = np.zeros(n, dtype=bool)
    result = {}
    for (fg_name, smarts, _) in _FG_DEFS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            result[fg_name] = np.nan
            continue
        matches = mol.GetSubstructMatches(patt)
        # Collect unassigned atoms from all matches
        fg_atoms = sorted({idx for m in matches for idx in m if not assigned[idx]})
        if fg_atoms:
            result[fg_name] = float(np.mean(w[fg_atoms]))
            assigned[fg_atoms] = True
        else:
            result[fg_name] = np.nan
    return result


def fig_feature_importance():
    """Functional group importance heatmap across the 6 representative compounds."""
    interp_path = os.path.join(RESULT_DIR, "interpretability_results.json")
    if not os.path.exists(interp_path):
        print("  Fig 7: interpretability results not found, skipping")
        return

    with open(interp_path) as f:
        results = json.load(f)

    fg_names  = [d[0] for d in _FG_DEFS]
    comp_names = [r['name'] for r in results]
    pred_vals  = [r.get('pred_pIC50', 0.0) for r in results]

    matrix = np.array([
        [_assign_fg(r['smiles'], r['atom_weights']).get(g, np.nan) for g in fg_names]
        for r in results
    ])

    # ── Heatmap ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6.5))

    cmap_hm = cm.get_cmap('RdYlBu_r').copy()
    cmap_hm.set_bad('#EEEEEE')   # gray for absent groups

    masked = np.ma.array(matrix, mask=np.isnan(matrix))
    im = ax.imshow(masked, cmap=cmap_hm, aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(fg_names)))
    ax.set_xticklabels(fg_names, fontsize=12)
    ax.set_yticks(range(len(comp_names)))
    ylabels = [f"{n}\n(pred pIC$_{{50}}$={p:.2f})" for n, p in zip(comp_names, pred_vals)]
    ax.set_yticklabels(ylabels, fontsize=12)

    # Cell annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                txt_col = 'white' if val > 0.55 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=13, color=txt_col, fontweight='bold')
            else:
                ax.text(j, i, '—', ha='center', va='center',
                        fontsize=14, color='#999999')

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Mean Atom Importance', fontsize=13)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Medium', 'High'])
    cbar.ax.tick_params(labelsize=12)

    ax.set_title('GNNExplainer: Pharmacophore Importance\n'
                 '(mean atom importance aggregated per functional group)',
                 fontsize=14, fontweight='bold', pad=12)

    # Vertical separators between group categories
    for x in [2.5, 4.5]:
        ax.axvline(x, color='white', linewidth=1.5, alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, 'fig7_fg_importance.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Fig 7 (FG heatmap) saved")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Generating all publication figures...\n")

    # Load data
    with open(os.path.join(RESULT_DIR, "processed_data.pkl"), 'rb') as f:
        data = pickle.load(f)

    df = data['df']

    print("  Loading results...")
    rand_results = scaf_results = williams_data = None

    rand_path = os.path.join(RESULT_DIR, "random_split_results.json")
    scaf_path = os.path.join(RESULT_DIR, "scaffold_split_results.json")
    will_path = os.path.join(RESULT_DIR, "williams_plot_data.json")

    if os.path.exists(rand_path):
        with open(rand_path) as f:
            rand_results = json.load(f)
    if os.path.exists(scaf_path):
        with open(scaf_path) as f:
            scaf_results = json.load(f)
    if os.path.exists(will_path):
        with open(will_path) as f:
            williams_data = json.load(f)

    print("\nFigure 1: pIC50 distribution")
    fig_distribution(df)

    if rand_results:
        print("Figure 2: 10-fold CV boxplot")
        fig_cv_boxplot(rand_results)

        print("Figure 3: Predicted vs actual scatter")
        fig_scatter(rand_results)

    if williams_data:
        print("Figure 4: Williams plot")
        fig_williams(williams_data)

    if rand_results:
        print("Figure 5: GNN model performance bar chart")
        fig_comparison(rand_results)

    print("Figure 7: Functional group importance heatmap")
    fig_feature_importance()

    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == '__main__':
    main()
