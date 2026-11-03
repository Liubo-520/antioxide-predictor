"""
Reviewer 3, comment 2: figure for the independent literature validation.

(a) experimental versus predicted pIC50 for all 196 independent compounds,
    coloured by the maximum Tanimoto similarity to the training set;
(b) RMSE and R2 as a function of a similarity cut-off, with the observed
    between-publication spread of the replicated compounds drawn as the noise
    floor that no model can beat;
(c) error distribution for phenolic versus non-phenolic compounds.

Outputs: revision_work/figures/fig_external_validation.png / .pdf
"""

import os, json, csv
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score, mean_squared_error
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
OUT = os.path.join(REV, "results")
FIG = os.path.join(REV, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})


def main():
    z = np.load(os.path.join(OUT, "external_arrays.npz"))
    y, p, sim = z['y'], z['hybrid'], z['max_sim']
    rows = list(csv.DictReader(open(os.path.join(OUT, "external_predictions.csv"),
                                    encoding='utf-8')))
    smiles = [r['smiles'] for r in rows]
    spread = np.array([float(r['pIC50_spread']) for r in rows])
    nmeas = np.array([int(r['n_measurements']) for r in rows])
    phenol_q = Chem.MolFromSmarts('[OX2H][c]')
    is_phenol = np.array([(lambda m: m is not None and m.HasSubstructMatch(phenol_q))
                          (Chem.MolFromSmiles(s)) for s in smiles])
    noise = float(np.median(spread[nmeas > 1])) if (nmeas > 1).any() else None

    fig = plt.figure(figsize=(11.4, 3.5))
    gs = fig.add_gridspec(1, 3, wspace=0.46, left=0.060, right=0.985,
                          top=0.90, bottom=0.17)

    # ---------------------------------------------------------------- (a)
    ax = fig.add_subplot(gs[0])
    sc = ax.scatter(y, p, c=sim, cmap='viridis', s=26, alpha=0.88,
                    edgecolor='white', linewidth=0.4, vmin=0.15, vmax=0.8)
    lo, hi = 0.0, 8.0
    ax.plot([lo, hi], [lo, hi], ls='--', color='#555555', lw=0.9, zorder=0)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('Experimental pIC$_{50}$ (literature)')
    ax.set_ylabel('Predicted pIC$_{50}$ (hybrid)')
    ax.set_title('(a) Independent literature set, $n$ = %d' % len(y), fontsize=9.5, loc='left')
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label('max Tanimoto to training set', fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    ax.text(0.04, 0.96, '$R^2$ = %.2f\nRMSE = %.2f\nSpearman = %.2f'
            % (r2_score(y, p), np.sqrt(mean_squared_error(y, p)),
               __import__('scipy.stats', fromlist=['spearmanr']).spearmanr(y, p)[0]),
            transform=ax.transAxes, va='top', ha='left', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.35', fc='#f4f6f8', ec='#b8c2cc', lw=0.6))

    # ---------------------------------------------------------------- (b)
    ax = fig.add_subplot(gs[1])
    thr = np.arange(0.15, 0.71, 0.05)
    rm, r2v, ns = [], [], []
    for t in thr:
        m = sim >= t
        if m.sum() >= 10:
            rm.append(np.sqrt(mean_squared_error(y[m], p[m])))
            r2v.append(r2_score(y[m], p[m]))
            ns.append(int(m.sum()))
        else:
            rm.append(np.nan); r2v.append(np.nan); ns.append(int(m.sum()))
    ax.plot(thr, rm, 'o-', color='#c0392b', lw=1.4, ms=4.5, label='RMSE')
    if noise:
        ax.axhline(noise, ls=':', color='#2c3e50', lw=1.2)
        ax.text(thr[-1], noise + 0.03, 'between-publication spread (%.2f)' % noise,
                ha='right', va='bottom', fontsize=7.4, color='#2c3e50')
    ax.set_xlabel('minimum Tanimoto similarity to training set')
    ax.set_ylabel('RMSE (pIC$_{50}$ units)', color='#c0392b')
    ax.tick_params(axis='y', labelcolor='#c0392b')
    ax.set_title('(b) Accuracy inside the similarity domain', fontsize=9.5, loc='left')
    ax2 = ax.twinx()
    ax2.plot(thr, r2v, 's--', color='#2471a3', lw=1.2, ms=4, label='$R^2$')
    ax2.axhline(0.0, color='#95a5a6', lw=0.7, ls='-')
    ax2.set_ylabel('$R^2$', color='#2471a3')
    ax2.tick_params(axis='y', labelcolor='#2471a3')
    for t, n, v in zip(thr[::2], ns[::2], rm[::2]):
        if not np.isnan(v):
            ax.annotate('n=%d' % n, (t, v), textcoords='offset points',
                        xytext=(0, 7), ha='center', fontsize=6.6, color='#555555')
    ax.legend(handles=[Line2D([], [], color='#c0392b', marker='o', ls='-', label='RMSE'),
                       Line2D([], [], color='#2471a3', marker='s', ls='--', label='$R^2$')],
              loc='upper center', fontsize=7.6, frameon=False, ncol=2)

    # ---------------------------------------------------------------- (c)
    ax = fig.add_subplot(gs[2])
    data = [np.abs(y - p)[is_phenol], np.abs(y - p)[~is_phenol]]
    labels = ['phenolic\n($n$ = %d)' % is_phenol.sum(),
              'non-phenolic\n($n$ = %d)' % (~is_phenol).sum()]
    bp = ax.boxplot(data, labels=labels, widths=0.55, patch_artist=True,
                    medianprops=dict(color='#2c3e50', lw=1.3), showfliers=False)
    for patch, col in zip(bp['boxes'], ['#aed6a3', '#f5b7a1']):
        patch.set_facecolor(col); patch.set_edgecolor('#5d6d7e'); patch.set_linewidth(0.8)
    for i, d in enumerate(data, 1):
        x = np.random.default_rng(1).normal(i, 0.055, len(d))
        ax.scatter(x, d, s=11, color='#34495e', alpha=0.45, zorder=3,
                   edgecolor='none')
    if noise:
        ax.axhline(noise, ls=':', color='#2c3e50', lw=1.2)
    ax.set_ylabel('|experimental $-$ predicted| (pIC$_{50}$)')
    ax.set_title('(c) Error by presence of a phenolic OH', fontsize=9.5, loc='left')
    ax.set_ylim(bottom=0)
    for i, d in enumerate(data, 1):
        ax.text(i, ax.get_ylim()[1] * 0.97, 'median %.2f' % np.median(d),
                ha='center', va='top', fontsize=7.4)

    for ext in ('png', 'pdf'):
        pth = os.path.join(FIG, 'fig_external_validation.' + ext)
        fig.savefig(pth, dpi=400 if ext == 'png' else None, bbox_inches='tight',
                    facecolor='white')
        print('wrote', pth)


if __name__ == '__main__':
    main()
