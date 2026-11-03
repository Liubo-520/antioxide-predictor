"""
Supplementary figure: predicted against experimental pIC50 for the GCN and GAT
baselines on the 383-compound external test set.

The submitted supporting information promised these panels but reproduced the
AttentiveFP scatter instead, with a note to regenerate them. This script draws
them from the stored test-set predictions.

Outputs: revision_work/figures/fig_si_gcn_gat.png / .pdf
"""

import os, json
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE); PROJ = os.path.dirname(REV)
FIG = os.path.join(REV, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})


def main():
    rows = json.load(open(os.path.join(PROJ, "results", "random_split_results.json")))
    want = [('gcn', 'GCN'), ('gat', 'GAT')]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.1))
    for ax, (key, label) in zip(axes, want):
        r = next(x for x in rows if x['model'] == key)
        y = np.array(r['targets']); p = np.array(r['predictions'])
        ax.scatter(y, p, s=20, alpha=0.55, color='#2471a3', edgecolor='white',
                   linewidth=0.3)
        lo, hi = 0.0, 8.0
        ax.plot([lo, hi], [lo, hi], ls='--', lw=0.9, color='#555555', zorder=0)
        m, b = np.polyfit(y, p, 1)
        xs = np.linspace(lo, hi, 10)
        ax.plot(xs, m * xs + b, color='#c0392b', lw=1.2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel('Experimental pIC$_{50}$')
        ax.set_ylabel('Predicted pIC$_{50}$')
        ax.set_title('%s ($n$ = %d)' % (label, len(y)), fontsize=10, loc='left')
        txt = ('$R^2$ = %.4f\nRMSE = %.4f\nMAE = %.4f\nslope = %.2f'
               % (r2_score(y, p), np.sqrt(mean_squared_error(y, p)),
                  mean_absolute_error(y, p), m))
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, va='top', ha='left',
                fontsize=8, bbox=dict(boxstyle='round,pad=0.35', fc='#f4f6f8',
                                      ec='#b8c2cc', lw=0.6))
        ax.grid(alpha=0.22, lw=0.6)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(FIG, 'fig_si_gcn_gat.' + ext)
        fig.savefig(p, dpi=360 if ext == 'png' else None, bbox_inches='tight',
                    facecolor='white')
        print('wrote', p)


if __name__ == '__main__':
    main()
