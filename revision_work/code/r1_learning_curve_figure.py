"""
Reviewer 1, comment 3: figure for the learning curve.

(a) external-test R2 against training-set size for a high-capacity and a
    low-capacity AttentiveFP, with the spread over three random subsamples;
(b) the same curves as RMSE, with the ensemble result marked, showing how much
    of the remaining gap is closed by averaging fold models rather than by
    adding data or capacity.

Outputs: revision_work/figures/fig_learning_curve.png / .pdf
"""

import os, json
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
OUT = os.path.join(REV, "results")
FIG = os.path.join(REV, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})

STYLE = {'tuned': dict(color='#1f4e79', marker='o', label='tuned (2.64 M parameters)'),
         'compact': dict(color='#c0702a', marker='s', label='compact (0.23 M parameters)')}


def main():
    lc = json.load(open(os.path.join(OUT, "learning_curve.json")))
    fh = json.load(open(os.path.join(OUT, "final_hybrid.json")))
    ens_r2 = fh['graph']['AttentiveFP tuned']['r2']
    ens_rmse = fh['graph']['AttentiveFP tuned']['rmse']

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for key in ('tuned', 'compact'):
        rs = lc['summary'][key]
        n = np.array([r['n_train'] for r in rs])
        for ax, mkey, skey in ((axes[0], 'test_r2_mean', 'test_r2_sd'),
                               (axes[1], 'test_rmse_mean', 'test_rmse_sd')):
            m = np.array([r[mkey] for r in rs])
            s = np.array([r[skey] for r in rs])
            ax.plot(n, m, lw=1.5, ms=5, **STYLE[key])
            ax.fill_between(n, m - s, m + s, color=STYLE[key]['color'], alpha=0.16, lw=0)

    axes[0].axhline(ens_r2, ls='--', color='#2e7d32', lw=1.2)
    axes[0].text(300, ens_r2 + 0.012, '30-model ensemble, full training set (%.3f)' % ens_r2,
                 fontsize=7.6, color='#2e7d32')
    axes[0].set_ylabel('External-test $R^2$')
    axes[0].set_title('(a) Accuracy against training-set size', fontsize=9.5, loc='left')
    axes[1].axhline(ens_rmse, ls='--', color='#2e7d32', lw=1.2)
    axes[1].text(300, ens_rmse - 0.028, '30-model ensemble (%.3f)' % ens_rmse,
                 fontsize=7.6, color='#2e7d32')
    axes[1].set_ylabel('External-test RMSE (pIC$_{50}$)')
    axes[1].set_title('(b) Error against training-set size', fontsize=9.5, loc='left')
    for ax in axes:
        ax.set_xlabel('Training compounds')
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(fontsize=7.8, frameon=False, loc='best')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(FIG, 'fig_learning_curve.' + ext)
        fig.savefig(p, dpi=400 if ext == 'png' else None, bbox_inches='tight',
                    facecolor='white')
        print('wrote', p)


if __name__ == '__main__':
    main()
