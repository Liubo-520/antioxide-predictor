"""
Reviewer 1, comment 4: schematic of the three GNN architectures with their
hyperparameters and trainable-parameter counts.

Draws revision_work/figures/fig_architectures.png (and .pdf).
Layer shapes and hyperparameters are read off code/models.py, so the figure
matches the implementation that produced the reported results.

Each panel draws the configuration SELECTED BY THE HYPERPARAMETER SEARCH,
because that is the model the paper reports; the configuration of the original
submission is given underneath as a second line. Drawing the submitted
configuration instead, as an earlier version of this figure did, makes the
diagram disagree with Table 2 and with the implementation.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings; warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
FIG = os.path.join(REV, "figures"); os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5})

C_IN = '#dfe7f2'      # input / output
C_MP = '#bcd4ea'      # message passing
C_RD = '#f6ddc0'      # readout / pooling
C_HD = '#dfe9d8'      # regression head
C_EDGE = '#3a4a5a'


def box(ax, x, y, w, h, text, color, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.025",
                                linewidth=0.9, edgecolor=C_EDGE, facecolor=color))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, label=None, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle='-|>', mutation_scale=9,
                                 linewidth=0.9, color=C_EDGE,
                                 connectionstyle="arc3,rad=%.2f" % rad))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.035, label, ha='center',
                va='bottom', fontsize=7.3, color='#30404f')


def hp_panel(ax, x, y, w, lines, title):
    h = 0.055 + 0.042 * len(lines)
    ax.add_patch(Rectangle((x, y - h), w, h, linewidth=0.7, edgecolor='#9aa7b4',
                           facecolor='#fbfbfc', linestyle=(0, (3, 2))))
    ax.text(x + 0.012, y - 0.032, title, fontsize=7.6, fontweight='bold',
            va='center', ha='left', color='#30404f')
    for i, ln in enumerate(lines):
        ax.text(x + 0.012, y - 0.032 - 0.042 * (i + 1), ln, fontsize=7.2,
                va='center', ha='left', color='#30404f')


def panel_frame(ax, tag, title):
    # All drawing sits between y = 0.27 and y = 0.98, so clip the empty lower
    # half of each panel rather than shipping a figure that is mostly blank.
    ax.set_xlim(0, 1); ax.set_ylim(0.25, 1.005); ax.axis('off')
    ax.text(0.0, 0.965, tag, fontsize=11, fontweight='bold', va='top', ha='left')
    ax.text(0.045, 0.965, title, fontsize=9.6, fontweight='bold', va='top', ha='left')


def main():
    # Keep the aspect ratio short enough that the float plus its caption fits
    # one text height at the width used in the manuscript; at 7.1 x 9.4 it
    # overflowed the page and the caption ran into the page number.
    fig = plt.figure(figsize=(7.1, 6.4))
    gs = fig.add_gridspec(3, 1, hspace=0.10, left=0.015, right=0.985,
                          top=0.985, bottom=0.012)

    # ---------------------------------------------------------------- (a) GCN
    ax = fig.add_subplot(gs[0]); panel_frame(ax, '(a)', 'Graph Convolutional Network (GCN)')
    y = 0.70
    box(ax, 0.02, y, 0.175, 0.14,
        'Molecular\ngraph\n$x_v \\in \\mathbb{R}^{40}$', C_IN)
    arrow(ax, 0.195, y + 0.07, 0.245, y + 0.07)
    box(ax, 0.245, y, 0.14, 0.14, 'Linear\n$40 \\to 512$\nReLU', C_IN)
    arrow(ax, 0.385, y + 0.07, 0.435, y + 0.07)
    box(ax, 0.435, y, 0.215, 0.14,
        'GCNConv $512 \\to 512$\nReLU + Dropout 0.2\n$\\times\\,4$ layers', C_MP, bold=True)
    arrow(ax, 0.65, y + 0.07, 0.70, y + 0.07)
    box(ax, 0.70, y, 0.135, 0.14, 'Global\nmean\npool', C_RD)
    arrow(ax, 0.835, y + 0.07, 0.86, y + 0.07)
    box(ax, 0.86, y, 0.125, 0.14, 'Linear\n$512\\!\\to\\!256\\!\\to\\!1$', C_HD)
    ax.text(0.922, y - 0.045, '$\\hat{y}$ = pIC$_{50}$', fontsize=8, ha='center')
    ax.text(0.5425, y - 0.055,
            '$H^{(l+1)}=\\sigma\\!\\left(\\tilde{D}^{-1/2}\\tilde{A}\\tilde{D}^{-1/2}H^{(l)}W^{(l)}\\right)$',
            fontsize=8.2, ha='center')
    hp_panel(ax, 0.02, 0.50, 0.96,
             ['selected: hidden nodes 512   |   graph conv. layers 4   |   batch 32   |   '
              'parameters 1,203,201',
              'as submitted: hidden nodes 256   |   layers 3   |   batch 64   |   parameters 240,897',
              'dropout 0.2   |   pooling global mean   |   bond features unused by GCNConv '
              '(topology only)'],
             'Hyperparameters (diagram shows the selected configuration)')
    # -------------------------------------------------------------- (b) GAT
    ax = fig.add_subplot(gs[1]); panel_frame(ax, '(b)', 'Graph Attention Network (GAT)')
    y = 0.70
    box(ax, 0.02, y, 0.175, 0.14, 'Molecular\ngraph\n$x_v \\in \\mathbb{R}^{40}$', C_IN)
    arrow(ax, 0.195, y + 0.07, 0.245, y + 0.07)
    box(ax, 0.245, y, 0.14, 0.14, 'Linear\n$40 \\to 1024$\nELU', C_IN)
    arrow(ax, 0.385, y + 0.07, 0.435, y + 0.07)
    box(ax, 0.435, y, 0.215, 0.14,
        'GATConv $1024 \\to 128\\!\\times\\!8$\nheads concatenated\nELU, $\\times\\,2$ layers', C_MP, bold=True)
    arrow(ax, 0.65, y + 0.07, 0.70, y + 0.07)
    box(ax, 0.70, y, 0.145, 0.14,
        'GATConv\n$1024 \\to 128$\n1 head, ELU', C_MP)
    arrow(ax, 0.845, y + 0.07, 0.868, y + 0.07)
    box(ax, 0.868, y, 0.117, 0.14, 'Mean pool\nLinear\n$128\\!\\to\\!64\\!\\to\\!1$', C_RD)
    ax.text(0.5425, y - 0.055,
            '$\\alpha_{vu}=\\mathrm{softmax}_u\\!\\left(\\mathrm{LeakyReLU}'
            '\\!\\left(a^{\\top}[Wh_v\\,\\Vert\\,Wh_u]\\right)\\right)$',
            fontsize=8.2, ha='center')
    hp_panel(ax, 0.02, 0.50, 0.96,
             ['selected: hidden nodes per head 128   |   attention heads 8   |   layers 3   |   '
              'parameters 2,285,057',
              'as submitted: hidden nodes per head 64   |   heads 4   |   layers 3   |   '
              'parameters 161,793',
              'layers 1-2 concatenate the heads, layer 3 uses a single head   |   '
              'dropout 0.2   |   pooling global mean'],
             'Hyperparameters (diagram shows the selected configuration)')
    # ------------------------------------------------------ (c) AttentiveFP
    ax = fig.add_subplot(gs[2]); panel_frame(ax, '(c)', 'AttentiveFP (primary model)')
    y = 0.70
    box(ax, 0.02, y, 0.16, 0.14,
        'Molecular\ngraph\n$x_v\\!\\in\\!\\mathbb{R}^{40},\\ e_{vu}\\!\\in\\!\\mathbb{R}^{6}$', C_IN, fs=8.0)
    arrow(ax, 0.18, y + 0.07, 0.225, y + 0.07)
    box(ax, 0.225, y, 0.135, 0.14, 'Atom\nembedding\n$40 \\to 300$', C_IN)
    arrow(ax, 0.36, y + 0.07, 0.405, y + 0.07)
    box(ax, 0.405, y, 0.225, 0.14,
        'Atom-level GRU\n+ graph attention\n(edge features used)\n$L = 3$ layers', C_MP, bold=True, fs=8.0)
    arrow(ax, 0.63, y + 0.07, 0.675, y + 0.07)
    box(ax, 0.675, y, 0.195, 0.14,
        'Super-node readout\nattention + GRU\n$T = 2$ timesteps', C_RD, bold=True, fs=8.0)
    arrow(ax, 0.87, y + 0.07, 0.893, y + 0.07)
    box(ax, 0.893, y, 0.092, 0.14, 'Linear\n$300\\!\\to\\!1$', C_HD)
    arrow(ax, 0.7725, y - 0.038, 0.5175, y - 0.038, rad=0.20)
    ax.text(0.645, y - 0.105, 'molecule embedding refined over $T$ timesteps',
            fontsize=7.3, ha='center', color='#30404f')
    hp_panel(ax, 0.02, 0.495, 0.96,
             ['selected (Table S5): hidden 300, graph attention layers $L$ = 3, '
              'readout timesteps $T$ = 2, dropout 0.2, parameters 2,635,201',
              'as submitted: hidden 200, $L$ = 2, $T$ = 2, dropout 0.2, parameters 895,001',
              'optimiser Adam, learning rate $10^{-3}$, weight decay $10^{-5}$, batch 64, '
              'ReduceLROnPlateau (patience 10, factor 0.5)',
              'early stopping on validation $R^{2}$ (patience 30, max 300 epochs); '
              'final prediction averages 30 fold models (3 seeds $\\times$ 10 folds)'],
             'Hyperparameters (diagram shows the selected configuration)')
    for ext in ('png', 'pdf'):
        p = os.path.join(FIG, 'fig_architectures.' + ext)
        fig.savefig(p, dpi=400 if ext == 'png' else None, bbox_inches='tight',
                    facecolor='white')
        print('wrote', p)


if __name__ == '__main__':
    main()
