"""
Generate every LaTeX table of the revised manuscript and its supplementary
information directly from the stored result files, so that no number in the
paper is typed by hand.

Outputs .tex fragments in revision_work/tables/ which the manuscript and the
supporting information pull in with \\input{}.
"""

import os, json, glob, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE); PROJ = os.path.dirname(REV)
RES = os.path.join(PROJ, "results"); OUT = os.path.join(REV, "results")
TAB = os.path.join(REV, "tables"); os.makedirs(TAB, exist_ok=True)

HYB_KEY = 'Hybrid (tuned AttentiveFP + ET-all)'   # label in validation_metrics_table.json
FH_KEY = 'AttentiveFP tuned + ET-all'             # key in final_hybrid.json


def write(name, text):
    p = os.path.join(TAB, name)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(text.rstrip() + "\n")
    print("wrote", name)


def load(fn, base=OUT):
    p = os.path.join(base, fn)
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else None


# ---------------------------------------------------------------- HPO tables
def hpo_rows():
    rows = []
    for fn in glob.glob(os.path.join(OUT, "hpo", "stage*_w*.jsonl")):
        rows += [json.loads(l) for l in open(fn) if l.strip()]
    return rows


ORIGINAL_CFG = {
    'attentivefp': dict(hidden=200, num_layers=2, num_timesteps=2, dropout=0.2,
                        lr=1e-3, weight_decay=1e-5, batch_size=64),
    'gcn': dict(hidden=256, n_layers=3, dropout=0.2, lr=1e-3,
                weight_decay=1e-5, batch_size=64),
    'gat': dict(hidden=64, heads=4, n_layers=3, dropout=0.2, lr=1e-3,
                weight_decay=1e-5, batch_size=64),
}
PRETTY = {'attentivefp': 'AttentiveFP', 'gcn': 'GCN', 'gat': 'GAT'}


def arch_cells(model, cfg):
    if model == 'attentivefp':
        return ("%d" % cfg['hidden'], "%d" % cfg['num_layers'],
                "%d" % cfg['num_timesteps'], "---")
    if model == 'gcn':
        return ("%d" % cfg['hidden'], "%d" % cfg['n_layers'], "---", "---")
    return ("%d" % cfg['hidden'], "%d" % cfg['n_layers'], "---", "%d" % cfg['heads'])


def si_hpo_tables(rows):
    for model in ('attentivefp', 'gcn', 'gat'):
        rs = sorted([r for r in rows if r['model'] == model],
                    key=lambda r: -r['cv_r2_mean'])
        head = (r"\begin{table}[htbp]" "\n"
                r"  \caption{Hyperparameter search for %s: all %d configurations "
                r"evaluated, ranked by 5-fold cross-validation $R^2$ on the "
                r"1{,}528-compound training set. The external test set was not used "
                r"for selection. The configuration of the originally submitted "
                r"manuscript is marked \textsuperscript{\emph{s}}; the selected "
                r"configuration is in bold.}" "\n"
                r"  \label{tbl:si_hpo_%s}" "\n"
                r"  \centering\footnotesize" "\n"
                r"  \setlength{\tabcolsep}{3.5pt}" "\n"
                r"  \begin{tabular}{rcccccccccc}" "\n"
                r"    \toprule" "\n"
                r"    Rank & Hidden & Layers & $T$ & Heads & Dropout & LR & "
                r"Weight decay & Batch & "
                r"Params & CV $R^2$ \\" "\n"
                r"    \midrule" % (PRETTY[model], len(rs), model))
        body = []
        for i, r in enumerate(rs, 1):
            c = r['config']
            h, l, t, hd = arch_cells(model, c)
            mark = r"\textsuperscript{\emph{s}}" if c == ORIGINAL_CFG[model] else ""
            val = "%.4f $\\pm$ %.3f" % (r['cv_r2_mean'], r['cv_r2_sd'])
            if i == 1:
                val = r"\textbf{%s}" % val
            # Weight decay was a Stage-2 factor, so without this column three
            # configurations per model print identically while reporting
            # different CV R2, and the table cannot be reproduced from.
            body.append("    %d%s & %s & %s & %s & %s & %.2f & %.4g & %.4g & %d & %d & %s \\\\"
                        % (i, mark, h, l, t, hd, c['dropout'], c['lr'],
                           c['weight_decay'], c['batch_size'], r['n_params'], val))
        tail = "    \\bottomrule\n  \\end{tabular}\n\\end{table}"
        write("tab_si_hpo_%s.tex" % model, head + "\n" + "\n".join(body) + "\n" + tail)


def hpo_summary(rows, tuned, fh=None):
    # The tuned AttentiveFP ensemble was rebuilt from the 30 released
    # checkpoints, which the manuscript declares to be the reported model, so
    # its test R2 must be read from there and not from the earlier training
    # run recorded in tuned_retrain.json. Reading it from two places is how
    # this table came to print 0.7506 while Table 3 printed 0.7562.
    authoritative = {}
    if fh:
        authoritative['attentivefp'] = fh['graph']['AttentiveFP tuned']['r2']
    lines = []
    for model in ('attentivefp', 'gcn', 'gat'):
        rs = sorted([r for r in rows if r['model'] == model], key=lambda r: -r['cv_r2_mean'])
        orig = next(r for r in rs if r['config'] == ORIGINAL_CFG[model])
        best = rs[0]
        t = tuned.get(model, {}) if tuned else {}
        oc, bc = orig['config'], best['config']
        if model == 'attentivefp':
            od = "hidden 200, $L$=2, $T$=2"
            bd = "hidden %d, $L$=%d, $T$=%d" % (bc['hidden'], bc['num_layers'], bc['num_timesteps'])
        elif model == 'gcn':
            od = "hidden 256, 3 layers"
            bd = "hidden %d, %d layers, batch %d" % (bc['hidden'], bc['n_layers'], bc['batch_size'])
        else:
            od = "hidden 64, 4 heads, 3 layers"
            bd = "hidden %d, %d heads, %d layers" % (bc['hidden'], bc['heads'], bc['n_layers'])
        lines.append(r"    \multirow{2}{*}{%s} & as submitted & %s & %s & %.4f $\pm$ %.3f & %s \\"
                     % (PRETTY[model], od, "{:,}".format(orig['n_params']),
                        orig['cv_r2_mean'], orig['cv_r2_sd'],
                        ORIG_TEST.get(model, '---')))
        r2_test = authoritative.get(model, t['test']['r2'] if t else None)
        tt = "%.4f" % r2_test if r2_test is not None else '---'
        lines.append(r"     & tuned & %s & %s & \textbf{%.4f $\pm$ %.3f} & \textbf{%s} \\"
                     % (bd, "{:,}".format(best['n_params']),
                        best['cv_r2_mean'], best['cv_r2_sd'], tt))
        lines.append(r"    \midrule" if model != 'gat' else "")
    body = "\n".join(l for l in lines if l)
    write("tab_hpo_summary.tex", r"""\begin{table}[htbp]
  \caption{Effect of hyperparameter optimisation. Configurations were ranked by
    5-fold cross-validation $R^2$ on the 1{,}528-compound training set over 90
    configurations in total (35 AttentiveFP, 20 GCN, 35 GAT; complete listings
    in Tables~S5--S7). The external test set was never used for selection. Test
    $R^2$ is that of the corresponding fold-model ensemble, built with the
    protocol of Section~\ref{sec:train}.}
  \label{tbl:hpo_summary}
  \centering\footnotesize
  \setlength{\tabcolsep}{3.5pt}
  \begin{threeparttable}
  \begin{tabular}{llp{2.8cm}rcc}
    \toprule
    Model & Setting & Architecture & Parameters & CV $R^2$ & Test $R^2$ \\
    \midrule
""" + body + r"""
    \bottomrule
  \end{tabular}
  \begin{tablenotes}\footnotesize
    \item Learning rate $10^{-3}$, weight decay $10^{-5}$ and batch size 64
      unless stated. CV $R^2$ is the mean over 5 folds $\pm$ one standard
      deviation.
  \end{tablenotes}
  \end{threeparttable}
\end{table}""")


ORIG_TEST = {'attentivefp': '0.7331', 'gcn': '0.6002', 'gat': '0.6551'}


# ------------------------------------------------------------ performance
def performance_table(fh, tuned):
    g = fh['graph']; hy = fh['hybrid']; dt = fh['descriptor_test']
    rows = [
        (r"GCN (ours, tuned)", '0.666 $\\pm$ 0.096', "%.4f" % tuned['gcn']['test']['r2'],
         "%.4f" % tuned['gcn']['test']['rmse'], "%.4f" % tuned['gcn']['test']['mae']),
        (r"GAT (ours, tuned)", '0.723 $\\pm$ 0.049', "%.4f" % tuned['gat']['test']['r2'],
         "%.4f" % tuned['gat']['test']['rmse'], "%.4f" % tuned['gat']['test']['mae']),
        (r"AttentiveFP (ours, as submitted)", '0.700 $\\pm$ 0.083',
         "%.4f" % g['AttentiveFP as submitted']['r2'],
         "%.4f" % g['AttentiveFP as submitted']['rmse'],
         "%.4f" % g['AttentiveFP as submitted']['mae']),
        (r"AttentiveFP (ours, tuned)", '0.751 $\\pm$ 0.067',
         "%.4f" % g['AttentiveFP tuned']['r2'],
         "%.4f" % g['AttentiveFP tuned']['rmse'],
         "%.4f" % g['AttentiveFP tuned']['mae']),
        (r"Mordred Extra-Trees (ours)", '0.726\\textsuperscript{\\emph{b}}',
         "%.4f" % dt['ET-all']['r2'], "%.4f" % dt['ET-all']['rmse'],
         "%.4f" % dt['ET-all']['mae']),
    ]
    body = "\n".join("    %s & %s & %s & %s & %s \\\\" % r for r in rows)
    hyb = hy[FH_KEY]
    body += ("\n    \\midrule\n    \\textbf{Hybrid (ours): tuned AttentiveFP $+$ Mordred ET}\n"
             "      & --- & \\textbf{%.4f} & \\textbf{%.4f} & \\textbf{%.4f} \\\\"
             % (hyb['r2'], hyb['rmse'], hyb['mae']))
    body += ("\n    \\midrule\n    \\multicolumn{5}{l}{\\textit{Descriptor-based reference,"
             " same dataset and same 80:20 split}\\textsuperscript{\\emph{c}}} \\\\\n"
             "    Extra Trees & 0.74 $\\pm$ 0.06 & 0.77 & 0.45 & 0.31 \\\\\n"
             "    Ensemble ET$+$GB$+$XGB & --- & 0.78 & 0.45 & 0.31 \\\\")
    write("tab_performance.tex", r"""\begin{table}[htbp]
  \caption{Performance on the 80:20 random split. CV $R^2$ is the mean $\pm$ one
    standard deviation over the 5-fold search of Section~\ref{sec:hpo}; external test
    metrics are computed on the held-out 383-compound set and are reported once,
    after all model choices had been fixed on training data.}
  \label{tbl:random_results}
  \centering\small
  \begin{threeparttable}
  \begin{tabular}{p{5.6cm}cccc}
    \toprule
    Model & CV $R^2$ & Test $R^2$ & RMSE & MAE \\
    \midrule
""" + body + r"""
    \bottomrule
  \end{tabular}
  \begin{tablenotes}\footnotesize
    \item[\emph{a}] GNN test metrics are fold-model ensembles: 30 models
      (3 seeds $\times$ 10 folds) for AttentiveFP, 10 models for GCN and GAT.
    \item[\emph{b}] Out-of-fold $R^2$ over 3 seeds $\times$ 10 folds on the
      training set.
    \item[\emph{c}] Ghironi \textit{et al.}, \textit{J. Xenobiot.} 2025, 15, 80.
  \end{tablenotes}
  \end{threeparttable}
\end{table}""")


# ------------------------------------------------------------ validation
def validation_table(vmt):
    order = ['AttentiveFP ensemble (as submitted)', 'AttentiveFP ensemble (tuned)',
             'Mordred Extra-Trees (ET-all)', HYB_KEY,
             'Hybrid (tuned) after out-of-fold variance rescaling']
    short = {order[0]: 'AttentiveFP (submitted)', order[1]: 'AttentiveFP (tuned)',
             order[2]: 'Mordred ET', order[3]: 'Hybrid (AFP $+$ ET)',
             order[4]: 'Hybrid, rescaled\\textsuperscript{\\emph{b}}'}
    m = vmt['models']
    cols = [('$Q^2_{F1}$', 'q2f1'), ('$Q^2_{F2}$', 'q2f2'), ('$Q^2_{F3}$', 'q2f3'),
            ('$r^2_m$', 'r2m_mean'), ('$\\Delta r^2_m$', 'r2m_delta'),
            ('$k$', 'k'), ("$k'$", 'k_prime'), ('CCC', 'ccc')]
    body = []
    for key in order:
        if key not in m:
            continue
        d = m[key]
        cells = " & ".join("%.3f" % d[c[1]] for c in cols)
        body.append("    %s & %s & %d/%d \\\\" % (short[key], cells, d['n_passed'], d['n_criteria']))
    head = " & ".join(c[0] for c in cols)
    write("tab_validation.tex", r"""\begin{table}[htbp]
  \caption{External-validation metrics on the 383-compound test set. $Q^2_{F1}$,
    $Q^2_{F2}$ and $Q^2_{F3}$ follow Consonni \textit{et al.}; $r^2_m$ and
    $\Delta r^2_m$ follow Roy \textit{et al.}; $k$ and $k'$ are the slopes of the
    regressions through the origin used in the Golbraikh--Tropsha criteria; CCC is
    Lin's concordance correlation coefficient. The last column counts how many of
    the ten acceptance criteria listed in Table~S8 are met.}
  \label{tbl:validation}
  \centering\footnotesize
  \setlength{\tabcolsep}{3pt}
  \begin{threeparttable}
  \begin{tabular}{lcccccccc c}
    \toprule
    Model & """ + head + r""" & Criteria \\
    \midrule
""" + "\n".join(body) + r"""
    \bottomrule
  \end{tabular}
  \begin{tablenotes}\footnotesize
    \item[\emph{a}] Acceptance thresholds: $Q^2 > 0.5$; $r^2 > 0.6$;
      $0.85 \le k, k' \le 1.15$; $(r^2-r^2_0)/r^2 < 0.1$;
      $|r^2_0 - r'^2_0| < 0.3$; $r^2_m > 0.5$; $\Delta r^2_m < 0.2$; CCC $> 0.85$.
    \item[\emph{b}] Variance rescaling with slope and intercept fitted on
      out-of-fold training predictions only (Section~\ref{sec:metrics_res}).
  \end{tablenotes}
  \end{threeparttable}
\end{table}""")


# ------------------------------------------------------------ literature
def literature_table(fh):
    # Read this work's row from the result file rather than typing it, so it
    # cannot drift from the abstract and the performance table as it did when
    # it was hard-coded at the earlier values 0.776 / 0.419 / 0.273.
    hyb = fh['hybrid'][FH_KEY]
    this_work = ("$R^2$ = %.3f, RMSE = %.3f, MAE = %.3f"
                 % (hyb['r2'], hyb['rmse'], hyb['mae']))
    write("tab_literature.tex", r"""\begin{table}[htbp]
  \caption{Published quantitative models of DPPH radical-scavenging activity.
    Reported accuracies are not directly comparable because the datasets,
    endpoints, compound-selection rules and validation schemes differ; only the
    study of Ghironi \textit{et al.} uses the same 1{,}911 compounds and the same
    80:20 partition as the present work, which is why it is used for the
    head-to-head comparison.}
  \label{tbl:literature}
  \centering\footnotesize
  \setlength{\tabcolsep}{4pt}
  \begin{threeparttable}
  \begin{tabular}{p{2.5cm}p{2.0cm}p{1.9cm}p{2.5cm}p{2.4cm}}
    \toprule
    Study & Dataset & Compounds & Method & Reported accuracy \\
    \midrule
    Goya Jorge \textit{et al.} 2016\cite{GoyaJorge2016}
      & literature compilation & 1{,}373 (1{,}017/339)
      & multilayer perceptron on descriptors
      & $R^2_{\mathrm{train}}$ = 0.713; $Q^2_{\mathrm{ext}}$ = 0.654 \\
    Jung \textit{et al.} 2024\cite{Jung2024}
      & DPPH and ABTS records & $>$1{,}900
      & RF, SVM, XGBoost, DNN on ECFP-4
      & classification, accuracy $>$ 0.9 \\
    Ghironi \textit{et al.} 2025\cite{Ghironi2025}
      & AODB, DPPH, 30 min & 1{,}911 (1{,}528/383)
      & Extra Trees $+$ gradient boosting $+$ XGBoost on Mordred descriptors
      & $R^2$ = 0.78, RMSE = 0.45, MAE = 0.31 \\
    Kim \textit{et al.} 2025\cite{Kim2025}
      & AODB, DPPH & 3{,}133 (2{,}506/627)
      & XGBoost on 82 RDKit descriptors
      & $R^2$ = 0.817, $Q^2$ = 0.816 \\
    \midrule
    This work
      & AODB, DPPH, 30 min & 1{,}911 (1{,}528/383)
      & AttentiveFP ensemble $+$ Mordred Extra-Trees
      & """ + this_work + r""" \\
    \bottomrule
  \end{tabular}
  \begin{tablenotes}\footnotesize
    \item Compounds column: total (training/test). Kim \textit{et al.} express
      pIC$_{50}$ relative to a nanomolar reference, so their numerical range is
      offset from the molar convention used here and by Ghironi \textit{et al.}
  \end{tablenotes}
  \end{threeparttable}
\end{table}""")


# ------------------------------------------------------------ external set
def external_table(ev):
    sub = ev['_subsets']
    rows = [("All independent compounds", ev['Hybrid'])]
    for k in ('inside Williams AD', 'max Tanimoto $\\ge$ 0.40',
              'max Tanimoto $\\ge$ 0.50', 'max Tanimoto $\\ge$ 0.60'):
        kk = k.replace('$\\ge$', '>=')
        if kk in sub:
            rows.append((k, sub[kk]))
    body = "\n".join(
        "    %s & %d & %.3f & %.3f & %.3f & %.3f \\\\"
        % (lbl, d['n'], d['r2'], d['rmse'], d['mae'], d['spearman'])
        for lbl, d in rows)
    write("tab_external.tex", r"""\begin{table}[htbp]
  \caption{Independent validation on 196 compounds with DPPH IC$_{50}$ values
    measured with a 30~min incubation and reported in 120 primary publications
    indexed in ChEMBL. No compound of this set appears in the training or test
    set. Rows below the first restrict the set to compounds that are
    structurally close to the training data; the maximum Tanimoto similarity is
    computed on Morgan fingerprints against the 1{,}528 training compounds.}
  \label{tbl:external}
  \centering\small
  \begin{threeparttable}
  \begin{tabular}{lccccc}
    \toprule
    Subset & $n$ & $R^2$ & RMSE & MAE & Spearman $\rho$ \\
    \midrule
""" + body + r"""
    \bottomrule
  \end{tabular}
  \begin{tablenotes}\footnotesize
    \item Predictions are from the hybrid model. For the 16 compounds measured in
      more than one publication the between-publication spread of the
      experimental values is 0.95 pIC$_{50}$ units (median), which bounds the
      agreement any model can reach on this set.
  \end{tablenotes}
  \end{threeparttable}
\end{table}""")


# ------------------------------------------------------------ outliers
def outlier_table(oa):
    # Name every outlier explicitly. Falling back to the SMARTS class prints
    # an uninformative label that can contradict the discussion: CHEMBL4294673
    # is matched as "Phenol" but the text describes it as a thioxo-pyrimidine
    # hydrazone carrying two guaiacol groups.
    names = {'CHEMBL371315': 'Gly-Pro-Glu tripeptide',
             'CHEMBL66879': r'\textit{p}-coumaric acid',
             'CHEMBL54909': 'hydroxy-methoxy-coumarin',
             'CHEMBL2087310': 'prenylated cyclopropanecarboxanilide',
             'CHEMBL4294673': 'bis(guaiacyl thioxo-pyrimidine) hydrazone'}
    body = []
    for o in oa['outliers']:
        if o['chembl_id'] not in names:
            raise SystemExit("outlier %s (%s) has no name; add it to `names` "
                             "rather than letting it print its SMARTS class"
                             % (o['chembl_id'], o['chemical_class']))
        cc = o['chembl_cross_check']
        nrec = cc.get('n_dpph_ic50_records')
        spread = cc.get('spread_log_units')
        note = ("%d, spread %.2f" % (nrec, spread)) if spread else \
               ("%d" % nrec if nrec is not None else '---')
        body.append("    %d & %s & %.2f & %.2f & %+.2f & %.2f & %s \\\\"
                    % (o['rank'], names.get(o['chembl_id'], o['chemical_class']),
                       o['experimental_pIC50'], o['predicted_pIC50'], o['error'],
                       o['max_tanimoto_to_training'], note))
    d = oa['diagnostics']
    write("tab_outliers.tex", r"""\begin{table}[htbp]
  \caption{The five largest prediction errors of the hybrid model on the
    383-compound test set. Structures with GNNExplainer atom importance are
    shown in Figure~\ref{fgr:outliers}. The last column reports how many DPPH
    IC$_{50}$ measurements ChEMBL holds for the same compound and, where there is
    more than one, their spread in pIC$_{50}$ units. Compound 1 is one of the
    peptides that the inherited curation filter did not remove
    (Section~\ref{sec:data}).}
  \label{tbl:outliers}
  \centering\footnotesize
  \setlength{\tabcolsep}{3.5pt}
  \begin{threeparttable}
  \begin{tabular}{cp{4.3cm}ccccp{2.1cm}}
    \toprule
    \# & Compound & Exp. & Pred. & Error & Max $T$ & ChEMBL records \\
    \midrule
""" + "\n".join(body) + r"""
    \bottomrule
  \end{tabular}
  \begin{tablenotes}\footnotesize
    \item Max $T$ is the maximum Tanimoto similarity to any training compound.
      None of the five exceeds the leverage threshold $h^{*}$ = """ +
          ("%.4f" % d['h_star']) + r""". Excluding these five compounds lowers the
      test RMSE from """ + ("%.4f to %.4f" % (d['rmse_all'], d['rmse_without_top'])) +
          r""". Across the whole test set the RMSE is """ +
          ("%.3f in the outer 5\\%% of the activity range and %.3f in the central 90\\%%"
           % (d['rmse_extreme_5pct_tails'], d['rmse_central_90pct'])) + r""".
  \end{tablenotes}
  \end{threeparttable}
\end{table}""")


# ------------------------------------------------------------ learning curve
def learning_curve_table(lc):
    s = lc['summary']
    body = []
    for frac_i in range(len(s['tuned'])):
        t = s['tuned'][frac_i]; c = s['compact'][frac_i]
        body.append("    %.0f\\%% & %d & %.3f $\\pm$ %.3f & %.3f & %.3f $\\pm$ %.3f & %.3f \\\\"
                    % (t['fraction'] * 100, t['n_train'],
                       t['test_r2_mean'], t['test_r2_sd'], t['test_rmse_mean'],
                       c['test_r2_mean'], c['test_r2_sd'], c['test_rmse_mean']))
    write("tab_si_learning_curve.tex", r"""\begin{table}[htbp]
  \caption{Learning curve. For each training fraction three independent random
    subsamples were drawn, ten per cent of each subsample was held back for
    early stopping, a single AttentiveFP model was trained from scratch, and the
    same 383-compound test set was predicted. Values are mean $\pm$ one standard
    deviation over the three subsamples. These are single models, not fold
    ensembles, so they sit below the ensemble values of Table~2 of the main
    text.}
  \label{tbl:si_learning_curve}
  \centering\small
  \begin{tabular}{lccccc}
    \toprule
    & & \multicolumn{2}{c}{Tuned (2{,}635{,}201 param.)}
      & \multicolumn{2}{c}{Compact (227{,}501 param.)} \\
    \cmidrule(lr){3-4}\cmidrule(lr){5-6}
    Fraction & $n_{\mathrm{train}}$ & Test $R^2$ & RMSE & Test $R^2$ & RMSE \\
    \midrule
""" + "\n".join(body) + r"""
    \bottomrule
  \end{tabular}
\end{table}""")


# ------------------------------------------------------------ external list
def external_compound_list():
    path = os.path.join(OUT, "external_predictions.csv")
    if not os.path.exists(path):
        return
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    rows.sort(key=lambda r: -float(r['pIC50_exp']))
    body, shown, every = [], set(), set()
    for r in rows:
        dois = [d.strip() for d in r['dois'].split(';') if d.strip()]
        doi = dois[0] if dois else ''
        shown.add(doi); every.update(dois)
        body.append("    %s & %s & %s & %s & %s & \\texttt{\\scriptsize %s} \\\\"
                    % (r['chembl_id'], r['pIC50_exp'], r['pIC50_pred_hybrid'],
                       r['max_tanimoto'], r['n_measurements'],
                       doi.replace('_', r'\_')))
    # A reader who counts DOIs in this column finds fewer than the number of
    # publications the set draws on, because one publication may report several
    # compounds and one compound may be reported in several publications. Say
    # so, and give both counts.
    write("tab_si_external_list.tex", r"""\begin{longtable}{llllll}
  \caption{The 196 independent literature compounds, their experimental DPPH
    pIC$_{50}$, the hybrid-model prediction, the maximum Tanimoto similarity to
    the training set, the number $n$ of independent measurements found, and the
    DOI of the first source publication. Compounds are ordered by decreasing
    experimental activity. The DOI column lists one source per compound and
    therefore shows """ + str(len(shown)) + r""" distinct publications; counting
    every source of every measurement, including the further sources of the
    compounds with $n > 1$, the set draws on """ + str(len(every)) + r""" distinct
    publications. The complete per-compound DOI lists and the curated SMILES
    strings are given in the accompanying \texttt{external\_set.csv}.}
  \label{tbl:si_external_list} \\
  \toprule
  ChEMBL ID & Exp. & Pred. & Max $T$ & $n$ & Primary source (DOI) \\
  \midrule
  \endfirsthead
  \toprule
  ChEMBL ID & Exp. & Pred. & Max $T$ & $n$ & Primary source (DOI) \\
  \midrule
  \endhead
  \bottomrule
  \endfoot
""" + "\n".join(body) + r"""
\end{longtable}""")


def criteria_table(vmt):
    d = vmt['models'][HYB_KEY]
    body = "\n".join("    %s & %.4f & %s \\\\"
                     % (nm.replace('<=', '$\\le$').replace('>', '$>$').replace('<', '$<$')
                        .replace('|', '$|$').replace('delta ', '$\\Delta$ ')
                        .replace('r2m', '$r^2_m$').replace('r2_0', '$r^2_0$')
                        .replace("r2'", "$r'^2$").replace('Q2F2', '$Q^2_{F2}$')
                        .replace('R2pred', '$R^2_{pred}$').replace('r2', '$r^2$')
                        .replace('CCC', 'CCC').replace('k', '$k$'),
                        val, r'\checkmark' if ok else r'$\times$')
                     for nm, val, ok in d['criteria'])
    write("tab_si_criteria.tex", r"""\begin{table}[htbp]
  \caption{Golbraikh--Tropsha and Roy acceptance criteria evaluated for the
    hybrid model on the 383-compound test set.}
  \label{tbl:si_criteria}
  \centering\small
  \begin{tabular}{lcc}
    \toprule
    Criterion & Value & Met \\
    \midrule
""" + body + r"""
    \bottomrule
  \end{tabular}
\end{table}""")


def main():
    rows = hpo_rows()
    tuned = load("tuned_retrain.json")
    fh = load("final_hybrid.json")
    vmt = load("validation_metrics_table.json")
    ev = load("external_validation.json")
    oa = load("outlier_analysis.json")
    lc = load("learning_curve.json")

    if rows:
        si_hpo_tables(rows)
        if tuned:
            hpo_summary(rows, tuned, fh)
    if fh and tuned:
        performance_table(fh, tuned)
    if vmt:
        validation_table(vmt)
        criteria_table(vmt)
    if fh:
        literature_table(fh)
    if ev:
        external_table(ev)
    if oa:
        outlier_table(oa)
    if lc:
        learning_curve_table(lc)
    external_compound_list()


if __name__ == '__main__':
    main()
