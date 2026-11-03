# AntioxFP — graph neural networks for DPPH antioxidant activity

Code, data and trained models for:

> *Graph Neural Network Models for Predicting the Antioxidant Activity of
> Chemical Compounds: Interpretability Analysis and Applicability Domain
> Assessment.* Submitted to *Chemical Engineering Research and Design*
> (CHERD-D-26-01950).

A live predictor is available at
<https://huggingface.co/spaces/AI4deeperScience/AntioxFP> — paste a SMILES
string and it returns an ensemble pIC50 with a fold-level uncertainty and the
maximum Tanimoto similarity to the training set.

---

## What this model does, and where it stops

The model predicts pIC50 for DPPH radical scavenging (30 min incubation) from a
SMILES string. It was trained on 1,911 compounds and evaluated on a held-out
set of 383.

Two models are reported, and they are not interchangeable. The **hybrid**
(AttentiveFP ensemble averaged with a Mordred Extra-Trees model) reaches
R² = 0.779 but needs the full Mordred descriptor calculation. The **standalone
AttentiveFP ensemble** reaches R² = 0.756 from the structure alone. The live
predictor serves the AttentiveFP ensemble, so its numbers correspond to 0.756,
not 0.779.

It is reliable **for phenolic antioxidants that resemble the training data**.
Validation on 196 compounds taken from 120 independent publications showed a
median absolute error of 0.33 pIC50 units on phenolic compounds and 1.00 on
non-phenolic ones, and R² below zero over that set as a whole. Leverage-based
applicability-domain tests do **not** catch this; fingerprint similarity does.
Treat a prediction as untrustworthy when the maximum Tanimoto similarity to the
training set is below about 0.4, and see `data/external_predictions.csv` for the
evidence behind that threshold. The 0.4 cut-off was chosen by inspecting that
same validation set, so treat it as an exploratory guide rather than a
validated threshold.

Two further limits are worth stating. The curated set inherits its peptide
exclusion from the source database and that filter is incomplete, so twelve
short linear peptides remain; they are kept because removing them would break
comparability with the reference study. And a model trained without a whole
compound class cannot rank that class, which the flavonoid holdout in
`results/scaffold_split_results.json` shows directly.

---

## Layout

    code/                  the original pipeline, numbered in run order
    revision_work/code/    the analyses added during peer review
    data/                  curated dataset, independent set, predictions, HPO results
    results/               result files needed to rebuild tables and figures
    revision_work/results/ hyperparameter search, validation, outliers
    hf_space/              the web predictor
    reproduce.py           single entry point, runs the stages in order

The repository carries sources and result files, not outputs. Rendered figures
are not tracked. The plotting sources are included in `code/` and
`revision_work/code/`; the recorded numerical results are included alongside them.

Trained network weights are **not** tracked either, because they are large and
because GPU training is not bit-wise reproducible. They are attached to the
tagged release of this repository as
`antioxfp_tuned_attentivefp_weights.zip` (the 30 checkpoints the paper reports)
and `antioxfp_submitted_model_weights.zip` (the originally submitted
architecture, kept for reference). The one large file that is tracked is
`results/mordred_cache.npz`, so that the descriptor component of the hybrid
refits in under a minute without recomputing Mordred descriptors.

---

## Reproducing

    pip install -r requirements.txt
    python reproduce.py --list      # see the stages
    python reproduce.py             # run them all

Stages marked GPU take hours on CPU. No retraining is needed to inspect the
released predictions. To regenerate tables from the stored numerical results,
run `python reproduce.py --stage tables`. To rerun downstream
analyses, first unpack the tuned weights into
`revision_work/results/tuned_models/` and run

    python reproduce.py --stage data
    python reproduce.py --stage hybrid --stage metrics --stage external \
                        --stage outliers --stage tables

---

## Citing

See `CITATION.cff`. Cite the paper rather than this repository.

## Licence

MIT, see `LICENSE`. The independent validation set is derived from ChEMBL and
is redistributed under CC BY-SA 3.0.

The release at <https://github.com/Liubo-520/antioxide-predictor/releases/tag/v1.0>
contains the two checkpoint archives and SHA-256 checksums. Publication drafts,
reviewer correspondence, local environment files and development backups are
excluded from this repository.
