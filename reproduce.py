"""Rebuild the results of the paper, stage by stage.

Run the whole pipeline with

    python reproduce.py

or a single stage with

    python reproduce.py --stage hpo

Stages run in the order listed by --list, because each consumes what the
previous one wrote. The GNN stages need a GPU to finish in reasonable time;
they also are not bit-wise reproducible on a GPU, which is why the trained
weights are distributed separately. Everything downstream of the weights
reproduces exactly from the shipped result files.
"""

import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))

STAGES = [
    ("data",      "code/01_data_processing.py",        "curate the dataset and build the 80:20 split"),
    ("baseline",  "code/00_baseline_sklearn.py",       "descriptor and fingerprint baselines"),
    ("train",     "code/03_train_random_split.py",    "train GCN, GAT and AttentiveFP (GPU)"),
    ("seeds",     "code/03b_train_extra_seeds.py",    "extra seeds for the 30-model ensemble (GPU)"),
    ("ensemble",  "code/03c_ensemble_predict.py",     "average the fold models on the test set"),
    ("holdout",   "code/04_train_scaffold_split.py",   "flavonoid class-holdout experiment (GPU)"),
    ("ad",        "code/05_applicability_domain.py",  "Williams plot and Tanimoto domain"),
    ("interpret", "code/06_interpretability.py",      "GNNExplainer atom importance"),
    ("space",     "code/07_chemical_space.py",         "UMAP and SMARTS categories"),
    ("hpo",       "revision_work/code/r1_hpo_search.py",  "90-configuration search (GPU)"),
    ("tuned",     "revision_work/code/r1_tuned_retrain.py","retrain at the selected settings (GPU)"),
    ("weights",   "revision_work/code/r1_save_tuned_weights.py","save the 30 released checkpoints (GPU)"),
    ("curve",     "revision_work/code/r1_learning_curve.py",  "accuracy against training-set size (GPU)"),
    ("blend",     "revision_work/code/r3_blend_cv.py",       "select the descriptor component out of fold"),
    ("hybrid",    "revision_work/code/r3_final_hybrid.py",    "final hybrid, test set predicted once"),
    ("hybrid-ad", "code/05_applicability_domain.py",          "refresh the Williams diagnostic for the final hybrid"),
    ("metrics",   "revision_work/code/r3_validation_metrics.py","Golbraikh-Tropsha and Roy statistics"),
    ("harvest",   "revision_work/code/r3_harvest_external.py","assemble the 196-compound set from ChEMBL"),
    ("external",  "revision_work/code/r3_external_predict.py", "predict the independent set"),
    ("outliers",  "revision_work/code/r1_outlier_analysis.py", "five largest errors with attributions"),
    ("tables",    "revision_work/code/make_tables.py",       "regenerate every LaTeX table"),
]


def run(script):
    path = os.path.join(HERE, script)
    if not os.path.exists(path):
        print("  missing, skipped: %s" % script)
        return 0
    print("\n=== %s ===" % script)
    return subprocess.run([sys.executable, path], cwd=HERE).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", action="append",
                    help="run only this stage; repeatable")
    ap.add_argument("--list", action="store_true",
                    help="list the stages and exit")
    a = ap.parse_args()
    if a.list:
        for name, script, what in STAGES:
            print("  %-10s %-45s %s" % (name, script, what))
        return 0
    todo = STAGES if not a.stage else [s for s in STAGES if s[0] in a.stage]
    if not todo:
        print("no such stage; use --list")
        return 1
    bad = 0
    for name, script, _ in todo:
        if run(script):
            print("stage %s failed" % name)
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
