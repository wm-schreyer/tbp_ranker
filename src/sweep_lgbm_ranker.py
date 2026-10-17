"""
sweep_lgbm_ranker.py
====================
Compare LambdaRank configurations on the out-of-fold (OOF) training scores: several
`--min-group-label` values x several ensemble sizes (seed counts).

The training frame is built once. For each min-group-label, models are trained once at the
largest seed count; smaller ensembles are the first n seeds of that run (seeds are
independent), so no extra training is needed. CV folds depend only on the seed, so configs
are compared on identical splits. Results are appended to a CSV after every config.

Usage
-----
    python sweep_lgbm_ranker.py --min-group-labels 0,1,2 --seed-counts 1,3,5,9
    python sweep_lgbm_ranker.py --min-group-labels 0,2 --seed-counts 3,5 --neg-per-group 50 --output sweep.csv

Note: LightGBM runs on CPU as configured in LGB_PARAMS (device="cpu"); a GPU is not used.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import train_lgbm_ranker as T
from retrain_predict import (
    _build_training_frame, set_seed, custom_metric_raw,
    TARGET_COL, GROUP_COL, SEED, COLUMNS_TO_DROP, TOP_FEATURES_PATH,
)


def int_list(text):
    return [int(x) for x in text.split(",")]


def main():
    ap = argparse.ArgumentParser(description="Sweep min-group-label x ensemble size for the LambdaRank ensemble.")
    ap.add_argument("--min-group-labels", type=int_list, default=[0, 1, 2],
                    help="Comma-separated --min-group-label values to compare.")
    ap.add_argument("--seed-counts", type=int_list, default=[1, 3, 5, 9],
                    help="Comma-separated ensemble sizes in seeds (5 models each); the max is trained.")
    ap.add_argument("--methods", default="mean,platt_logit", help="Comma-separated averaging methods to report.")
    ap.add_argument("--output", default="ranker_sweep.csv", help="Results CSV (appended to after each config).")
    # Training options, same meaning as in train_lgbm_ranker.py.
    ap.add_argument("--label-gain", default=T.DEFAULT_LABEL_GAIN)
    ap.add_argument("--objective", choices=["lambdarank", "rank_xendcg"], default="lambdarank")
    ap.add_argument("--query-by", choices=["patient", "chunk"], default="patient")
    ap.add_argument("--chunk-size", type=int, default=5000)
    ap.add_argument("--neg-frac", type=float, default=0.25)
    ap.add_argument("--neg-per-group", type=int, default=0)
    ap.add_argument("--truncation-level", type=int, default=30)
    ap.add_argument("--eval-at", type=int, default=15)
    ap.add_argument("--lof-by-site", action="store_true")
    ap.add_argument("--top-features-path", type=Path, default=TOP_FEATURES_PATH)
    args = ap.parse_args()
    args.label_gain = [float(g) for g in args.label_gain.split(",")]
    if len(args.label_gain) != 4:
        ap.error("--label-gain needs exactly 4 comma-separated values.")
    methods = args.methods.split(",")
    args.n_seeds = max(args.seed_counts)

    set_seed(SEED)
    with open(args.top_features_path, "r") as f:
        top_features_dict = json.load(f)
    df, feature_cols, _, _, _ = _build_training_frame(args.lof_by_site, top_features_dict)
    use_cols = [c for c in feature_cols if c not in COLUMNS_TO_DROP]
    label = T.build_relevance_labels(df)
    target = df[TARGET_COL].astype(int).values
    patients = df[GROUP_COL].values
    mel = (label == 3).astype(int)

    out_path = Path(args.output)
    rows = []
    for mgl in args.min_group_labels:
        print(f"\n=== min_group_label={mgl}: training {args.n_seeds * T.N_FOLDS} models ===")
        args.min_group_label = mgl
        _, _, val_ndcg, oof, oof_logit = T.train_rankers(df, use_cols, label, target, args)
        new = []
        for n in args.seed_counts:
            for method in methods:
                scores = T.aggregate_scores(oof[:n], oof_logit[:n], method)
                rec_mal, rec_mel = T.topk_recall(scores, patients, label, args.eval_at)
                new.append({
                    "min_group_label": mgl, "seeds": n, "models": n * T.N_FOLDS, "method": method,
                    "pauc_malignant": custom_metric_raw(scores, target),
                    "pauc_melanoma": custom_metric_raw(scores, mel),
                    f"top{args.eval_at}_malignant": rec_mal, f"top{args.eval_at}_melanoma": rec_mel,
                    f"val_ndcg@{args.eval_at}": val_ndcg.mean(),
                })
        rows += new
        pd.DataFrame(new).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + res.round(4).to_string(index=False))
    print(f"\nResults appended to {out_path}")


if __name__ == "__main__":
    main()
