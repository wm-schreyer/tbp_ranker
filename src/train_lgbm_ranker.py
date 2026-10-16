"""
train_lgbm_ranker.py
====================
Train and evaluate a LightGBM LambdaRank ensemble as an alternative to the binary-
classification GBDT stack in `retrain_predict.py` (which is left untouched).

Relevance labels (built from train-metadata.csv):
    0  benign lesion                                  target == 0
    1  benign lesion chosen for dermoscopy            target == 0 and non-blank lesion_id
    2  malignant, not melanoma                        target == 1 and 'melanoma' not in iddx_3
    3  confirmed melanoma                             target == 1 and 'melanoma' in iddx_3

Training mirrors the original stack's CV scheme: for each of `--n-seeds` seeds, a
StratifiedGroupKFold(5) split grouped by patient (no patient leakage), one LGBMRanker per
fold. Every row is predicted out-of-fold once per seed, so the OOF scores are used to
evaluate single models and the seed-averaged ensemble (three ways of averaging).

Each ranker also gets a Platt calibrator (1-feature logistic regression on its held-out
fold's scores vs. the binary malignant target), putting every model on a common scale
before averaging.

Usage
-----
    python train_lgbm_ranker.py                                  # label_gain 0,1,3,7
    python train_lgbm_ranker.py --label-gain 0,1,10,100 --model_prefix gain100_
    python train_lgbm_ranker.py --query-by chunk --chunk-size 5000 --neg-frac 0.2
    python train_lgbm_ranker.py --objective rank_xendcg
    python train_lgbm_ranker.py --neg-per-group 50 --truncation-level 30 --eval-at 15
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from retrain_predict import (
    _build_training_frame, set_seed, custom_metric_raw,
    ID_COL, TARGET_COL, GROUP_COL, SEED,
    COLUMNS_TO_DROP, IMAGE_FEATURE_COLS, LGB_PARAMS, ARTIFACT_DIR, TOP_FEATURES_PATH,
)

# LightGBM >= 4.7 deprecates eval_set in favour of eval_X/eval_y; older versions only accept eval_set.
warnings.filterwarnings("ignore", message=".*'eval_set' is deprecated.*")

N_FOLDS = 5
DEFAULT_LABEL_GAIN = "0,1,3,7"


# ---------------------------------------------------------------------------
# Labels and query groups
# ---------------------------------------------------------------------------
def build_relevance_labels(df):
    """Return the 0-3 relevance label for every row (see module docstring)."""
    target = df[TARGET_COL].astype(int).values
    has_lesion_id = df["lesion_id"].fillna("").astype(str).str.strip().ne("").values
    is_melanoma = df["iddx_3"].fillna("").astype(str).str.contains("melanoma", case=False).values

    label = np.zeros(len(df), dtype=int)
    label[(target == 0) & has_lesion_id] = 1
    label[target == 1] = 2
    label[(target == 1) & is_melanoma] = 3
    return label


def make_query_ids(df, query_by, chunk_size, seed):
    """Query id per row. 'patient': one query per patient. 'chunk': whole patients are
    shuffled and packed into queries of ~chunk_size lesions, so scores are compared
    across patients (closer to the global pAUC metric) while patients stay intact."""
    patients = df[GROUP_COL].values
    if query_by == "patient":
        return pd.factorize(patients)[0]
    counts = pd.Series(patients).value_counts()
    order = np.random.default_rng(seed).permutation(len(counts))
    chunk_of_patient = pd.Series(np.cumsum(counts.values[order]) // chunk_size, index=counts.index[order])
    return chunk_of_patient.loc[patients].values


def group_sizes(sorted_qids):
    """Consecutive group sizes for LightGBM, given query ids already sorted."""
    return np.unique(sorted_qids, return_counts=True)[1]


def subsample_train(tr_idx, label, qids, rng, args):
    """Keep every label > 0 row and thin the label-0 rows: at most --neg-per-group random
    zeros per query if set, otherwise a random --neg-frac fraction of them."""
    zero = label[tr_idx] == 0
    if args.neg_per_group > 0:
        order = pd.Series(rng.random(zero.sum())).groupby(qids[tr_idx][zero]).rank(method="first").values
        keep = np.ones(len(tr_idx), dtype=bool)
        keep[np.flatnonzero(zero)[order > args.neg_per_group]] = False
    else:
        keep = ~zero | (rng.random(len(tr_idx)) < args.neg_frac)
    return tr_idx[keep]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_rankers(df, use_cols, label, target, args):
    """n_seeds x 5 LGBMRankers, each with a Platt calibrator fit on its held-out fold.
    Returns models, calibrators, per-model final validation NDCG@k, and (n_seeds, N) OOF
    raw scores and calibrated logits."""
    params = {k: v for k, v in LGB_PARAMS.items() if k != "scale_pos_weight"}  # n/a to lambdarank
    params.update(objective=args.objective, metric="ndcg", label_gain=args.label_gain)
    if args.objective == "lambdarank":  # truncation applies to lambdarank only
        params["lambdarank_truncation_level"] = args.truncation_level
    ndcg_key = f"ndcg@{args.eval_at}"

    n = len(df)
    oof = np.full((args.n_seeds, n), np.nan)
    oof_logit = np.full((args.n_seeds, n), np.nan)
    models, calibrators, val_ndcg = [], [], []

    with tqdm(total=args.n_seeds * N_FOLDS, desc="  lgbm-ranker") as pbar:
        for s in range(args.n_seeds):
            seed = s + 1
            rng = np.random.default_rng(seed)
            qids = make_query_ids(df, args.query_by, args.chunk_size, seed)
            skf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed)
            for tr_idx, va_idx in skf.split(df, y=label, groups=df[GROUP_COL]):
                tr_idx = subsample_train(tr_idx, label, qids, rng, args)
                tr_idx = tr_idx[np.argsort(qids[tr_idx], kind="stable")]  # groups must be contiguous
                va_idx = va_idx[np.argsort(qids[va_idx], kind="stable")]  # validation uses FULL groups

                tr_x = df.iloc[tr_idx][use_cols].reset_index(drop=True)
                for col in IMAGE_FEATURE_COLS:  # same light jitter as the original stack
                    tr_x[col] = tr_x[col] + np.random.normal(0, 0.1, size=tr_x.shape[0])
                va_x = df.iloc[va_idx][use_cols]

                ranker = lgb.LGBMRanker(**{**params, "random_state": seed})
                ranker.fit(tr_x, label[tr_idx], group=group_sizes(qids[tr_idx]),
                           eval_set=[(va_x, label[va_idx])], eval_group=[group_sizes(qids[va_idx])],
                           eval_at=[args.eval_at])  # eval_at only sets the reported NDCG@k
                val_ndcg.append(ranker.evals_result_["valid_0"][ndcg_key][-1])

                score = ranker.predict(va_x)
                cal = LogisticRegression(C=1e6).fit(score.reshape(-1, 1), target[va_idx])
                oof[s, va_idx] = score
                oof_logit[s, va_idx] = cal.decision_function(score.reshape(-1, 1))
                models.append(ranker)
                calibrators.append(cal)
                pbar.update(1)
    return models, calibrators, np.array(val_ndcg), oof, oof_logit


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
METHODS = ("mean", "zscore", "rank", "platt_logit", "platt_prob")


def aggregate_scores(oof, oof_logit, method):
    """Combine per-seed OOF score vectors: raw mean, z-scored mean, mean percentile rank,
    or the mean of Platt-calibrated logits / probabilities."""
    if method == "mean":
        return oof.mean(axis=0)
    if method == "zscore":
        return ((oof - oof.mean(axis=1, keepdims=True)) / oof.std(axis=1, keepdims=True)).mean(axis=0)
    if method == "rank":
        return np.mean([pd.Series(row).rank(pct=True).values for row in oof], axis=0)
    if method == "platt_logit":
        return oof_logit.mean(axis=0)
    if method == "platt_prob":
        return (1.0 / (1.0 + np.exp(-oof_logit))).mean(axis=0)
    raise ValueError(method)


def topk_recall(scores, patients, label, k):
    """Share of malignant (label >= 2) and melanoma (label 3) lesions that land in their
    patient's top-k by score, ranking within each patient's FULL set of lesions."""
    rank_in_patient = pd.Series(scores).groupby(patients).rank(ascending=False, method="first").values
    in_top = rank_in_patient <= k
    return in_top[label >= 2].mean(), in_top[label == 3].mean()


def evaluate(oof, oof_logit, patients, target, label, k):
    """Competition pAUC (any malignant), melanoma-only pAUC and per-patient top-k recall,
    for single models and for each ensemble averaging method."""
    mel = (label == 3).astype(int)

    def row(scores):
        rec = topk_recall(scores, patients, label, k)
        return [custom_metric_raw(scores, target), custom_metric_raw(scores, mel), *rec]

    single = np.array([row(r) for r in oof])
    m, sd = single.mean(axis=0), single.std(axis=0)
    print(f"\n  {'':22s}{'pAUC malig':>12s}{'pAUC mel':>12s}{f'top{k} malig':>13s}{f'top{k} mel':>12s}")
    print(f"  {'single model (avg)':22s}" + "".join(f"{v:>12.4f}" for v in m[:2]) + "".join(f"{v:>13.4f}" if i == 0 else f"{v:>12.4f}" for i, v in enumerate(m[2:])))
    for method in METHODS:
        r = row(aggregate_scores(oof, oof_logit, method))
        print(f"  {'ensemble ' + method:22s}" + "".join(f"{v:>12.4f}" for v in r[:2]) + "".join(f"{v:>13.4f}" if i == 0 else f"{v:>12.4f}" for i, v in enumerate(r[2:])))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train/evaluate a LightGBM LambdaRank ensemble on ISIC-2024.")
    ap.add_argument("--label-gain", default=DEFAULT_LABEL_GAIN,
                    help="Comma-separated gains for labels 0,1,2,3 (e.g. 0,1,3,7 or 0,1,10,100).")
    ap.add_argument("--query-by", choices=["patient", "chunk"], default="patient",
                    help="Query groups: one per patient, or random whole-patient chunks of --chunk-size lesions.")
    ap.add_argument("--chunk-size", type=int, default=5000, help="Approx. lesions per query for --query-by chunk.")
    ap.add_argument("--neg-frac", type=float, default=1.0,
                    help="Fraction of label-0 training rows kept (labels 1-3 are always kept).")
    ap.add_argument("--neg-per-group", type=int, default=0,
                    help="If > 0, keep at most this many label-0 rows per query (overrides --neg-frac).")
    ap.add_argument("--objective", choices=["lambdarank", "rank_xendcg"], default="lambdarank",
                    help="rank_xendcg is cheaper and has no truncation level; as I recall LightGBM then "
                         "ignores --label-gain for training (it still uses it for the NDCG metric).")
    ap.add_argument("--truncation-level", type=int, default=30,
                    help="lambdarank_truncation_level: pairs are formed only if one item is in the model's "
                         "current top-N for its query. This is what focuses training on the top of each group.")
    ap.add_argument("--eval-at", type=int, default=15,
                    help="k for the reported validation NDCG@k and the per-patient top-k recall.")
    ap.add_argument("--n-seeds", type=int, default=9, help="CV repeats (x5 folds = models).")
    ap.add_argument("--artifact-dir", default=Path(ARTIFACT_DIR), help="Where models/OOF scores are saved.")
    ap.add_argument("--model_prefix", type=str, default="", help="Name prepended to saved files.")
    ap.add_argument("--lof-by-site", action="store_true", help="Compute per-anatomic-site LOF scores.")
    ap.add_argument("--top-features-path", type=Path, default=TOP_FEATURES_PATH,
                    help=f"Per-site top-features JSON (default: {TOP_FEATURES_PATH}).")
    args = ap.parse_args()
    args.label_gain = [float(g) for g in args.label_gain.split(",")]
    if len(args.label_gain) != 4:
        ap.error("--label-gain needs exactly 4 comma-separated values (labels 0-3).")

    set_seed(SEED)
    with open(args.top_features_path, "r") as f:
        top_features_dict = json.load(f)

    df, feature_cols, onehot_cols, encoder, scaling = _build_training_frame(args.lof_by_site, top_features_dict)
    use_cols = [c for c in feature_cols if c not in COLUMNS_TO_DROP]

    label = build_relevance_labels(df)
    print("  [labels] counts:", dict(zip(*np.unique(label, return_counts=True))))
    target = df[TARGET_COL].astype(int).values

    models, calibrators, val_ndcg, oof, oof_logit = train_rankers(df, use_cols, label, target, args)
    print(f"\n  validation NDCG@{args.eval_at} per model (full groups): {val_ndcg.mean():.4f} +/- {val_ndcg.std():.4f}")
    evaluate(oof, oof_logit, df[GROUP_COL].values, target, label, args.eval_at)

    out_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = args.model_prefix
    joblib.dump(models, out_dir / f"{pre}ranker_models.joblib")
    joblib.dump(calibrators, out_dir / f"{pre}ranker_calibrators.joblib")
    joblib.dump({"feature_cols": feature_cols, "use_cols": use_cols, "cat_cols": onehot_cols,
                 "columns_to_drop": COLUMNS_TO_DROP, "scaling": scaling, "encoder": encoder,
                 "label_gain": args.label_gain}, out_dir / f"{pre}ranker_meta.joblib")
    oof_df = df[[ID_COL, GROUP_COL, TARGET_COL]].assign(label=label)
    for s in range(len(oof)):
        oof_df[f"oof_seed{s + 1}"] = oof[s]
        oof_df[f"oof_logit_seed{s + 1}"] = oof_logit[s]
    oof_df.to_parquet(out_dir / f"{pre}ranker_oof.parquet", index=False)
    print(f"Saved {len(models)} rankers, calibrators, metadata and OOF scores to {out_dir}")


if __name__ == "__main__":
    main()
