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

Usage
-----
    python train_lgbm_ranker.py                                  # label_gain 0,1,3,7
    python train_lgbm_ranker.py --label-gain 0,1,10,100 --model_prefix gain100_
    python train_lgbm_ranker.py --query-by chunk --chunk-size 5000 --neg-frac 0.2
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from retrain_predict import (
    _build_training_frame, set_seed, custom_metric_raw,
    ID_COL, TARGET_COL, GROUP_COL, SEED,
    COLUMNS_TO_DROP, IMAGE_FEATURE_COLS, LGB_PARAMS, ARTIFACT_DIR, TOP_FEATURES_PATH,
)

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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_rankers(df, use_cols, label, args):
    """45 (n_seeds x 5) LGBMRankers. Returns the models and an (n_seeds, N) OOF score matrix."""
    params = {k: v for k, v in LGB_PARAMS.items() if k != "scale_pos_weight"}  # n/a to lambdarank
    params.update(objective="lambdarank", metric="ndcg", label_gain=args.label_gain)

    n = len(df)
    oof = np.full((args.n_seeds, n), np.nan)
    models = []

    with tqdm(total=args.n_seeds * N_FOLDS, desc="  lgbm-ranker") as pbar:
        for s in range(args.n_seeds):
            seed = s + 1
            rng = np.random.default_rng(seed)
            qids = make_query_ids(df, args.query_by, args.chunk_size, seed)
            skf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed)
            for tr_idx, va_idx in skf.split(df, y=label, groups=df[GROUP_COL]):
                # Keep every lesion with label > 0; subsample the plain-benign rows.
                keep = (label[tr_idx] > 0) | (rng.random(len(tr_idx)) < args.neg_frac)
                tr_idx = tr_idx[keep]
                tr_idx = tr_idx[np.argsort(qids[tr_idx], kind="stable")]  # groups must be contiguous

                tr_x = df.iloc[tr_idx][use_cols].reset_index(drop=True)
                for col in IMAGE_FEATURE_COLS:  # same light jitter as the original stack
                    tr_x[col] = tr_x[col] + np.random.normal(0, 0.1, size=tr_x.shape[0])

                ranker = lgb.LGBMRanker(**{**params, "random_state": seed})
                ranker.fit(tr_x, label[tr_idx], group=group_sizes(qids[tr_idx]))
                oof[s, va_idx] = ranker.predict(df.iloc[va_idx][use_cols])
                models.append(ranker)
                pbar.update(1)
    return models, oof


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def aggregate_scores(oof, method):
    """Combine per-seed OOF score vectors: raw mean, z-scored mean, or mean percentile rank."""
    if method == "mean":
        return oof.mean(axis=0)
    if method == "zscore":
        return ((oof - oof.mean(axis=1, keepdims=True)) / oof.std(axis=1, keepdims=True)).mean(axis=0)
    if method == "rank":
        return np.mean([pd.Series(row).rank(pct=True).values for row in oof], axis=0)
    raise ValueError(method)


def evaluate(oof, target, label):
    """Competition pAUC (any malignant) and melanoma-only pAUC, for single models and ensembles."""
    mel = (label == 3).astype(int)
    single = np.array([[custom_metric_raw(row, target), custom_metric_raw(row, mel)] for row in oof])
    print(f"\n  single model (per-seed OOF, n={len(oof)}):  "
          f"pAUC malignant {single[:, 0].mean():.5f} +/- {single[:, 0].std():.5f} | "
          f"pAUC melanoma {single[:, 1].mean():.5f} +/- {single[:, 1].std():.5f}")
    for method in ("mean", "zscore", "rank"):
        agg = aggregate_scores(oof, method)
        print(f"  ensemble [{method:>6}]:               "
              f"pAUC malignant {custom_metric_raw(agg, target):.5f} | "
              f"pAUC melanoma {custom_metric_raw(agg, mel):.5f}")


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
    ap.add_argument("--neg-frac", type=float, default=0.1,
                    help="Fraction of label-0 training rows kept (labels 1-3 are always kept).")
    ap.add_argument("--n-seeds", type=int, default=9, help="CV repeats (x5 folds = models).")
    ap.add_argument("--artifact-dir", default=Path(ARTIFACT_DIR) / "ranker", help="Where models/OOF scores are saved.")
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

    models, oof = train_rankers(df, use_cols, label, args)
    evaluate(oof, target, label)

    out_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = args.model_prefix
    joblib.dump(models, out_dir / f"{pre}ranker_models.joblib")
    joblib.dump({"feature_cols": feature_cols, "use_cols": use_cols, "cat_cols": onehot_cols,
                 "columns_to_drop": COLUMNS_TO_DROP, "scaling": scaling, "encoder": encoder,
                 "label_gain": args.label_gain}, out_dir / f"{pre}ranker_meta.joblib")
    oof_df = df[[ID_COL, GROUP_COL, TARGET_COL]].assign(label=label)
    for s in range(len(oof)):
        oof_df[f"oof_seed{s + 1}"] = oof[s]
    oof_df.to_parquet(out_dir / f"{pre}ranker_oof.parquet", index=False)
    print(f"Saved {len(models)} rankers, metadata and OOF scores to {out_dir}")


if __name__ == "__main__":
    main()
