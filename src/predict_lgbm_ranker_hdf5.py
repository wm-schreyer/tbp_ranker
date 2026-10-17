"""
predict_lgbm_ranker_hdf5.py
===========================
Score and evaluate new data with the LightGBM LambdaRank ensemble trained by
`train_lgbm_ranker.py`. Mirrors `retrain_predict_hdf5.py`: images come from an HDF5 file
keyed by `isic_id`, tabular metadata from a CSV, and the same feature pipeline (tabular
features, LOF, EVA/EdgeNeXt image features) is run before the rankers.

Needs the artifacts written by train_lgbm_ranker.py in --artifact-dir:
    {prefix}ranker_models.joblib, {prefix}ranker_calibrators.joblib, {prefix}ranker_meta.joblib

The slow part is the CNN inference (EVA02 3-class 'old set' model, EVA02 5 folds, EdgeNeXt 5
folds). Their RAW outputs are cached per isic_id in a parquet (--cnn-cache, default
`<input>_cnn_outputs.parquet`): computed and saved when missing, loaded when present. Lesions
absent from an existing cache are computed and appended. Scaling and patient-normalisation are
applied after loading, so the cache stays valid across rankers; delete it (or use
--refresh-cnn-cache) if the CNN weights or the images change.

Output CSV: isic_id, patient_id, `prediction` (the --ensemble you chose) and one column per
averaging method (mean, zscore, rank, platt_logit, platt_prob). Notes:
  * zscore and rank are computed over THIS input file, so they depend on the dataset scored;
    mean / platt_logit / platt_prob are per-lesion and do not.
  * If the CSV has a `target` column, pAUC and per-patient top-k recall are printed for every
    method. Melanoma metrics also need `lesion_id` and `iddx_3`.

Usage
-----
    python predict_lgbm_ranker_hdf5.py --input meta.csv --images images.hdf5 \\
        --model_prefix gain100_ --output ranker_preds.csv
"""
import argparse
import gc
import json
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import torch

from retrain_predict import (
    ISICModel, ISICModelEdgnet, add_patient_norm, read_data, get_lof_score_pat_norm,
    get_lof_bysite_pat_norm, fill_inf, custom_metric_raw,
    MODELS_DIR, EVA_MODEL_NAME, EVA_IMG_SIZE, EDG_MODEL_NAME, EDG_IMG_SIZE,
    cat_cols_base, ID_COL, TARGET_COL, GROUP_COL, ARTIFACT_DIR, TOP_FEATURES_PATH,
)
from retrain_predict_hdf5 import _infer_hdf5
from train_lgbm_ranker import METHODS, aggregate_scores, build_relevance_labels, topk_recall


def load_ranker_artifacts(artifact_dir, prefix):
    artifact_dir = Path(artifact_dir)
    models = joblib.load(artifact_dir / f"{prefix}ranker_models.joblib")
    calibrators = joblib.load(artifact_dir / f"{prefix}ranker_calibrators.joblib")
    meta = joblib.load(artifact_dir / f"{prefix}ranker_meta.joblib")
    print(f"Loaded {len(models)} rankers from {artifact_dir}")
    return models, calibrators, meta


def _run_cnn(model, ckpt, ids, hdf5_path, img_size, device, batch_size, num_workers):
    """Load one checkpoint, run it over `ids`, free it. (_infer_hdf5 squeezes, so 1 id is padded.)"""
    model = model.to(device)
    model.load_state_dict(torch.load(MODELS_DIR / ckpt, weights_only=True))
    padded = ids if len(ids) > 1 else ids * 2
    out = _infer_hdf5(model, padded, hdf5_path, img_size, device, batch_size, num_workers)[:len(ids)]
    model.to("cpu"); del model; gc.collect(); torch.cuda.empty_cache()
    return out


def compute_raw_cnn_outputs(ids, hdf5_path, device, batch_size, num_workers):
    """Raw (unscaled) outputs of the CNNs: old_0..2, eva_0..4, edg_0..4, one row per isic_id."""
    out = {ID_COL: list(ids)}
    print("  [image] EVA02 3-class 'old set' model ...")
    preds3 = _run_cnn(ISICModel(EVA_MODEL_NAME, pretrained=False, num_classes=3), "ema_small_pretrained",
                      ids, hdf5_path, EVA_IMG_SIZE, device, batch_size, num_workers)
    for k in range(3):
        out[f"old_{k}"] = preds3[:, k]
    print("  [image] EVA02 5 folds ...")
    for fold in range(5):
        out[f"eva_{fold}"] = _run_cnn(ISICModel(EVA_MODEL_NAME, pretrained=False, num_classes=1), f"eva_model__{fold}",
                                      ids, hdf5_path, EVA_IMG_SIZE, device, batch_size, num_workers)
    print("  [image] EdgeNeXt 5 folds ...")
    for fold in range(5):
        out[f"edg_{fold}"] = _run_cnn(ISICModelEdgnet(EDG_MODEL_NAME, pretrained=False, num_classes=1), f"edg_model__{fold}",
                                      ids, hdf5_path, EDG_IMG_SIZE, device, batch_size, num_workers)
    return pd.DataFrame(out)


def load_or_compute_cnn_outputs(ids, cache_path, hdf5_path, device, batch_size, num_workers, refresh=False):
    """Raw CNN outputs for `ids` (in that order): from the parquet cache where present, computed
    (and appended to the cache) for any ids that are missing."""
    cache_path = Path(cache_path)
    cache = pd.read_parquet(cache_path) if cache_path.exists() and not refresh else None
    have = set(cache[ID_COL]) if cache is not None else set()
    missing = [i for i in ids if i not in have]
    if missing:
        print(f"  [cnn cache] computing {len(missing)} of {len(ids)} lesions ...")
        new = compute_raw_cnn_outputs(missing, hdf5_path, device, batch_size, num_workers)
        cache = new if cache is None else pd.concat([cache, new], ignore_index=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache.to_parquet(cache_path, index=False)
        print(f"  [cnn cache] saved {len(cache)} lesions to {cache_path}")
    else:
        print(f"  [cnn cache] loaded all {len(ids)} lesions from {cache_path}")
    return cache.drop_duplicates(ID_COL).set_index(ID_COL).loc[ids].reset_index()


def image_features_from_cnn(df, cnn, scaling):
    """Same features as compute_image_features_hdf5, built from raw CNN outputs: old_set_0..2 (+ _m),
    predictions_eva (+ _m), predictions_edg (+ _m). `df` has isic_id and patient_id."""
    cnn = cnn.set_index(ID_COL).loc[df[ID_COL]]
    for k in range(3):
        df[f"old_set_{k}"] = cnn[f"old_{k}"].values
        df = add_patient_norm(df, f"old_set_{k}", f"old_set_{k}_m")
    for name, key in (("eva", "predictions_eva"), ("edg", "predictions_edg")):
        scaled = [(cnn[f"{name}_{f}"].values - scaling[f"{name}_mean"]) / scaling[f"{name}_std"] for f in range(5)]
        df[key] = np.mean(scaled, axis=0)
        df = add_patient_norm(df, key, f"{key}_m")
    return df


def build_feature_matrix(input_csv, hdf5_path, meta, device, batch_size, num_workers, lof_by_site, features_dict,
                         cnn_cache, refresh_cnn_cache=False):
    """Same feature pipeline as predict_hdf5. Returns (frame, X) with X = the model columns."""
    raw = pd.read_csv(input_csv)
    for required in (ID_COL, GROUP_COL):
        if required not in raw.columns:
            raise ValueError(f"Input CSV must contain a '{required}' column.")
    if not Path(hdf5_path).exists():
        raise FileNotFoundError(f"HDF5 image file not found: {hdf5_path}")
    with h5py.File(hdf5_path, "r") as fp:
        keys = set(fp.keys())
    missing = [i for i in raw[ID_COL].astype(str) if i not in keys]
    if missing:
        raise KeyError(f"{len(missing)} isic_id(s) missing from HDF5 (e.g. {missing[:5]}).")

    print("Engineering tabular features ...")
    df = read_data(input_csv).reset_index()
    onehot_cols = meta["cat_cols"]
    df[onehot_cols] = meta["encoder"].transform(df[cat_cols_base])
    df[onehot_cols] = df[onehot_cols].astype("category")

    print("Computing LOF ...")
    df = df.set_index(ID_COL)
    if lof_by_site:
        df = get_lof_bysite_pat_norm(df, top_features_dict=features_dict, output_col_name="of",
                                     patient_col="patient_id", site_col="anatom_site_general")
    else:
        df = get_lof_score_pat_norm(df)
    df = df.reset_index()

    print("Running image models ...")
    img = raw[[ID_COL, GROUP_COL]].copy()
    cnn = load_or_compute_cnn_outputs(raw[ID_COL].tolist(), cnn_cache, hdf5_path, device, batch_size,
                                      num_workers, refresh_cnn_cache)
    img = image_features_from_cnn(img, cnn, meta["scaling"])
    image_cols = [c for c in img.columns if c not in (ID_COL, GROUP_COL)]
    df = df.merge(img[[ID_COL] + image_cols], how="left", on=ID_COL)

    df = fill_inf(df, meta["feature_cols"])
    use_cols = meta["use_cols"]
    missing_cols = [c for c in use_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Engineered frame is missing expected feature columns: {missing_cols[:10]} ...")
    return df, df[use_cols]


def score_models(models, calibrators, X):
    """(n_models, N) raw ranker scores and Platt-calibrated logits."""
    scores = np.vstack([m.predict(X) for m in models])
    logits = np.vstack([c.decision_function(s.reshape(-1, 1)) for c, s in zip(calibrators, scores)])
    return scores, logits


def report_metrics(df, ensembles, k):
    """pAUC + per-patient top-k recall for every averaging method, when labels are available."""
    target = df[TARGET_COL].astype(int).values
    patients = df[GROUP_COL].values
    full_labels = {"lesion_id", "iddx_3"} <= set(df.columns)
    label = build_relevance_labels(df) if full_labels else target * 2  # without iddx_3: malignant only
    mel = (label == 3).astype(int)

    print(f"\n  {'':16s}{'pAUC malig':>12s}{'pAUC mel':>12s}{f'top{k} malig':>13s}{f'top{k} mel':>12s}")
    for method, scores in ensembles.items():
        rec_mal, rec_mel = topk_recall(scores, patients, label, k)
        pauc_mal = custom_metric_raw(scores, target)
        pauc_mel = custom_metric_raw(scores, mel) if full_labels and mel.any() else float("nan")
        print(f"  {method:16s}{pauc_mal:>12.4f}{pauc_mel:>12.4f}{rec_mal:>13.4f}"
              f"{(rec_mel if full_labels else float('nan')):>12.4f}")


def main():
    ap = argparse.ArgumentParser(description="Score/evaluate new data with the LightGBM LambdaRank ensemble.")
    ap.add_argument("--input", required=True, help="Metadata CSV (isic_id, patient_id, tabular columns; target optional).")
    ap.add_argument("--images", required=True, help="HDF5 file of images keyed by isic_id (JPEG bytes).")
    ap.add_argument("--output", default="ranker_predictions.csv", help="Where to write the prediction CSV.")
    ap.add_argument("--artifact-dir", default=Path(ARTIFACT_DIR) / "ranker", help="Directory with the ranker artifacts.")
    ap.add_argument("--model_prefix", type=str, default="", help="Prefix used when the rankers were trained.")
    ap.add_argument("--ensemble", choices=METHODS, default="platt_logit",
                    help="Averaging method written to the `prediction` column (all methods are saved).")
    ap.add_argument("--cnn-cache", type=Path, default=Path(ARTIFACT_DIR) / "test_cnn_outputs.parquet",
                    help="Parquet cache of raw CNN outputs (default: <artifact_dir>/test_cnn_outputs.parquet).")
    ap.add_argument("--refresh-cnn-cache", action="store_true", help="Recompute CNN outputs even if cached.")
    ap.add_argument("--eval-at", type=int, default=15, help="k for per-patient top-k recall.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lof-by-site", action="store_true",
                    help="Per-anatomic-site LOF; must match how the rankers were trained.")
    ap.add_argument("--top-features-path", type=Path, default=TOP_FEATURES_PATH)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    with open(args.top_features_path, "r") as f:
        top_features_dict = json.load(f)

    cnn_cache = args.cnn_cache
    models, calibrators, meta = load_ranker_artifacts(args.artifact_dir, args.model_prefix)
    df, X = build_feature_matrix(args.input, args.images, meta, device, args.batch_size, args.num_workers,
                                 args.lof_by_site, top_features_dict, cnn_cache, args.refresh_cnn_cache)

    print(f"Scoring {len(X)} lesions with {len(models)} rankers ...")
    scores, logits = score_models(models, calibrators, X)
    ensembles = {m: aggregate_scores(scores, logits, m) for m in METHODS}

    out = df[[ID_COL, GROUP_COL]].copy()
    out["prediction"] = ensembles[args.ensemble]
    for m in METHODS:
        out[m] = ensembles[m]
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} predictions to {args.output} (prediction = {args.ensemble})")

    if TARGET_COL in df.columns:
        report_metrics(df, ensembles, args.eval_at)
    print("All done!")


if __name__ == "__main__":
    main()
