"""
retrain_predict_hdf5.py
===============
HDF5 variant of `predict.py` with new outlier factor scoring.

Identical pipeline and cached GBDT artifacts as `retrain_predict.py`, but the images are
read from an HDF5 file instead of individual JPEG files. This matches the original
`first_place_winner_script.py` data contract:

    * the metadata CSV has an `isic_id` column (plus `patient_id` and the usual
      ISIC tabular columns; `target` optional);
    * the HDF5 file contains one dataset per image, keyed by `isic_id`, whose value
      is the raw JPEG byte string -> decoded with PIL via BytesIO.

All the heavy lifting (feature engineering, LOF, the cached CatBoost/LGBM/XGB stack,
the ensemble and the metric) is imported unchanged from `predict.py`; only the image
loader differs. Train the GBDT stack once with either script:

    python retrain_predict.py --train-only          # or: python retrain_predict_hdf5.py --train-only

Usage
-----
    python retrain_predict_hdf5.py --input meta.csv --images images.hdf5 --artifacts  --output preds.csv
"""

import argparse
import gc
from io import BytesIO
from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

# Reuse every shared piece of the pipeline from predict.py.
import retrain_predict
from retrain_predict import (
    ISICModel, ISICModelEdgnet, _build_transforms, add_patient_norm,
    read_data, get_lof_score_pat_norm, get_lof_bysite_pat_norm, fill_inf, rank_average, 
    custom_metric_raw, ensure_gbdt_artifacts, ensemble_predict,
    MODELS_DIR, EVA_MODEL_NAME, EVA_IMG_SIZE, EDG_MODEL_NAME, EDG_IMG_SIZE,
    cat_cols_base, ID_COL, TARGET_COL, GROUP_COL, TOP_FEATURES_PATH
)

# ---------------------------------------------------------------------------
# HDF5-backed image dataset (mirrors the original ISICDataset reader)
# ---------------------------------------------------------------------------
class HDF5Dataset(Dataset):
    """Reads JPEG bytes stored in an HDF5 file under keys equal to each `isic_id`."""

    def __init__(self, isic_ids, hdf5_path, transforms):
        self.isic_ids = list(isic_ids)
        self.hdf5_path = str(hdf5_path)
        self.transforms = transforms
        self._fp = None  # opened lazily, once per worker process (h5py isn't fork-safe)

    def __len__(self):
        return len(self.isic_ids)

    def _file(self):
        if self._fp is None:
            self._fp = h5py.File(self.hdf5_path, mode="r")
        return self._fp

    def __getitem__(self, index):
        isic_id = self.isic_ids[index]
        raw = self._file()[isic_id][()]
        img = np.array(Image.open(BytesIO(raw)).convert("RGB"))
        img = self.transforms(image=img)["image"]
        return {"image": img}


@torch.inference_mode()
def _infer_hdf5(model, isic_ids, hdf5_path, img_size, device, batch_size=64, num_workers=4):
    """Run one image model over the given isic_ids (read from HDF5). Returns a numpy
    array shaped [N] for sigmoid heads, [N, C] for the softmax head."""
    model.eval()
    loader = DataLoader(
        HDF5Dataset(isic_ids, hdf5_path, _build_transforms(img_size)),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True,
    )
    out = []
    for data in tqdm(loader, total=len(loader), leave=False):
        images = data["image"].to(device, dtype=torch.float)
        out.append(model(images).squeeze().cpu().numpy())
    return np.concatenate(out)


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


def get_lof_bysite(
        dataframe, 
        top_features_dict, 
        output_col_name="of", 
        patient_col="patient_id",
        site_col="anatom_site_general",
        k=15,
        null=0.0
        ):
    """
    Calculates a Local Outlier Factor score for TBP tiles using features dervied from VECTRA lesion analyzer measurements.
    Arguments:
        - Dataframe with lesion analyzer measurements
        - List of features to use for LOF calculation
        - Name of output score column
        - Groupby levels for background tiles, can be str or list
        - Minimum # of samples for valid LOF score
    """
    group_keys = [patient_col, site_col]
    outlier_factors = []

    for (patient_id, site), idx in dataframe.groupby(group_keys, observed=True).indices.items():
        row_ids = dataframe.index.values[idx]

        # No matching site key: return null LOF score for all rows in this group
        if site not in top_features_dict:
            outlier_factors.append(
                pd.DataFrame({
                    "isic_id": row_ids,
                    output_col_name: null,
                    #f"exception_{output_col_name}": "no_site_key",
                })
            )
            continue

        top_site_features = top_features_dict[site]

        # LOF cannot handle NaN/inf. Median-impute the feature matrix first; this does not occur
        # when the data is already clean.
        lof_feat = dataframe[top_site_features].replace([np.inf, -np.inf], np.nan)
        lof_feat = lof_feat.fillna(lof_feat.median()).fillna(0.0)
        scaled_array = np.nan_to_num(StandardScaler().fit_transform(lof_feat))

        # Insufficient tiles: return null LOF score
        if len(idx) <= k:
            outlier_factors.append(
                pd.DataFrame({
                    "isic_id": row_ids,
                    output_col_name: null,
                    #f"exception_{output_col_name}": "insufficient_tiles",
                })
            )
            continue

        # Valid group: compute LOF
        #clf = LocalOutlierFactor(n_neighbors=min(35, len(idx)))
        clf = LocalOutlierFactor(n_neighbors=k)
        clf.fit_predict(scaled_array[idx])
        outlier_factors.append(
            pd.DataFrame({
                "isic_id": row_ids,
                output_col_name: clf.negative_outlier_factor_,
                #f"exception_{output_col_name}": None,
            })
        ) 

    # Edge case: no groups processed at all
    if len(outlier_factors) == 0:
        dataframe[output_col_name] = null
        #dataframe[f"exception_{output_col_name}"] = "no_valid_groups"
        return dataframe
    
    # Stitch all groups back together and merge onto original dataframe
    result = pd.concat(outlier_factors, ignore_index=True)
    dataframe = dataframe.merge(
        result.set_index("isic_id"),
        how="left",
        left_index=True,
        right_index=True,
    )

    # Catch any rows that fell through the merge (e.g. null site values)
    # and fill with null so the GBDT stack never sees a null
    dataframe[output_col_name] = dataframe[output_col_name].fillna(null)
    #dataframe[f"exception_{output_col_name}"] = dataframe[f"exception_{output_col_name}"].fillna("null_site")

    return dataframe


def predict_hdf5(
        input_csv, 
        hdf5_path, 
        output_csv, 
        artifacts, 
        device, 
        batch_size=64, 
        num_workers=4,
        lof_by_site=False,
        lof_null=0.0,
        features_dict=None,
        ensemble="rank",
        cnn_cache=None,
        refresh_cnn_cache=False
        ):
    feature_cols = artifacts["feature_cols"]
    columns_to_drop = artifacts["columns_to_drop"]
    encoder = artifacts["encoder"]
    onehot_cols = artifacts["cat_cols"]

    raw = pd.read_csv(input_csv)
    for required in (ID_COL, GROUP_COL):
        if required not in raw.columns:
            raise ValueError(f"Input CSV must contain a '{required}' column.")
    if not Path(hdf5_path).exists():
        raise FileNotFoundError(f"HDF5 image file not found: {hdf5_path}")
    has_target = TARGET_COL in raw.columns

    # Verify every isic_id has a matching dataset in the HDF5 file.
    with h5py.File(hdf5_path, "r") as fp:
        keys = set(fp.keys())
    missing = [i for i in raw[ID_COL].astype(str) if i not in keys]
    if missing:
        raise KeyError(f"{len(missing)} isic_id(s) missing from HDF5 (e.g. {missing[:5]}).")

    # --- tabular features ------------------------------------------------------
    print("Engineering tabular features ...")
    df = read_data(input_csv).reset_index()
    df[onehot_cols] = encoder.transform(df[cat_cols_base])
    df[onehot_cols] = df[onehot_cols].astype("category")

    print("Computing LOF ...")
    df = df.set_index(ID_COL)
    if lof_by_site:
        # restrict LOF scoring to lesions within body site
        df = get_lof_bysite_pat_norm(
            df,
            top_features_dict=features_dict,
            output_col_name="of",
            patient_col="patient_id",
            site_col="anatom_site_general",
            null=lof_null
        )
    else:
        df = get_lof_score_pat_norm(df, null=lof_null)
    df = df.reset_index()

    # --- image features (neural nets, read from HDF5) --------------------------
    print("Running image models ...")
    img = raw[[ID_COL, GROUP_COL]].copy()
    cnn_cache = cnn_cache or Path(input_csv).with_name(f"{Path(input_csv).stem}_cnn_outputs.parquet")
    cnn = load_or_compute_cnn_outputs(raw[ID_COL].tolist(), cnn_cache, hdf5_path, device, batch_size,
                                      num_workers, refresh_cnn_cache)
    img = image_features_from_cnn(img, cnn, artifacts["scaling"])
    image_cols = [c for c in img.columns if c not in (ID_COL, GROUP_COL)]
    df = df.merge(img[[ID_COL] + image_cols], how="left", on=ID_COL)

    # --- assemble model matrix -------------------------------------------------
    df = fill_inf(df, feature_cols)
    use_cols = [c for c in feature_cols if c not in columns_to_drop]
    missing_cols = [c for c in use_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Engineered frame is missing expected feature columns: {missing_cols[:10]} ...")
    X = df[use_cols]

    # --- ensemble --------------------------------------------------------------
    print("Predicting with GBDT stack ...")
    predictions = ensemble_predict(artifacts, X, ensemble)

    out = pd.DataFrame({ID_COL: df[ID_COL].values, "prediction": predictions})
    out.to_csv(output_csv, index=False)
    print(f"Wrote {len(out)} predictions to {output_csv}")

    if has_target:
        merged = out.merge(raw[[ID_COL, TARGET_COL]], on=ID_COL, how="left")
        score = custom_metric_raw(merged["prediction"].values, merged[TARGET_COL].values)
        print(f"Partial AUC (pAUC above 80% TPR): {score:.5f}")

    return out


def main():
    ap = argparse.ArgumentParser(description="ISIC-2024 1st-place inference (HDF5 images + CSV metadata).")
    ap.add_argument("--input", help="Metadata CSV (needs isic_id, patient_id, tabular columns).")
    ap.add_argument("--images", help="HDF5 file of images keyed by isic_id (JPEG bytes).")
    ap.add_argument("--output", default="predictions.csv", help="Where to write the prediction CSV.")
    ap.add_argument("--train-only", action="store_true", help="Only build/refresh cached GBDT artifacts.")
    ap.add_argument("--force-retrain", action="store_true", help="Retrain the GBDT stack even if cached.")
    ap.add_argument("--artifact-dir", default=None, help="Directory where trained GBDT .joblib files are stored.")
    ap.add_argument("--model_prefix", type=str, help="Name prepended to trained model, if desired (e.g. 'new_feature_')")
    ap.add_argument("--catboost-device", choices=["gpu", "cpu"], default="gpu",
                    help="CatBoost training device (auto-falls back to CPU on GPU failure).")
    ap.add_argument("--batch-size", type=int, default=64, help="Image inference batch size.")
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    ap.add_argument("--lof-by-site", action="store_true",
                    help="Also compute per-anatomic-site LOF scores (requires 'anatom_site_general' "
                         "in metadata and a top_features_by_location.json file).")
    ap.add_argument("--lof-null", type=float, default=0.0)
    ap.add_argument("--top-features-path", type=Path, default=TOP_FEATURES_PATH,
                    help=f"Path to the per-site top-features JSON (default: {TOP_FEATURES_PATH}).")
    ap.add_argument("--ensemble", choices=["rank", "calibrated"], default="rank",
                    help="How to combine GBDT models: percentile-rank average (default) or average of "
                         "per-model Platt-calibrated probabilities.")
    ap.add_argument("--cnn-cache", type=Path, default=None,
                    help="Parquet cache of raw CNN outputs, computed if missing and loaded if present "
                         "(default: <input>_cnn_outputs.parquet).")
    ap.add_argument("--refresh-cnn-cache", action="store_true", help="Recompute CNN outputs even if cached.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | count: {torch.cuda.device_count()}")

    # load site specific top features
    with open(args.top_features_path, "r") as file:
        top_features_dict = json.load(file)
    
    arti_dir = Path(args.artifact_dir) # Path object for appending with '/'
    
    artifacts = ensure_gbdt_artifacts(
        force_retrain=args.force_retrain, 
        catboost_device=args.catboost_device,
        artifact_dir=arti_dir,
        model_prefix=args.model_prefix,
        lof_by_site=args.lof_by_site,
        lof_null=args.lof_null,
        features_dict=top_features_dict,
        )

    if args.train_only:
        print("train-only mode: artifacts ready, exiting.")
        return
    if not args.input or not args.images:
        ap.error("--input and --images are both required unless --train-only is given.")

    predict_hdf5(
        args.input, 
        args.images, 
        args.output, 
        artifacts, 
        device, 
        args.batch_size, 
        args.num_workers,
        lof_by_site=args.lof_by_site,
        lof_null=args.lof_null,
        features_dict=top_features_dict,
        ensemble=args.ensemble,
        cnn_cache=args.cnn_cache,
        refresh_cnn_cache=args.refresh_cnn_cache
        )
    print("All done!")


if __name__ == "__main__":
    main()
