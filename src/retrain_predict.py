"""
retrain_predict.py
==========
Slimmed, reusable inference pipeline for the ISIC-2024 1st-place solution.

This is a refactor of `first_place_winner_script.py`. The original script mixed
*training* and *inference* in one flat top-to-bottom run and retrained 135 gradient
-boosted models every time it produced a prediction. This version splits the work
into two clearly separated phases:

  PHASE 1  ensure_gbdt_artifacts()
  -------------------------------
  Train the CatBoost / LightGBM / XGBoost stack ONCE from `train-metadata.csv`
  plus the precomputed out-of-fold (OOF) parquet files, then cache everything
  (the 135 fitted models, the one-hot encoder, the final feature list and the
  image-score scaling stats) under `skin-models-base/predict_artifacts/`.
  Subsequent runs detect the cache and skip straight to prediction.

  NOTE: the neural-network image models are NOT needed for training. The training
  side's image features come entirely from the OOF parquets
  (`oof_forecasts_eva.parquet`, `oof_forecasts_edgenext_base.parquet`) and
  `old_data_model_forecast.parquet`.

  PHASE 2  predict()
  ------------------
  Given a CSV describing new images (a `file_path` column pointing at each JPEG,
  plus the usual ISIC metadata columns and `patient_id` / `isic_id`), run the
  image models on the new images, engineer the tabular features, and produce one
  malignancy probability per image. If the input CSV also carries a `target`
  column, the partial-AUC competition metric is printed.

Usage
-----
    # Train artifacts (only needed once) and predict on a new CSV:
    python predict.py --input /path/to/new_data.csv --output predictions.csv

    # Force a full retrain of the GBDT stack:
    python predict.py --input new_data.csv --force-retrain

    # Only build/refresh the cached GBDT artifacts, no prediction:
    python predict.py --train-only

The new-data CSV is expected to contain (at minimum):
    file_path, isic_id, patient_id, and the ISIC tabular columns used as features
    (age_approx, sex, anatom_site_general, all the tbp_lv_* columns, ...).
    A `target` column is optional; if present, metrics are reported.
"""

import argparse
import gc
import os
import random
from pathlib import Path

import h5py  # noqa: F401  (kept for parity / optional HDF5 inputs)
import joblib
import json
import numpy as np
import pandas as pd
import polars as pl
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import lightgbm as lgb
import catboost as cb
import xgboost as xgb

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Paths & global constants
# ---------------------------------------------------------------------------
# Root of the original Kaggle data (used only when training the GBDT stack).
ROOT = Path("/home/gillism/Programs/ISIC Challenge 2024 Winners/kaggle/")
PROJECT = Path("/home/gillism/Programs/ISIC Challenge 2024 Winners/1st Place New/")

TRAIN_META_PATH = Path("/fast_data/schreym/data/raw/slice3d/train-metadata.csv") # read from /fast_data

# Directory holding the neural-net weights + OOF parquets shipped with the model.
MODELS_DIR = PROJECT / "skin-models-base"
# Where the trained GBDT stack (and the encoder / metadata) get cached.
ARTIFACT_DIR = MODELS_DIR / "predict_artifacts"
# Pre-computed top LightGBM features per body site
TOP_FEATURES_PATH = "/home/schreym/isic2024/top_features_by_location.json"

# Parquet inputs required to TRAIN the GBDT stack (image features for the train set).
OOF_EVA_PATH = MODELS_DIR / "oof_forecasts_eva.parquet"
OOF_EDG_PATH = MODELS_DIR / "oof_forecasts_edgenext_base.parquet"
OLD_PREDS_TRAIN_PATH = PROJECT / "old_data_model_forecast.parquet"

ID_COL = "isic_id"
TARGET_COL = "target"
GROUP_COL = "patient_id"

ERR = 1e-5            # small constant guarding divisions / std normalisation
SAMPLING_RATIO = 0.01  # RandomUnderSampler target ratio (matches original)
SEED = 42

# EVA02 image model (used for both the 3-class "old set" model and the 5 folds).
EVA_MODEL_NAME = "eva02_small_patch14_336.mim_in22k_ft_in1k"
EVA_IMG_SIZE = 336
# EdgeNeXt image model.
EDG_MODEL_NAME = "edgenext_base.in21k_ft_in1k"
EDG_IMG_SIZE = 256

# ImageNet-style normalisation used during the original training (keep identical).
NORM_MEAN = [0.4815, 0.4578, 0.4082]
NORM_STD = [0.2686, 0.2613, 0.2758]


# ---------------------------------------------------------------------------
# Feature column definitions (verbatim from the original script)
# ---------------------------------------------------------------------------
num_cols = [
    "age_approx", "clin_size_long_diam_mm", "tbp_lv_A", "tbp_lv_Aext", "tbp_lv_B",
    "tbp_lv_Bext", "tbp_lv_C", "tbp_lv_Cext", "tbp_lv_H", "tbp_lv_Hext", "tbp_lv_L",
    "tbp_lv_Lext", "tbp_lv_areaMM2", "tbp_lv_area_perim_ratio", "tbp_lv_color_std_mean",
    "tbp_lv_deltaA", "tbp_lv_deltaB", "tbp_lv_deltaL", "tbp_lv_deltaLB",
    "tbp_lv_deltaLBnorm", "tbp_lv_eccentricity", "tbp_lv_minorAxisMM",
    "tbp_lv_nevi_confidence", "tbp_lv_norm_border", "tbp_lv_norm_color",
    "tbp_lv_perimeterMM", "tbp_lv_radial_color_std_max", "tbp_lv_stdL",
    "tbp_lv_stdLExt", "tbp_lv_symm_2axis", "tbp_lv_symm_2axis_angle", "tbp_lv_x",
    "tbp_lv_y", "tbp_lv_z",
]

new_num_cols = [
    "lesion_size_ratio", "lesion_shape_index", "hue_contrast", "luminance_contrast",
    "lesion_color_difference", "border_complexity", "color_uniformity",
    "position_distance_3d", "perimeter_to_area_ratio", "area_to_perimeter_ratio",
    "lesion_visibility_score", "symmetry_border_consistency", "consistency_symmetry_border",
    "color_consistency", "consistency_color", "size_age_interaction",
    "hue_color_std_interaction", "lesion_severity_index", "shape_complexity_index",
    "color_contrast_index", "log_lesion_area", "normalized_lesion_size",
    "mean_hue_difference", "std_dev_contrast", "color_shape_composite_index",
    "lesion_orientation_3d", "overall_color_difference", "symmetry_perimeter_interaction",
    "comprehensive_lesion_index", "color_variance_ratio", "border_color_interaction",
    "border_color_interaction_2", "size_color_contrast_ratio",
    "age_normalized_nevi_confidence", "age_normalized_nevi_confidence_2",
    "color_asymmetry_index", "volume_approximation_3d", "color_range",
    "shape_color_consistency", "border_length_ratio", "age_size_symmetry_index",
    "index_age_size_symmetry",
]

cat_cols_base = [
    "sex", "anatom_site_general", "tbp_tile_type", "tbp_lv_location",
    "tbp_lv_location_simple", "attribution",
]
norm_cols = [f"{c}_patient_norm" for c in num_cols + new_num_cols]
special_cols = ["count_per_patient", "tbp_lv_areaMM2_patient", "tbp_lv_areaMM2_bp"]

# Columns the winning model intentionally drops from the GBDT inputs.
COLUMNS_TO_DROP = [
    "tbp_lv_B", "tbp_lv_C", "tbp_lv_H", "tbp_lv_L", "tbp_lv_radial_color_std_max",
    "tbp_lv_y", "tbp_lv_z", "luminance_contrast", "lesion_color_difference",
    "normalized_lesion_size", "tbp_lv_norm_border_patient_norm",
    "lesion_color_difference_patient_norm",
    "age_normalized_nevi_confidence_2_patient_norm", "tbp_lv_deltaA",
]

# Image-derived columns that get a little Gaussian jitter during GBDT training
# (a light regularisation trick from the original solution).
IMAGE_FEATURE_COLS = ["predictions_edg", "predictions_edg_m", "predictions_eva", "predictions_eva_m"]


# ---------------------------------------------------------------------------
# Gradient-boosted model hyper-parameters (verbatim from the original)
# ---------------------------------------------------------------------------
LGB_PARAMS = {
    "objective": "binary", "verbosity": -1, "n_iter": 200, "boosting_type": "gbdt",
    "device": "cpu", "random_state": SEED,
    "lambda_l1": 0.08758718919397321, "lambda_l2": 0.0039689175176025465,
    "learning_rate": 0.03231007103195577, "max_depth": 4, "num_leaves": 103,
    "colsample_bytree": 0.8329551585827726, "colsample_bynode": 0.4025961355653304,
    "bagging_fraction": 0.7738954452473223, "bagging_freq": 4,
    "min_data_in_leaf": 85, "scale_pos_weight": 2.7984184778875543,
}

XGB_PARAMS = {
    "enable_categorical": True, "tree_method": "hist", "random_state": SEED,
    "learning_rate": 0.08501257473292347, "lambda": 8.879624125465703,
    "alpha": 0.6779926606782505, "max_depth": 6, "subsample": 0.6012681388711075,
    "colsample_bytree": 0.8437772277074493, "colsample_bylevel": 0.5476090898823716,
    "colsample_bynode": 0.9928601203635129, "scale_pos_weight": 3.29440313334688,
    "device": "cpu",
}

# Best CatBoost params (line 1114 of the original). NOTE: in the original,
# `random_strength` was tuned but silently dropped (the pydantic config had no such
# field) so it never reached CatBoost. We keep that behaviour for exact parity and
# do NOT pass random_strength.
CB_PARAMS = {
    "iterations": 2000,
    "learning_rate": 0.02606161517843435,
    "l2_leaf_reg": 18.04422276698195,
    "depth": 6,
    "bagging_temperature": 0.8735940473548339,
    "border_count": 256,
    "grow_policy": "Lossguide",
    "min_data_in_leaf": 38,
    "loss_function": "Logloss",
}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(random_seed=SEED):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ===========================================================================
# Neural-network image models
# ===========================================================================
class ISICModel(nn.Module):
    """EVA02 backbone with a fresh linear head. num_classes=1 -> sigmoid,
    num_classes>1 -> softmax (used by the 3-class 'old set' model)."""

    def __init__(self, model_name, num_classes=1, pretrained=False, checkpoint_path=None):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,   # fixed: original had a typo 'heckpoint_path'
        )
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)
        # dim=1 is explicit (the original used a bare nn.Softmax() which warns).
        self.activation = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)

    def forward(self, images):
        return self.activation(self.model(images))


class ISICModelEdgnet(nn.Module):
    """EdgeNeXt backbone (timm builds the classifier head internally)."""

    def __init__(self, model_name, num_classes=1, pretrained=False):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes, global_pool="avg"
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        return self.sigmoid(self.model(images))


class ImagePathDataset(Dataset):
    """Loads images from file paths (new-data contract: a `file_path` column).

    Mirrors the original HDF5 dataset's preprocessing: open the image, convert to
    an RGB numpy array, then apply the albumentations transform.
    """

    def __init__(self, file_paths, transforms):
        self.file_paths = list(file_paths)
        self.transforms = transforms

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        img = np.array(Image.open(self.file_paths[index]).convert("RGB"))
        img = self.transforms(image=img)["image"]
        return {"image": img}


def _build_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD, max_pixel_value=255.0, p=1.0),
        ToTensorV2(),
    ], p=1.0)


@torch.inference_mode()
def _infer(model, file_paths, img_size, device, batch_size=64, num_workers=4):
    """Run one image model over `file_paths`, returning a numpy array of outputs
    (shape [N] for sigmoid heads, [N, C] for the softmax head)."""
    model.eval()
    loader = DataLoader(
        ImagePathDataset(file_paths, _build_transforms(img_size)),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True,
    )
    out = []
    for data in tqdm(loader, total=len(loader), leave=False):
        images = data["image"].to(device, dtype=torch.float)
        out.append(model(images).squeeze().cpu().numpy())
    return np.concatenate(out)


def add_patient_norm(df, column_name, column_name_new):
    """Divide a column by its per-patient mean (the original `add_pation_norm`)."""
    means = df.groupby(GROUP_COL)[column_name].mean().rename(column_name_new).reset_index()
    df = df.merge(means, how="left", on=GROUP_COL)
    df[column_name_new] = df[column_name] / df[column_name_new]
    return df


def compute_image_features(df, device, scaling, batch_size=64, num_workers=4):
    """Run all neural-net image models on new data and attach their feature columns.

    `df` must contain `file_path`, `isic_id` and `patient_id`. `scaling` holds the
    fold-0 mean/std for the EVA and EdgeNeXt OOF scores (saved at training time).

    Adds: old_set_0/1/2 (+ _m), predictions_eva (+ _m), predictions_edg (+ _m).
    """
    file_paths = df["file_path"].values

    # --- 1) EVA02 3-class "old set" model --------------------------------------
    print("  [image] EVA02 3-class 'old set' model ...")
    m = ISICModel(EVA_MODEL_NAME, pretrained=False, num_classes=3).to(device)
    m.load_state_dict(torch.load(MODELS_DIR / "ema_small_pretrained", weights_only=True))
    preds3 = _infer(m, file_paths, EVA_IMG_SIZE, device, batch_size, num_workers)
    m.to("cpu"); del m; gc.collect(); torch.cuda.empty_cache()
    for k in range(3):
        df[f"old_set_{k}"] = preds3[:, k]
        df = add_patient_norm(df, f"old_set_{k}", f"old_set_{k}_m")

    # --- 2) EVA02 5-fold malignancy model --------------------------------------
    print("  [image] EVA02 5 folds ...")
    eva_scaled = []
    for fold in range(5):
        m = ISICModel(EVA_MODEL_NAME, pretrained=False, num_classes=1).to(device)
        m.load_state_dict(torch.load(MODELS_DIR / f"eva_model__{fold}", weights_only=True))
        raw = _infer(m, file_paths, EVA_IMG_SIZE, device, batch_size, num_workers)
        m.to("cpu"); del m; gc.collect(); torch.cuda.empty_cache()
        # z-scale using fold-0 OOF stats (the original scales every fold by fold 0).
        eva_scaled.append((raw - scaling["eva_mean"]) / scaling["eva_std"])
    df["predictions_eva"] = np.mean(eva_scaled, axis=0)
    df = add_patient_norm(df, "predictions_eva", "predictions_eva_m")

    # --- 3) EdgeNeXt 5-fold malignancy model -----------------------------------
    print("  [image] EdgeNeXt 5 folds ...")
    edg_scaled = []
    for fold in range(5):
        m = ISICModelEdgnet(EDG_MODEL_NAME, pretrained=False, num_classes=1).to(device)
        m.load_state_dict(torch.load(MODELS_DIR / f"edg_model__{fold}", weights_only=True))
        raw = _infer(m, file_paths, EDG_IMG_SIZE, device, batch_size, num_workers)
        m.to("cpu"); del m; gc.collect(); torch.cuda.empty_cache()
        edg_scaled.append((raw - scaling["edg_mean"]) / scaling["edg_std"])
    df["predictions_edg"] = np.mean(edg_scaled, axis=0)
    df = add_patient_norm(df, "predictions_edg", "predictions_edg_m")

    return df


# ===========================================================================
# Tabular feature engineering (polars) — verbatim logic from the original
# ===========================================================================
def read_data(path):
    """Engineer the tabular features from a metadata CSV. Returns a pandas frame
    indexed by isic_id. Missing floats are filled with the column median (computed
    on whatever data is passed in, matching the original behaviour)."""
    return (
        pl.read_csv(path)
        .with_columns(
            pl.col("age_approx").cast(pl.String).replace("NA", np.nan).cast(pl.Float64),
        )
        .with_columns(
            pl.col(pl.Float64).fill_nan(pl.col(pl.Float64).median()),
        )
        .with_columns(
            lesion_size_ratio=pl.col("tbp_lv_minorAxisMM") / pl.col("clin_size_long_diam_mm"),
            lesion_shape_index=pl.col("tbp_lv_areaMM2") / (pl.col("tbp_lv_perimeterMM") ** 2),
            hue_contrast=(pl.col("tbp_lv_H") - pl.col("tbp_lv_Hext")).abs(),
            luminance_contrast=(pl.col("tbp_lv_L") - pl.col("tbp_lv_Lext")).abs(),
            lesion_color_difference=(pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2 + pl.col("tbp_lv_deltaL") ** 2).sqrt(),
            border_complexity=pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_symm_2axis"),
            color_uniformity=pl.col("tbp_lv_color_std_mean") / (pl.col("tbp_lv_radial_color_std_max") + ERR),
        )
        .with_columns(
            position_distance_3d=(pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2).sqrt(),
            perimeter_to_area_ratio=pl.col("tbp_lv_perimeterMM") / pl.col("tbp_lv_areaMM2"),
            area_to_perimeter_ratio=pl.col("tbp_lv_areaMM2") / pl.col("tbp_lv_perimeterMM"),
            lesion_visibility_score=pl.col("tbp_lv_deltaLBnorm") + pl.col("tbp_lv_norm_color"),
            combined_anatomical_site=pl.col("anatom_site_general") + "_" + pl.col("tbp_lv_location"),
            symmetry_border_consistency=pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_norm_border"),
            consistency_symmetry_border=pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_norm_border") / (pl.col("tbp_lv_symm_2axis") + pl.col("tbp_lv_norm_border")),
        )
        .with_columns(
            color_consistency=pl.col("tbp_lv_stdL") / pl.col("tbp_lv_Lext"),
            consistency_color=pl.col("tbp_lv_stdL") * pl.col("tbp_lv_Lext") / (pl.col("tbp_lv_stdL") + pl.col("tbp_lv_Lext")),
            size_age_interaction=pl.col("clin_size_long_diam_mm") * pl.col("age_approx"),
            hue_color_std_interaction=pl.col("tbp_lv_H") * pl.col("tbp_lv_color_std_mean"),
            lesion_severity_index=(pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_norm_color") + pl.col("tbp_lv_eccentricity")) / 3,
            shape_complexity_index=pl.col("border_complexity") + pl.col("lesion_shape_index"),
            color_contrast_index=pl.col("tbp_lv_deltaA") + pl.col("tbp_lv_deltaB") + pl.col("tbp_lv_deltaL") + pl.col("tbp_lv_deltaLBnorm"),
        )
        .with_columns(
            log_lesion_area=(pl.col("tbp_lv_areaMM2") + 1).log(),
            normalized_lesion_size=pl.col("clin_size_long_diam_mm") / pl.col("age_approx"),
            mean_hue_difference=(pl.col("tbp_lv_H") + pl.col("tbp_lv_Hext")) / 2,
            std_dev_contrast=((pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2 + pl.col("tbp_lv_deltaL") ** 2) / 3).sqrt(),
            color_shape_composite_index=(pl.col("tbp_lv_color_std_mean") + pl.col("tbp_lv_area_perim_ratio") + pl.col("tbp_lv_symm_2axis")) / 3,
            lesion_orientation_3d=pl.arctan2(pl.col("tbp_lv_y"), pl.col("tbp_lv_x")),
            overall_color_difference=(pl.col("tbp_lv_deltaA") + pl.col("tbp_lv_deltaB") + pl.col("tbp_lv_deltaL")) / 3,
        )
        .with_columns(
            symmetry_perimeter_interaction=pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_perimeterMM"),
            comprehensive_lesion_index=(pl.col("tbp_lv_area_perim_ratio") + pl.col("tbp_lv_eccentricity") + pl.col("tbp_lv_norm_color") + pl.col("tbp_lv_symm_2axis")) / 4,
            color_variance_ratio=pl.col("tbp_lv_color_std_mean") / pl.col("tbp_lv_stdLExt"),
            border_color_interaction=pl.col("tbp_lv_norm_border") * pl.col("tbp_lv_norm_color"),
            border_color_interaction_2=pl.col("tbp_lv_norm_border") * pl.col("tbp_lv_norm_color") / (pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_norm_color")),
            size_color_contrast_ratio=pl.col("clin_size_long_diam_mm") / pl.col("tbp_lv_deltaLBnorm"),
            age_normalized_nevi_confidence=pl.col("tbp_lv_nevi_confidence") / pl.col("age_approx"),
            age_normalized_nevi_confidence_2=(pl.col("clin_size_long_diam_mm") ** 2 + pl.col("age_approx") ** 2).sqrt(),
            color_asymmetry_index=pl.col("tbp_lv_radial_color_std_max") * pl.col("tbp_lv_symm_2axis"),
        )
        .with_columns(
            volume_approximation_3d=pl.col("tbp_lv_areaMM2") * (pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2).sqrt(),
            color_range=(pl.col("tbp_lv_L") - pl.col("tbp_lv_Lext")).abs() + (pl.col("tbp_lv_A") - pl.col("tbp_lv_Aext")).abs() + (pl.col("tbp_lv_B") - pl.col("tbp_lv_Bext")).abs(),
            shape_color_consistency=pl.col("tbp_lv_eccentricity") * pl.col("tbp_lv_color_std_mean"),
            border_length_ratio=pl.col("tbp_lv_perimeterMM") / (2 * np.pi * (pl.col("tbp_lv_areaMM2") / np.pi).sqrt()),
            age_size_symmetry_index=pl.col("age_approx") * pl.col("clin_size_long_diam_mm") * pl.col("tbp_lv_symm_2axis"),
            index_age_size_symmetry=pl.col("age_approx") * pl.col("tbp_lv_areaMM2") * pl.col("tbp_lv_symm_2axis"),
        )
        .with_columns(
            ((pl.col(col) - pl.col(col).mean().over(GROUP_COL)) / (pl.col(col).std().over(GROUP_COL) + ERR)).alias(f"{col}_patient_norm")
            for col in (num_cols + new_num_cols)
        )
        .with_columns(count_per_patient=pl.col("isic_id").count().over(GROUP_COL))
        .with_columns(tbp_lv_areaMM2_patient=pl.col("tbp_lv_areaMM2").sum().over(GROUP_COL))
        .with_columns(tbp_lv_areaMM2_bp=pl.col("tbp_lv_areaMM2").sum().over([GROUP_COL, "anatom_site_general"]))
        .with_columns(pl.col(cat_cols_base).cast(pl.Categorical))
        .to_pandas()
        .set_index(ID_COL)
    )


def get_lof_score_pat_norm(df, k=15):
    """Per-patient Local Outlier Factor score (`of`). Patients with <= k images are
    left at the default 0.0. Expects `isic_id` as the index and a `patient_id` col."""
    top_lof_features = [
        "tbp_lv_H", "hue_contrast", "age_normalized_nevi_confidence_2", "tbp_lv_deltaB",
        "color_uniformity", "tbp_lv_z", "clin_size_long_diam_mm", "tbp_lv_y",
        "position_distance_3d", "hue_contrast", "tbp_lv_stdLExt", "mean_hue_difference",
        "age_normalized_nevi_confidence", "lesion_visibility_score", "position_distance_3d",
        "tbp_lv_minorAxisMM", "tbp_lv_Hext",
    ]
    # new StandardScaler can't handle duplicate feature columns
    top_lof_features_dedupe = list(dict.fromkeys(top_lof_features))
    
    # LOF cannot handle NaN/inf. Median-impute the feature matrix first; this is a
    # no-op when the data is already clean (so training parity is preserved).
    lof_feat = df[top_lof_features_dedupe].replace([np.inf, -np.inf], np.nan)
        
    factors = []
    for patient_id in tqdm(df[GROUP_COL].unique(), leave=False):
        mask = (df[GROUP_COL] == patient_id).values
        n = int(mask.sum())
        if n <= k: # does not meet minimum tile images required for LOF scoring
            continue

        # subset patient features
        X_pat = lof_feat[mask]
        X_pat = X_pat.fillna(X_pat.median()).fillna(0.0)

        # scale once per patient
        pat_scaled = np.nan_to_num(StandardScaler().fit_transform(X_pat))
        
        clf = LocalOutlierFactor(n_neighbors=k)
        clf.fit_predict(pat_scaled)
        
        factors.append(pd.DataFrame({
            "isic_id": df[mask].index.values,
            "of": clf.negative_outlier_factor_,
        }))

    if not factors:
        df["of"] = 0.0 # null value outside of typical LOF range
        return df
    
    factors = pd.concat(factors).set_index("isic_id")
    df = df.merge(factors, how="left", left_index=True, right_index=True)
    
    df["of"] = df["of"].fillna(0.0) # replace any remaining NA values
    
    return df


def get_lof_bysite_pat_norm(
        dataframe, 
        top_features_dict, 
        output_col_name="of_site", 
        patient_col="patient_id",
        site_col="anatom_site_general",
        k=15
        ):
    """
    Calculates a Local Outlier Factor score for TBP tiles using features dervied from VECTRA lesion analyzer measurements.
    Arguments:
        - Dataframe with lesion analyzer measurements
        - List of features to use for LOF calculation
        - Name of output score column
        - Groupby levels for background tiles, can be str or list
        - Minimum # of samples (k_neighbors) for valid LOF score, all samples used for LOF when n_samples == k+1
    """
    outlier_factors = []

    patient_groups = dataframe.groupby(patient_col, observed=True).indices
    for _, patient_pos in tqdm(patient_groups.items()):
        patient_df = dataframe.iloc[patient_pos]
        uniq_sites = patient_df[site_col].unique()
        
        # Union of features needed for any site
        feat_union = set()
        for site in uniq_sites:
            if site in top_features_dict:
                feat_union.update(top_features_dict[site])

        # no sites match our top features dict
        if not feat_union:
            outlier_factors.append(
                pd.DataFrame({
                    "isic_id": site_ids,
                    output_col_name: 0.0,
                })
            )
            continue
        
        feat_union = list(feat_union)

        # LOF can't handle NaN/inf. Median-impute the feature matrix first; this does not occur
        # when the data is already clean.
        lof_feat = patient_df[feat_union].replace([np.inf, -np.inf], np.nan)
        lof_feat = lof_feat.fillna(lof_feat.median()).fillna(0.0)
        
        # scale once per patient
        scaled_array = np.nan_to_num(StandardScaler().fit_transform(lof_feat))
        scaled_df = pd.DataFrame(
            scaled_array, index=patient_df.index, columns=feat_union
        )
        
        # loop through sites
        site_groups = patient_df.groupby(site_col, observed=True).indices
        for site, site_pos in site_groups.items():
            site_ids = patient_df.index.values[site_pos]

            if site not in top_features_dict: # handle new sites as subset of patient
                outlier_factors.append(
                    pd.DataFrame({
                        "isic_id": site_ids,
                        output_col_name: 0.0,
                    })
                )
                continue

            top_site_features = top_features_dict[site]
            site_scaled = scaled_df.loc[site_ids, top_site_features].values

            if len(site_pos) <= k: # skip cases where too few lesions
                outlier_factors.append(
                    pd.DataFrame({
                        "isic_id": site_ids,
                        output_col_name: 0.0,
                    })
                )
                continue

            clf = LocalOutlierFactor(n_neighbors=k)
            clf.fit_predict(site_scaled)
            outlier_factors.append(
                pd.DataFrame({
                    "isic_id": site_ids,
                    output_col_name: clf.negative_outlier_factor_,
                })
            )

    # Edge case: no groups processed at all
    if len(outlier_factors) == 0:
        dataframe[output_col_name] = 0.0
        return dataframe
    
    # Stitch all groups back together and merge onto original dataframe
    result = pd.concat(outlier_factors, ignore_index=True)
    dataframe = dataframe.merge(
        result.set_index("isic_id"),
        how="left",
        left_index=True,
        right_index=True,
    )

    return dataframe


def fill_inf(df, cols):
    """Replace +/-inf with NaN across the given columns (GBDTs treat NaN as missing).
    This is the corrected version of the original's single-column fill."""
    present = [c for c in cols if c in df.columns]
    df[present] = df[present].replace([np.inf, -np.inf], np.nan)
    return df


# ===========================================================================
# Custom competition metric (partial AUC above 80% TPR)
# ===========================================================================
def custom_metric_raw(y_hat, y_true):
    min_tpr = 0.80
    max_fpr = abs(1 - min_tpr)
    v_gt = abs(np.asarray(y_true) - 1)
    v_pred = 1.0 - np.asarray(y_hat)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    return 0.5 * max_fpr ** 2 + (max_fpr - 0.5 * max_fpr ** 2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)


# ===========================================================================
# PHASE 1 — train the GBDT stack once and cache it
# ===========================================================================
def _build_training_frame(lof_by_site, features_dict):
    """Assemble df_train exactly as the original script did, but pulling the image
    features from the precomputed OOF parquets (no neural nets needed here)."""
    print("  [train] engineering tabular features ...")
    df = read_data(TRAIN_META_PATH).reset_index()

    # One-hot encode the categorical columns; fit the encoder on the training data.
    encoder = OneHotEncoder(sparse_output=False, dtype=np.int32, handle_unknown="ignore")
    encoder.fit(df[cat_cols_base])
    onehot_cols = [f"onehot_{i}" for i in range(len(encoder.get_feature_names_out()))]
    df[onehot_cols] = encoder.transform(df[cat_cols_base])
    df[onehot_cols] = df[onehot_cols].astype("category")

    # Feature list mirrors the original: numeric + engineered + onehot + norm + special.
    feature_cols = num_cols + new_num_cols + onehot_cols + norm_cols + special_cols

    # Local Outlier Factor.
    df = df.set_index(ID_COL)
    if lof_by_site is True:
        print("  [train] computing LOF body site...")
        df = get_lof_bysite_pat_norm(
            df,
            top_features_dict=features_dict,
            output_col_name="of",
            patient_col="patient_id",
            site_col="anatom_site_general",
            )
    else:
        print("  [train] computing LOF total body...")
        df = get_lof_score_pat_norm(df)
    df = df.reset_index()
    feature_cols = feature_cols + ["of"]

    # Old-data 3-class model predictions (precomputed parquet for the train set).
    old = pd.read_parquet(OLD_PREDS_TRAIN_PATH)
    df = df.merge(old, how="left", on=ID_COL)
    for k in range(3):
        means = df.groupby(GROUP_COL)[f"old_set_{k}"].mean().rename(f"old_set_{k}_m").reset_index()
        df = df.merge(means, how="left", on=GROUP_COL)
        df[f"old_set_{k}_m"] = df[f"old_set_{k}"] / df[f"old_set_{k}_m"]
    feature_cols = feature_cols + [f"old_set_{k}" for k in range(3)] + [f"old_set_{k}_m" for k in range(3)]

    # EVA + EdgeNeXt OOF image scores (train side uses the ranked OOF columns), and
    # the fold-0 mean/std used to scale the corresponding test-time NN outputs.
    eva = pd.read_parquet()
    eva_agg = eva.groupby("fold_n").agg(
        mean_preds=pd.NamedAgg("tmp_predictions_all", "mean"),
        std_preds=pd.NamedAgg("tmp_predictions_all", "std"),
    )
    edg = pd.read_parquet(OOF_EDG_PATH)
    edg_agg = edg.groupby("fold_n").agg(
        mean_preds=pd.NamedAgg("tmp_predictions_all", "mean"),
        std_preds=pd.NamedAgg("tmp_predictions_all", "std"),
    )
    scaling = {
        "eva_mean": float(eva_agg.loc[0].mean_preds), "eva_std": float(eva_agg.loc[0].std_preds),
        "edg_mean": float(edg_agg.loc[0].mean_preds), "edg_std": float(edg_agg.loc[0].std_preds),
    }

    eva = eva[[ID_COL, GROUP_COL, "tmp_predictions_all__pr"]].rename(columns={"tmp_predictions_all__pr": "predictions_eva"})
    eva = add_patient_norm(eva, "predictions_eva", "predictions_eva_m")
    df = df.merge(eva[[ID_COL, "predictions_eva", "predictions_eva_m"]], how="left", on=ID_COL)
    feature_cols = feature_cols + ["predictions_eva", "predictions_eva_m"]

    edg = edg[[ID_COL, GROUP_COL, "tmp_predictions_all__pr"]].rename(columns={"tmp_predictions_all__pr": "predictions_edg"})
    edg = add_patient_norm(edg, "predictions_edg", "predictions_edg_m")
    df = df.merge(edg[[ID_COL, "predictions_edg", "predictions_edg_m"]], how="left", on=ID_COL)
    feature_cols = feature_cols + ["predictions_edg", "predictions_edg_m"]

    return df, feature_cols, onehot_cols, encoder, scaling


def _train_catboost(df, feature_cols, cat_cols, catboost_device):
    """45 CatBoost models (9 seeds x 5 folds). Tries GPU, falls back to CPU."""
    use_cols = [c for c in feature_cols if c not in COLUMNS_TO_DROP]
    models = []
    task_type, devices = ("GPU", "0") if catboost_device == "gpu" else ("CPU", None)
    gpu_failed = False

    with tqdm(total=45, desc="  catboost") as pbar:
        for seed in range(1, 10):
            skf = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
            for tr_idx, va_idx in skf.split(df, y=df[TARGET_COL], groups=df[GROUP_COL]):
                tr_x = df.iloc[tr_idx][use_cols].reset_index(drop=True)
                tr_y = df.iloc[tr_idx][TARGET_COL].reset_index(drop=True)
                va_x = df.iloc[va_idx][use_cols].reset_index(drop=True)
                va_y = df.iloc[va_idx][TARGET_COL].reset_index(drop=True)

                # Over/under-sample, THEN jitter the image features (original order).
                sampler = Pipeline([
                    ("over", RandomOverSampler(sampling_strategy=0.003, random_state=seed)),
                    ("under", RandomUnderSampler(sampling_strategy=SAMPLING_RATIO, random_state=seed)),
                ])
                tr_x, tr_y = sampler.fit_resample(tr_x, tr_y)
                for col in IMAGE_FEATURE_COLS:
                    tr_x[col] = tr_x[col] + np.random.normal(0, 0.1, size=tr_x.shape[0])

                clf = cb.CatBoostClassifier(
                    eval_metric="AUC", od_wait=100, random_state=seed,
                    task_type=("CPU" if gpu_failed else task_type),
                    devices=(None if gpu_failed else devices),
                    **CB_PARAMS,
                )
                train_pool = cb.Pool(tr_x, tr_y.values, cat_features=cat_cols)
                val_pool = cb.Pool(va_x, va_y.values, cat_features=cat_cols)
                try:
                    clf.fit(train_pool, eval_set=val_pool, verbose=False)
                except Exception as e:  # GPU kernel/driver issue -> fall back to CPU
                    if gpu_failed:
                        raise
                    print(f"\n  [catboost] GPU training failed ({e}); falling back to CPU.")
                    gpu_failed = True
                    clf = cb.CatBoostClassifier(
                        eval_metric="AUC", od_wait=100, random_state=seed,
                        task_type="CPU", **CB_PARAMS,
                    )
                    clf.fit(train_pool, eval_set=val_pool, verbose=False)
                models.append(clf)
                pbar.update(1)
    return models


def _train_sklearn_family(df, feature_cols, make_classifier, seed_offset, desc):
    """45 LightGBM/XGBoost models wrapped in imblearn sampling pipelines.
    Jitter is applied to the image features BEFORE sampling (original order)."""
    use_cols = [c for c in feature_cols if c not in COLUMNS_TO_DROP]
    models = []
    with tqdm(total=45, desc=desc) as pbar:
        for seed in range(1, 10):
            rs = seed * 10 + seed_offset
            skf = StratifiedGroupKFold(5, shuffle=True, random_state=rs)
            for tr_idx, va_idx in skf.split(df, y=df[TARGET_COL], groups=df[GROUP_COL]):
                tr_x = df.iloc[tr_idx][use_cols].reset_index(drop=True)
                tr_y = df.iloc[tr_idx][TARGET_COL].reset_index(drop=True)
                for col in IMAGE_FEATURE_COLS:
                    tr_x[col] = tr_x[col] + np.random.normal(0, 0.1, size=tr_x.shape[0])

                model = Pipeline([
                    ("over", RandomOverSampler(sampling_strategy=0.003, random_state=rs)),
                    ("under", RandomUnderSampler(sampling_strategy=SAMPLING_RATIO, random_state=rs)),
                    ("classifier", make_classifier()),
                ])
                model.fit(tr_x, tr_y)
                models.append(model)
                pbar.update(1)
    return models


def ensure_gbdt_artifacts(
        force_retrain=False, 
        catboost_device="gpu", 
        artifact_dir=None, 
        model_prefix="", 
        lof_by_site=None,
        features_dict=None,
        ):
    """Load the cached GBDT stack, or train + cache it if missing."""
    manifest = artifact_dir / f"{model_prefix}manifest.joblib"
    if manifest.exists() and not force_retrain:
        print(f"Found cached GBDT artifacts in {artifact_dir} — skipping training.")
        meta = joblib.load(manifest)
        return {
            "models_cb": joblib.load(artifact_dir / f"{model_prefix}models_cb.joblib"),
            "models_lgb": joblib.load(artifact_dir / f"{model_prefix}models_lgb.joblib"),
            "models_xgb": joblib.load(artifact_dir / f"{model_prefix}models_xgb.joblib"),
            "encoder": joblib.load(artifact_dir / f"{model_prefix}encoder.joblib"),
            **meta,
        }

    print("Training GBDT stack from scratch (one-time) ...")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    df, feature_cols, onehot_cols, encoder, scaling = _build_training_frame(lof_by_site, features_dict)

    print("  [train] CatBoost (45) ...")
    models_cb = _train_catboost(df, feature_cols, onehot_cols, catboost_device)
    print("  [train] LightGBM (45) ...")
    models_lgb = _train_sklearn_family(df, feature_cols, lambda: lgb.LGBMClassifier(**LGB_PARAMS), 17, "  lightgbm")
    print("  [train] XGBoost (45) ...")
    models_xgb = _train_sklearn_family(df, feature_cols, lambda: xgb.XGBClassifier(**XGB_PARAMS), 88, "  xgboost")

    # Persist everything needed for prediction.
    joblib.dump(models_cb, artifact_dir / f"{model_prefix}models_cb.joblib")
    joblib.dump(models_lgb, artifact_dir / f"{model_prefix}models_lgb.joblib")
    joblib.dump(models_xgb, artifact_dir / f"{model_prefix}models_xgb.joblib")
    joblib.dump(encoder, artifact_dir / f"{model_prefix}encoder.joblib")
    meta = {
        "feature_cols": feature_cols,
        "cat_cols": onehot_cols,
        "columns_to_drop": COLUMNS_TO_DROP,
        "scaling": scaling,
    }
    joblib.dump(meta, artifact_dir / f"{model_prefix}manifest.joblib")
    print(f"Saved GBDT artifacts to {artifact_dir}")

    return {"models_cb": models_cb, "models_lgb": models_lgb, "models_xgb": models_xgb,
            "encoder": encoder, **meta}


# ===========================================================================
# PHASE 2 — predict on new data
# ===========================================================================
def rank_average(models, X):
    """Average the percentile-ranked predict_proba across a family of models
    (the original `get_model_predictions`)."""
    acc = None
    for model in models:
        p = pd.Series(model.predict_proba(X)[:, 1]).rank(pct=True).values
        acc = p if acc is None else acc + p
    return acc / len(models)


def predict(
        input_csv, 
        output_csv, 
        artifacts, 
        device, 
        batch_size=64, 
        num_workers=4,
        lof_by_site=False,
        features_dict=None
        ):
    feature_cols = artifacts["feature_cols"]
    columns_to_drop = artifacts["columns_to_drop"]
    encoder = artifacts["encoder"]
    onehot_cols = artifacts["cat_cols"]

    raw = pd.read_csv(input_csv)
    for required in ("file_path", ID_COL, GROUP_COL):
        if required not in raw.columns:
            raise ValueError(f"Input CSV must contain a '{required}' column.")
    has_target = TARGET_COL in raw.columns

    # --- tabular features ------------------------------------------------------
    print("Engineering tabular features ...")
    df = read_data(input_csv).reset_index()

    # one-hot encode using the SAVED encoder (handle_unknown='ignore').
    df[onehot_cols] = encoder.transform(df[cat_cols_base])
    df[onehot_cols] = df[onehot_cols].astype("category")

    # Local Outlier Factor.
    df = df.set_index(ID_COL)
    if lof_by_site == True:
        print("Computing LOF body site...")
        df = get_lof_bysite_pat_norm(
            df,
            top_features_dict=features_dict,
            output_col_name="of",
            patient_col="patient_id",
            site_col="anatom_site_general",
            )
    else:
        print("Computing LOF total body...")
        df = get_lof_score_pat_norm(df)
    df = df.reset_index()
    
    # --- image features (neural nets) -----------------------------------------
    print("Running image models ...")
    img = raw[[ID_COL, GROUP_COL, "file_path"]].copy()
    img = compute_image_features(img, device, artifacts["scaling"], batch_size, num_workers)
    image_cols = [c for c in img.columns if c not in (ID_COL, GROUP_COL, "file_path")]
    df = df.merge(img[[ID_COL] + image_cols], how="left", on=ID_COL)

    # --- assemble model matrix -------------------------------------------------
    df = fill_inf(df, feature_cols)
    use_cols = [c for c in feature_cols if c not in columns_to_drop]
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Engineered frame is missing expected feature columns: {missing[:10]} ...")
    X = df[use_cols]

    # --- ensemble --------------------------------------------------------------
    print("Predicting with GBDT stack ...")
    p_cb = rank_average(artifacts["models_cb"], X)
    p_lgb = rank_average(artifacts["models_lgb"], X)
    p_xgb = rank_average(artifacts["models_xgb"], X)
    predictions = (p_cb + p_lgb + p_xgb) / 3.0

    out = pd.DataFrame({ID_COL: df[ID_COL].values, "prediction": predictions})
    out.to_csv(output_csv, index=False)
    print(f"Wrote {len(out)} predictions to {output_csv}")

    if has_target:
        merged = out.merge(raw[[ID_COL, TARGET_COL]], on=ID_COL, how="left")
        score = custom_metric_raw(merged["prediction"].values, merged[TARGET_COL].values)
        print(f"Partial AUC (pAUC above 80% TPR): {score:.5f}")

    return out


# ===========================================================================
# CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="ISIC-2024 1st-place inference on new data.")
    ap.add_argument("--input", help="CSV of new data (needs file_path, isic_id, patient_id, metadata).")
    ap.add_argument("--output", default="predictions.csv", help="Where to write the prediction CSV.")
    ap.add_argument("--train-only", action="store_true", help="Only build/refresh cached GBDT artifacts.")
    ap.add_argument("--force-retrain", action="store_true", help="Retrain the GBDT stack even if cached.")
    ap.add_argument("--artifact-dir", default=ARTIFACT_DIR, help="Directory where trained GBDT .joblib files are stored.")
    ap.add_argument("--model_prefix", type=str, help="Name prepended to trained model, if desired (e.g. 'new_feature_')")
    ap.add_argument("--catboost-device", choices=["gpu", "cpu"], default="gpu",
                    help="CatBoost training device (auto-falls back to CPU on GPU failure).")
    ap.add_argument("--batch-size", type=int, default=64, help="Image inference batch size.")
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    ap.add_argument("--lof-by-site", action="store_true",
                        help="Also compute per-anatomic-site LOF scores (requires 'anatom_site_general' "
                             "in metadata and a top_features_by_location.json file).")
    ap.add_argument("--top-features-path", type=Path, default=TOP_FEATURES_PATH,
                    help=f"Path to the per-site top-features JSON (default: {TOP_FEATURES_PATH}).")
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
        features_dict=top_features_dict
        )

    if args.train_only:
        print("train-only mode: artifacts ready, exiting.")
        return
    if not args.input:
        ap.error("--input is required unless --train-only is given.")

    predict(
        args.input, 
        args.output, 
        artifacts, 
        device, 
        args.batch_size, 
        args.num_workers,
        lof_by_site=args.lof_by_site,
        features_dict=top_features_dict
        )
    print("All done!")


if __name__ == "__main__":
    main()
