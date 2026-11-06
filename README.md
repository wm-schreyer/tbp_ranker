# Aligning the ISIC'24 Challenge Winner to Clinical High-Risk Melanoma Screening

## Navigation
* [Background](#background)
* [Results](#results)
* [Using this Repository](#using-this-repository)

## Disclaimer
This software is for research purposes only and should not be used to treat patients. It is not a medical device and has not been approved by any regulatory body. Anyone who modifies, uses or distributes this software does so entirely at their own risk and takes on all responsibility for compliance with applicable laws and regulations.

## Background
The [ISIC 2024 Challenge](https://www.kaggle.com/competitions/isic-2024-challenge/overview) was an AI/ML competition hosted on Kaggle with the objective of predicting skin cancer. The data was sourced from 9 medical centers across Europe, Australia and North America and images were captured using Vectra WB360 total body photography (TBP) systems. 

Kaggle challenge winners often use [highly optimized ensemble stragtegies](https://www.kaggle.com/code/yeemeitsang/unreasonably-effective-ensemble-learning) which may not be suited for clinical use cases like screening individual patients at risk of developing melanoma. To align the existing model with this clinical setting, my objectives were to:
  1. Adapt the ISIC'24 Challenge winning model to output an ensemble average, enabling predictions for one patient at a time
  2. Calibrate model outputs so they are interpretable as malignancy probabilities
  3. Implement a learn-to-rank strategy that could better prioritize melanomas over benign lesions

## Results
I calibrated the original ISIC'24 model using Platt scaling which fits a logistic regression curve to the ensemble outputs. The un-calibrated original model uses a rank-averaging approach across the entire test dataset, which we can see causes the output score distribution to center around 0.5 for benign cases (below, left). Of course, we shouldn't interpret these scores for benign lesions as having on average a predicted 50% probability of being malignant as this would cause nearly every lesion to be sent for biopsy. With Platt scaling, the calibrated model gives probabilities of ~0.0 for benign lesions which is what we'd need for clinically-interpretable scoring (below, right). I used the ISIC 2024 challenge public leaderboard dataset to validate model performance with and without calibration, finding that the partial area under the ROC curve >TPR80% was nearly identical (0.1858 vs. 0.1857).

<p align="center">
<img width="450" alt="isic24_calibration" src="https://github.com/user-attachments/assets/39f28112-2553-453a-bb54-42ecb2b67b3f" />
</p>

In addition to a competition scoring strategy, the original model was trained to predict a binary "Malignant" vs. "Benign" label using an ensemble of gradient-boosted decision trees (GBDT). In practice, some skin cancers like melanoma are more deadly than others. We would therefore like to prioritize their detection by pushing their scores higher relative to the other (non-melanoma) scores in a patient's lesion set. I thought a GBDT strategy that used a [learn-to-rank (LTR)](https://en.wikipedia.org/wiki/Learning_to_rank) algorithm like LambdaMART may be a better fit because it makes many pairwise comparisons to optimize the ranking order.

I fitted an ensemble (m=50) of LightGBM Rankers using the original model's intermediate image-based and tabular features and achieved a higher top-15 melanoma sensitivity on the private leaderboard than my calibrated version of the original model (0.780 vs. 0.703). This approached the best top-15 melanoma sensitivity (0.792) out of all challenge submissions while using a GBDT ensemble that was 3x smaller than the 1st place solution.

<p align="center">
<img width="450" alt="melanoma_topk_sens_3models" src="https://github.com/user-attachments/assets/c7e75dcd-4aaf-457f-80d9-04da99724a4e" />
</p>

Optimization of the ranking approach and diversifying the LTR ensemble with algorithms like XE<sub>ndcg</sub> should yield further improvement.

## Using this Repository

### Setup

1. Create your environment using Conda and the requirements file.
```
conda create --name your_env python pip
conda activate your_env
pip install -r requirements.txt
```
2. Download the Kaggle [competition dataset](https://www.kaggle.com/competitions/isic-2024-challenge/data) in HDF5 format.
3. Download the trained ISIC'24 image models from this [Google Drive link](https://drive.google.com/drive/folders/1uX-6cG02IqMuuf_tZPB-k2-BtRvVsMb-?usp=sharing).
4. Edit the paths in the `paths.example.json` and rename the file to `paths.json`.

### Training
Use the CLI to re-train the original model with a calibrated GBDT stack via Platt scaling.
```
python3 /your/path/src/retrain_predict_hdf5.py \
    --input /your/path/train-metadata.csv \
    --images /your/path/train-image.hdf5 \
    --artifact-dir /your/path/calibrated_gbdt \
    --train-only \
    --lof-null -1.0 \
    --ensemble calibrated
```

Train the LightGBM Ranker ensemble which optimizes lesion ordering at the patient level.
```
python3 /your/path/train_lgbm_ranker.py \
    --input /your/path/train-metadata.csv \
    --images /your/path/train-image.hdf5 \
    --artifact-dir /your/path/lgbm_ranker \
    --label-gain 0,1,10,100 \
    --query-by patient
```

### Evaluation
Run the trained models on both challenge test sets.
```
python3 /your/path/src/retrain_predict_hdf5.py \
    --input /your/path/test-metadata.csv \
    --images /your/path/test-image.hdf5 \
    --artifact-dir /your/path/calibrated_gbdt \
    --cnn-cache /your/path/test_cnn_outputs.parquet \ # use this option to speed up repeated inference
    --output /your/path/calibrated_gbdt_preds.csv \
    --lof-null -1.0 \
    --ensemble calibrated

python3 /your/path/src/predict_lgbm_ranker_hdf5.py \
    --input /your/path/test-metadata.csv \
    --images /your/path/test-image.hdf5 \
    --artifact-dir /your/path/lgbm_ranker \
    --cnn-cache /your/path/test_cnn_outputs.parquet \
    --output /your/path/lgbm_ranker_preds.csv
```
