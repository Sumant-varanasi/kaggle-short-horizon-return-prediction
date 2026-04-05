# Short-Horizon Return Prediction Challenge by iRage

## Kaggle Competition Solution

**Competition:** [Short-Horizon Return Prediction Challenge by iRage](https://www.kaggle.com/competitions/short-horizon-return-prediction-challenge-by-i-rage)

**Kaggle Notebook:** [notebooka6568cb810](https://www.kaggle.com/code/sumantvaranasi/notebooka6568cb810)

---

## Overview

This repository contains the solution for the Kaggle "Short-Horizon Return Prediction Challenge by iRage" competition. The goal is to predict short-horizon percentage returns of financial instruments using OHLCV-style lagged features.

## Repository Structure

| File | Description |
|------|-------------|
| `notebook.py` | Main solution — 3-model ensemble (LGB × 2 + Ridge) with stability-based feature selection |
| `notebook_catboost.py` | Alternate solution using CatBoost (V8, score: -0.00012) |
| `README.md` | This file |

---

## Pipeline (notebook.py)

The pipeline runs in 3 phases:

### Phase 1 — Stability-Based Feature Selection

A quick LightGBM model is trained across 5-fold `GroupKFold` CV to rank features by their mean gain importance. The top 150 features are selected based on average rank stability across folds, reducing the full feature set down to the most predictive signals.

- CV: 5-fold `GroupKFold` (grouped by `CV_GROUP`)
- Rounds: up to 200 with early stopping (patience = 20)
- Features kept: top 150 by mean rank

### Phase 2 — 3-Model Ensemble Training

Three models are trained on the selected 150 features using 5-fold `GroupKFold` CV:

| Model | Objective | Key Params |
|-------|-----------|------------|
| LGB1 | `regression` (RMSE) | lr=0.02, leaves=15, depth=4, min_samples=500 |
| LGB2 | `huber` (δ=0.5) | lr=0.02, leaves=20, depth=5, min_samples=400 |
| Ridge | L2 regression | α=100, StandardScaler applied |

- Max boosting rounds: 1000 (LGB), early stopping patience: 50
- Out-of-fold (OOF) predictions collected for all 3 models

### Phase 3 — Optimal Ensemble Weights + Shrinkage

A grid search (step=0.05) over all valid weight triplets (w1, w2, w3) summing to 1.0 selects the combination maximising OOF R². The final test predictions are blended using these weights, then multiplied by a shrinkage factor of **0.6** to reduce overfitting to the training distribution.

---

## Data

| Split | Rows | Format |
|-------|------|--------|
| Train | 661,574 | `.parquet` |
| Test | 410,139 | `.parquet` |

- **Target:** `TARGET` — short-horizon percentage return
- **Group column:** `CV_GROUP` (used for grouped CV to prevent leakage)
- **Features:** OHLCV-style lagged features (LagT1, LagT2, LagT3 variants)

---

## Target Processing

The target is symmetrically winsorized at the 99th percentile of absolute values before training to reduce outlier impact:

```python
y_clipped = np.clip(y, -np.percentile(np.abs(y), 99), np.percentile(np.abs(y), 99))
```

---

## Competition Scoring

- **Formula:** `2 × mean(batch R²) − full_test_R²`
- **Constraints:** CPU only, no internet access, < 30 min runtime

---

## Results

| Version | Score | Notes |
|---------|-------|-------|
| V7 (notebook.py) | -0.00014 | Feature selection + LGB ensemble + shrinkage |
| V8 (notebook_catboost.py) | -0.00012 | CatBoost model |

---

## Usage

This notebook is designed to run on Kaggle. To use it:

1. Upload `notebook.py` (or `notebook_catboost.py`) to a Kaggle notebook
2. Attach the competition dataset (`short-horizon-return-prediction-challenge-by-i-rage`)
3. Set the accelerator to **CPU** (GPU not required)
4. Run all cells — total runtime is under 30 minutes
