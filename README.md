# Short-Horizon Return Prediction Challenge by iRage

## Kaggle Competition Solution

**Competition:** [Short-Horizon Return Prediction Challenge by iRage](https://www.kaggle.com/competitions/short-horizon-return-prediction-challenge-by-i-rage)

**Kaggle Notebook:** [short-horizon-improved-v2](https://www.kaggle.com/code/sumantvaranasi/short-horizon-improved-v2)

---

## Overview

This repository contains the solution for the Kaggle "Short-Horizon Return Prediction Challenge by iRage" competition. The goal is to predict short-horizon percentage returns of financial instruments using OHLCV-style lagged features.

## Repository Structure

| File | Description |
|------|-------------|
| `notebook.py` | Main solution — 3-model ensemble (LGB x 2 + Ridge) with stability-based feature selection |
| `notebook_advanced_ensemble.py` | **V15** — 6-model advanced ensemble (4xLGB + Ridge + ElasticNet) with feature engineering and competition-metric optimization |
| `notebook_catboost.py` | CatBoost solution (V8, score: -0.00012) |
| `notebook_extra_trees.py` | ExtraTrees + PCA + Interactions model (V13, score: -0.00051) |
| `notebook_metric_optimized.py` | Metric-optimized LGB+CB with rank/group features (V14, score: -0.00053) |
| `notebook_neural_net.py` | PyTorch DeepNet ensemble model (V12, score: -0.00048) |
| `notebook_super_ensemble.py` | Super-Ensemble: LGB+CatBoost+Ridge (V10, score: -0.00001) |
| `notebook_ultimate_v2.py` | Ultimate V2: LGB+CB+XGB ensemble (V11, score: -0.00000) |
| `README.md` | This file |

---

## Pipeline (notebook_advanced_ensemble.py) — V15

The latest pipeline runs in 5 phases:

### Phase 1 — Feature Engineering
- Lag acceleration features (`LagT2 - LagT1`) capturing rate-of-change dynamics
- Second-order acceleration (`LagT3 - LagT2`)
- Momentum features (`LagT3 - LagT1`)
- Price percentage change ratios
- `SO3_T` interaction features (cross-product with Price, squared)
- Feature family group statistics (mean/std across same-prefix features)

### Phase 2 — Stability-Based Feature Selection
- Signal-to-noise ratio ranking (`mean_importance / std_importance`) selects features that are both important AND stable across folds
- Keeps top 200 features from ~780 total (raw + engineered)

### Phase 3 — 6-Model Diverse Ensemble Training
| Model | Objective | Key Params |
|-------|-----------|------------|
| LGB Conservative | `regression` (RMSE) | lr=0.01, leaves=15, depth=4, min_samples=500 |
| LGB Huber | `huber` (delta=0.3) | lr=0.01, leaves=20, depth=5, min_samples=400 |
| LGB Aggressive | `regression` (RMSE) | lr=0.02, leaves=31, depth=6, min_samples=200 |
| LGB MAE | `mae` | lr=0.02, leaves=24, depth=5, min_samples=300 |
| Ridge | L2 regression | alpha=50, RobustScaler |
| ElasticNet | L1+L2 | alpha=0.01, l1_ratio=0.5 |

### Phase 4 — Competition-Metric-Aware Optimization
- Directly optimizes `2 x mean(batch_R2) - full_R2` using Nelder-Mead
- Jointly optimizes ensemble weights AND shrinkage factor
- 10 random restarts to avoid local minima

### Phase 5 — Safety Net
- Compares against zero-prediction baseline
- Auto-blends towards zero if model hurts the score
- 3-sigma prediction clipping

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

## Competition Scoring

- **Formula:** `2 x mean(batch R2) - full_test_R2`
- **Constraints:** CPU only, no internet access, < 30 min runtime

---

## Results

| Version | Score | Notes |
|---------|-------|-------|
| V15 (notebook_advanced_ensemble.py) | N/A (exceeded 30min) | 6-model ensemble + feature engineering + metric optimization. Runtime: 1h 51m. Needs runtime optimization |
| V14 (notebook_metric_optimized.py) | -0.00053 | Metric-optimized LGB+CB with rank/group features |
| V13 (notebook_extra_trees.py) | -0.00051 | ExtraTrees+PCA+Interactions model |
| V12 (notebook_neural_net.py) | -0.00048 | PyTorch DeepNet ensemble model |
| V11 (notebook_ultimate_v2.py) | -0.00000 | Ultimate 7-model: LGB+CB+XGB+MLP+Ridge, shrinkage=0.35 |
| V10 (notebook_super_ensemble.py) | -0.00001 | Super-Ensemble: LGB+CatBoost+Ridge |
| V8 (notebook_catboost.py) | -0.00012 | CatBoost 3-model ensemble |
| V7 (notebook.py) | -0.00014 | Feature selection + LGB ensemble + shrinkage |

---

## Usage

This notebook is designed to run on Kaggle. To use it:

1. Upload the desired `.py` file to a Kaggle notebook
2. Attach the competition dataset (`short-horizon-return-prediction-challenge-by-i-rage`)
3. Set the accelerator to **CPU** (GPU not required)
4. Run all cells — ensure runtime is under 30 minutes

**Note:** `notebook_advanced_ensemble.py` (V15) currently exceeds the 30-minute runtime limit. To fix, reduce N_BOOST from 1500 to 500, N_KEEP from 200 to 100, and remove ElasticNet model.
