# Short-Horizon Return Prediction Challenge by iRage

## Kaggle Competition Solution

**Competition:** [Short-Horizon Return Prediction Challenge by iRage](https://www.kaggle.com/competitions/short-horizon-return-prediction-challenge-by-i-rage)

**Kaggle Notebook:** [notebooka6568cb810](https://www.kaggle.com/code/sumantvaranasi/notebooka6568cb810)

---

## Overview

This repository contains the solution for the Kaggle "Short-Horizon Return Prediction Challenge by iRage" competition. The goal is to predict short-horizon percentage returns of financial instruments using OHLCV-style lagged features.

## Approach

### Model
- **Algorithm:** LightGBM (Gradient Boosted Decision Trees)
- **Cross-Validation:** 5-fold KFold (shuffle=True, random_state=42)
- **Target Processing:** Clipped at [1st, 99th] percentiles to reduce outlier impact

### Feature Engineering
- **Base Features:** ~446 OHLCV-style features with LagT1, LagT2, LagT3 variants
- **Ratio Features:** 50 additional LagT1/LagT2 ratio features (clipped to [-10, 10])
- **Total Features:** 496

### Hyperparameters
- n_estimators: 600
- learning_rate: 0.05
- num_leaves: 127
- min_child_samples: 50
- subsample: 0.8
- colsample_bytree: 0.8
- early_stopping_rounds: 50

### Results
- **OOF R²:** ~0.194591
- **Mean Fold R²:** ~0.194604 ± 0.001770

## Competition Details

- **Scoring Formula:** 2 × mean(batch R²) - full_test_R²
- **Constraints:** CPU only, no internet, < 30 min runtime
- **Data:** Train (661,574 rows) and Test (410,139 rows) parquet files

## Files

- `notebook.py` - Main solution script (equivalent to Kaggle notebook)
- `README.md` - This file

## Usage

This notebook is designed to run on Kaggle. Upload to Kaggle, attach the competition dataset, and run all cells.
