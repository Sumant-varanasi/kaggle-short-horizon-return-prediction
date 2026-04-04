# =============================================================
# Short-Horizon Return Prediction Challenge by iRage
# =============================================================
# Competition: Predict short-horizon % return of Price
# Metric: R2 Score (higher is better)
# Constraint: Must run offline, CPU only, under 30 minutes
# =============================================================

import numpy as np
import pandas as pd
import warnings
import os
import gc
warnings.filterwarnings('ignore')

print("Libraries imported successfully")
print("Python environment ready")

# Create placeholder files to prevent FileNotFoundError from debug cells
for _f in ['out3.txt', 'output.txt', 'out2.txt']: open(f'/kaggle/working/{_f}', 'w').close()

# =============================================================
# Cell 2: Load Data
# =============================================================
DATA_PATH = '/kaggle/input/competitions/short-horizon-return-prediction-challenge-by-i-rage/'

print("Loading training data...")
train = pd.read_parquet(DATA_PATH + 'train.parquet')
print(f"Train shape: {train.shape}")

print("Loading test data...")
test = pd.read_parquet(DATA_PATH + 'test.parquet')
print(f"Test shape: {test.shape}")

print(f"Train columns: {list(train.columns)}")
print(f"Test columns: {list(test.columns)}")

# =============================================================
# Cell 3: Feature Engineering & EDA
# =============================================================
print("\n" + "=" * 60)
print("TARGET Statistics:")
print("=" * 60)
print(train['TARGET'].describe())
print(f"\nSkewness: {train['TARGET'].skew():.4f}")
print(f"Kurtosis: {train['TARGET'].kurt():.4f}")

print("\n" + "=" * 60)
print("Missing Values Check:")
print("=" * 60)
train_nulls = train.isnull().sum().sum()
test_nulls = test.isnull().sum().sum()
print(f"Train total nulls: {train_nulls}")
print(f"Test total nulls: {test_nulls}")

print("\n" + "=" * 60)
print("Target outlier analysis:")
print("=" * 60)
q01 = train['TARGET'].quantile(0.01)
q99 = train['TARGET'].quantile(0.99)
print(f"1st percentile: {q01:.4f}")
print(f"99th percentile: {q99:.4f}")
pct_outliers = ((train['TARGET'] < q01) | (train['TARGET'] > q99)).mean() * 100
print(f"Outliers outside [1%, 99%]: {pct_outliers:.2f}%")

print("\n" + "=" * 60)
print("Feature count breakdown:")
print("=" * 60)
feature_cols = [c for c in train.columns if c not in ['ID', 'TARGET'] and 'batch_id' not in c.lower()]
lag1_cols = [c for c in feature_cols if '_LagT1' in c]
lag2_cols = [c for c in feature_cols if '_LagT2' in c]
lag3_cols = [c for c in feature_cols if '_LagT3' in c]
print(f"Total feature columns: {len(feature_cols)}")
print(f"  LagT1 features: {len(lag1_cols)}")
print(f"  LagT2 features: {len(lag2_cols)}")
print(f"  LagT3 features: {len(lag3_cols)}")

# Feature engineering: Create ratio features LagT1/LagT2
print("\n" + "=" * 60)
print("Feature Engineering: Ratio features")
print("=" * 60)
train_fe = train.copy()
test_fe = test.copy()

ratio_count = 0
for c1 in lag1_cols:
    base_name = c1.replace('_LagT1', '')
    c2 = base_name + '_LagT2'
    if c2 in train_fe.columns:
        nm = base_name + '_Ratio_T1T2'
        train_fe[nm] = (train_fe[c1] / (train_fe[c2].abs() + 1e-8)).clip(-10, 10)
        test_fe[nm] = (test_fe[c1] / (test_fe[c2].abs() + 1e-8)).clip(-10, 10)
        ratio_count += 1

final_features = [c for c in train_fe.columns if c not in ['ID', 'TARGET']]
print(f"Added {ratio_count} ratio features. Total features: {len(final_features)}")

# =============================================================
# Cell 4: Target Clipping
# =============================================================
print("\nClipping target at [1%, 99%] percentiles...")
y = train_fe['TARGET'].values.copy()
y_clipped = np.clip(y, q01, q99)
print(f"Target range before clip: [{y.min():.6f}, {y.max():.6f}]")
print(f"Target range after clip:  [{y_clipped.min():.6f}, {y_clipped.max():.6f}]")

# =============================================================
# Cell 5: LightGBM 5-Fold CV Training
# =============================================================
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

X = train_fe[final_features].values
X_test = test_fe[[c for c in final_features if c in test_fe.columns]].values
y_c = y_clipped

p = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 600,
    'learning_rate': 0.05,
    'num_leaves': 127,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'verbose': -1,
    'n_jobs': -1
}

kf = KFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
tp = np.zeros(len(X_test))
models = []
fs = []

for fold, (ti, vi) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}/5")
    m = lgb.LGBMRegressor(**p)
    m.fit(X[ti], y_c[ti], eval_set=[(X[vi], y_c[vi])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])
    oof[vi] = m.predict(X[vi])
    s = r2_score(y_c[vi], oof[vi])
    fs.append(s)
    tp += m.predict(X_test) / 5
    print(f"  R2={s:.6f} trees={m.n_estimators}")
    del m
    gc.collect()

print(f"OOF R2={r2_score(y_c, oof):.6f} Mean={np.mean(fs):.6f} +/-{np.std(fs):.6f}")

# =============================================================
# Cell 6: Submission
# =============================================================
sub = pd.DataFrame({'ID': test['ID'], 'TARGET': tp})
print(f"Submission shape: {sub.shape}")
print(sub.head())
print(f"NaN count: {sub['TARGET'].isna().sum()}")
sub.to_csv('submission.csv', index=False)
print(f"Saved! Size: {os.path.getsize('submission.csv')/1e6:.2f} MB")
print("submission.csv ready for competition!")
