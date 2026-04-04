import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import gc, time, warnings
warnings.filterwarnings('ignore')

start_time = time.time()

# ============================================================
# 1. LOAD DATA
# ============================================================
DATA = '/kaggle/input/competitions/short-horizon-return-prediction-challenge-by-i-rage/'
train = pd.read_parquet(DATA + 'train.parquet')
test = pd.read_parquet(DATA + 'test.parquet')

TARGET_COL = 'TARGET'
GROUP_COL = 'CV_GROUP'
ID_COL = 'ID'

print(f'Train shape: {train.shape}, Test shape: {test.shape}')

y = train[TARGET_COL].values
groups = train[GROUP_COL].values
test_ids = test[ID_COL].values

exclude_cols = [TARGET_COL, GROUP_COL, ID_COL]
feature_cols = [c for c in train.columns if c not in exclude_cols]
print(f'Number of features: {len(feature_cols)}')

# ============================================================
# 2. TARGET CLIPPING
# ============================================================
y_clip_val = float(np.percentile(np.abs(y), 99))
y_clipped = np.clip(y, -y_clip_val, y_clip_val)
print(f'Target clipped to [-{y_clip_val:.6f}, {y_clip_val:.6f}]')

X_train = train[feature_cols].values.astype(np.float32)
X_test = test[feature_cols].values.astype(np.float32)

# Replace NaN with -999 for CatBoost
X_train = np.nan_to_num(X_train, nan=-999.0)
X_test = np.nan_to_num(X_test, nan=-999.0)

del train, test
gc.collect()

# ============================================================
# 3. CATBOOST MODELS with GroupKFold
# ============================================================
print('\nTraining CatBoost models...')
n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

# Model 1: Conservative CatBoost with Ordered boosting
cb_params_1 = {
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 4,
    'l2_leaf_reg': 10.0,
    'min_data_in_leaf': 500,
    'subsample': 0.6,
    'colsample_bylevel': 0.5,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'verbose': 0,
    'task_type': 'CPU',
    'boosting_type': 'Ordered',
    'early_stopping_rounds': 50,
}

# Model 2: Different config with Plain boosting
cb_params_2 = {
    'iterations': 1000,
    'learning_rate': 0.02,
    'depth': 5,
    'l2_leaf_reg': 5.0,
    'min_data_in_leaf': 300,
    'subsample': 0.65,
    'colsample_bylevel': 0.6,
    'loss_function': 'MAE',
    'eval_metric': 'RMSE',
    'random_seed': 123,
    'verbose': 0,
    'task_type': 'CPU',
    'boosting_type': 'Plain',
    'early_stopping_rounds': 50,
}

# Model 3: Quantile-like with Huber
cb_params_3 = {
    'iterations': 1000,
    'learning_rate': 0.025,
    'depth': 3,
    'l2_leaf_reg': 15.0,
    'min_data_in_leaf': 600,
    'subsample': 0.55,
    'colsample_bylevel': 0.4,
    'loss_function': 'Huber:delta=0.5',
    'eval_metric': 'RMSE',
    'random_seed': 456,
    'verbose': 0,
    'task_type': 'CPU',
    'boosting_type': 'Ordered',
    'early_stopping_rounds': 50,
}

oof_cb1 = np.zeros(len(X_train))
oof_cb2 = np.zeros(len(X_train))
oof_cb3 = np.zeros(len(X_train))

test_cb1 = np.zeros(len(X_test))
test_cb2 = np.zeros(len(X_test))
test_cb3 = np.zeros(len(X_test))

for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_clipped, groups)):
    print(f'\n--- Fold {fold_i+1}/{n_splits} ---')
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y_clipped[tr_idx], y_clipped[va_idx]

    train_pool = Pool(X_tr, y_tr)
    val_pool = Pool(X_va, y_va)

    # CatBoost Model 1 (Ordered RMSE)
    cb1 = CatBoostRegressor(**cb_params_1)
    cb1.fit(train_pool, eval_set=val_pool, use_best_model=True)
    oof_cb1[va_idx] = cb1.predict(X_va)
    test_cb1 += cb1.predict(X_test) / n_splits
    print(f'  CB1 (Ordered RMSE): iter={cb1.best_iteration_}, val_R2={r2_score(y_va, oof_cb1[va_idx]):.6f}')

    # CatBoost Model 2 (Plain MAE)
    cb2 = CatBoostRegressor(**cb_params_2)
    cb2.fit(train_pool, eval_set=val_pool, use_best_model=True)
    oof_cb2[va_idx] = cb2.predict(X_va)
    test_cb2 += cb2.predict(X_test) / n_splits
    print(f'  CB2 (Plain MAE): iter={cb2.best_iteration_}, val_R2={r2_score(y_va, oof_cb2[va_idx]):.6f}')

    # CatBoost Model 3 (Ordered Huber)
    cb3 = CatBoostRegressor(**cb_params_3)
    cb3.fit(train_pool, eval_set=val_pool, use_best_model=True)
    oof_cb3[va_idx] = cb3.predict(X_va)
    test_cb3 += cb3.predict(X_test) / n_splits
    print(f'  CB3 (Ordered Huber): iter={cb3.best_iteration_}, val_R2={r2_score(y_va, oof_cb3[va_idx]):.6f}')

    del cb1, cb2, cb3, train_pool, val_pool
    gc.collect()

print('\n--- OOF Scores ---')
print(f'CB1 OOF R2: {r2_score(y_clipped, oof_cb1):.6f}')
print(f'CB2 OOF R2: {r2_score(y_clipped, oof_cb2):.6f}')
print(f'CB3 OOF R2: {r2_score(y_clipped, oof_cb3):.6f}')

# ============================================================
# 4. OPTIMAL ENSEMBLE WEIGHTS
# ============================================================
print('\nFinding optimal ensemble weights...')
best_r2 = -999
best_w = (0.33, 0.33, 0.34)

for w1 in np.arange(0.0, 1.05, 0.05):
    for w2 in np.arange(0.0, 1.05 - w1, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < -0.01:
            continue
        w3 = max(w3, 0.0)
        blend = w1 * oof_cb1 + w2 * oof_cb2 + w3 * oof_cb3
        r2 = r2_score(y_clipped, blend)
        if r2 > best_r2:
            best_r2 = r2
            best_w = (w1, w2, w3)

print(f'Best weights: CB1={best_w[0]:.2f}, CB2={best_w[1]:.2f}, CB3={best_w[2]:.2f}')
print(f'Best OOF R2: {best_r2:.6f}')

# ============================================================
# 5. FINAL PREDICTION with shrinkage
# ============================================================
test_pred = best_w[0] * test_cb1 + best_w[1] * test_cb2 + best_w[2] * test_cb3

SHRINKAGE = 0.5
test_pred_final = test_pred * SHRINKAGE
print(f'\nShrinkage factor: {SHRINKAGE}')
print(f'Pred stats before shrinkage: mean={test_pred.mean():.6f}, std={test_pred.std():.6f}')
print(f'Pred stats after shrinkage: mean={test_pred_final.mean():.6f}, std={test_pred_final.std():.6f}')

# ============================================================
# 6. CREATE SUBMISSION
# ============================================================
submission = pd.DataFrame({
    'ID': test_ids,
    'TARGET': test_pred_final
})
submission.to_csv('submission.csv', index=False)
print(f'\nSubmission shape: {submission.shape}')
print(f'Submission head:\n{submission.head()}')

elapsed = time.time() - start_time
print(f'\nTotal runtime: {elapsed:.1f}s')
print('Done!')
