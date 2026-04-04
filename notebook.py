import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
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
# 2. TARGET CLIPPING (symmetric winsorization at 99th pct of abs)
# ============================================================
y_clip_val = float(np.percentile(np.abs(y), 99))
y_clipped = np.clip(y, -y_clip_val, y_clip_val)
print(f'Target clipped to [-{y_clip_val:.6f}, {y_clip_val:.6f}]')

X_train = train[feature_cols].values
X_test = test[feature_cols].values

del train, test
gc.collect()

# ============================================================
# 3. FEATURE SELECTION - Stability-based
# ============================================================
print('\nPhase 1: Feature selection via quick LightGBM...')
n_splits_fs = 5
gkf_fs = GroupKFold(n_splits=n_splits_fs)

quick_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'n_jobs': -1,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 5,
    'min_child_samples': 200,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
}

fold_feat_ranks = []
for fold_i, (tr_idx, va_idx) in enumerate(gkf_fs.split(X_train, y_clipped, groups)):
    ds_tr = lgb.Dataset(X_train[tr_idx], y_clipped[tr_idx])
    ds_va = lgb.Dataset(X_train[va_idx], y_clipped[va_idx], reference=ds_tr)
    bst = lgb.train(
        quick_params, ds_tr, num_boost_round=200,
        valid_sets=[ds_va],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
    )
    imp = bst.feature_importance(importance_type='gain')
    ranks = np.argsort(np.argsort(-imp))
    fold_feat_ranks.append(ranks)
    print(f'  FS fold {fold_i+1}: best_iter={bst.best_iteration}')

fold_feat_ranks = np.array(fold_feat_ranks)
mean_ranks = fold_feat_ranks.mean(axis=0)

N_KEEP = min(150, len(feature_cols))
top_feat_idx = np.argsort(mean_ranks)[:N_KEEP]
selected_features = [feature_cols[i] for i in top_feat_idx]
print(f'Selected {len(selected_features)} features based on stability ranking')

X_train_sel = X_train[:, top_feat_idx]
X_test_sel = X_test[:, top_feat_idx]
del X_train, X_test
gc.collect()

elapsed = time.time() - start_time
print(f'Feature selection took {elapsed:.1f}s')

# ============================================================
# 4. MODEL TRAINING with selected features
# ============================================================
print('\nPhase 2: Training models with selected features...')
n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

lgb_params_1 = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'n_jobs': -1,
    'learning_rate': 0.02,
    'num_leaves': 15,
    'max_depth': 4,
    'min_child_samples': 500,
    'subsample': 0.6,
    'colsample_bytree': 0.5,
    'reg_alpha': 5.0,
    'reg_lambda': 5.0,
    'min_gain_to_split': 0.01,
    'feature_pre_filter': False,
}

lgb_params_2 = {
    'objective': 'huber',
    'huber_delta': 0.5,
    'metric': 'rmse',
    'verbosity': -1,
    'n_jobs': -1,
    'learning_rate': 0.02,
    'num_leaves': 20,
    'max_depth': 5,
    'min_child_samples': 400,
    'subsample': 0.65,
    'colsample_bytree': 0.5,
    'reg_alpha': 3.0,
    'reg_lambda': 3.0,
    'min_gain_to_split': 0.005,
    'feature_pre_filter': False,
}

N_BOOST = 1000
EARLY_STOP = 50

oof_lgb1 = np.zeros(len(X_train_sel))
oof_lgb2 = np.zeros(len(X_train_sel))
oof_ridge = np.zeros(len(X_train_sel))

test_lgb1 = np.zeros(len(X_test_sel))
test_lgb2 = np.zeros(len(X_test_sel))
test_ridge = np.zeros(len(X_test_sel))

for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X_train_sel, y_clipped, groups)):
    print(f'\n--- Fold {fold_i+1}/{n_splits} ---')
    X_tr, X_va = X_train_sel[tr_idx], X_train_sel[va_idx]
    y_tr, y_va = y_clipped[tr_idx], y_clipped[va_idx]

    ds_tr = lgb.Dataset(X_tr, y_tr)
    ds_va_lgb = lgb.Dataset(X_va, y_va, reference=ds_tr)

    bst1 = lgb.train(
        lgb_params_1, ds_tr, num_boost_round=N_BOOST,
        valid_sets=[ds_va_lgb],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb1[va_idx] = bst1.predict(X_va)
    test_lgb1 += bst1.predict(X_test_sel) / n_splits
    print(f'  LGB1 (RMSE): iter={bst1.best_iteration}, val_R2={r2_score(y_va, oof_lgb1[va_idx]):.6f}')

    bst2 = lgb.train(
        lgb_params_2, ds_tr, num_boost_round=N_BOOST,
        valid_sets=[ds_va_lgb],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb2[va_idx] = bst2.predict(X_va)
    test_lgb2 += bst2.predict(X_test_sel) / n_splits
    print(f'  LGB2 (Huber): iter={bst2.best_iteration}, val_R2={r2_score(y_va, oof_lgb2[va_idx]):.6f}')

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(np.nan_to_num(X_tr, nan=0.0))
    X_va_sc = scaler.transform(np.nan_to_num(X_va, nan=0.0))
    X_test_sc = scaler.transform(np.nan_to_num(X_test_sel, nan=0.0))

    ridge = Ridge(alpha=100.0)
    ridge.fit(X_tr_sc, y_tr)
    oof_ridge[va_idx] = ridge.predict(X_va_sc)
    test_ridge += ridge.predict(X_test_sc) / n_splits
    print(f'  Ridge: val_R2={r2_score(y_va, oof_ridge[va_idx]):.6f}')

print('\n--- OOF Scores ---')
print(f'LGB1 OOF R2: {r2_score(y_clipped, oof_lgb1):.6f}')
print(f'LGB2 OOF R2: {r2_score(y_clipped, oof_lgb2):.6f}')
print(f'Ridge OOF R2: {r2_score(y_clipped, oof_ridge):.6f}')

# ============================================================
# 5. OPTIMAL ENSEMBLE WEIGHTS via grid search on OOF
# ============================================================
print('\nPhase 3: Finding optimal ensemble weights...')
best_r2 = -999
best_w = (0.33, 0.33, 0.34)

for w1 in np.arange(0.0, 1.05, 0.05):
    for w2 in np.arange(0.0, 1.05 - w1, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < -0.01:
            continue
        w3 = max(w3, 0.0)
        blend = w1 * oof_lgb1 + w2 * oof_lgb2 + w3 * oof_ridge
        r2 = r2_score(y_clipped, blend)
        if r2 > best_r2:
            best_r2 = r2
            best_w = (w1, w2, w3)

print(f'Best weights: LGB1={best_w[0]:.2f}, LGB2={best_w[1]:.2f}, Ridge={best_w[2]:.2f}')
print(f'Best OOF R2: {best_r2:.6f}')

# ============================================================
# 6. FINAL PREDICTION with shrinkage
# ============================================================
test_pred = best_w[0] * test_lgb1 + best_w[1] * test_lgb2 + best_w[2] * test_ridge

SHRINKAGE = 0.6
test_pred_final = test_pred * SHRINKAGE
print(f'\nShrinkage factor: {SHRINKAGE}')
print(f'Pred stats before shrinkage: mean={test_pred.mean():.6f}, std={test_pred.std():.6f}')
print(f'Pred stats after shrinkage: mean={test_pred_final.mean():.6f}, std={test_pred_final.std():.6f}')

# ============================================================
# 7. CREATE SUBMISSION
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
