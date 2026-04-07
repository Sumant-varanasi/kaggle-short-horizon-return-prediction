import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import gc, time, warnings, os
warnings.filterwarnings('ignore')

start_time = time.time()

# ============================================================
# 1. LOAD DATA
# ============================================================
BASE_PATHS = [
    '/kaggle/input/competitions/short-horizon-return-prediction-challenge-by-i-rage/',
    '/kaggle/input/short-horizon-return-prediction-challenge-by-i-rage/',
]
DATA = None
for p in BASE_PATHS:
    if os.path.exists(p):
        DATA = p
        break
if DATA is None:
    raise FileNotFoundError("Data directory not found")

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
print(f'Number of raw features: {len(feature_cols)}')

# ============================================================
# 2. ANALYZE FEATURE STRUCTURE
# ============================================================
# Separate features by type
base_features = []
lag1_features = []
lag2_features = []
lag3_features = []
special_features = []

for c in feature_cols:
    if c.endswith('_LagT1'):
        lag1_features.append(c)
    elif c.endswith('_LagT2'):
        lag2_features.append(c)
    elif c.endswith('_LagT3'):
        lag3_features.append(c)
    elif c in ['SO3_T', 'Price']:
        special_features.append(c)
    else:
        base_features.append(c)

print(f'Base features: {len(base_features)}')
print(f'LagT1 features: {len(lag1_features)}')
print(f'LagT2 features: {len(lag2_features)}')
print(f'LagT3 features: {len(lag3_features)}')
print(f'Special features: {special_features}')

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print('\nPhase 1: Feature Engineering...')

# 3a. Lag acceleration features (LagT2 - LagT1 = acceleration of change)
# These capture whether the rate of change is accelerating or decelerating
eng_cols_train = []
eng_cols_test = []
eng_names = []

for lag1_col in lag1_features:
    base_name = lag1_col.replace('_LagT1', '')
    lag2_col = base_name + '_LagT2'
    lag3_col = base_name + '_LagT3'

    if lag2_col in feature_cols:
        # Acceleration: difference of differences
        accel_name = base_name + '_Accel12'
        train[accel_name] = train[lag2_col] - train[lag1_col]
        test[accel_name] = test[lag2_col] - test[lag1_col]
        eng_names.append(accel_name)

    if lag3_col in feature_cols and lag2_col in feature_cols:
        # Second acceleration
        accel_name2 = base_name + '_Accel23'
        train[accel_name2] = train[lag3_col] - train[lag2_col]
        test[accel_name2] = test[lag3_col] - test[lag2_col]
        eng_names.append(accel_name2)

    if lag3_col in feature_cols:
        # Total momentum: lag3 captures the full window
        mom_name = base_name + '_Mom13'
        train[mom_name] = train[lag3_col] - train[lag1_col]
        test[mom_name] = test[lag3_col] - test[lag1_col]
        eng_names.append(mom_name)

print(f'Created {len(eng_names)} engineered lag features')

# 3b. Ratio features for Price
if 'Price' in feature_cols:
    for lag_col in ['Price_LagT1', 'Price_LagT2', 'Price_LagT3']:
        if lag_col in feature_cols:
            ratio_name = lag_col + '_PctChg'
            denom = train['Price'].replace(0, np.nan)
            train[ratio_name] = train[lag_col] / denom
            denom_test = test['Price'].replace(0, np.nan)
            test[ratio_name] = test[lag_col] / denom_test
            eng_names.append(ratio_name)

# 3c. SO3_T interaction features
if 'SO3_T' in feature_cols:
    # SO3_T * Price
    train['SO3T_x_Price'] = train['SO3_T'] * train['Price']
    test['SO3T_x_Price'] = test['SO3_T'] * test['Price']
    eng_names.append('SO3T_x_Price')

    # SO3_T squared
    train['SO3T_sq'] = train['SO3_T'] ** 2
    test['SO3T_sq'] = test['SO3_T'] ** 2
    eng_names.append('SO3T_sq')

# 3d. Cross-feature statistics (for top base features - computed later after selection)
# Add rolling-style aggregations across feature families
# Group features by prefix
from collections import defaultdict
prefix_groups = defaultdict(list)
for c in base_features:
    prefix = c.split('_')[0] if '_' in c else c
    prefix_groups[prefix].append(c)

# For each group with multiple features, add mean/std/range
for prefix, cols in prefix_groups.items():
    if len(cols) >= 3 and len(cols) <= 20:
        grp_name_mean = f'{prefix}_grp_mean'
        grp_name_std = f'{prefix}_grp_std'
        train[grp_name_mean] = train[cols].mean(axis=1)
        test[grp_name_mean] = test[cols].mean(axis=1)
        train[grp_name_std] = train[cols].std(axis=1)
        test[grp_name_std] = test[cols].std(axis=1)
        eng_names.extend([grp_name_mean, grp_name_std])

print(f'Total engineered features: {len(eng_names)}')

# Update feature list
all_feature_cols = feature_cols + eng_names
print(f'Total features (raw + engineered): {len(all_feature_cols)}')

# ============================================================
# 4. TARGET PROCESSING
# ============================================================
# Symmetric winsorization at 99th percentile
y_clip_val = float(np.percentile(np.abs(y), 99))
y_clipped = np.clip(y, -y_clip_val, y_clip_val)
print(f'\nTarget clipped to [-{y_clip_val:.6f}, {y_clip_val:.6f}]')
print(f'Target stats: mean={y.mean():.6f}, std={y.std():.6f}, min={y.min():.6f}, max={y.max():.6f}')

# ============================================================
# 5. FEATURE SELECTION - Multi-method stability
# ============================================================
print('\nPhase 2: Feature Selection...')

X_train_all = train[all_feature_cols].values.astype(np.float32)
X_test_all = test[all_feature_cols].values.astype(np.float32)

del train, test
gc.collect()

# Quick LightGBM importance across folds
n_splits_fs = 5
gkf_fs = GroupKFold(n_splits=n_splits_fs)

quick_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'n_jobs': -1,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': 7,
    'min_child_samples': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
}

fold_importances = []
for fold_i, (tr_idx, va_idx) in enumerate(gkf_fs.split(X_train_all, y_clipped, groups)):
    ds_tr = lgb.Dataset(X_train_all[tr_idx], y_clipped[tr_idx])
    ds_va = lgb.Dataset(X_train_all[va_idx], y_clipped[va_idx], reference=ds_tr)
    bst = lgb.train(
        quick_params, ds_tr, num_boost_round=300,
        valid_sets=[ds_va],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    imp = bst.feature_importance(importance_type='gain')
    fold_importances.append(imp)
    r2_val = r2_score(y_clipped[va_idx], bst.predict(X_train_all[va_idx]))
    print(f'  FS fold {fold_i+1}: best_iter={bst.best_iteration}, val_R2={r2_val:.6f}')

fold_importances = np.array(fold_importances)

# Use both mean importance and stability (low variance = stable)
mean_imp = fold_importances.mean(axis=0)
std_imp = fold_importances.std(axis=0) + 1e-10
stability_score = mean_imp / std_imp  # Higher = more stable and important

# Select top features by stability-weighted importance
N_KEEP = 200
top_feat_idx = np.argsort(stability_score)[::-1][:N_KEEP]
selected_features = [all_feature_cols[i] for i in top_feat_idx]
print(f'Selected {len(selected_features)} features')

# Check how many engineered features made it
eng_selected = [f for f in selected_features if f in eng_names]
print(f'  Of which {len(eng_selected)} are engineered features')

X_train = X_train_all[:, top_feat_idx]
X_test = X_test_all[:, top_feat_idx]
del X_train_all, X_test_all
gc.collect()

elapsed = time.time() - start_time
print(f'Feature selection took {elapsed:.1f}s')

# ============================================================
# 6. MODEL TRAINING - Diverse ensemble
# ============================================================
print('\nPhase 3: Training diverse model ensemble...')
n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

# Model 1: LightGBM with RMSE (conservative)
lgb_params_conservative = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'n_jobs': -1,
    'learning_rate': 0.01,
    'num_leaves': 15,
    'max_depth': 4,
    'min_child_samples': 500,
    'subsample': 0.6,
    'colsample_bytree': 0.4,
    'reg_alpha': 5.0,
    'reg_lambda': 5.0,
    'min_gain_to_split': 0.01,
    'feature_pre_filter': False,
}

# Model 2: LightGBM with Huber loss (robust to outliers)
lgb_params_huber = {
    'objective': 'huber',
    'huber_delta': 0.3,
    'metric': 'rmse',
    'verbosity': -1,
    'n_jobs': -1,
    'learning_rate': 0.01,
    'num_leaves': 20,
    'max_depth': 5,
    'min_child_samples': 400,
    'subsample': 0.65,
    'colsample_bytree': 0.45,
    'reg_alpha': 3.0,
    'reg_lambda': 3.0,
    'min_gain_to_split': 0.005,
    'feature_pre_filter': False,
}

# Model 3: LightGBM aggressive (higher capacity)
lgb_params_aggressive = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'n_jobs': -1,
    'learning_rate': 0.02,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 200,
    'subsample': 0.7,
    'colsample_bytree': 0.5,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'min_gain_to_split': 0.001,
    'feature_pre_filter': False,
}

# Model 4: LightGBM with MAE loss
lgb_params_mae = {
    'objective': 'mae',
    'metric': 'mae',
    'verbosity': -1,
    'n_jobs': -1,
    'learning_rate': 0.02,
    'num_leaves': 24,
    'max_depth': 5,
    'min_child_samples': 300,
    'subsample': 0.65,
    'colsample_bytree': 0.45,
    'reg_alpha': 2.0,
    'reg_lambda': 2.0,
    'feature_pre_filter': False,
}

N_BOOST = 1500
EARLY_STOP = 80

model_configs = {
    'LGB_Conservative': (lgb_params_conservative, X_train, X_test),
    'LGB_Huber': (lgb_params_huber, X_train, X_test),
    'LGB_Aggressive': (lgb_params_aggressive, X_train, X_test),
    'LGB_MAE': (lgb_params_mae, X_train, X_test),
}

oof_preds = {}
test_preds = {}

for model_name, (params, X_tr_data, X_te_data) in model_configs.items():
    t0 = time.time()
    oof = np.zeros(len(y_clipped))
    tpred = np.zeros(len(X_te_data))

    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_data, y_clipped, groups)):
        ds_tr = lgb.Dataset(X_tr_data[tr_idx], y_clipped[tr_idx])
        ds_va = lgb.Dataset(X_tr_data[va_idx], y_clipped[va_idx], reference=ds_tr)

        bst = lgb.train(
            params, ds_tr, num_boost_round=N_BOOST,
            valid_sets=[ds_va],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va_idx] = bst.predict(X_tr_data[va_idx])
        tpred += bst.predict(X_te_data) / n_splits

    r2 = r2_score(y_clipped, oof)
    oof_preds[model_name] = oof
    test_preds[model_name] = tpred
    print(f'  {model_name}: OOF R2={r2:.6f} (best iters avg, {time.time()-t0:.0f}s)')

# Model 5: Ridge Regression
print('  Training Ridge...')
t0 = time.time()
oof_ridge = np.zeros(len(y_clipped))
tpred_ridge = np.zeros(len(X_test))

for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_clipped, groups)):
    scaler = RobustScaler()
    X_tr_sc = scaler.fit_transform(np.nan_to_num(X_train[tr_idx], nan=0.0))
    X_va_sc = scaler.transform(np.nan_to_num(X_train[va_idx], nan=0.0))
    X_te_sc = scaler.transform(np.nan_to_num(X_test, nan=0.0))

    ridge = Ridge(alpha=50.0)
    ridge.fit(X_tr_sc, y_clipped[tr_idx])
    oof_ridge[va_idx] = ridge.predict(X_va_sc)
    tpred_ridge += ridge.predict(X_te_sc) / n_splits

r2 = r2_score(y_clipped, oof_ridge)
oof_preds['Ridge'] = oof_ridge
test_preds['Ridge'] = tpred_ridge
print(f'  Ridge: OOF R2={r2:.6f} ({time.time()-t0:.0f}s)')

# Model 6: ElasticNet
print('  Training ElasticNet...')
t0 = time.time()
oof_enet = np.zeros(len(y_clipped))
tpred_enet = np.zeros(len(X_test))

for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_clipped, groups)):
    scaler = RobustScaler()
    X_tr_sc = scaler.fit_transform(np.nan_to_num(X_train[tr_idx], nan=0.0))
    X_va_sc = scaler.transform(np.nan_to_num(X_train[va_idx], nan=0.0))
    X_te_sc = scaler.transform(np.nan_to_num(X_test, nan=0.0))

    enet = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=1000)
    enet.fit(X_tr_sc, y_clipped[tr_idx])
    oof_enet[va_idx] = enet.predict(X_va_sc)
    tpred_enet += enet.predict(X_te_sc) / n_splits

r2 = r2_score(y_clipped, oof_enet)
oof_preds['ElasticNet'] = oof_enet
test_preds['ElasticNet'] = tpred_enet
print(f'  ElasticNet: OOF R2={r2:.6f} ({time.time()-t0:.0f}s)')

# ============================================================
# 7. COMPETITION-METRIC-AWARE ENSEMBLE OPTIMIZATION
# ============================================================
print('\nPhase 4: Competition-metric-aware ensemble optimization...')

def compute_competition_metric(y_true, y_pred, cv_groups):
    """Compute: 2 * mean(batch_R2) - full_R2"""
    unique_g = np.unique(cv_groups)
    batch_r2s = []
    for g in unique_g:
        mask = cv_groups == g
        if np.std(y_true[mask]) > 1e-12 and len(y_true[mask]) > 10:
            r2 = r2_score(y_true[mask], y_pred[mask])
            batch_r2s.append(r2)
    mean_batch_r2 = np.mean(batch_r2s) if batch_r2s else 0.0
    full_r2 = r2_score(y_true, y_pred)
    return 2.0 * mean_batch_r2 - full_r2

# Filter models with reasonable R2
print('\nIndividual model scores:')
for name, oof in oof_preds.items():
    r2 = r2_score(y_clipped, oof)
    comp_score = compute_competition_metric(y_clipped, oof, groups)
    print(f'  {name}: R2={r2:.6f}, CompScore={comp_score:.6f}')

# Use scipy minimize for optimal weights + shrinkage jointly
from scipy.optimize import minimize

candidate_names = list(oof_preds.keys())
n_models = len(candidate_names)

def objective(params):
    weights = np.abs(params[:n_models])
    weights = weights / (weights.sum() + 1e-12)
    shrinkage = np.clip(params[n_models], 0.01, 1.0)

    blend = np.zeros(len(y_clipped))
    for i, name in enumerate(candidate_names):
        blend += weights[i] * oof_preds[name]
    blend *= shrinkage

    score = compute_competition_metric(y_clipped, blend, groups)
    return -score  # minimize negative score

# Try multiple initializations
best_result = None
best_neg_score = float('inf')

for trial in range(10):
    np.random.seed(trial * 42)
    init_weights = np.random.dirichlet(np.ones(n_models))
    init_shrinkage = np.random.uniform(0.05, 0.8)
    init_params = np.concatenate([init_weights, [init_shrinkage]])

    result = minimize(
        objective, init_params,
        method='Nelder-Mead',
        options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12}
    )

    if result.fun < best_neg_score:
        best_neg_score = result.fun
        best_result = result

raw_weights = np.abs(best_result.x[:n_models])
best_weights = raw_weights / (raw_weights.sum() + 1e-12)
best_shrinkage = np.clip(best_result.x[n_models], 0.01, 1.0)

print(f'\nBest competition metric (OOF): {-best_neg_score:.8f}')
print(f'Best shrinkage: {best_shrinkage:.4f}')
for i, name in enumerate(candidate_names):
    print(f'  {name}: weight={best_weights[i]:.4f}')

# ============================================================
# 8. BUILD FINAL PREDICTIONS
# ============================================================
print('\nPhase 5: Building final predictions...')

# Blend OOF
blend_oof = np.zeros(len(y_clipped))
blend_test = np.zeros(len(test_ids))
for i, name in enumerate(candidate_names):
    blend_oof += best_weights[i] * oof_preds[name]
    blend_test += best_weights[i] * test_preds[name]

blend_oof *= best_shrinkage
blend_test *= best_shrinkage

# Clip extreme predictions (3-sigma)
pred_mean = np.mean(blend_oof)
pred_std = np.std(blend_oof)
clip_lo = pred_mean - 3.0 * pred_std
clip_hi = pred_mean + 3.0 * pred_std

blend_oof_clipped = np.clip(blend_oof, clip_lo, clip_hi)
blend_test_clipped = np.clip(blend_test, clip_lo, clip_hi)

# Compare clipped vs unclipped
score_unclipped = compute_competition_metric(y_clipped, blend_oof, groups)
score_clipped = compute_competition_metric(y_clipped, blend_oof_clipped, groups)
print(f'Competition score (unclipped): {score_unclipped:.8f}')
print(f'Competition score (clipped):   {score_clipped:.8f}')

if score_clipped >= score_unclipped:
    final_test = blend_test_clipped
    final_score = score_clipped
    print('Using CLIPPED predictions')
else:
    final_test = blend_test
    final_score = score_unclipped
    print('Using UNCLIPPED predictions')

# Also try pure zero (predicting mean) as a baseline comparison
zero_score = compute_competition_metric(y_clipped, np.zeros(len(y_clipped)), groups)
print(f'\nBaseline (all zeros) competition score: {zero_score:.8f}')
print(f'Our competition score: {final_score:.8f}')

# If our model is worse than zero, blend towards zero
if final_score < zero_score:
    print('\nWARNING: Model worse than baseline. Trying blend with zero...')
    best_alpha = 0.0
    best_blended_score = final_score
    for alpha in np.arange(0.0, 1.01, 0.05):
        blended = (1.0 - alpha) * blend_oof + alpha * 0.0  # blend towards zero
        s = compute_competition_metric(y_clipped, blended, groups)
        if s > best_blended_score:
            best_blended_score = s
            best_alpha = alpha

    if best_alpha > 0:
        final_test = final_test * (1.0 - best_alpha)
        print(f'Blending {best_alpha:.0%} towards zero. New score: {best_blended_score:.8f}')
        final_score = best_blended_score

# ============================================================
# 9. CREATE SUBMISSION
# ============================================================
submission = pd.DataFrame({
    'ID': test_ids,
    'TARGET': final_test
})
submission.to_csv('submission.csv', index=False)

print(f'\n{"="*60}')
print(f'FINAL RESULTS')
print(f'{"="*60}')
print(f'Submission shape: {submission.shape}')
print(f'Prediction stats: mean={final_test.mean():.8f}, std={final_test.std():.8f}')
print(f'Prediction range: [{final_test.min():.8f}, {final_test.max():.8f}]')
print(f'Final competition score (OOF): {final_score:.8f}')
print(f'Total runtime: {time.time() - start_time:.1f}s')
print('Done!')
