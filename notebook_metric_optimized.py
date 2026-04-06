import numpy as np
import pandas as pd
import os, time, warnings, gc
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

T0 = time.time()

# 1. Load data
base = '/kaggle/input/competitions/short-horizon-return-prediction-challenge-by-i-rage'
for ext in ['parquet','csv']:
    p = os.path.join(base, f'train.{ext}')
    if os.path.exists(p):
        train = pd.read_parquet(p) if ext=='parquet' else pd.read_csv(p)
        test  = pd.read_parquet(p.replace('train','test')) if ext=='parquet' else pd.read_csv(p.replace('train','test'))
        print(f'Loading {p}')
        break
print(f'Train: {train.shape}, Test: {test.shape}')

target_col = 'TARGET'
id_col = 'ID'
cv_col = 'CV_GROUP'
y = train[target_col].values
cv_groups = train[cv_col].values
test_ids = test[id_col].values

drop_cols = [c for c in [target_col, id_col, cv_col] if c in train.columns]
X = train.drop(columns=drop_cols)
X_test = test.drop(columns=[c for c in [id_col, cv_col] if c in test.columns])
feature_names = list(X.columns)
N_FEAT = len(feature_names)
print(f'Features: {N_FEAT}, CV column: {cv_col}')

# 2. Feature Selection (stability-based)
from sklearn.ensemble import ExtraTreesRegressor
np.random.seed(42)
importances = np.zeros(N_FEAT)
for s in [42, 123, 777]:
    idx = np.random.choice(len(y), min(50000, len(y)), replace=False)
    et = ExtraTreesRegressor(n_estimators=200, max_depth=8, n_jobs=-1, random_state=s)
    et.fit(X.iloc[idx], y[idx])
    importances += et.feature_importances_
importances /= 3
TOP_K = 120
top_idx = np.argsort(importances)[::-1][:TOP_K]
sel_features = [feature_names[i] for i in top_idx]
X_sel = X[sel_features].values.astype(np.float32)
X_test_sel = X_test[sel_features].values.astype(np.float32)
print(f'Feature selection: {N_FEAT} -> {TOP_K} in {time.time()-T0:.0f}s')

# Strategy 2: Rank-transformed features
def rank_transform(arr):
    from scipy.stats import rankdata
    return rankdata(arr, method="average") / len(arr) - 0.5

X_rank = np.zeros_like(X_sel)
X_test_rank = np.zeros_like(X_test_sel)
for j in range(X_sel.shape[1]):
    X_rank[:, j] = rank_transform(X_sel[:, j])
    X_test_rank[:, j] = rank_transform(X_test_sel[:, j])
print(f'Rank features created in {time.time()-T0:.0f}s')

# Strategy 5: CV_GROUP-based features
group_means = {}
group_stds = {}
unique_groups = np.unique(cv_groups)
for g in unique_groups:
    mask = cv_groups == g
    group_means[g] = np.nanmean(X_sel[mask], axis=0)
    group_stds[g] = np.nanstd(X_sel[mask], axis=0) + 1e-8

X_group_dev = np.zeros((len(y), TOP_K), dtype=np.float32)
for g in unique_groups:
    mask = cv_groups == g
    X_group_dev[mask] = (X_sel[mask] - group_means[g]) / group_stds[g]

global_mean = np.nanmean(X_sel, axis=0)
global_std = np.nanstd(X_sel, axis=0) + 1e-8
X_test_group_dev = (X_test_sel - global_mean) / global_std

# Combine: original + rank + group_dev = 360 features
X_aug = np.hstack([X_sel, X_rank, X_group_dev])
X_test_aug = np.hstack([X_test_sel, X_test_rank, X_test_group_dev])
aug_names = sel_features + [f"{f}_rank" for f in sel_features] + [f"{f}_gdev" for f in sel_features]
print(f'Augmented features: {X_aug.shape[1]} in {time.time()-T0:.0f}s')

# 3. CV Setup
unique_cv = np.sort(np.unique(cv_groups))
N_FOLDS = len(unique_cv)
print(f'CV folds: {N_FOLDS}')

# 4. LightGBM models
import lightgbm as lgb

lgb_configs = {
    "LGB_RMSE": {
        "objective": "regression", "metric": "rmse",
        "learning_rate": 0.03, "num_leaves": 31,
        "feature_fraction": 0.6, "bagging_fraction": 0.8,
        "bagging_freq": 1, "min_child_samples": 100,
        "lambda_l1": 0.1, "lambda_l2": 1.0,
        "max_depth": 6, "verbose": -1, "n_jobs": -1,
        "feature_pre_filter": False,
    },
    "LGB_Huber": {
        "objective": "huber", "metric": "rmse",
        "huber_delta": 0.5,
        "learning_rate": 0.03, "num_leaves": 31,
        "feature_fraction": 0.6, "bagging_fraction": 0.8,
        "bagging_freq": 1, "min_child_samples": 100,
        "lambda_l1": 0.1, "lambda_l2": 1.0,
        "max_depth": 6, "verbose": -1, "n_jobs": -1,
        "feature_pre_filter": False,
    },
    "LGB_Rank": {
        "objective": "regression", "metric": "rmse",
        "learning_rate": 0.03, "num_leaves": 24,
        "feature_fraction": 0.5, "bagging_fraction": 0.7,
        "bagging_freq": 1, "min_child_samples": 150,
        "lambda_l1": 0.2, "lambda_l2": 2.0,
        "max_depth": 5, "verbose": -1, "n_jobs": -1,
        "feature_pre_filter": False,
    },
}

oof_preds = {}
test_preds = {}

for name, params in lgb_configs.items():
    t0 = time.time()
    oof = np.zeros(len(y))
    tpred = np.zeros(len(test_ids))
    if name == "LGB_Rank":
        Xm, Xm_test = X_rank, X_test_rank
        fnames = [f"{f}_rank" for f in sel_features]
    else:
        Xm, Xm_test = X_aug, X_test_aug
        fnames = aug_names
    for fold_i, gval in enumerate(unique_cv):
        val_mask = cv_groups == gval
        trn_mask = ~val_mask
        dtrain = lgb.Dataset(Xm[trn_mask], label=y[trn_mask], feature_name=fnames)
        dval = lgb.Dataset(Xm[val_mask], label=y[val_mask], reference=dtrain, feature_name=fnames)
        model = lgb.train(params, dtrain, num_boost_round=500,
                          valid_sets=[dval], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[val_mask] = model.predict(Xm[val_mask])
        tpred += model.predict(Xm_test) / N_FOLDS
    r2 = r2_score(y, oof)
    oof_preds[name] = oof
    test_preds[name] = tpred
    print(f'{name}: R2={r2:.6f} ({time.time()-t0:.0f}s)')

# 5. CatBoost models
from catboost import CatBoostRegressor

cb_configs = {
    "CB_RMSE": {
        "iterations": 500, "learning_rate": 0.05,
        "depth": 6, "l2_leaf_reg": 3.0,
        "random_strength": 0.5, "bagging_temperature": 0.3,
        "rsm": 0.6, "loss_function": "RMSE",
        "boosting_type": "Plain", "verbose": 0,
        "thread_count": -1,
    },
    "CB_MAE": {
        "iterations": 500, "learning_rate": 0.05,
        "depth": 5, "l2_leaf_reg": 5.0,
        "random_strength": 0.8, "bagging_temperature": 0.5,
        "rsm": 0.5, "loss_function": "MAE",
        "boosting_type": "Plain", "verbose": 0,
        "thread_count": -1,
    },
}

for name, params in cb_configs.items():
    t0 = time.time()
    oof = np.zeros(len(y))
    tpred = np.zeros(len(test_ids))
    for fold_i, gval in enumerate(unique_cv):
        val_mask = cv_groups == gval
        trn_mask = ~val_mask
        model = CatBoostRegressor(**params, random_seed=42+fold_i)
        model.fit(X_aug[trn_mask], y[trn_mask],
                  eval_set=(X_aug[val_mask], y[val_mask]),
                  early_stopping_rounds=50, verbose=0)
        oof[val_mask] = model.predict(X_aug[val_mask])
        tpred += model.predict(X_test_aug) / N_FOLDS
    r2 = r2_score(y, oof)
    oof_preds[name] = oof
    test_preds[name] = tpred
    print(f'{name}: R2={r2:.6f} ({time.time()-t0:.0f}s)')

# Strategy 4: Competition-metric-aware ensemble optimization
def competition_metric_oof(weights, oof_dict, y_true, cv_groups, shrinkage):
    names = list(oof_dict.keys())
    w = np.array(weights)
    w = np.abs(w) / (np.abs(w).sum() + 1e-12)
    blend = np.zeros(len(y_true))
    for i, name in enumerate(names):
        blend += w[i] * oof_dict[name]
    blend *= shrinkage
    unique_g = np.unique(cv_groups)
    batch_r2s = []
    for g in unique_g:
        mask = cv_groups == g
        if np.std(y_true[mask]) > 1e-12:
            r2 = r2_score(y_true[mask], blend[mask])
            batch_r2s.append(r2)
    mean_batch_r2 = np.mean(batch_r2s) if batch_r2s else 0.0
    full_r2 = r2_score(y_true, blend)
    score = 2.0 * mean_batch_r2 - full_r2
    return -score

# Filter candidates with R2 > -0.005
candidates = {k: v for k, v in oof_preds.items() if r2_score(y, v) > -0.005}
cand_names = list(candidates.keys())
print(f'\nEnsemble candidates: {cand_names}')

best_score = -999
best_weights = None
best_shrinkage = None

for shrink in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    n = len(cand_names)
    init_w = np.ones(n) / n
    result = minimize(
        competition_metric_oof, init_w,
        args=(candidates, y, cv_groups, shrink),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10}
    )
    score = -result.fun
    if score > best_score:
        best_score = score
        raw_w = np.abs(result.x)
        best_weights = raw_w / (raw_w.sum() + 1e-12)
        best_shrinkage = shrink

print(f'\nBest competition metric score (OOF): {best_score:.8f}')
print(f'Best shrinkage: {best_shrinkage}')
for i, name in enumerate(cand_names):
    print(f'  {name}: {best_weights[i]:.4f}')

# Build final blend
blend_oof = np.zeros(len(y))
blend_test = np.zeros(len(test_ids))
for i, name in enumerate(cand_names):
    blend_oof += best_weights[i] * candidates[name]
    blend_test += best_weights[i] * test_preds[name]
blend_oof *= best_shrinkage
blend_test *= best_shrinkage

# Strategy 3: Clip extreme predictions
pred_std = np.std(blend_oof)
pred_mean = np.mean(blend_oof)
clip_lo = pred_mean - 3.0 * pred_std
clip_hi = pred_mean + 3.0 * pred_std
blend_oof_clipped = np.clip(blend_oof, clip_lo, clip_hi)
blend_test_clipped = np.clip(blend_test, clip_lo, clip_hi)

# Calculate final OOF metrics with clipping
final_r2 = r2_score(y, blend_oof_clipped)
unique_g = np.unique(cv_groups)
batch_r2s = []
for g in unique_g:
    mask = cv_groups == g
    r2 = r2_score(y[mask], blend_oof_clipped[mask])
    batch_r2s.append(r2)
mean_batch_r2 = np.mean(batch_r2s)
simulated_score = 2.0 * mean_batch_r2 - final_r2
print(f'\nFinal OOF R2: {final_r2:.6f}')
print(f'Mean batch R2: {mean_batch_r2:.6f}')
print(f'Simulated competition score (clipped): {simulated_score:.8f}')
print(f'Clip range: [{clip_lo:.8f}, {clip_hi:.8f}]')

# Also try without clipping
final_r2_nc = r2_score(y, blend_oof)
batch_r2s_nc = []
for g in unique_g:
    mask = cv_groups == g
    batch_r2s_nc.append(r2_score(y[mask], blend_oof[mask]))
sim_score_nc = 2.0 * np.mean(batch_r2s_nc) - final_r2_nc
print(f'Simulated competition score (unclipped): {sim_score_nc:.8f}')

if simulated_score >= sim_score_nc:
    final_test = blend_test_clipped
    print('Using CLIPPED predictions')
else:
    final_test = blend_test
    print('Using UNCLIPPED predictions')

# 6. Save submission
sub = pd.DataFrame({id_col: test_ids, target_col: final_test})
sub.to_csv('submission.csv', index=False)
print(f'\nSubmission saved: {sub.shape}')
print(f'Pred stats: mean={final_test.mean():.8f}, std={final_test.std():.8f}')
print(f'Total runtime: {time.time()-T0:.0f}s')