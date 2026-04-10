import numpy as np
import pandas as pd
import time, os, warnings, gc
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
warnings.filterwarnings('ignore')

T0 = time.time()

for p in ['/kaggle/input/competitions/short-horizon-return-prediction-challenge-by-i-rage/',
          '/kaggle/input/short-horizon-return-prediction-challenge-by-i-rage/']:
    if os.path.exists(p): DATA=p; break
else: raise FileNotFoundError('Data not found')

train = pd.read_parquet(DATA + 'train.parquet')
test = pd.read_parquet(DATA + 'test.parquet')
print(f'Train: {train.shape}, Test: {test.shape} [{time.time()-T0:.0f}s]')

y = train['TARGET'].values
groups = train['CV_GROUP'].values
test_ids = test['ID'].values
fc = [c for c in train.columns if c not in ['TARGET', 'CV_GROUP', 'ID']]

# TARGET = 100 * (Price[t+H] - Price[t]) / Price[t]
# LagT1 = current - value_T1_ago (DIFFERENCES not levels)
print(f'Target: mean={y.mean():.6f} std={y.std():.6f} min={y.min():.4f} max={y.max():.4f}')

# Clip outliers at 99th percentile
clip = float(np.percentile(np.abs(y), 99))
y_clip = np.clip(y, -clip, clip)
print(f'Clipped at 99th: +/-{clip:.4f}')

X = train[fc].values.astype(np.float32)
Xt = test[fc].values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
Xt = np.nan_to_num(Xt, nan=0, posinf=0, neginf=0)
del train, test; gc.collect()

ug = np.unique(groups)
nf = len(ug) if len(ug) <= 5 else 5
gkf = GroupKFold(n_splits=nf)
print(f'CV: {nf}-fold, {len(ug)} groups, {X.shape[1]} features')

# ============================================================
# THE KEY INSIGHT: Optimize for ACTUAL R2, not batch metric!
# R2 = 1 - SS_res / SS_tot
# We want predictions that REDUCE residual sum of squares
# NOT predictions close to zero
# ============================================================

oof_all = {}
test_all = {}

# ============================================================
# 1. LightGBM with different configs (NO excessive shrinkage)
# ============================================================
print(f'\n=== LIGHTGBM MODELS ===')

lgb_configs = {
    'LGB_Huber': {
        'objective': 'huber', 'huber_delta': 1.0,
        'metric': 'rmse', 'verbosity': -1, 'n_jobs': -1,
        'learning_rate': 0.05, 'num_leaves': 31,
        'max_depth': 6, 'min_child_samples': 100,
        'subsample': 0.8, 'colsample_bytree': 0.6,
        'reg_alpha': 0.5, 'reg_lambda': 1.0,
        'feature_pre_filter': False,
    },
    'LGB_MSE': {
        'objective': 'regression', 'metric': 'rmse',
        'verbosity': -1, 'n_jobs': -1,
        'learning_rate': 0.05, 'num_leaves': 31,
        'max_depth': 6, 'min_child_samples': 100,
        'subsample': 0.8, 'colsample_bytree': 0.6,
        'reg_alpha': 0.5, 'reg_lambda': 1.0,
        'feature_pre_filter': False,
    },
    'LGB_MAE': {
        'objective': 'mae', 'metric': 'mae',
        'verbosity': -1, 'n_jobs': -1,
        'learning_rate': 0.05, 'num_leaves': 24,
        'max_depth': 5, 'min_child_samples': 150,
        'subsample': 0.7, 'colsample_bytree': 0.5,
        'reg_alpha': 1.0, 'reg_lambda': 2.0,
        'feature_pre_filter': False,
    },
    'LGB_Deep': {
        'objective': 'huber', 'huber_delta': 0.5,
        'metric': 'rmse', 'verbosity': -1, 'n_jobs': -1,
        'learning_rate': 0.03, 'num_leaves': 63,
        'max_depth': 8, 'min_child_samples': 50,
        'subsample': 0.7, 'colsample_bytree': 0.5,
        'reg_alpha': 1.0, 'reg_lambda': 2.0,
        'feature_pre_filter': False,
    },
}

BOOST = 800
ES = 50

for nm, pp in lgb_configs.items():
    t0 = time.time()
    oo = np.zeros(len(y))
    tp = np.zeros(len(Xt))
    for _, (ti, vi) in enumerate(gkf.split(X, y_clip, groups)):
        dt = lgb.Dataset(X[ti], y_clip[ti])
        dv = lgb.Dataset(X[vi], y_clip[vi], reference=dt)
        m = lgb.train(pp, dt, num_boost_round=BOOST, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(ES, verbose=False),
                                 lgb.log_evaluation(0)])
        oo[vi] = m.predict(X[vi])
        tp += m.predict(Xt) / nf
    oof_all[nm] = oo
    test_all[nm] = tp
    r2 = r2_score(y, oo)
    print(f'  {nm}: OOF R2={r2:.6f} [{time.time()-t0:.0f}s]')

# ============================================================
# 2. Ridge regression (fast, stable)
# ============================================================
print(f'\n=== RIDGE MODELS ===')
for alpha in [10, 100, 1000]:
    key = f'Ridge_a{alpha}'
    t0 = time.time()
    oo = np.zeros(len(y))
    tp = np.zeros(len(Xt))
    for fi, (ti, vi) in enumerate(gkf.split(X, y_clip, groups)):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[ti])
        Xva = sc.transform(X[vi])
        Xte = sc.transform(Xt)
        model = Ridge(alpha=alpha)
        model.fit(Xtr, y_clip[ti])
        oo[vi] = model.predict(Xva)
        tp += model.predict(Xte) / nf
    oof_all[key] = oo
    test_all[key] = tp
    r2 = r2_score(y, oo)
    print(f'  {key}: OOF R2={r2:.6f} [{time.time()-t0:.0f}s]')

print(f'\nAll models done [{time.time()-T0:.0f}s]')

# ============================================================
# 3. OPTIMIZE FOR ACTUAL R2 (not batch metric!)
# ============================================================
print(f'\n=== R2 OPTIMIZATION ===')

# Find best single model by R2
best_r2 = -999
best_key = None
for key, oo in oof_all.items():
    r2 = r2_score(y, oo)
    print(f'  {key}: R2={r2:.6f}')
    if r2 > best_r2:
        best_r2 = r2
        best_key = key

print(f'\nBest single: {best_key} R2={best_r2:.6f}')

# Try shrinkage (but optimize R2, not batch metric)
best_r2_shrunk = -999
best_shrinkage = 1.0
for s in np.arange(0.5, 1.51, 0.01):
    r2 = r2_score(y, oof_all[best_key] * s)
    if r2 > best_r2_shrunk:
        best_r2_shrunk = r2
        best_shrinkage = s
print(f'Best shrinkage: {best_shrinkage:.2f} R2={best_r2_shrunk:.6f}')

# ============================================================
# 4. ENSEMBLE: Optimize weights for R2
# ============================================================
print(f'\n=== ENSEMBLE OPTIMIZATION ===')
from scipy.optimize import minimize

names = list(oof_all.keys())
n_models = len(names)
oof_matrix = np.column_stack([oof_all[k] for k in names])
test_matrix = np.column_stack([test_all[k] for k in names])

def neg_r2(weights):
    w = np.abs(weights)
    w = w / (w.sum() + 1e-12)
    pred = oof_matrix @ w
    return -r2_score(y, pred)

best_result = None
best_neg = float('inf')
for trial in range(20):
    np.random.seed(trial)
    w0 = np.random.dirichlet(np.ones(n_models))
    res = minimize(neg_r2, w0, method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
    if res.fun < best_neg:
        best_neg = res.fun
        best_result = res

raw_w = np.abs(best_result.x)
best_weights = raw_w / (raw_w.sum() + 1e-12)
ensemble_r2 = -best_neg

print(f'Ensemble R2: {ensemble_r2:.6f}')
for i, nm in enumerate(names):
    if best_weights[i] > 0.01:
        print(f'  {nm}: {best_weights[i]:.4f}')

# Ensemble predictions
oof_ensemble = oof_matrix @ best_weights
test_ensemble = test_matrix @ best_weights

# Try shrinkage on ensemble
best_ens_r2 = -999
best_ens_s = 1.0
for s in np.arange(0.5, 1.51, 0.01):
    r2 = r2_score(y, oof_ensemble * s)
    if r2 > best_ens_r2:
        best_ens_r2 = r2
        best_ens_s = s
print(f'Ensemble + shrinkage: s={best_ens_s:.2f} R2={best_ens_r2:.6f}')

# ============================================================
# 5. PICK BEST AND SUBMIT
# ============================================================
print(f'\n=== FINAL COMPARISON ===')
zero_r2 = r2_score(y, np.zeros(len(y)))
print(f'Zero baseline R2: {zero_r2:.6f}')
print(f'Best single R2:   {best_r2_shrunk:.6f} ({best_key} s={best_shrinkage:.2f})')
print(f'Ensemble R2:      {best_ens_r2:.6f} (s={best_ens_s:.2f})')

if best_ens_r2 > best_r2_shrunk and best_ens_r2 > zero_r2:
    final = test_ensemble * best_ens_s
    print('Using ENSEMBLE')
elif best_r2_shrunk > zero_r2:
    final = test_all[best_key] * best_shrinkage
    print(f'Using SINGLE: {best_key}')
else:
    final = np.zeros(len(test_ids))
    print('Using ZERO (no model beats baseline)')

# Clip extreme predictions
if np.std(final) > 0:
    p99 = np.percentile(np.abs(final[final!=0]), 99) if np.any(final!=0) else 0
    if p99 > 0:
        final = np.clip(final, -p99*1.5, p99*1.5)

sub = pd.DataFrame({'ID': test_ids, 'TARGET': final})
sub.to_csv('submission.csv', index=False)
print(f'\nSub: {sub.shape} mean={final.mean():.6f} std={final.std():.6f}')
print(f'Non-zero: {np.sum(final != 0)} / {len(final)}')
print(f'Time: {time.time()-T0:.0f}s ({(time.time()-T0)/60:.1f}min)')
print('Done!')
# V22: R2-optimized model with scipy ensemble optimization
# Score: -0.00085 (overfit - too aggressive)
# Key: 4 LGB configs + 3 Ridge, scipy-optimized ensemble weights
# Lesson: Aggressive models overfit on this low-SNR data

print('V22: R2-optimized 4xLGB + 3xRidge ensemble with scipy weight optimization')
