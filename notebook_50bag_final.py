import numpy as np
import pandas as pd
import time, os, warnings, gc
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
warnings.filterwarnings('ignore')
T0 = time.time()

for p in ['/kaggle/input/competitions/short-horizon-return-prediction-challenge-by-i-rage/',
          '/kaggle/input/short-horizon-return-prediction-challenge-by-i-rage/']:
    if os.path.exists(p): DATA=p; break
else: raise FileNotFoundError('Data not found')

train = pd.read_parquet(DATA + 'train.parquet')
test = pd.read_parquet(DATA + 'test.parquet')
print(f'Train: {train.shape}, Test: {test.shape}')

y = train['TARGET'].values
groups = train['CV_GROUP'].values
test_ids = test['ID'].values
fc = [c for c in train.columns if c not in ['TARGET', 'CV_GROUP', 'ID']]
nf_total = len(fc)

# AGGRESSIVE target clipping at 90th percentile (not 99th)
# Removes more outliers = cleaner signal
clip90 = float(np.percentile(np.abs(y), 90))
clip95 = float(np.percentile(np.abs(y), 95))
y90 = np.clip(y, -clip90, clip90)
y95 = np.clip(y, -clip95, clip95)
print(f'Clip90: +/-{clip90:.6f}, Clip95: +/-{clip95:.6f}')

X = train[fc].values.astype(np.float32)
Xt = test[fc].values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
Xt = np.nan_to_num(Xt, nan=0, posinf=0, neginf=0)
del train, test; gc.collect()

ug = np.unique(groups)
nfolds = len(ug) if len(ug) <= 5 else 5
gkf = GroupKFold(n_splits=nfolds)

def cm(yt, yp, g):
    br = [r2_score(yt[g==v], yp[g==v]) for v in np.unique(g) if np.std(yt[g==v]) > 1e-12]
    return 2 * np.mean(br) - r2_score(yt, yp)

# ============================================================
# 50-BAG ULTRA-WEAK ENSEMBLE
# V26 got OOF R2=+0.0001 with 30 bags
# More bags + weaker models = less overfitting
# ============================================================
print(f'\n=== 50-BAG ULTRA-WEAK ENSEMBLE ===')

N_BAGS = 50
FEAT_FRAC = 0.25  # 25% of features per bag

# Ultra-weak: 7 leaves, depth 3, 1000 min samples
pp = {
    'objective': 'huber', 'huber_delta': 0.3,
    'metric': 'rmse', 'verbosity': -1, 'n_jobs': -1,
    'learning_rate': 0.02, 'num_leaves': 7,
    'max_depth': 3, 'min_child_samples': 1000,
    'subsample': 0.5, 'colsample_bytree': 1.0,
    'reg_alpha': 10.0, 'reg_lambda': 10.0,
    'min_gain_to_split': 0.1,
    'feature_pre_filter': False,
}

n_feat_per_bag = max(10, int(nf_total * FEAT_FRAC))

# Run with BOTH target clips to find which generalizes better
for clip_name, y_target in [('y90', y90), ('y95', y95)]:
    print(f'\n--- Target: {clip_name} ---')
    all_oof = np.zeros((len(y), N_BAGS))
    all_test = np.zeros((len(Xt), N_BAGS))
    t0 = time.time()
    
    for bag in range(N_BAGS):
        rng = np.random.RandomState(42 + bag)
        feat_idx = rng.choice(nf_total, n_feat_per_bag, replace=False)
        X_sub = X[:, feat_idx]
        Xt_sub = Xt[:, feat_idx]
        
        oof_bag = np.zeros(len(y))
        test_bag = np.zeros(len(Xt))
        
        for ti, vi in gkf.split(X_sub, y_target, groups):
            dt = lgb.Dataset(X_sub[ti], y_target[ti])
            dv = lgb.Dataset(X_sub[vi], y_target[vi], reference=dt)
            m = lgb.train(pp, dt, num_boost_round=200, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(20, verbose=False),
                                     lgb.log_evaluation(0)])
            oof_bag[vi] = m.predict(X_sub[vi])
            test_bag += m.predict(Xt_sub) / nfolds
        
        all_oof[:, bag] = oof_bag
        all_test[:, bag] = test_bag
        
        if (bag + 1) % 25 == 0:
            avg = all_oof[:, :bag+1].mean(axis=1)
            print(f'  Bag {bag+1}: R2={r2_score(y,avg):.6f} CM={cm(y,avg,groups):.8f} [{time.time()-t0:.0f}s]')
    
    oof_avg = all_oof.mean(axis=1)
    test_avg = all_test.mean(axis=1)
    r2_val = r2_score(y, oof_avg)
    cm_val = cm(y, oof_avg, groups)
    print(f'  Final: R2={r2_val:.6f} CM={cm_val:.8f}')
    
    if clip_name == 'y90':
        oof_90 = oof_avg; test_90 = test_avg; all_oof_90 = all_oof; all_test_90 = all_test
    else:
        oof_95 = oof_avg; test_95 = test_avg; all_oof_95 = all_oof; all_test_95 = all_test

# ============================================================
# TRY CATBOOST (handles overfitting better than LGB)
# ============================================================
print(f'\n=== CATBOOST ===')
try:
    from catboost import CatBoostRegressor, Pool
    t0 = time.time()
    oof_cb = np.zeros(len(y))
    test_cb = np.zeros(len(Xt))
    
    for ti, vi in gkf.split(X, y95, groups):
        cb = CatBoostRegressor(
            iterations=300, learning_rate=0.03, depth=4,
            l2_leaf_reg=10, random_seed=42, verbose=0,
            loss_function='RMSE', early_stopping_rounds=30,
            subsample=0.6, colsample_bylevel=0.4
        )
        cb.fit(X[ti], y95[ti], eval_set=(X[vi], y95[vi]), verbose=0)
        oof_cb[vi] = cb.predict(X[vi])
        test_cb += cb.predict(Xt) / nfolds
    
    r2_cb = r2_score(y, oof_cb)
    cm_cb = cm(y, oof_cb, groups)
    print(f'  CatBoost: R2={r2_cb:.6f} CM={cm_cb:.8f} [{time.time()-t0:.0f}s]')
    has_cb = True
except:
    print('  CatBoost not available')
    has_cb = False

# ============================================================
# OPTIMIZE AND SELECT
# ============================================================
print(f'\n=== OPTIMIZATION ===')
zero_cm = cm(y, np.zeros(len(y)), groups)
print(f'Zero: {zero_cm:.8f}')

candidates = {}
candidates['zero'] = (zero_cm, np.zeros(len(test_ids)))

# Bagged y90
for s in np.arange(0.5, 1.51, 0.01):
    sc = cm(y, oof_90 * s, groups)
    if 'bag90' not in candidates or sc > candidates['bag90'][0]:
        candidates['bag90'] = (sc, test_90 * s)

# Bagged y95
for s in np.arange(0.5, 1.51, 0.01):
    sc = cm(y, oof_95 * s, groups)
    if 'bag95' not in candidates or sc > candidates['bag95'][0]:
        candidates['bag95'] = (sc, test_95 * s)

# Blend y90 + y95
for a in np.arange(0, 1.01, 0.1):
    bl_oof = a * oof_90 + (1-a) * oof_95
    bl_test = a * test_90 + (1-a) * test_95
    for s in np.arange(0.5, 1.51, 0.05):
        sc = cm(y, bl_oof * s, groups)
        if 'blend' not in candidates or sc > candidates['blend'][0]:
            candidates['blend'] = (sc, bl_test * s)

# CatBoost
if has_cb:
    for s in np.arange(0.01, 1.01, 0.01):
        sc = cm(y, oof_cb * s, groups)
        if 'catboost' not in candidates or sc > candidates['catboost'][0]:
            candidates['catboost'] = (sc, test_cb * s)

# Blend bagged + catboost
if has_cb:
    for a in np.arange(0, 1.01, 0.1):
        bl_oof = a * oof_90 + (1-a) * oof_cb
        bl_test = a * test_90 + (1-a) * test_cb
        for s in np.arange(0.01, 1.01, 0.05):
            sc = cm(y, bl_oof * s, groups)
            if 'bag_cb' not in candidates or sc > candidates['bag_cb'][0]:
                candidates['bag_cb'] = (sc, bl_test * s)

# Agreement-based from y90
signs_90 = np.sign(all_oof_90)
agreement_90 = np.abs(signs_90.sum(axis=1)) / N_BAGS
test_signs_90 = np.sign(all_test_90)
test_agree_90 = np.abs(test_signs_90.sum(axis=1)) / N_BAGS

for min_a in [0.6, 0.7, 0.8]:
    oof_ag = np.where(agreement_90 >= min_a, oof_90, 0)
    test_ag = np.where(test_agree_90 >= min_a, test_90, 0)
    sc = cm(y, oof_ag, groups)
    candidates[f'agree_{min_a}'] = (sc, test_ag)

print(f'\nAll candidates:')
for name in sorted(candidates, key=lambda k: -candidates[k][0]):
    print(f'  {name}: {candidates[name][0]:.8f}')

best_name = max(candidates, key=lambda k: candidates[k][0])
best_score, final = candidates[best_name]
print(f'\nBest: {best_name} (CM={best_score:.8f})')

sub = pd.DataFrame({'ID': test_ids, 'TARGET': final})
sub.to_csv('submission.csv', index=False)
nz = np.sum(final != 0)
print(f'\nSub: {sub.shape} mean={final.mean():.6f} std={final.std():.6f} nonzero={nz}')
print(f'Time: {time.time()-T0:.0f}s ({(time.time()-T0)/60:.1f}min)')
print('Done!')
