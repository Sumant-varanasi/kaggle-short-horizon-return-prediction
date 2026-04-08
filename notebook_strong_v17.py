import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import gc, time, warnings, os
warnings.filterwarnings('ignore')

T0 = time.time()

# ============================================================
# 1. LOAD DATA
# ============================================================
for p in ['/kaggle/input/competitions/short-horizon-return-prediction-challenge-by-i-rage/',
          '/kaggle/input/short-horizon-return-prediction-challenge-by-i-rage/']:
    if os.path.exists(p):
        DATA = p
        break
else:
    raise FileNotFoundError('Data not found')

train = pd.read_parquet(DATA + 'train.parquet')
test = pd.read_parquet(DATA + 'test.parquet')
print(f'Train: {train.shape}, Test: {test.shape} [{time.time()-T0:.0f}s]')

y_raw = train['TARGET'].values
groups = train['CV_GROUP'].values
test_ids = test['ID'].values
fc = [c for c in train.columns if c not in ['TARGET', 'CV_GROUP', 'ID']]

# ============================================================
# 2. TARGETED FEATURE ENGINEERING (only high-impact, cheap)
# ============================================================
# Price lag ratios - directly related to target
if 'Price' in fc:
    for lag in ['Price_LagT1', 'Price_LagT2', 'Price_LagT3']:
        if lag in fc:
            n = lag + '_ratio'
            train[n] = train[lag] / train['Price'].replace(0, np.nan)
            test[n] = test[lag] / test['Price'].replace(0, np.nan)
            fc.append(n)

# SO3_T interaction
if 'SO3_T' in train.columns:
    train['SO3T_xP'] = train['SO3_T'] * train['Price']
    test['SO3T_xP'] = test['SO3_T'] * test['Price']
    fc.append('SO3T_xP')

# Lag acceleration features (only for top features - cheap, high signal)
lag1 = [c for c in fc if c.endswith('_LagT1')]
for l1 in lag1:
    base = l1.replace('_LagT1', '')
    l2 = base + '_LagT2'
    l3 = base + '_LagT3'
    if l2 in fc:
        n = base + '_acc12'
        train[n] = train[l2] - train[l1]
        test[n] = test[l2] - test[l1]
        fc.append(n)
    if l3 in fc and l2 in fc:
        n = base + '_acc23'
        train[n] = train[l3] - train[l2]
        test[n] = test[l3] - test[l2]
        fc.append(n)

print(f'Features: {len(fc)} [{time.time()-T0:.0f}s]')

# ============================================================
# 3. TARGET CLIPPING
# ============================================================
cv = float(np.percentile(np.abs(y_raw), 99))
y = np.clip(y_raw, -cv, cv)
print(f'Target clipped +/-{cv:.6f}')

X = train[fc].values.astype(np.float32)
Xt = test[fc].values.astype(np.float32)
del train, test
gc.collect()

# ============================================================
# 4. CV SETUP
# ============================================================
ug = np.unique(groups)
nf = len(ug) if len(ug) <= 5 else 5
gkf = GroupKFold(n_splits=nf)
print(f'CV: {nf}-fold GroupKFold, {len(ug)} groups')

# ============================================================
# 5. THREE BEST MODELS (from V15 analysis)
# ============================================================
# Huber was best individual, MAE + Aggressive completed the ensemble
configs = {
    'Huber': {
        'objective': 'huber', 'huber_delta': 0.5,
        'metric': 'rmse', 'verbosity': -1, 'n_jobs': -1,
        'learning_rate': 0.03, 'num_leaves': 31,
        'max_depth': 6, 'min_child_samples': 300,
        'subsample': 0.7, 'colsample_bytree': 0.5,
        'reg_alpha': 1.5, 'reg_lambda': 2.5,
        'min_gain_to_split': 0.005,
        'feature_pre_filter': False,
    },
    'Aggressive': {
        'objective': 'regression', 'metric': 'rmse',
        'verbosity': -1, 'n_jobs': -1,
        'learning_rate': 0.03, 'num_leaves': 31,
        'max_depth': 6, 'min_child_samples': 200,
        'subsample': 0.7, 'colsample_bytree': 0.5,
        'reg_alpha': 1.0, 'reg_lambda': 2.0,
        'min_gain_to_split': 0.001,
        'feature_pre_filter': False,
    },
    'MAE': {
        'objective': 'mae', 'metric': 'mae',
        'verbosity': -1, 'n_jobs': -1,
        'learning_rate': 0.03, 'num_leaves': 24,
        'max_depth': 5, 'min_child_samples': 300,
        'subsample': 0.65, 'colsample_bytree': 0.45,
        'reg_alpha': 2.0, 'reg_lambda': 3.0,
        'feature_pre_filter': False,
    },
}

BOOST = 600
ES = 40
oa = {}
ta = {}

for nm, pp in configs.items():
    t0 = time.time()
    oo = np.zeros(len(y))
    tp = np.zeros(len(Xt))
    for _, (ti, vi) in enumerate(gkf.split(X, y, groups)):
        dt = lgb.Dataset(X[ti], y[ti])
        dv = lgb.Dataset(X[vi], y[vi], reference=dt)
        m = lgb.train(pp, dt, num_boost_round=BOOST, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(ES, verbose=False),
                                 lgb.log_evaluation(0)])
        oo[vi] = m.predict(X[vi])
        tp += m.predict(Xt) / nf
    oa[nm] = oo
    ta[nm] = tp
    print(f'  {nm}: R2={r2_score(y, oo):.6f} [{time.time()-t0:.0f}s]')

print(f'Models done [{time.time()-T0:.0f}s]')

# ============================================================
# 6. COMPETITION METRIC OPTIMIZATION
# ============================================================
def cm(yt, yp, g):
    br = [r2_score(yt[g == v], yp[g == v])
          for v in np.unique(g) if np.std(yt[g == v]) > 1e-12]
    return 2 * np.mean(br) - r2_score(yt, yp)

# Optimize 3 weights + shrinkage via grid search
names = list(oa.keys())
bs = -999
bw = [0.33, 0.33, 0.34]
bsh = 0.5

# Coarse grid
for w0 in np.arange(0, 1.01, 0.1):
    for w1 in np.arange(0, 1.01 - w0, 0.1):
        w2 = 1.0 - w0 - w1
        if w2 < -0.01:
            continue
        w2 = max(w2, 0)
        bl = w0 * oa[names[0]] + w1 * oa[names[1]] + w2 * oa[names[2]]
        for s in np.arange(0.05, 1.01, 0.05):
            sc = cm(y, bl * s, groups)
            if sc > bs:
                bs = sc
                bw = [w0, w1, w2]
                bsh = s

# Fine grid around best
for w0 in np.arange(max(0, bw[0]-0.1), min(1, bw[0]+0.11), 0.02):
    for w1 in np.arange(max(0, bw[1]-0.1), min(1-w0, bw[1]+0.11), 0.02):
        w2 = 1.0 - w0 - w1
        if w2 < -0.01 or w2 > 1.01:
            continue
        w2 = np.clip(w2, 0, 1)
        bl = w0 * oa[names[0]] + w1 * oa[names[1]] + w2 * oa[names[2]]
        for s in np.arange(max(0.05, bsh-0.1), min(1.0, bsh+0.11), 0.01):
            sc = cm(y, bl * s, groups)
            if sc > bs:
                bs = sc
                bw = [w0, w1, w2]
                bsh = s

print(f'\nBest: {names[0]}={bw[0]:.3f} {names[1]}={bw[1]:.3f} {names[2]}={bw[2]:.3f}')
print(f'Shrinkage={bsh:.3f} Score={bs:.8f}')

# ============================================================
# 7. ZERO BASELINE CHECK + FINAL PREDICTIONS
# ============================================================
zs = cm(y, np.zeros(len(y)), groups)
print(f'Zero: {zs:.8f}, Ours: {bs:.8f}')

fo = (bw[0]*oa[names[0]] + bw[1]*oa[names[1]] + bw[2]*oa[names[2]]) * bsh
ft = (bw[0]*ta[names[0]] + bw[1]*ta[names[1]] + bw[2]*ta[names[2]]) * bsh

if bs < zs:
    print('Blending towards zero...')
    ba = 0.0
    bbs = bs
    for a in np.arange(0, 1.01, 0.05):
        sc = cm(y, fo * (1-a), groups)
        if sc > bbs:
            bbs = sc
            ba = a
    if ba > 0:
        ft = ft * (1-ba)
        print(f'Blended {ba:.0%} to zero. Score: {bbs:.8f}')

# ============================================================
# 8. SUBMISSION
# ============================================================
sub = pd.DataFrame({'ID': test_ids, 'TARGET': ft})
sub.to_csv('submission.csv', index=False)
print(f'\nSub: {sub.shape} mean={ft.mean():.8f} std={ft.std():.8f}')
print(f'Time: {time.time()-T0:.0f}s ({(time.time()-T0)/60:.1f}min)')
print('Done!')
