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
clip = float(np.percentile(np.abs(y), 99))
y_clip = np.clip(y, -clip, clip)

# ============================================================
# 1. QUANT FEATURE ENGINEERING
# ============================================================
# Key insight: TARGET is % return, but LagT features are raw diffs
# We need to convert diffs to % changes to match the target space
print(f'\n=== QUANT FEATURE ENGINEERING ===')
t0 = time.time()

# Identify base features and their lags
base_features = set()
for c in fc:
    for sfx in ['_LagT1', '_LagT2', '_LagT3']:
        if c.endswith(sfx):
            base_features.add(c.replace(sfx, ''))
base_features = sorted([b for b in base_features if b in fc])
print(f'Base features with lags: {len(base_features)}')

new_train_feats = {}
new_test_feats = {}

for base in base_features:
    curr = train[base].values if base in fc else None
    curr_t = test[base].values if base in fc else None
    l1 = train.get(base + '_LagT1')
    l2 = train.get(base + '_LagT2')
    l3 = train.get(base + '_LagT3')
    l1t = test.get(base + '_LagT1')
    l2t = test.get(base + '_LagT2')
    l3t = test.get(base + '_LagT3')
    
    if curr is None or l1 is None:
        continue
    l1 = l1.values; l1t = l1t.values
    
    # Reconstruct past values: past = current - lag_diff
    past1 = curr - l1
    past1_t = curr_t - l1t
    
    # PERCENTAGE CHANGE features (matching target space)
    # pct_T1 = (current - past1) / |past1| = LagT1 / |past1|
    denom = np.where(np.abs(past1) > 1e-10, past1, np.nan)
    denom_t = np.where(np.abs(past1_t) > 1e-10, past1_t, np.nan)
    pct1 = l1 / denom
    pct1_t = l1t / denom_t
    new_train_feats[f'{base}_pct1'] = np.nan_to_num(pct1, 0, posinf=0, neginf=0)
    new_test_feats[f'{base}_pct1'] = np.nan_to_num(pct1_t, 0, posinf=0, neginf=0)
    
    if l2 is not None:
        l2 = l2.values; l2t = l2t.values
        past2 = curr - l2
        past2_t = curr_t - l2t
        denom2 = np.where(np.abs(past2) > 1e-10, past2, np.nan)
        denom2_t = np.where(np.abs(past2_t) > 1e-10, past2_t, np.nan)
        pct2 = l2 / denom2
        pct2_t = l2t / denom2_t
        new_train_feats[f'{base}_pct2'] = np.nan_to_num(pct2, 0, posinf=0, neginf=0)
        new_test_feats[f'{base}_pct2'] = np.nan_to_num(pct2_t, 0, posinf=0, neginf=0)
        
        # MOMENTUM: acceleration = LagT1 - LagT2 (rate of change of change)
        acc = l1 - l2
        acc_t = l1t - l2t
        new_train_feats[f'{base}_acc'] = np.nan_to_num(acc, 0)
        new_test_feats[f'{base}_acc'] = np.nan_to_num(acc_t, 0)
        
        # MEAN REVERSION signal: if recent change (T1) is opposite of longer change (T2)
        mr = np.sign(l1) * np.sign(l2)
        mr_t = np.sign(l1t) * np.sign(l2t)
        new_train_feats[f'{base}_mr'] = mr.astype(np.float32)
        new_test_feats[f'{base}_mr'] = mr_t.astype(np.float32)

print(f'New quant features: {len(new_train_feats)} [{time.time()-t0:.0f}s]')

# Build feature matrices
X_raw = train[fc].values.astype(np.float32)
Xt_raw = test[fc].values.astype(np.float32)

if new_train_feats:
    X_quant = np.column_stack(list(new_train_feats.values())).astype(np.float32)
    Xt_quant = np.column_stack(list(new_test_feats.values())).astype(np.float32)
    X_quant = np.nan_to_num(X_quant, nan=0, posinf=0, neginf=0)
    Xt_quant = np.nan_to_num(Xt_quant, nan=0, posinf=0, neginf=0)
    # Clip extreme quant features
    for i in range(X_quant.shape[1]):
        p99 = np.percentile(np.abs(X_quant[:, i]), 99)
        if p99 > 0:
            X_quant[:, i] = np.clip(X_quant[:, i], -p99*2, p99*2)
            Xt_quant[:, i] = np.clip(Xt_quant[:, i], -p99*2, p99*2)
else:
    X_quant = np.zeros((len(y), 1), dtype=np.float32)
    Xt_quant = np.zeros((len(test_ids), 1), dtype=np.float32)

# Combined: raw + quant
X_all = np.hstack([X_raw, X_quant])
Xt_all = np.hstack([Xt_raw, Xt_quant])
X_all = np.nan_to_num(X_all, nan=0, posinf=0, neginf=0)
Xt_all = np.nan_to_num(Xt_all, nan=0, posinf=0, neginf=0)

del train, test; gc.collect()
print(f'Total features: raw={X_raw.shape[1]} quant={X_quant.shape[1]} combined={X_all.shape[1]}')

ug = np.unique(groups)
nf = len(ug) if len(ug) <= 5 else 5
gkf = GroupKFold(n_splits=nf)

def cm(yt, yp, g):
    br = [r2_score(yt[g==v], yp[g==v]) for v in np.unique(g) if np.std(yt[g==v]) > 1e-12]
    return 2 * np.mean(br) - r2_score(yt, yp)

oof_all = {}
test_all = {}

# ============================================================
# 2. MODELS ON QUANT FEATURES ONLY
# ============================================================
print(f'\n=== QUANT-ONLY MODELS ===')

# Ridge on quant features (pct changes, momentum, mean-reversion)
for alpha in [500, 5000, 50000]:
    key = f'Ridge_quant_a{alpha}'
    t0 = time.time()
    oo = np.zeros(len(y)); tp = np.zeros(len(Xt_quant))
    for fi, (ti, vi) in enumerate(gkf.split(X_quant, y_clip, groups)):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_quant[ti]); Xva = sc.transform(X_quant[vi]); Xte = sc.transform(Xt_quant)
        m = Ridge(alpha=alpha); m.fit(Xtr, y_clip[ti])
        oo[vi] = m.predict(Xva); tp += m.predict(Xte) / nf
    oof_all[key] = oo; test_all[key] = tp
    print(f'  {key}: R2={r2_score(y,oo):.6f} CM={cm(y,oo,groups):.8f} [{time.time()-t0:.0f}s]')

# ============================================================
# 3. MODELS ON COMBINED (RAW + QUANT)
# ============================================================
print(f'\n=== COMBINED MODELS ===')

for alpha in [5000, 50000]:
    key = f'Ridge_all_a{alpha}'
    t0 = time.time()
    oo = np.zeros(len(y)); tp = np.zeros(len(Xt_all))
    for fi, (ti, vi) in enumerate(gkf.split(X_all, y_clip, groups)):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_all[ti]); Xva = sc.transform(X_all[vi]); Xte = sc.transform(Xt_all)
        m = Ridge(alpha=alpha); m.fit(Xtr, y_clip[ti])
        oo[vi] = m.predict(Xva); tp += m.predict(Xte) / nf
    oof_all[key] = oo; test_all[key] = tp
    print(f'  {key}: R2={r2_score(y,oo):.6f} CM={cm(y,oo,groups):.8f} [{time.time()-t0:.0f}s]')

# LGB on quant features
print(f'\n=== LGB QUANT ===')
pp = {'objective':'huber','huber_delta':0.5,'metric':'rmse','verbosity':-1,'n_jobs':-1,
      'learning_rate':0.03,'num_leaves':15,'max_depth':4,'min_child_samples':500,
      'subsample':0.6,'colsample_bytree':0.4,'reg_alpha':5.0,'reg_lambda':5.0,
      'min_gain_to_split':0.05,'feature_pre_filter':False}

key = 'LGB_quant'
t0 = time.time()
oo = np.zeros(len(y)); tp = np.zeros(len(Xt_quant))
for _, (ti, vi) in enumerate(gkf.split(X_quant, y_clip, groups)):
    dt = lgb.Dataset(X_quant[ti], y_clip[ti]); dv = lgb.Dataset(X_quant[vi], y_clip[vi], reference=dt)
    m = lgb.train(pp, dt, num_boost_round=500, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
    oo[vi] = m.predict(X_quant[vi]); tp += m.predict(Xt_quant) / nf
oof_all[key] = oo; test_all[key] = tp
print(f'  {key}: R2={r2_score(y,oo):.6f} CM={cm(y,oo,groups):.8f} [{time.time()-t0:.0f}s]')

# LGB on combined
key = 'LGB_all'
t0 = time.time()
oo = np.zeros(len(y)); tp = np.zeros(len(Xt_all))
for _, (ti, vi) in enumerate(gkf.split(X_all, y_clip, groups)):
    dt = lgb.Dataset(X_all[ti], y_clip[ti]); dv = lgb.Dataset(X_all[vi], y_clip[vi], reference=dt)
    m = lgb.train(pp, dt, num_boost_round=500, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
    oo[vi] = m.predict(X_all[vi]); tp += m.predict(Xt_all) / nf
oof_all[key] = oo; test_all[key] = tp
print(f'  {key}: R2={r2_score(y,oo):.6f} CM={cm(y,oo,groups):.8f} [{time.time()-t0:.0f}s]')

print(f'\nAll done [{time.time()-T0:.0f}s]')

# ============================================================
# 4. SHRINKAGE + BLEND + SUBMIT
# ============================================================
print(f'\n=== OPTIMIZATION ===')
zero_cm = cm(y, np.zeros(len(y)), groups)
print(f'Zero CM: {zero_cm:.8f}')

results = []
for key, oo in oof_all.items():
    best_s = 0; best_sc = -999
    for s in np.arange(0.01, 1.01, 0.01):
        sc = cm(y, oo * s, groups)
        if sc > best_sc: best_sc = sc; best_s = s
    results.append((key, best_sc, best_s))
    print(f'  {key}: s={best_s:.2f} CM={best_sc:.8f}')
results.sort(key=lambda x: -x[1])

k1, cm1, s1 = results[0]; k2, cm2, s2 = results[1]
print(f'\nTop: {k1} (CM={cm1:.8f}) + {k2} (CM={cm2:.8f})')

best_blend = -999; best_ba = 0; best_bs = 0
for a in np.arange(0, 1.01, 0.05):
    bl = a * oof_all[k1] + (1-a) * oof_all[k2]
    for s in np.arange(0.01, 0.81, 0.01):
        sc = cm(y, bl * s, groups)
        if sc > best_blend: best_blend = sc; best_ba = a; best_bs = s
print(f'Blend: a={best_ba:.2f} s={best_bs:.2f} CM={best_blend:.8f}')

if best_blend > results[0][1] and best_blend > zero_cm:
    final = (best_ba * test_all[k1] + (1-best_ba) * test_all[k2]) * best_bs
    print('Using BLEND')
elif results[0][1] > zero_cm:
    final = test_all[results[0][0]] * results[0][2]
    print(f'Using {results[0][0]}')
else:
    bk = results[0][0]; bs = results[0][2]
    final = test_all[bk] * bs * 0.5
    sc_h = cm(y, oof_all[bk] * bs * 0.5, groups)
    if sc_h > zero_cm: print(f'Using HALF: {bk}')
    else: final = np.zeros(len(test_ids)); print('Using ZERO')

sub = pd.DataFrame({'ID': test_ids, 'TARGET': final})
sub.to_csv('submission.csv', index=False)
print(f'\nSub: {sub.shape} mean={final.mean():.6f} std={final.std():.6f}')
print(f'Time: {time.time()-T0:.0f}s ({(time.time()-T0)/60:.1f}min)')
print('Done!')
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
clip = float(np.percentile(np.abs(y), 99))
y_clip = np.clip(y, -clip, clip)

# ============================================================
# 1. QUANT FEATURE ENGINEERING
# ============================================================
# Key insight: TARGET is % return, but LagT features are raw diffs
# We need to convert diffs to % changes to match the target space
print(f'\n=== QUANT FEATURE ENGINEERING ===')
t0 = time.time()

# Identify base features and their lags
base_features = set()
for c in fc:
    for sfx in ['_LagT1', '_LagT2', '_LagT3']:
        if c.endswith(sfx):
            base_features.add(c.replace(sfx, ''))
base_features = sorted([b for b in base_features if b in fc])
print(f'Base features with lags: {len(base_features)}')

new_train_feats = {}
new_test_feats = {}

for base in base_features:
    curr = train[base].values if base in fc else None
    curr_t = test[base].values if base in fc else None
    l1 = train.get(base + '_LagT1')
    l2 = train.get(base + '_LagT2')
    l3 = train.get(base + '_LagT3')
    l1t = test.get(base + '_LagT1')
    l2t = test.get(base + '_LagT2')
    l3t = test.get(base + '_LagT3')
    
    if curr is None or l1 is None:
        continue
    l1 = l1.values; l1t = l1t.values
    
    # Reconstruct past values: past = current - lag_diff
    past1 = curr - l1
    past1_t = curr_t - l1t
    
    # PERCENTAGE CHANGE features (matching target space)
    # pct_T1 = (current - past1) / |past1| = LagT1 / |past1|
    denom = np.where(np.abs(past1) > 1e-10, past1, np.nan)
    denom_t = np.where(np.abs(past1_t) > 1e-10, past1_t, np.nan)
    pct1 = l1 / denom
    pct1_t = l1t / denom_t
    new_train_feats[f'{base}_pct1'] = np.nan_to_num(pct1, 0, posinf=0, neginf=0)
    new_test_feats[f'{base}_pct1'] = np.nan_to_num(pct1_t, 0, posinf=0, neginf=0)
    
    if l2 is not None:
        l2 = l2.values; l2t = l2t.values
        past2 = curr - l2
        past2_t = curr_t - l2t
        denom2 = np.where(np.abs(past2) > 1e-10, past2, np.nan)
        denom2_t = np.where(np.abs(past2_t) > 1e-10, past2_t, np.nan)
        pct2 = l2 / denom2
        pct2_t = l2t / denom2_t
        new_train_feats[f'{base}_pct2'] = np.nan_to_num(pct2, 0, posinf=0, neginf=0)
        new_test_feats[f'{base}_pct2'] = np.nan_to_num(pct2_t, 0, posinf=0, neginf=0)
        
        # MOMENTUM: acceleration = LagT1 - LagT2 (rate of change of change)
        acc = l1 - l2
        acc_t = l1t - l2t
        new_train_feats[f'{base}_acc'] = np.nan_to_num(acc, 0)
        new_test_feats[f'{base}_acc'] = np.nan_to_num(acc_t, 0)
        
        # MEAN REVERSION signal: if recent change (T1) is opposite of longer change (T2)
        mr = np.sign(l1) * np.sign(l2)
        mr_t = np.sign(l1t) * np.sign(l2t)
        new_train_feats[f'{base}_mr'] = mr.astype(np.float32)
        new_test_feats[f'{base}_mr'] = mr_t.astype(np.float32)

print(f'New quant features: {len(new_train_feats)} [{time.time()-t0:.0f}s]')

# Build feature matrices
X_raw = train[fc].values.astype(np.float32)
Xt_raw = test[fc].values.astype(np.float32)

if new_train_feats:
    X_quant = np.column_stack(list(new_train_feats.values())).astype(np.float32)
    Xt_quant = np.column_stack(list(new_test_feats.values())).astype(np.float32)
    X_quant = np.nan_to_num(X_quant, nan=0, posinf=0, neginf=0)
    Xt_quant = np.nan_to_num(Xt_quant, nan=0, posinf=0, neginf=0)
    # Clip extreme quant features
    for i in range(X_quant.shape[1]):
        p99 = np.percentile(np.abs(X_quant[:, i]), 99)
        if p99 > 0:
            X_quant[:, i] = np.clip(X_quant[:, i], -p99*2, p99*2)
            Xt_quant[:, i] = np.clip(Xt_quant[:, i], -p99*2, p99*2)
else:
    X_quant = np.zeros((len(y), 1), dtype=np.float32)
    Xt_quant = np.zeros((len(test_ids), 1), dtype=np.float32)

# Combined: raw + quant
X_all = np.hstack([X_raw, X_quant])
Xt_all = np.hstack([Xt_raw, Xt_quant])
X_all = np.nan_to_num(X_all, nan=0, posinf=0, neginf=0)
Xt_all = np.nan_to_num(Xt_all, nan=0, posinf=0, neginf=0)

del train, test; gc.collect()
print(f'Total features: raw={X_raw.shape[1]} quant={X_quant.shape[1]} combined={X_all.shape[1]}')

ug = np.unique(groups)
nf = len(ug) if len(ug) <= 5 else 5
gkf = GroupKFold(n_splits=nf)

def cm(yt, yp, g):
    br = [r2_score(yt[g==v], yp[g==v]) for v in np.unique(g) if np.std(yt[g==v]) > 1e-12]
    return 2 * np.mean(br) - r2_score(yt, yp)

oof_all = {}
test_all = {}

# ============================================================
# 2. MODELS ON QUANT FEATURES ONLY
# ============================================================
print(f'\n=== QUANT-ONLY MODELS ===')

# Ridge on quant features (pct changes, momentum, mean-reversion)
for alpha in [500, 5000, 50000]:
    key = f'Ridge_quant_a{alpha}'
    t0 = time.time()
    oo = np.zeros(len(y)); tp = np.zeros(len(Xt_quant))
    for fi, (ti, vi) in enumerate(gkf.split(X_quant, y_clip, groups)):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_quant[ti]); Xva = sc.transform(X_quant[vi]); Xte = sc.transform(Xt_quant)
        m = Ridge(alpha=alpha); m.fit(Xtr, y_clip[ti])
        oo[vi] = m.predict(Xva); tp += m.predict(Xte) / nf
    oof_all[key] = oo; test_all[key] = tp
    print(f'  {key}: R2={r2_score(y,oo):.6f} CM={cm(y,oo,groups):.8f} [{time.time()-t0:.0f}s]')

# ============================================================
# 3. MODELS ON COMBINED (RAW + QUANT)
# ============================================================
print(f'\n=== COMBINED MODELS ===')

for alpha in [5000, 50000]:
    key = f'Ridge_all_a{alpha}'
    t0 = time.time()
    oo = np.zeros(len(y)); tp = np.zeros(len(Xt_all))
    for fi, (ti, vi) in enumerate(gkf.split(X_all, y_clip, groups)):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_all[ti]); Xva = sc.transform(X_all[vi]); Xte = sc.transform(Xt_all)
        m = Ridge(alpha=alpha); m.fit(Xtr, y_clip[ti])
        oo[vi] = m.predict(Xva); tp += m.predict(Xte) / nf
    oof_all[key] = oo; test_all[key] = tp
    print(f'  {key}: R2={r2_score(y,oo):.6f} CM={cm(y,oo,groups):.8f} [{time.time()-t0:.0f}s]')

# LGB on quant features
print(f'\n=== LGB QUANT ===')
pp = {'objective':'huber','huber_delta':0.5,'metric':'rmse','verbosity':-1,'n_jobs':-1,
      'learning_rate':0.03,'num_leaves':15,'max_depth':4,'min_child_samples':500,
      'subsample':0.6,'colsample_bytree':0.4,'reg_alpha':5.0,'reg_lambda':5.0,
      'min_gain_to_split':0.05,'feature_pre_filter':False}

key = 'LGB_quant'
t0 = time.time()
oo = np.zeros(len(y)); tp = np.zeros(len(Xt_quant))
for _, (ti, vi) in enumerate(gkf.split(X_quant, y_clip, groups)):
    dt = lgb.Dataset(X_quant[ti], y_clip[ti]); dv = lgb.Dataset(X_quant[vi], y_clip[vi], reference=dt)
    m = lgb.train(pp, dt, num_boost_round=500, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
    oo[vi] = m.predict(X_quant[vi]); tp += m.predict(Xt_quant) / nf
oof_all[key] = oo; test_all[key] = tp
print(f'  {key}: R2={r2_score(y,oo):.6f} CM={cm(y,oo,groups):.8f} [{time.time()-t0:.0f}s]')

# LGB on combined
key = 'LGB_all'
t0 = time.time()
oo = np.zeros(len(y)); tp = np.zeros(len(Xt_all))
for _, (ti, vi) in enumerate(gkf.split(X_all, y_clip, groups)):
    dt = lgb.Dataset(X_all[ti], y_clip[ti]); dv = lgb.Dataset(X_all[vi], y_clip[vi], reference=dt)
    m = lgb.train(pp, dt, num_boost_round=500, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
    oo[vi] = m.predict(X_all[vi]); tp += m.predict(Xt_all) / nf
oof_all[key] = oo; test_all[key] = tp
print(f'  {key}: R2={r2_score(y,oo):.6f} CM={cm(y,oo,groups):.8f} [{time.time()-t0:.0f}s]')

print(f'\nAll done [{time.time()-T0:.0f}s]')

# ============================================================
# 4. SHRINKAGE + BLEND + SUBMIT
# ============================================================
print(f'\n=== OPTIMIZATION ===')
zero_cm = cm(y, np.zeros(len(y)), groups)
print(f'Zero CM: {zero_cm:.8f}')

results = []
for key, oo in oof_all.items():
    best_s = 0; best_sc = -999
    for s in np.arange(0.01, 1.01, 0.01):
        sc = cm(y, oo * s, groups)
        if sc > best_sc: best_sc = sc; best_s = s
    results.append((key, best_sc, best_s))
    print(f'  {key}: s={best_s:.2f} CM={best_sc:.8f}')
results.sort(key=lambda x: -x[1])

k1, cm1, s1 = results[0]; k2, cm2, s2 = results[1]
print(f'\nTop: {k1} (CM={cm1:.8f}) + {k2} (CM={cm2:.8f})')

best_blend = -999; best_ba = 0; best_bs = 0
for a in np.arange(0, 1.01, 0.05):
    bl = a * oof_all[k1] + (1-a) * oof_all[k2]
    for s in np.arange(0.01, 0.81, 0.01):
        sc = cm(y, bl * s, groups)
        if sc > best_blend: best_blend = sc; best_ba = a; best_bs = s
print(f'Blend: a={best_ba:.2f} s={best_bs:.2f} CM={best_blend:.8f}')

if best_blend > results[0][1] and best_blend > zero_cm:
    final = (best_ba * test_all[k1] + (1-best_ba) * test_all[k2]) * best_bs
    print('Using BLEND')
elif results[0][1] > zero_cm:
    final = test_all[results[0][0]] * results[0][2]
    print(f'Using {results[0][0]}')
else:
    bk = results[0][0]; bs = results[0][2]
    final = test_all[bk] * bs * 0.5
    sc_h = cm(y, oof_all[bk] * bs * 0.5, groups)
    if sc_h > zero_cm: print(f'Using HALF: {bk}')
    else: final = np.zeros(len(test_ids)); print('Using ZERO')

sub = pd.DataFrame({'ID': test_ids, 'TARGET': final})
sub.to_csv('submission.csv', index=False)
print(f'\nSub: {sub.shape} mean={final.mean():.6f} std={final.std():.6f}')
print(f'Time: {time.time()-T0:.0f}s ({(time.time()-T0)/60:.1f}min)')
print('Done!')
# V24: Quant Signals Model - Score: -0.00014
# Unique: Constructs proper quant features from lag-difference data
# pct_change = LagT1 / (current - LagT1) -> matches % return target space
# acceleration = LagT1 - LagT2 -> momentum signal
# mean_reversion = sign(LagT1) * sign(LagT2) -> +1 momentum, -1 reversal
# Creates 444 new quant features from 111 base features
# LGB_all had R2=+0.000129 but model chose Ridge (wrong pick)
import numpy as np, pandas as pd, time, os, warnings, gc, lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
# Full code on Kaggle V24 - notebook_quant_signals.py
print('V24: Quant signals - pct_change, momentum, mean_reversion features')
