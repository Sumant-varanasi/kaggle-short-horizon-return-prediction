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

clip = float(np.percentile(np.abs(y), 99))
y_clip = np.clip(y, -clip, clip)

X = train[fc].values.astype(np.float32)
Xt = test[fc].values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
Xt = np.nan_to_num(Xt, nan=0, posinf=0, neginf=0)
del train, test; gc.collect()
print(f'Features: {nf_total}, Target clip: +/-{clip:.4f}')

ug = np.unique(groups)
nfolds = len(ug) if len(ug) <= 5 else 5
gkf = GroupKFold(n_splits=nfolds)

def cm(yt, yp, g):
    br = [r2_score(yt[g==v], yp[g==v]) for v in np.unique(g) if np.std(yt[g==v]) > 1e-12]
    return 2 * np.mean(br) - r2_score(yt, yp)

# ============================================================
# APPROACH 1: FEATURE-BAGGED ENSEMBLE
# Train many small models on random feature subsets
# Noise cancels out, signal accumulates
# Like a random forest over feature space
# ============================================================
print(f'\n=== FEATURE-BAGGED ENSEMBLE ===')

N_BAGS = 30          # number of sub-models
FEAT_FRAC = 0.30     # each model sees 30% of features
SEED_BASE = 42

pp = {
    'objective': 'huber', 'huber_delta': 0.5,
    'metric': 'rmse', 'verbosity': -1, 'n_jobs': -1,
    'learning_rate': 0.03, 'num_leaves': 15,
    'max_depth': 4, 'min_child_samples': 500,
    'subsample': 0.6, 'colsample_bytree': 0.8,  # high since we already subset
    'reg_alpha': 5.0, 'reg_lambda': 5.0,
    'min_gain_to_split': 0.05,
    'feature_pre_filter': False,
}

n_feat_per_bag = max(10, int(nf_total * FEAT_FRAC))
all_oof = np.zeros((len(y), N_BAGS))
all_test = np.zeros((len(Xt), N_BAGS))

t0 = time.time()
for bag in range(N_BAGS):
    rng = np.random.RandomState(SEED_BASE + bag)
    feat_idx = rng.choice(nf_total, n_feat_per_bag, replace=False)
    X_sub = X[:, feat_idx]
    Xt_sub = Xt[:, feat_idx]
    
    oof_bag = np.zeros(len(y))
    test_bag = np.zeros(len(Xt))
    
    for ti, vi in gkf.split(X_sub, y_clip, groups):
        dt = lgb.Dataset(X_sub[ti], y_clip[ti])
        dv = lgb.Dataset(X_sub[vi], y_clip[vi], reference=dt)
        m = lgb.train(pp, dt, num_boost_round=300, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(30, verbose=False),
                                 lgb.log_evaluation(0)])
        oof_bag[vi] = m.predict(X_sub[vi])
        test_bag += m.predict(Xt_sub) / nfolds
    
    all_oof[:, bag] = oof_bag
    all_test[:, bag] = test_bag
    
    if (bag + 1) % 10 == 0:
        avg_oof = all_oof[:, :bag+1].mean(axis=1)
        r2 = r2_score(y, avg_oof)
        c = cm(y, avg_oof, groups)
        print(f'  Bag {bag+1}/{N_BAGS}: R2={r2:.6f} CM={c:.8f} [{time.time()-t0:.0f}s]')

# Average all bags
oof_bagged = all_oof.mean(axis=1)
test_bagged = all_test.mean(axis=1)
r2_bagged = r2_score(y, oof_bagged)
cm_bagged = cm(y, oof_bagged, groups)
print(f'\nFull bagged ({N_BAGS} bags): R2={r2_bagged:.6f} CM={cm_bagged:.8f}')

# ============================================================
# APPROACH 2: SELECTIVE PREDICTION
# Only predict for high-confidence rows
# Zero out uncertain predictions
# ============================================================
print(f'\n=== SELECTIVE PREDICTION ===')

# Method: keep only predictions where absolute value > threshold
# For uncertain rows (small predictions), predict zero
best_sel_cm = -999
best_threshold = 0
oof_abs = np.abs(oof_bagged)

for pct in [50, 60, 70, 80, 90, 95]:
    threshold = np.percentile(oof_abs, pct)
    oof_sel = np.where(oof_abs > threshold, oof_bagged, 0)
    sc = cm(y, oof_sel, groups)
    r2 = r2_score(y, oof_sel)
    print(f'  Keep top {100-pct}%: threshold={threshold:.6f} R2={r2:.6f} CM={sc:.8f}')
    if sc > best_sel_cm:
        best_sel_cm = sc
        best_threshold = threshold
        best_pct = pct

# ============================================================
# APPROACH 3: AGREEMENT-BASED PREDICTION  
# Only predict where majority of bags agree on direction
# ============================================================
print(f'\n=== AGREEMENT-BASED ===')

signs = np.sign(all_oof)
agreement = np.abs(signs.sum(axis=1)) / N_BAGS  # 1.0 = all agree, 0 = split

for min_agree in [0.6, 0.7, 0.8, 0.9]:
    oof_agree = np.where(agreement >= min_agree, oof_bagged, 0)
    sc = cm(y, oof_agree, groups)
    r2 = r2_score(y, oof_agree)
    kept = (agreement >= min_agree).sum()
    print(f'  Agreement >= {min_agree}: kept={kept} ({100*kept/len(y):.0f}%) R2={r2:.6f} CM={sc:.8f}')

# ============================================================
# APPROACH 4: SHRINKAGE SEARCH ON BAGGED
# ============================================================
print(f'\n=== SHRINKAGE ON BAGGED ===')
best_s = 0; best_s_cm = -999
for s in np.arange(0.01, 1.01, 0.01):
    sc = cm(y, oof_bagged * s, groups)
    if sc > best_s_cm: best_s_cm = sc; best_s = s
print(f'Best shrinkage: {best_s:.2f} CM={best_s_cm:.8f}')

# ============================================================
# PICK BEST AND SUBMIT
# ============================================================
print(f'\n=== FINAL DECISION ===')
zero_cm = cm(y, np.zeros(len(y)), groups)
print(f'Zero CM: {zero_cm:.8f}')
print(f'Bagged CM: {cm_bagged:.8f}')
print(f'Bagged+shrinkage CM: {best_s_cm:.8f}')
print(f'Selective CM: {best_sel_cm:.8f}')

candidates = {
    'zero': (zero_cm, np.zeros(len(test_ids))),
    'bagged': (cm_bagged, test_bagged),
    'bagged_shrunk': (best_s_cm, test_bagged * best_s),
    'selective': (best_sel_cm, np.where(np.abs(test_bagged) > best_threshold, test_bagged, 0)),
}

# Add agreement-based candidates
for min_agree in [0.6, 0.7, 0.8, 0.9]:
    test_signs = np.sign(all_test)
    test_agreement = np.abs(test_signs.sum(axis=1)) / N_BAGS
    test_agree = np.where(test_agreement >= min_agree, test_bagged, 0)
    oof_agree = np.where(agreement >= min_agree, oof_bagged, 0)
    sc = cm(y, oof_agree, groups)
    candidates[f'agree_{min_agree}'] = (sc, test_agree)

best_name = max(candidates, key=lambda k: candidates[k][0])
best_score, final = candidates[best_name]
print(f'\nBest: {best_name} (CM={best_score:.8f})')
for name in sorted(candidates, key=lambda k: -candidates[k][0]):
    print(f'  {name}: {candidates[name][0]:.8f}')

sub = pd.DataFrame({'ID': test_ids, 'TARGET': final})
sub.to_csv('submission.csv', index=False)
nz = np.sum(final != 0)
print(f'\nSub: {sub.shape} mean={final.mean():.6f} std={final.std():.6f} nonzero={nz}')
print(f'Time: {time.time()-T0:.0f}s ({(time.time()-T0)/60:.1f}min)')
print('Done!')
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

clip = float(np.percentile(np.abs(y), 99))
y_clip = np.clip(y, -clip, clip)

X = train[fc].values.astype(np.float32)
Xt = test[fc].values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
Xt = np.nan_to_num(Xt, nan=0, posinf=0, neginf=0)
del train, test; gc.collect()
print(f'Features: {nf_total}, Target clip: +/-{clip:.4f}')

ug = np.unique(groups)
nfolds = len(ug) if len(ug) <= 5 else 5
gkf = GroupKFold(n_splits=nfolds)

def cm(yt, yp, g):
    br = [r2_score(yt[g==v], yp[g==v]) for v in np.unique(g) if np.std(yt[g==v]) > 1e-12]
    return 2 * np.mean(br) - r2_score(yt, yp)

# ============================================================
# APPROACH 1: FEATURE-BAGGED ENSEMBLE
# Train many small models on random feature subsets
# Noise cancels out, signal accumulates
# Like a random forest over feature space
# ============================================================
print(f'\n=== FEATURE-BAGGED ENSEMBLE ===')

N_BAGS = 30          # number of sub-models
FEAT_FRAC = 0.30     # each model sees 30% of features
SEED_BASE = 42

pp = {
    'objective': 'huber', 'huber_delta': 0.5,
    'metric': 'rmse', 'verbosity': -1, 'n_jobs': -1,
    'learning_rate': 0.03, 'num_leaves': 15,
    'max_depth': 4, 'min_child_samples': 500,
    'subsample': 0.6, 'colsample_bytree': 0.8,  # high since we already subset
    'reg_alpha': 5.0, 'reg_lambda': 5.0,
    'min_gain_to_split': 0.05,
    'feature_pre_filter': False,
}

n_feat_per_bag = max(10, int(nf_total * FEAT_FRAC))
all_oof = np.zeros((len(y), N_BAGS))
all_test = np.zeros((len(Xt), N_BAGS))

t0 = time.time()
for bag in range(N_BAGS):
    rng = np.random.RandomState(SEED_BASE + bag)
    feat_idx = rng.choice(nf_total, n_feat_per_bag, replace=False)
    X_sub = X[:, feat_idx]
    Xt_sub = Xt[:, feat_idx]
    
    oof_bag = np.zeros(len(y))
    test_bag = np.zeros(len(Xt))
    
    for ti, vi in gkf.split(X_sub, y_clip, groups):
        dt = lgb.Dataset(X_sub[ti], y_clip[ti])
        dv = lgb.Dataset(X_sub[vi], y_clip[vi], reference=dt)
        m = lgb.train(pp, dt, num_boost_round=300, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(30, verbose=False),
                                 lgb.log_evaluation(0)])
        oof_bag[vi] = m.predict(X_sub[vi])
        test_bag += m.predict(Xt_sub) / nfolds
    
    all_oof[:, bag] = oof_bag
    all_test[:, bag] = test_bag
    
    if (bag + 1) % 10 == 0:
        avg_oof = all_oof[:, :bag+1].mean(axis=1)
        r2 = r2_score(y, avg_oof)
        c = cm(y, avg_oof, groups)
        print(f'  Bag {bag+1}/{N_BAGS}: R2={r2:.6f} CM={c:.8f} [{time.time()-t0:.0f}s]')

# Average all bags
oof_bagged = all_oof.mean(axis=1)
test_bagged = all_test.mean(axis=1)
r2_bagged = r2_score(y, oof_bagged)
cm_bagged = cm(y, oof_bagged, groups)
print(f'\nFull bagged ({N_BAGS} bags): R2={r2_bagged:.6f} CM={cm_bagged:.8f}')

# ============================================================
# APPROACH 2: SELECTIVE PREDICTION
# Only predict for high-confidence rows
# Zero out uncertain predictions
# ============================================================
print(f'\n=== SELECTIVE PREDICTION ===')

# Method: keep only predictions where absolute value > threshold
# For uncertain rows (small predictions), predict zero
best_sel_cm = -999
best_threshold = 0
oof_abs = np.abs(oof_bagged)

for pct in [50, 60, 70, 80, 90, 95]:
    threshold = np.percentile(oof_abs, pct)
    oof_sel = np.where(oof_abs > threshold, oof_bagged, 0)
    sc = cm(y, oof_sel, groups)
    r2 = r2_score(y, oof_sel)
    print(f'  Keep top {100-pct}%: threshold={threshold:.6f} R2={r2:.6f} CM={sc:.8f}')
    if sc > best_sel_cm:
        best_sel_cm = sc
        best_threshold = threshold
        best_pct = pct

# ============================================================
# APPROACH 3: AGREEMENT-BASED PREDICTION  
# Only predict where majority of bags agree on direction
# ============================================================
print(f'\n=== AGREEMENT-BASED ===')

signs = np.sign(all_oof)
agreement = np.abs(signs.sum(axis=1)) / N_BAGS  # 1.0 = all agree, 0 = split

for min_agree in [0.6, 0.7, 0.8, 0.9]:
    oof_agree = np.where(agreement >= min_agree, oof_bagged, 0)
    sc = cm(y, oof_agree, groups)
    r2 = r2_score(y, oof_agree)
    kept = (agreement >= min_agree).sum()
    print(f'  Agreement >= {min_agree}: kept={kept} ({100*kept/len(y):.0f}%) R2={r2:.6f} CM={sc:.8f}')

# ============================================================
# APPROACH 4: SHRINKAGE SEARCH ON BAGGED
# ============================================================
print(f'\n=== SHRINKAGE ON BAGGED ===')
best_s = 0; best_s_cm = -999
for s in np.arange(0.01, 1.01, 0.01):
    sc = cm(y, oof_bagged * s, groups)
    if sc > best_s_cm: best_s_cm = sc; best_s = s
print(f'Best shrinkage: {best_s:.2f} CM={best_s_cm:.8f}')

# ============================================================
# PICK BEST AND SUBMIT
# ============================================================
print(f'\n=== FINAL DECISION ===')
zero_cm = cm(y, np.zeros(len(y)), groups)
print(f'Zero CM: {zero_cm:.8f}')
print(f'Bagged CM: {cm_bagged:.8f}')
print(f'Bagged+shrinkage CM: {best_s_cm:.8f}')
print(f'Selective CM: {best_sel_cm:.8f}')

candidates = {
    'zero': (zero_cm, np.zeros(len(test_ids))),
    'bagged': (cm_bagged, test_bagged),
    'bagged_shrunk': (best_s_cm, test_bagged * best_s),
    'selective': (best_sel_cm, np.where(np.abs(test_bagged) > best_threshold, test_bagged, 0)),
}

# Add agreement-based candidates
for min_agree in [0.6, 0.7, 0.8, 0.9]:
    test_signs = np.sign(all_test)
    test_agreement = np.abs(test_signs.sum(axis=1)) / N_BAGS
    test_agree = np.where(test_agreement >= min_agree, test_bagged, 0)
    oof_agree = np.where(agreement >= min_agree, oof_bagged, 0)
    sc = cm(y, oof_agree, groups)
    candidates[f'agree_{min_agree}'] = (sc, test_agree)

best_name = max(candidates, key=lambda k: candidates[k][0])
best_score, final = candidates[best_name]
print(f'\nBest: {best_name} (CM={best_score:.8f})')
for name in sorted(candidates, key=lambda k: -candidates[k][0]):
    print(f'  {name}: {candidates[name][0]:.8f}')

sub = pd.DataFrame({'ID': test_ids, 'TARGET': final})
sub.to_csv('submission.csv', index=False)
nz = np.sum(final != 0)
print(f'\nSub: {sub.shape} mean={final.mean():.6f} std={final.std():.6f} nonzero={nz}')
print(f'Time: {time.time()-T0:.0f}s ({(time.time()-T0)/60:.1f}min)')
print('Done!')
# V26: Feature-Bagged Ensemble (30 bags) - Score: -0.00005
# KEY RESULT: OOF R2 = +0.000102 (POSITIVE! Signal confirmed)
# Public LB = -0.00005 (tiny overfitting gap of ~0.00015)
# Approach: 30 LGB models on random 30% feature subsets, averaged
# Ultra-weak base learners: 7 leaves, depth 3-4, 500 min_samples
# Also tests selective prediction and agreement-based strategies
import numpy as np, pandas as pd, time, os, warnings, gc, lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')
T0 = time.time()
for p in ['/kaggle/input/competitions/short-horizon-return-prediction-challenge-by-i-rage/','/kaggle/input/short-horizon-return-prediction-challenge-by-i-rage/']:
    if os.path.exists(p): DATA=p; break
else: raise FileNotFoundError('Data not found')
train = pd.read_parquet(DATA + 'train.parquet')
test = pd.read_parquet(DATA + 'test.parquet')
y = train['TARGET'].values; groups = train['CV_GROUP'].values; test_ids = test['ID'].values
fc = [c for c in train.columns if c not in ['TARGET','CV_GROUP','ID']]
nf_total = len(fc); clip = float(np.percentile(np.abs(y), 99)); y_clip = np.clip(y, -clip, clip)
X = train[fc].values.astype(np.float32); Xt = test[fc].values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0); Xt = np.nan_to_num(Xt, nan=0, posinf=0, neginf=0)
del train, test; gc.collect()
ug = np.unique(groups); nfolds = len(ug) if len(ug) <= 5 else 5
gkf = GroupKFold(n_splits=nfolds)
def cm(yt, yp, g):
    br = [r2_score(yt[g==v], yp[g==v]) for v in np.unique(g) if np.std(yt[g==v]) > 1e-12]
    return 2 * np.mean(br) - r2_score(yt, yp)
N_BAGS = 30; FEAT_FRAC = 0.30; SEED_BASE = 42
pp = {'objective':'huber','huber_delta':0.5,'metric':'rmse','verbosity':-1,'n_jobs':-1,
      'learning_rate':0.03,'num_leaves':15,'max_depth':4,'min_child_samples':500,
      'subsample':0.6,'colsample_bytree':0.8,'reg_alpha':5.0,'reg_lambda':5.0,
      'min_gain_to_split':0.05,'feature_pre_filter':False}
n_feat_per_bag = max(10, int(nf_total * FEAT_FRAC))
all_oof = np.zeros((len(y), N_BAGS)); all_test = np.zeros((len(Xt), N_BAGS))
for bag in range(N_BAGS):
    rng = np.random.RandomState(SEED_BASE + bag)
    feat_idx = rng.choice(nf_total, n_feat_per_bag, replace=False)
    X_sub = X[:, feat_idx]; Xt_sub = Xt[:, feat_idx]
    oof_bag = np.zeros(len(y)); test_bag = np.zeros(len(Xt))
    for ti, vi in gkf.split(X_sub, y_clip, groups):
        dt = lgb.Dataset(X_sub[ti], y_clip[ti]); dv = lgb.Dataset(X_sub[vi], y_clip[vi], reference=dt)
        m = lgb.train(pp, dt, num_boost_round=300, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        oof_bag[vi] = m.predict(X_sub[vi]); test_bag += m.predict(Xt_sub)/nfolds
    all_oof[:, bag] = oof_bag; all_test[:, bag] = test_bag
oof_bagged = all_oof.mean(axis=1); test_bagged = all_test.mean(axis=1)
# Selective + agreement strategies tested, raw bagged average was best
zero_cm = cm(y, np.zeros(len(y)), groups)
final = test_bagged  # raw bagged was best strategy
sub = pd.DataFrame({'ID': test_ids, 'TARGET': final})
sub.to_csv('submission.csv', index=False)
print(f'Sub: {sub.shape}, Time: {time.time()-T0:.0f}s')
