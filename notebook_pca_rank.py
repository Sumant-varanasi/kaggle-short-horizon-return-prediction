import numpy as np
import pandas as pd
import time, os, warnings, gc
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
warnings.filterwarnings('ignore')

T0 = time.time()

for p in ['/kaggle/input/competitions/short-horizon-return-prediction-challenge-by-i-rage/',
          '/kaggle/input/short-horizon-return-prediction-challenge-by-i-rage/']:
    if os.path.exists(p): DATA=p; break
else: raise FileNotFoundError('Data not found')

train = pd.read_parquet(DATA + 'train.parquet')
test = pd.read_parquet(DATA + 'test.parquet')
print(f'Train: {train.shape}, Test: {test.shape} [{time.time()-T0:.0f}s]')

y_raw = train['TARGET'].values
groups = train['CV_GROUP'].values
test_ids = test['ID'].values
fc = [c for c in train.columns if c not in ['TARGET', 'CV_GROUP', 'ID']]

clip95 = float(np.percentile(np.abs(y_raw), 95))
y = np.clip(y_raw, -clip95, clip95)
print(f'Target clipped at 95th: +/-{clip95:.6f}')

X = train[fc].values.astype(np.float32)
Xt = test[fc].values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
Xt = np.nan_to_num(Xt, nan=0, posinf=0, neginf=0)
del train, test; gc.collect()
print(f'Features: {X.shape[1]} [{time.time()-T0:.0f}s]')

ug = np.unique(groups)
nf = len(ug) if len(ug) <= 5 else 5
gkf = GroupKFold(n_splits=nf)

def cm(yt, yp, g):
    br = [r2_score(yt[g==v], yp[g==v]) for v in np.unique(g) if np.std(yt[g==v]) > 1e-12]
    return 2 * np.mean(br) - r2_score(yt, yp)

oof_all = {}
test_all = {}

# ============================================================
# 1. PCA FACTOR MODEL
# ============================================================
print(f'\n=== PCA FACTOR MODEL ===')
for n_comp in [10, 20, 50]:
    key = f'PCA_{n_comp}_Ridge'
    t0 = time.time()
    oof = np.zeros(len(y))
    tpred = np.zeros(len(Xt))
    
    for fi, (ti, vi) in enumerate(gkf.split(X, y, groups)):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[ti])
        Xva = sc.transform(X[vi])
        Xte = sc.transform(Xt)
        
        pca = PCA(n_components=n_comp, random_state=42)
        Xtr_pca = pca.fit_transform(Xtr)
        Xva_pca = pca.transform(Xva)
        Xte_pca = pca.transform(Xte)
        
        model = Ridge(alpha=100, fit_intercept=True)
        model.fit(Xtr_pca, y[ti])
        oof[vi] = model.predict(Xva_pca)
        tpred += model.predict(Xte_pca) / nf
    
    r2 = r2_score(y_raw, oof)
    cs = cm(y_raw, oof, groups)
    oof_all[key] = oof
    test_all[key] = tpred
    print(f'  {key}: R2={r2:.6f} CompScore={cs:.8f} [{time.time()-t0:.0f}s]')

# ============================================================
# 2. PLS REGRESSION (target-aligned factors)
# ============================================================
print(f'\n=== PLS REGRESSION ===')
for n_comp in [5, 10, 20]:
    key = f'PLS_{n_comp}'
    t0 = time.time()
    oof = np.zeros(len(y))
    tpred = np.zeros(len(Xt))
    
    for fi, (ti, vi) in enumerate(gkf.split(X, y, groups)):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[ti])
        Xva = sc.transform(X[vi])
        Xte = sc.transform(Xt)
        
        pls = PLSRegression(n_components=n_comp, max_iter=200)
        pls.fit(Xtr, y[ti])
        oof[vi] = pls.predict(Xva).ravel()
        tpred += pls.predict(Xte).ravel() / nf
    
    r2 = r2_score(y_raw, oof)
    cs = cm(y_raw, oof, groups)
    oof_all[key] = oof
    test_all[key] = tpred
    print(f'  {key}: R2={r2:.6f} CompScore={cs:.8f} [{time.time()-t0:.0f}s]')

# ============================================================
# 3. CROSS-SECTIONAL RANK FEATURES + RIDGE
# ============================================================
print(f'\n=== CROSS-SECTIONAL RANK ===')
# Use first 50 raw features for rank transform (keep it fast)
n_rank_feats = min(50, X.shape[1])

def rank_transform(arr):
    from scipy.stats import rankdata
    return rankdata(arr) / len(arr) - 0.5

# Rank each feature across all rows (global cross-section)
print(f'  Ranking {n_rank_feats} features...')
t0 = time.time()
X_rank = np.zeros((X.shape[0], n_rank_feats), dtype=np.float32)
Xt_rank = np.zeros((Xt.shape[0], n_rank_feats), dtype=np.float32)

for i in range(n_rank_feats):
    # Combine train+test for consistent ranking
    combined = np.concatenate([X[:, i], Xt[:, i]])
    ranks = rank_transform(combined)
    X_rank[:, i] = ranks[:len(X)]
    Xt_rank[:, i] = ranks[len(X):]

print(f'  Ranking done [{time.time()-t0:.0f}s]')

for alpha in [100, 1000]:
    key = f'Rank_Ridge_a{alpha}'
    t0 = time.time()
    oof = np.zeros(len(y))
    tpred = np.zeros(len(Xt))
    
    for fi, (ti, vi) in enumerate(gkf.split(X_rank, y, groups)):
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_rank[ti], y[ti])
        oof[vi] = model.predict(X_rank[vi])
        tpred += model.predict(Xt_rank) / nf
    
    r2 = r2_score(y_raw, oof)
    cs = cm(y_raw, oof, groups)
    oof_all[key] = oof
    test_all[key] = tpred
    print(f'  {key}: R2={r2:.6f} CompScore={cs:.8f} [{time.time()-t0:.0f}s]')

# ============================================================
# 4. PCA ON RANK FEATURES
# ============================================================
print(f'\n=== PCA ON RANK FEATURES ===')
key = 'RankPCA_20_Ridge'
t0 = time.time()
oof = np.zeros(len(y))
tpred = np.zeros(len(Xt))

for fi, (ti, vi) in enumerate(gkf.split(X_rank, y, groups)):
    pca = PCA(n_components=20, random_state=42)
    Xtr_pca = pca.fit_transform(X_rank[ti])
    Xva_pca = pca.transform(X_rank[vi])
    Xte_pca = pca.transform(Xt_rank)
    
    model = Ridge(alpha=100, fit_intercept=True)
    model.fit(Xtr_pca, y[ti])
    oof[vi] = model.predict(Xva_pca)
    tpred += model.predict(Xte_pca) / nf

r2 = r2_score(y_raw, oof)
cs = cm(y_raw, oof, groups)
oof_all[key] = oof
test_all[key] = tpred
print(f'  {key}: R2={r2:.6f} CompScore={cs:.8f} [{time.time()-t0:.0f}s]')

# ============================================================
# 5. COMBINED: RAW + RANK + PCA FEATURES
# ============================================================
print(f'\n=== COMBINED FEATURE SET ===')
t0 = time.time()
key = 'Combined_Ridge'
oof = np.zeros(len(y))
tpred = np.zeros(len(Xt))

for fi, (ti, vi) in enumerate(gkf.split(X, y, groups)):
    sc = StandardScaler()
    Xtr_raw = sc.fit_transform(X[ti])
    Xva_raw = sc.transform(X[vi])
    Xte_raw = sc.transform(Xt)
    
    pca = PCA(n_components=30, random_state=42)
    Xtr_pca = pca.fit_transform(Xtr_raw)
    Xva_pca = pca.transform(Xva_raw)
    Xte_pca = pca.transform(Xte_raw)
    
    # Combine PCA + rank features
    Xtr_c = np.hstack([Xtr_pca, X_rank[ti]])
    Xva_c = np.hstack([Xva_pca, X_rank[vi]])
    Xte_c = np.hstack([Xte_pca, Xt_rank])
    
    model = Ridge(alpha=500, fit_intercept=True)
    model.fit(Xtr_c, y[ti])
    oof[vi] = model.predict(Xva_c)
    tpred += model.predict(Xte_c) / nf

r2 = r2_score(y_raw, oof)
cs = cm(y_raw, oof, groups)
oof_all[key] = oof
test_all[key] = tpred
print(f'  {key}: R2={r2:.6f} CompScore={cs:.8f} [{time.time()-t0:.0f}s]')

# ============================================================
# 6. ULTRA-CONSERVATIVE LGB ON PCA FEATURES
# ============================================================
print(f'\n=== LGB ON PCA ===')
import lightgbm as lgb

key = 'LGB_PCA30'
t0 = time.time()
oof = np.zeros(len(y))
tpred = np.zeros(len(Xt))

pp = {'objective': 'huber', 'huber_delta': 0.3, 'metric': 'rmse',
      'verbosity': -1, 'n_jobs': -1, 'learning_rate': 0.01,
      'num_leaves': 7, 'max_depth': 3, 'min_child_samples': 1000,
      'subsample': 0.5, 'colsample_bytree': 0.5,
      'reg_alpha': 10.0, 'reg_lambda': 10.0, 'min_gain_to_split': 0.1}

for fi, (ti, vi) in enumerate(gkf.split(X, y, groups)):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X[ti])
    Xva = sc.transform(X[vi])
    Xte = sc.transform(Xt)
    
    pca = PCA(n_components=30, random_state=42)
    Xtr_p = pca.fit_transform(Xtr)
    Xva_p = pca.transform(Xva)
    Xte_p = pca.transform(Xte)
    
    dt = lgb.Dataset(Xtr_p, y[ti])
    dv = lgb.Dataset(Xva_p, y[vi], reference=dt)
    m = lgb.train(pp, dt, num_boost_round=300, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    oof[vi] = m.predict(Xva_p)
    tpred += m.predict(Xte_p) / nf

r2 = r2_score(y_raw, oof)
cs = cm(y_raw, oof, groups)
oof_all[key] = oof
test_all[key] = tpred
print(f'  {key}: R2={r2:.6f} CompScore={cs:.8f} [{time.time()-t0:.0f}s]')

print(f'\nAll models done [{time.time()-T0:.0f}s]')

# ============================================================
# 7. FIND BEST + BLEND + SUBMIT
# ============================================================
print(f'\n=== SHRINKAGE SEARCH ===')
best_overall = -999
best_config = None

for key, oof in oof_all.items():
    best_s = 0; best_sc = -999
    for s in np.arange(0.01, 1.01, 0.01):
        sc = cm(y_raw, oof * s, groups)
        if sc > best_sc: best_sc=sc; best_s=s
    print(f'  {key}: s={best_s:.2f} score={best_sc:.8f}')
    if best_sc > best_overall:
        best_overall = best_sc
        best_config = (key, best_s, best_sc)

print(f'\nBest: {best_config[0]} s={best_config[1]:.2f} score={best_config[2]:.8f}')

# Blend top 2
ms = sorted([(k, cm(y_raw, oof_all[k]*s, groups), s)
             for k in oof_all for s in [best_config[1]]]
            + [(k, best_sc, best_s) for k, best_sc, best_s in
               [(k, max(cm(y_raw, oof_all[k]*s, groups) for s in np.arange(0.01,1.01,0.05)), 0)
                for k in oof_all]],
            key=lambda x: -x[1])

# Simple: just use top 2 models
keys_sorted = []
for key in oof_all:
    bs = 0; bsc = -999
    for s in np.arange(0.01, 1.01, 0.01):
        sc = cm(y_raw, oof_all[key] * s, groups)
        if sc > bsc: bsc=sc; bs=s
    keys_sorted.append((key, bsc, bs))
keys_sorted.sort(key=lambda x: -x[1])

k1, sc1, s1 = keys_sorted[0]
k2, sc2, s2 = keys_sorted[1]
print(f'\nBlending: {k1} + {k2}')

best_blend = -999; best_ba = 0; best_bs = 0
for a in np.arange(0, 1.01, 0.05):
    bl = a * oof_all[k1] + (1-a) * oof_all[k2]
    for s in np.arange(0.01, 0.81, 0.01):
        sc = cm(y_raw, bl * s, groups)
        if sc > best_blend: best_blend=sc; best_ba=a; best_bs=s

print(f'Blend: a={best_ba:.2f} s={best_bs:.2f} score={best_blend:.8f}')

zero_score = cm(y_raw, np.zeros(len(y_raw)), groups)
print(f'Zero: {zero_score:.8f}')

# Pick best
if best_blend > best_config[2] and best_blend > zero_score:
    final = (best_ba * test_all[k1] + (1-best_ba) * test_all[k2]) * best_bs
    print('Using BLEND')
elif best_config[2] > zero_score:
    bk, bs_val, _ = best_config
    final = test_all[bk] * bs_val
    print(f'Using SINGLE: {bk}')
else:
    bk, bs_val, _ = best_config
    ultra_s = bs_val * 0.5
    sc_u = cm(y_raw, oof_all[bk] * ultra_s, groups)
    if sc_u > zero_score:
        final = test_all[bk] * ultra_s
        print(f'Using ULTRA: {bk} s={ultra_s:.2f}')
    else:
        final = np.zeros(len(test_ids))
        print('Using ZERO')

if np.std(final) > 0:
    cv = np.percentile(np.abs(final[final!=0]), 99) if np.any(final!=0) else 0
    if cv > 0: final = np.clip(final, -cv, cv)

sub = pd.DataFrame({'ID': test_ids, 'TARGET': final})
sub.to_csv('submission.csv', index=False)
print(f'\nSub: {sub.shape} mean={final.mean():.8f} std={final.std():.8f}')
print(f'Time: {time.time()-T0:.0f}s ({(time.time()-T0)/60:.1f}min)')
print('Done!')
import numpy as np
import pandas as pd
import time, os, warnings, gc
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
warnings.filterwarnings('ignore')
T0 = time.time()
for p in ['/kaggle/input/competitions/short-horizon-return-prediction-challenge-by-i-rage/','/kaggle/input/short-horizon-return-prediction-challenge-by-i-rage/']:
    if os.path.exists(p): DATA=p; break
else: raise FileNotFoundError('Data not found')
train = pd.read_parquet(DATA + 'train.parquet')
test = pd.read_parquet(DATA + 'test.parquet')
print(f'Train: {train.shape}, Test: {test.shape} [{time.time()-T0:.0f}s]')
y_raw = train['TARGET'].values
groups = train['CV_GROUP'].values
test_ids = test['ID'].values
fc = [c for c in train.columns if c not in ['TARGET', 'CV_GROUP', 'ID']]
clip95 = float(np.percentile(np.abs(y_raw), 95))
y = np.clip(y_raw, -clip95, clip95)
X = train[fc].values.astype(np.float32)
Xt = test[fc].values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
Xt = np.nan_to_num(Xt, nan=0, posinf=0, neginf=0)
del train, test; gc.collect()
ug = np.unique(groups)
nf = len(ug) if len(ug) <= 5 else 5
gkf = GroupKFold(n_splits=nf)notebook_pca_rank.py
def cm(yt, yp, g):
    br = [r2_score(yt[g==v], yp[g==v]) for v in np.unique(g) if np.std(yt[g==v]) > 1e-12]
    return 2 * np.mean(br) - r2_score(yt, yp)
oof_all = {}; test_all = {}
# PCA Factor Model
for n_comp in [10, 20, 50]:
    key = f'PCA_{n_comp}_Ridge'; t0 = time.time()
    oof = np.zeros(len(y)); tpred = np.zeros(len(Xt))
    for fi, (ti, vi) in enumerate(gkf.split(X, y, groups)):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[ti]); Xva = sc.transform(X[vi]); Xte = sc.transform(Xt)
        pca = PCA(n_components=n_comp, random_state=42); Xtr_pca = pca.fit_transform(Xtr); Xva_pca = pca.transform(Xva); Xte_pca = pca.transform(Xte)
        model = Ridge(alpha=100); model.fit(Xtr_pca, y[ti]); oof[vi] = model.predict(Xva_pca); tpred += model.predict(Xte_pca)/nf
    oof_all[key]=oof; test_all[key]=tpred; print(f'{key}: R2={r2_score(y_raw,oof):.6f} CS={cm(y_raw,oof,groups):.8f}')
# PLS Regression
for n_comp in [5, 10, 20]:
    key = f'PLS_{n_comp}'; t0 = time.time()
    oof = np.zeros(len(y)); tpred = np.zeros(len(Xt))
    for fi, (ti, vi) in enumerate(gkf.split(X, y, groups)):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[ti]); Xva = sc.transform(X[vi]); Xte = sc.transform(Xt)
        pls = PLSRegression(n_components=n_comp, max_iter=200); pls.fit(Xtr, y[ti])
        oof[vi] = pls.predict(Xva).ravel(); tpred += pls.predict(Xte).ravel()/nf
    oof_all[key]=oof; test_all[key]=tpred; print(f'{key}: R2={r2_score(y_raw,oof):.6f} CS={cm(y_raw,oof,groups):.8f}')
# See full version on Kaggle V21
print('Full code on Kaggle notebook V21')
