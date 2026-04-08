import numpy as np
import pandas as pd
import time, os, warnings, gc
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
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
clip99 = float(np.percentile(np.abs(y_raw), 99))
y95 = np.clip(y_raw, -clip95, clip95)
y99 = np.clip(y_raw, -clip99, clip99)
print(f'Target 95th clip: +/-{clip95:.6f}')
print(f'Target 99th clip: +/-{clip99:.6f}')

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

alphas = [10, 50, 100, 500, 1000, 5000]
targets = {'y95': y95, 'y99': y99}

print(f'\n=== RIDGE ENSEMBLE ===')
oof_all = {}
test_all = {}

for tname, y_target in targets.items():
    for alpha in alphas:
        key = f'Ridge_{tname}_a{alpha}'
        t0 = time.time()
        oof = np.zeros(len(y_raw))
        tpred = np.zeros(len(Xt))
        
        for fi, (ti, vi) in enumerate(gkf.split(X, y_target, groups)):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[ti])
            Xva = scaler.transform(X[vi])
            Xte = scaler.transform(Xt)
            
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(Xtr, y_target[ti])
            oof[vi] = model.predict(Xva)
            tpred += model.predict(Xte) / nf
        
        r2 = r2_score(y_raw, oof)
        cs = cm(y_raw, oof, groups)
        oof_all[key] = oof
        test_all[key] = tpred
        print(f'  {key}: R2={r2:.6f} CompScore={cs:.8f} [{time.time()-t0:.0f}s]')

import lightgbm as lgb

lgb_configs = {
    'LGB_ultra_conservative': {
        'objective': 'huber', 'huber_delta': 0.3,
        'metric': 'rmse', 'verbosity': -1, 'n_jobs': -1,
        'learning_rate': 0.01, 'num_leaves': 7,
        'max_depth': 3, 'min_child_samples': 1000,
        'subsample': 0.5, 'colsample_bytree': 0.3,
        'reg_alpha': 10.0, 'reg_lambda': 10.0,
        'min_gain_to_split': 0.1,
        'feature_pre_filter': False,
    },
}

for tname in ['y95']:
    y_target = targets[tname]
    for nm, pp in lgb_configs.items():
        key = f'{nm}_{tname}'
        t0 = time.time()
        oo = np.zeros(len(y_raw))
        tp = np.zeros(len(Xt))
        for _, (ti, vi) in enumerate(gkf.split(X, y_target, groups)):
            dt = lgb.Dataset(X[ti], y_target[ti])
            dv = lgb.Dataset(X[vi], y_target[vi], reference=dt)
            m = lgb.train(pp, dt, num_boost_round=300, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(30, verbose=False),
                                     lgb.log_evaluation(0)])
            oo[vi] = m.predict(X[vi])
            tp += m.predict(Xt) / nf
        oof_all[key] = oo
        test_all[key] = tp
        r2 = r2_score(y_raw, oo)
        cs = cm(y_raw, oo, groups)
        print(f'  {key}: R2={r2:.6f} CompScore={cs:.8f} [{time.time()-t0:.0f}s]')

print(f'\nAll models done [{time.time()-T0:.0f}s]')

print(f'\n=== SINGLE MODEL + SHRINKAGE SEARCH ===')
best_overall = -999
best_config = None

for key, oof in oof_all.items():
    best_s = 0; best_sc = -999
    for s in np.arange(0.01, 1.01, 0.01):
        sc = cm(y_raw, oof * s, groups)
        if sc > best_sc: best_sc=sc; best_s=s
    print(f'  {key}: best_s={best_s:.2f} CompScore={best_sc:.8f}')
    if best_sc > best_overall:
        best_overall = best_sc
        best_config = (key, best_s, best_sc)

print(f'\nBest single: {best_config[0]} s={best_config[1]:.2f} score={best_config[2]:.8f}')

print(f'\n=== BLENDING TOP MODELS ===')
model_scores = []
for key, oof in oof_all.items():
    best_s = 0; best_sc = -999
    for s in np.arange(0.01, 1.01, 0.01):
        sc = cm(y_raw, oof * s, groups)
        if sc > best_sc: best_sc=sc; best_s=s
    model_scores.append((key, best_sc, best_s))

model_scores.sort(key=lambda x: -x[1])
top3 = model_scores[:3]
print(f'Top 3 models:')
for k, sc, s in top3:
    print(f'  {k}: score={sc:.8f} shrinkage={s:.2f}')

k1, _, s1 = top3[0]
k2, _, s2 = top3[1]
best_blend = -999
best_ba = 0; best_bs = 0
for a in np.arange(0, 1.01, 0.05):
    bl = a * oof_all[k1] + (1-a) * oof_all[k2]
    for s in np.arange(0.01, 0.81, 0.01):
        sc = cm(y_raw, bl * s, groups)
        if sc > best_blend:
            best_blend = sc
            best_ba = a; best_bs = s

print(f'Best blend: a={best_ba:.2f} s={best_bs:.2f} score={best_blend:.8f}')

print(f'\n=== FINAL COMPARISON ===')
zero_score = cm(y_raw, np.zeros(len(y_raw)), groups)
print(f'Zero baseline:  {zero_score:.8f}')
print(f'Best single:    {best_config[2]:.8f} ({best_config[0]})')
print(f'Best blend:     {best_blend:.8f}')

if best_blend > best_config[2] and best_blend > zero_score:
    final_oof = (best_ba * oof_all[k1] + (1-best_ba) * oof_all[k2]) * best_bs
    final_test = (best_ba * test_all[k1] + (1-best_ba) * test_all[k2]) * best_bs
    print(f'Using BLEND')
elif best_config[2] > zero_score:
    bk, bs_val, _ = best_config
    final_oof = oof_all[bk] * bs_val
    final_test = test_all[bk] * bs_val
    print(f'Using SINGLE: {bk}')
else:
    bk, bs_val, _ = best_config
    ultra_s = bs_val * 0.5
    sc_ultra = cm(y_raw, oof_all[bk] * ultra_s, groups)
    if sc_ultra > zero_score:
        final_oof = oof_all[bk] * ultra_s
        final_test = test_all[bk] * ultra_s
        print(f'Using ULTRA-CONSERVATIVE: {bk} s={ultra_s:.2f}')
    else:
        final_test = np.zeros(len(test_ids))
        print(f'Using ZERO (nothing beats baseline)')

if np.std(final_test) > 0:
    clip_val = np.percentile(np.abs(final_test[final_test != 0]), 99) if np.any(final_test != 0) else 0
    if clip_val > 0:
        final_test = np.clip(final_test, -clip_val, clip_val)

sub = pd.DataFrame({'ID': test_ids, 'TARGET': final_test})
sub.to_csv('submission.csv', index=False)
print(f'\nSub: {sub.shape} mean={final_test.mean():.8f} std={final_test.std():.8f}')
print(f'Non-zero predictions: {np.sum(final_test != 0)} / {len(final_test)}')
print(f'Time: {time.time()-T0:.0f}s ({(time.time()-T0)/60:.1f}min)')
print('Done!')
