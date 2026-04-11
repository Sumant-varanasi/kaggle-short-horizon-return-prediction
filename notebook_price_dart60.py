# V29: Price-focused 60-bag DART+GBDT model (Score: -0.00002, 3rd best)
# OOF R2: DART=+0.000029, GBDT=+0.000020 (both positive!)
# Runtime: 63 min (exceeded 30min limit but still scored)
# Key: Top 150 features, 85th percentile clip, 40 DART + 20 GBDT bags
import numpy as np, pandas as pd, time, os, warnings, gc, lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')
T0 = time.time()
for p in ['/kaggle/input/competitions/short-horizon-return-prediction-challenge-by-i-rage/','/kaggle/input/short-horizon-return-prediction-challenge-by-i-rage/']:
    if os.path.exists(p): DATA=p; break
else: raise FileNotFoundError('Data not found')
train = pd.read_parquet(DATA + 'train.parquet'); test = pd.read_parquet(DATA + 'test.parquet')
print(f'Train: {train.shape}, Test: {test.shape}')
y = train['TARGET'].values; groups = train['CV_GROUP'].values; test_ids = test['ID'].values
fc = [c for c in train.columns if c not in ['TARGET', 'CV_GROUP', 'ID']]
clip85 = float(np.percentile(np.abs(y), 85)); y_clip = np.clip(y, -clip85, clip85)
print(f'Target clip85: +/-{clip85:.6f}')
X = train[fc].values.astype(np.float32); Xt = test[fc].values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0); Xt = np.nan_to_num(Xt, nan=0, posinf=0, neginf=0)
price_idx = [i for i, c in enumerate(fc) if 'Price' in c]
print(f'Price features: {len(price_idx)} -> {[fc[i] for i in price_idx]}')
print(f'\n=== FEATURE IMPORTANCE ===')
imp = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31, subsample=0.5, colsample_bytree=0.5, verbose=-1, n_jobs=-1)
imp.fit(X, y_clip); importances = imp.feature_importances_; top_idx = np.argsort(importances)[::-1]
top150 = list(top_idx[:150])
for pi in price_idx:
    if pi not in top150: top150.append(pi)
top150 = np.array(top150[:150])
print(f'Top features: {len(top150)}')
del train, test; gc.collect()
ug = np.unique(groups); nfolds = len(ug) if len(ug) <= 5 else 5; gkf = GroupKFold(n_splits=nfolds)
def cm(yt, yp, g):
    br = [r2_score(yt[g==v], yp[g==v]) for v in np.unique(g) if np.std(yt[g==v]) > 1e-12]
    return 2 * np.mean(br) - r2_score(yt, yp)
print(f'\n=== 40-BAG DART ===')
N_BAGS = 40; N_FEAT_BAG = 40
pp_dart = {'boosting_type':'dart','objective':'huber','huber_delta':0.3,'metric':'rmse','verbosity':-1,'n_jobs':-1,'learning_rate':0.05,'num_leaves':7,'max_depth':3,'min_child_samples':1000,'subsample':0.5,'colsample_bytree':1.0,'reg_alpha':10.0,'reg_lambda':10.0,'drop_rate':0.3,'skip_drop':0.5,'min_gain_to_split':0.1,'feature_pre_filter':False}
X_top = X[:, top150]; Xt_top = Xt[:, top150]; n_top = X_top.shape[1]
all_oof = np.zeros((len(y), N_BAGS)); all_test = np.zeros((len(Xt), N_BAGS))
t0 = time.time()
for bag in range(N_BAGS):
    rng = np.random.RandomState(100+bag); bag_idx = rng.choice(n_top, N_FEAT_BAG, replace=False)
    Xb = X_top[:, bag_idx]; Xtb = Xt_top[:, bag_idx]
    ob = np.zeros(len(y)); tb = np.zeros(len(Xt))
    for ti, vi in gkf.split(Xb, y_clip, groups):
        dt = lgb.Dataset(Xb[ti], y_clip[ti]); dv = lgb.Dataset(Xb[vi], y_clip[vi], reference=dt)
        m = lgb.train(pp_dart, dt, num_boost_round=200, valid_sets=[dv], callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)])
        ob[vi] = m.predict(Xb[vi]); tb += m.predict(Xtb)/nfolds
    all_oof[:, bag] = ob; all_test[:, bag] = tb
    if (bag+1)%10==0:
        a = all_oof[:,:bag+1].mean(axis=1)
        print(f'  Bag {bag+1}: R2={r2_score(y,a):.6f} CM={cm(y,a,groups):.8f} [{time.time()-t0:.0f}s]')
oof_dart = all_oof.mean(axis=1); test_dart = all_test.mean(axis=1)
print(f'DART: R2={r2_score(y,oof_dart):.6f} CM={cm(y,oof_dart,groups):.8f}')

print(f'\n=== 20-BAG GBDT ===')
pp_gbdt = {'objective':'huber','huber_delta':0.3,'metric':'rmse','verbosity':-1,'n_jobs':-1,'learning_rate':0.03,'num_leaves':7,'max_depth':3,'min_child_samples':1000,'subsample':0.5,'colsample_bytree':1.0,'reg_alpha':10.0,'reg_lambda':10.0,'min_gain_to_split':0.1,'feature_pre_filter':False}
all_oof2 = np.zeros((len(y), 20)); all_test2 = np.zeros((len(Xt), 20))
t0 = time.time()
for bag in range(20):
    rng = np.random.RandomState(200+bag); bag_idx = rng.choice(n_top, N_FEAT_BAG, replace=False)
    Xb = X_top[:, bag_idx]; Xtb = Xt_top[:, bag_idx]
    ob = np.zeros(len(y)); tb = np.zeros(len(Xt))
    for ti, vi in gkf.split(Xb, y_clip, groups):
        dt = lgb.Dataset(Xb[ti], y_clip[ti]); dv = lgb.Dataset(Xb[vi], y_clip[vi], reference=dt)
        m = lgb.train(pp_gbdt, dt, num_boost_round=200, valid_sets=[dv], callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)])
        ob[vi] = m.predict(Xb[vi]); tb += m.predict(Xtb)/nfolds
    all_oof2[:, bag] = ob; all_test2[:, bag] = tb
    if (bag+1)%10==0:
        a = all_oof2[:,:bag+1].mean(axis=1)
        print(f'  Bag {bag+1}/20: R2={r2_score(y,a):.6f} CM={cm(y,a,groups):.8f} [{time.time()-t0:.0f}s]')
oof_gbdt = all_oof2.mean(axis=1); test_gbdt = all_test2.mean(axis=1)
print(f'GBDT: R2={r2_score(y,oof_gbdt):.6f} CM={cm(y,oof_gbdt,groups):.8f}')
print(f'\n=== OPTIMIZATION ===')
zero_cm = cm(y, np.zeros(len(y)), groups); print(f'Zero: {zero_cm:.8f}')
cands = {'zero': (zero_cm, np.zeros(len(test_ids)))}
for nm, oof, tp in [('dart', oof_dart, test_dart), ('gbdt', oof_gbdt, test_gbdt)]:
    for s in np.arange(0.3, 1.51, 0.01):
        sc = cm(y, oof*s, groups)
        if nm not in cands or sc > cands[nm][0]: cands[nm] = (sc, tp*s)
for a in np.arange(0, 1.01, 0.05):
    bl_o = a*oof_dart + (1-a)*oof_gbdt; bl_t = a*test_dart + (1-a)*test_gbdt
    for s in np.arange(0.3, 1.51, 0.02):
        sc = cm(y, bl_o*s, groups)
        if 'blend' not in cands or sc > cands['blend'][0]: cands['blend'] = (sc, bl_t*s)
for nm, oof, tp in [('dart', oof_dart, test_dart), ('gbdt', oof_gbdt, test_gbdt)]:
    p90 = np.percentile(np.abs(oof[oof!=0]), 90) if np.any(oof!=0) else 0
    if p90 > 0:
        oo_w = np.clip(oof, -p90, p90); tp_w = np.clip(tp, -p90, p90)
        for s in np.arange(0.5, 1.51, 0.01):
            sc = cm(y, oo_w*s, groups)
            if f'{nm}_win' not in cands or sc > cands[f'{nm}_win'][0]: cands[f'{nm}_win'] = (sc, tp_w*s)
all_c = np.hstack([all_oof, all_oof2]); all_tc = np.hstack([all_test, all_test2])
oof60 = all_c.mean(axis=1); test60 = all_tc.mean(axis=1)
for s in np.arange(0.3, 1.51, 0.01):
    sc = cm(y, oof60*s, groups)
    if 'all60' not in cands or sc > cands['all60'][0]: cands['all60'] = (sc, test60*s)
signs = np.sign(all_c); agr = np.abs(signs.sum(axis=1))/60
for ma in [0.6, 0.7]:
    oo_a = np.where(agr>=ma, oof60, 0)
    tt_a = np.where(np.abs(np.sign(all_tc).sum(axis=1))/60>=ma, test60, 0)
    cands[f'agr60_{ma}'] = (cm(y, oo_a, groups), tt_a)
print(f'\nCandidates:')
for n in sorted(cands, key=lambda k: -cands[k][0])[:10]: print(f'  {n}: {cands[n][0]:.8f}')
best = max(cands, key=lambda k: cands[k][0]); final = cands[best][1]
print(f'\nBest: {best} (CM={cands[best][0]:.8f})')
sub = pd.DataFrame({'ID': test_ids, 'TARGET': final}); sub.to_csv('submission.csv', index=False)
nz = np.sum(final != 0)
print(f'\nSub: {sub.shape} mean={final.mean():.6f} std={final.std():.6f} nonzero={nz}')
print(f'Time: {time.time()-T0:.0f}s ({(time.time()-T0)/60:.1f}min)')
print('Done!')
