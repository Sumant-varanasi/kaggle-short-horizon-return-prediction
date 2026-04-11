# V32: V11 Replica - ALL features, 99th clip, 7-model ensemble, shrinkage search
# Score: pending (running on Kaggle)
# This is an exact replica of V11's winning approach:
# - ALL 445 features (no feature selection)
# - 99th percentile target clipping (not 85th)
# - 7 diverse models: 4 LGB (Huber 0.3, Huber 1.0, MSE, MAE) + 2 CatBoost + Ridge
# - Equal-weight ensemble
# - Shrinkage search from 0.05 to 0.70 (V11 found 0.35 optimal)
import numpy as np, pandas as pd, time, os, warnings, gc, lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
T0 = time.time()
for p in ['/kaggle/input/competitions/short-horizon-return-prediction-challenge-by-i-rage/','/kaggle/input/short-horizon-return-prediction-challenge-by-i-rage/']:
    if os.path.exists(p): DATA=p; break
else: raise FileNotFoundError('Data not found')
train = pd.read_parquet(DATA + 'train.parquet'); test = pd.read_parquet(DATA + 'test.parquet')
y = train['TARGET'].values; groups = train['CV_GROUP'].values; test_ids = test['ID'].values
fc = [c for c in train.columns if c not in ['TARGET', 'CV_GROUP', 'ID']]
clip99 = float(np.percentile(np.abs(y), 99)); y99 = np.clip(y, -clip99, clip99)
clip95 = float(np.percentile(np.abs(y), 95)); y95 = np.clip(y, -clip95, clip95)
X = train[fc].values.astype(np.float32); Xt = test[fc].values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0); Xt = np.nan_to_num(Xt, nan=0, posinf=0, neginf=0)
del train, test; gc.collect()
ug = np.unique(groups); nf = len(ug) if len(ug) <= 5 else 5; gkf = GroupKFold(n_splits=nf)
def cm(yt, yp, g):
    br = [r2_score(yt[g==v], yp[g==v]) for v in np.unique(g) if np.std(yt[g==v]) > 1e-12]
    return 2 * np.mean(br) - r2_score(yt, yp)
models_oof = []; models_test = []
lgb_cfgs = [
    ('LGB_h03', y99, {'objective':'huber','huber_delta':0.3,'metric':'rmse','verbosity':-1,'n_jobs':-1,'learning_rate':0.01,'num_leaves':15,'max_depth':4,'min_child_samples':500,'subsample':0.7,'colsample_bytree':0.5,'reg_alpha':1.0,'reg_lambda':1.0,'feature_pre_filter':False}),
    ('LGB_h10', y99, {'objective':'huber','huber_delta':1.0,'metric':'rmse','verbosity':-1,'n_jobs':-1,'learning_rate':0.01,'num_leaves':20,'max_depth':5,'min_child_samples':400,'subsample':0.7,'colsample_bytree':0.5,'reg_alpha':0.5,'reg_lambda':1.0,'feature_pre_filter':False}),
    ('LGB_mse', y99, {'objective':'regression','metric':'rmse','verbosity':-1,'n_jobs':-1,'learning_rate':0.01,'num_leaves':15,'max_depth':4,'min_child_samples':500,'subsample':0.7,'colsample_bytree':0.5,'reg_alpha':1.0,'reg_lambda':1.0,'feature_pre_filter':False}),
    ('LGB_mae', y95, {'objective':'mae','metric':'mae','verbosity':-1,'n_jobs':-1,'learning_rate':0.01,'num_leaves':15,'max_depth':4,'min_child_samples':500,'subsample':0.7,'colsample_bytree':0.5,'reg_alpha':1.0,'reg_lambda':1.0,'feature_pre_filter':False}),
]
for nm, yt, pp in lgb_cfgs:
    oo = np.zeros(len(y)); tp = np.zeros(len(Xt))
    for ti, vi in gkf.split(X, yt, groups):
        dt = lgb.Dataset(X[ti], yt[ti]); dv = lgb.Dataset(X[vi], yt[vi], reference=dt)
        m = lgb.train(pp, dt, num_boost_round=500, valid_sets=[dv], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oo[vi] = m.predict(X[vi]); tp += m.predict(Xt)/nf
    models_oof.append(oo); models_test.append(tp)
    print(f'{nm}: R2={r2_score(y,oo):.6f}')
try:
    from catboost import CatBoostRegressor
    for depth, lr, nm in [(4, 0.02, 'CB_d4'), (3, 0.01, 'CB_d3')]:
        oo = np.zeros(len(y)); tp = np.zeros(len(Xt))
        for ti, vi in gkf.split(X, y99, groups):
            cb = CatBoostRegressor(iterations=300, learning_rate=lr, depth=depth, l2_leaf_reg=5, random_seed=42, verbose=0, early_stopping_rounds=30, subsample=0.7, colsample_bylevel=0.5)
            cb.fit(X[ti], y99[ti], eval_set=(X[vi], y99[vi]), verbose=0)
            oo[vi] = cb.predict(X[vi]); tp += cb.predict(Xt)/nf
        models_oof.append(oo); models_test.append(tp)
        print(f'{nm}: R2={r2_score(y,oo):.6f}')
except: pass
oo = np.zeros(len(y)); tp = np.zeros(len(Xt))
for ti, vi in gkf.split(X, y99, groups):
    sc = StandardScaler(); Xtr = sc.fit_transform(X[ti]); Xva = sc.transform(X[vi]); Xte = sc.transform(Xt)
    m = Ridge(alpha=100); m.fit(Xtr, y99[ti]); oo[vi] = m.predict(Xva); tp += m.predict(Xte)/nf
models_oof.append(oo); models_test.append(tp)
oof_ens = np.mean(models_oof, axis=0); test_ens = np.mean(models_test, axis=0)
zero_cm = cm(y, np.zeros(len(y)), groups); best_s = 0; best_cm = zero_cm
for s in np.arange(0.05, 0.70, 0.005):
    sc = cm(y, oof_ens*s, groups)
    if sc > best_cm: best_cm = sc; best_s = s
print(f'Shrinkage: {best_s:.3f} CM={best_cm:.8f}')
if best_cm > zero_cm: final = test_ens * best_s
else: final = np.zeros(len(test_ids))
sub = pd.DataFrame({'ID': test_ids, 'TARGET': final}); sub.to_csv('submission.csv', index=False)
print(f'Time: {time.time()-T0:.0f}s')
