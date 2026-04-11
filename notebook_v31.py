# V31: 7-model top100 + extreme shrinkage (Score: -0.00021)
# Attempted to replicate V11 approach on top 100 features with 85th clip
# Key learning: top 100 features + 85th clip hurts compared to ALL features + 99th clip
# V11 used ALL 445 features + 99th clip = -0.00000, this used top100 + 85th = -0.00021
# Confirms: feature selection HURTS in this low-SNR regime
import numpy as np, pandas as pd, time, os, warnings, gc, lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')
T0 = time.time()
for p in ['/kaggle/input/competitions/short-horizon-return-prediction-challenge-by-i-rage/','/kaggle/input/short-horizon-return-prediction-challenge-by-i-rage/']:
    if os.path.exists(p): DATA=p; break
else: raise FileNotFoundError('Data not found')
train = pd.read_parquet(DATA + 'train.parquet'); test = pd.read_parquet(DATA + 'test.parquet')
y = train['TARGET'].values; groups = train['CV_GROUP'].values; test_ids = test['ID'].values
fc = [c for c in train.columns if c not in ['TARGET', 'CV_GROUP', 'ID']]
clip85 = float(np.percentile(np.abs(y), 85)); y85 = np.clip(y, -clip85, clip85)
X = train[fc].values.astype(np.float32); Xt = test[fc].values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0); Xt = np.nan_to_num(Xt, nan=0, posinf=0, neginf=0)
imp = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31, subsample=0.5, colsample_bytree=0.5, verbose=-1, n_jobs=-1)
imp.fit(X, y85); top_idx = np.argsort(imp.feature_importances_)[::-1][:100]
X_top = X[:, top_idx]; Xt_top = Xt[:, top_idx]
del train, test; gc.collect()
ug = np.unique(groups); nf = len(ug) if len(ug) <= 5 else 5; gkf = GroupKFold(n_splits=nf)
def cm(yt, yp, g):
    br = [r2_score(yt[g==v], yp[g==v]) for v in np.unique(g) if np.std(yt[g==v]) > 1e-12]
    return 2 * np.mean(br) - r2_score(yt, yp)
configs = [
    ('LGB_h03', {'objective':'huber','huber_delta':0.3,'metric':'rmse','verbosity':-1,'n_jobs':-1,'learning_rate':0.01,'num_leaves':7,'max_depth':3,'min_child_samples':1000,'subsample':0.5,'colsample_bytree':0.5,'reg_alpha':10.0,'reg_lambda':10.0,'min_gain_to_split':0.1,'feature_pre_filter':False}),
    ('LGB_h10', {'objective':'huber','huber_delta':1.0,'metric':'rmse','verbosity':-1,'n_jobs':-1,'learning_rate':0.01,'num_leaves':15,'max_depth':4,'min_child_samples':500,'subsample':0.6,'colsample_bytree':0.5,'reg_alpha':5.0,'reg_lambda':5.0,'min_gain_to_split':0.05,'feature_pre_filter':False}),
    ('LGB_mse', {'objective':'regression','metric':'rmse','verbosity':-1,'n_jobs':-1,'learning_rate':0.01,'num_leaves':7,'max_depth':3,'min_child_samples':1000,'subsample':0.5,'colsample_bytree':0.5,'reg_alpha':10.0,'reg_lambda':10.0,'min_gain_to_split':0.1,'feature_pre_filter':False}),
    ('LGB_mae', {'objective':'mae','metric':'mae','verbosity':-1,'n_jobs':-1,'learning_rate':0.01,'num_leaves':7,'max_depth':3,'min_child_samples':1000,'subsample':0.5,'colsample_bytree':0.5,'reg_alpha':10.0,'reg_lambda':10.0,'min_gain_to_split':0.1,'feature_pre_filter':False}),
]
models_oof = []; models_test = []
for nm, pp in configs:
    oo = np.zeros(len(y)); tp = np.zeros(len(Xt))
    for ti, vi in gkf.split(X_top, y85, groups):
        dt = lgb.Dataset(X_top[ti], y85[ti]); dv = lgb.Dataset(X_top[vi], y85[vi], reference=dt)
        m = lgb.train(pp, dt, num_boost_round=500, valid_sets=[dv], callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        oo[vi] = m.predict(X_top[vi]); tp += m.predict(Xt_top)/nf
    models_oof.append(oo); models_test.append(tp)
    print(f'{nm}: R2={r2_score(y,oo):.6f}')
try:
    from catboost import CatBoostRegressor
    for depth, lr, nm in [(3, 0.01, 'CB_d3'), (4, 0.02, 'CB_d4'), (5, 0.03, 'CB_d5')]:
        oo = np.zeros(len(y)); tp = np.zeros(len(Xt))
        for ti, vi in gkf.split(X_top, y85, groups):
            cb = CatBoostRegressor(iterations=300, learning_rate=lr, depth=depth, l2_leaf_reg=10, random_seed=42, verbose=0, early_stopping_rounds=30, subsample=0.5, colsample_bylevel=0.4)
            cb.fit(X_top[ti], y85[ti], eval_set=(X_top[vi], y85[vi]), verbose=0)
            oo[vi] = cb.predict(X_top[vi]); tp += cb.predict(Xt_top)/nf
        models_oof.append(oo); models_test.append(tp)
        print(f'{nm}: R2={r2_score(y,oo):.6f}')
except: pass
oof_ens = np.mean(models_oof, axis=0); test_ens = np.mean(models_test, axis=0)
zero_cm = cm(y, np.zeros(len(y)), groups); best_s = 0; best_cm = zero_cm
for s in np.arange(0.01, 0.80, 0.005):
    sc = cm(y, oof_ens*s, groups)
    if sc > best_cm: best_cm = sc; best_s = s
print(f'Shrinkage: {best_s:.3f} CM={best_cm:.8f}')
if best_cm > zero_cm: final = test_ens * best_s
else: final = np.zeros(len(test_ids))
sub = pd.DataFrame({'ID': test_ids, 'TARGET': final}); sub.to_csv('submission.csv', index=False)
print(f'Time: {time.time()-T0:.0f}s')
