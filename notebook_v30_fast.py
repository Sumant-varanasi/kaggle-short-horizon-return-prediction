# V30: Fast Diverse Ensemble (Score: -0.00060)
# CatBoost OOF R2=+0.000123 (strongest ever) but didn't generalize
# Key lesson: Strong OOF signal from CatBoost overfits on test
# Models: LGB Huber, LGB MAE, Ridge x2, 20-bag feature-bagged, CatBoost
# Best candidate: blend2 (CatBoost + LGB Huber)
# Runtime: 25.5 min (under 30min limit)
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
clip85 = float(np.percentile(np.abs(y), 85)); y85 = np.clip(y, -clip85, clip85)
X = train[fc].values.astype(np.float32); Xt = test[fc].values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0); Xt = np.nan_to_num(Xt, nan=0, posinf=0, neginf=0)
imp = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31, subsample=0.5, colsample_bytree=0.5, verbose=-1, n_jobs=-1)
imp.fit(X, np.clip(y,-float(np.percentile(np.abs(y),95)),float(np.percentile(np.abs(y),95))))
top_idx = np.argsort(imp.feature_importances_)[::-1][:100]; X_top = X[:, top_idx]; Xt_top = Xt[:, top_idx]
del train, test; gc.collect()
ug = np.unique(groups); nfolds = len(ug) if len(ug) <= 5 else 5; gkf = GroupKFold(n_splits=nfolds)
def cm(yt, yp, g):
    br = [r2_score(yt[g==v], yp[g==v]) for v in np.unique(g) if np.std(yt[g==v]) > 1e-12]
    return 2 * np.mean(br) - r2_score(yt, yp)
oof_all = {}; test_all = {}
pp1 = {'objective':'huber','huber_delta':0.3,'metric':'rmse','verbosity':-1,'n_jobs':-1,'learning_rate':0.03,'num_leaves':7,'max_depth':3,'min_child_samples':1000,'subsample':0.5,'colsample_bytree':0.6,'reg_alpha':10.0,'reg_lambda':10.0,'min_gain_to_split':0.1,'feature_pre_filter':False}
for nm, pp in [('lgb_huber', pp1), ('lgb_mae', {**pp1, 'objective':'mae','metric':'mae'})]:
    oo = np.zeros(len(y)); tp = np.zeros(len(Xt))
    for ti, vi in gkf.split(X_top, y85, groups):
        dt = lgb.Dataset(X_top[ti], y85[ti]); dv = lgb.Dataset(X_top[vi], y85[vi], reference=dt)
        m = lgb.train(pp, dt, num_boost_round=300, valid_sets=[dv], callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        oo[vi] = m.predict(X_top[vi]); tp += m.predict(Xt_top)/nfolds
    oof_all[nm] = oo; test_all[nm] = tp
for alpha in [10000, 50000]:
    nm = f'ridge_a{alpha}'; oo = np.zeros(len(y)); tp = np.zeros(len(Xt))
    for ti, vi in gkf.split(X_top, y85, groups):
        sc = StandardScaler(); Xtr = sc.fit_transform(X_top[ti]); Xva = sc.transform(X_top[vi]); Xte = sc.transform(Xt_top)
        m = Ridge(alpha=alpha); m.fit(Xtr, y85[ti]); oo[vi] = m.predict(Xva); tp += m.predict(Xte)/nfolds
    oof_all[nm] = oo; test_all[nm] = tp
# 20-bag feature-bagged
all_oof_bag = np.zeros((len(y), 20)); all_test_bag = np.zeros((len(Xt), 20))
for bag in range(20):
    rng = np.random.RandomState(42+bag); idx = rng.choice(100, 30, replace=False)
    Xb = X_top[:, idx]; Xtb = Xt_top[:, idx]; ob = np.zeros(len(y)); tb = np.zeros(len(Xt))
    for ti, vi in gkf.split(Xb, y85, groups):
        dt = lgb.Dataset(Xb[ti], y85[ti]); dv = lgb.Dataset(Xb[vi], y85[vi], reference=dt)
        m = lgb.train(pp1, dt, num_boost_round=200, valid_sets=[dv], callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)])
        ob[vi] = m.predict(Xb[vi]); tb += m.predict(Xtb)/nfolds
    all_oof_bag[:, bag] = ob; all_test_bag[:, bag] = tb
oof_all['bag20'] = all_oof_bag.mean(axis=1); test_all['bag20'] = all_test_bag.mean(axis=1)
try:
    from catboost import CatBoostRegressor
    oo = np.zeros(len(y)); tp = np.zeros(len(Xt))
    for ti, vi in gkf.split(X_top, y85, groups):
        cb = CatBoostRegressor(iterations=200, learning_rate=0.03, depth=4, l2_leaf_reg=10, random_seed=42, verbose=0, loss_function='RMSE', early_stopping_rounds=30, subsample=0.6, colsample_bylevel=0.5)
        cb.fit(X_top[ti], y85[ti], eval_set=(X_top[vi], y85[vi]), verbose=0)
        oo[vi] = cb.predict(X_top[vi]); tp += cb.predict(Xt_top)/nfolds
    oof_all['catboost'] = oo; test_all['catboost'] = tp
except: pass
# Optimization
zero_cm = cm(y, np.zeros(len(y)), groups); cands = {'zero': (zero_cm, np.zeros(len(test_ids)))}
for nm, oo in oof_all.items():
    tp = test_all[nm]
    for s in np.arange(0.1, 1.51, 0.01):
        sc = cm(y, oo*s, groups)
        if nm not in cands or sc > cands[nm][0]: cands[nm] = (sc, tp*s)
names = list(oof_all.keys()); oof_eq = np.column_stack([oof_all[k] for k in names]).mean(axis=1)
test_eq = np.column_stack([test_all[k] for k in names]).mean(axis=1)
for s in np.arange(0.1, 1.51, 0.01):
    sc = cm(y, oof_eq*s, groups)
    if 'ensemble' not in cands or sc > cands['ensemble'][0]: cands['ensemble'] = (sc, test_eq*s)
singles = sorted([(nm, cands[nm][0]) for nm in names if nm in cands], key=lambda x:-x[1])
if len(singles) >= 2:
    k1, k2 = singles[0][0], singles[1][0]
    for a in np.arange(0, 1.01, 0.05):
        bl_o = a*oof_all[k1]+(1-a)*oof_all[k2]; bl_t = a*test_all[k1]+(1-a)*test_all[k2]
        for s in np.arange(0.1, 1.51, 0.02):
            sc = cm(y, bl_o*s, groups)
            if 'blend2' not in cands or sc > cands['blend2'][0]: cands['blend2'] = (sc, bl_t*s)
best = max(cands, key=lambda k: cands[k][0]); final = cands[best][1]
sub = pd.DataFrame({'ID': test_ids, 'TARGET': final}); sub.to_csv('submission.csv', index=False)
print(f'Best: {best} | Time: {time.time()-T0:.0f}s')
