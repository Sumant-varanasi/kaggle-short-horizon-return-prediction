# V33: Strong LGB + CatBoost (no shrinkage) — Score: -0.00042
# OOF R2: Ensemble=+0.001313, CatBoost=+0.001194, LGB1=+0.000494, LGB2=+0.000875
# KEY FINDING: Even with strong models (63-127 leaves, depth 7-9, 2000 rounds),
# GroupKFold OOF R2 is only ~0.1%. The features genuinely have near-zero
# predictive power for TARGET across temporal groups.
# Top LB teams (R2=0.86+) are exploiting test-set structure (ranking, KNN)
# which the final eval formula penalizes: Final = 2*mean(batch_R2) - full_R2
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
X = train[fc].values.astype(np.float32); Xt = test[fc].values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0); Xt = np.nan_to_num(Xt, nan=0, posinf=0, neginf=0)
del train, test; gc.collect()
ug = np.unique(groups); nf = len(ug) if len(ug) <= 5 else 5; gkf = GroupKFold(n_splits=nf)
# Strong LGB
pp1 = {'objective':'regression','metric':'rmse','verbosity':-1,'n_jobs':-1,
       'learning_rate':0.05,'num_leaves':63,'max_depth':7,
       'min_child_samples':100,'subsample':0.8,'colsample_bytree':0.8,
       'reg_alpha':0.1,'reg_lambda':0.1,'feature_pre_filter':False}
oo1 = np.zeros(len(y)); tp1 = np.zeros(len(Xt))
for ti, vi in gkf.split(X, y, groups):
    dt = lgb.Dataset(X[ti], y[ti]); dv = lgb.Dataset(X[vi], y[vi], reference=dt)
    m = lgb.train(pp1, dt, num_boost_round=2000, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    oo1[vi] = m.predict(X[vi]); tp1 += m.predict(Xt)/nf
# Deeper LGB
pp2 = {'objective':'regression','metric':'rmse','verbosity':-1,'n_jobs':-1,
       'learning_rate':0.03,'num_leaves':127,'max_depth':9,
       'min_child_samples':50,'subsample':0.7,'colsample_bytree':0.7,
       'reg_alpha':0.5,'reg_lambda':0.5,'feature_pre_filter':False}
oo2 = np.zeros(len(y)); tp2 = np.zeros(len(Xt))
for ti, vi in gkf.split(X, y, groups):
    dt = lgb.Dataset(X[ti], y[ti]); dv = lgb.Dataset(X[vi], y[vi], reference=dt)
    m = lgb.train(pp2, dt, num_boost_round=2000, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    oo2[vi] = m.predict(X[vi]); tp2 += m.predict(Xt)/nf
# CatBoost
try:
    from catboost import CatBoostRegressor
    oo3 = np.zeros(len(y)); tp3 = np.zeros(len(Xt))
    for ti, vi in gkf.split(X, y, groups):
        cb = CatBoostRegressor(iterations=2000, learning_rate=0.05, depth=8, l2_leaf_reg=3,
                               random_seed=42, verbose=0, early_stopping_rounds=50, subsample=0.8)
        cb.fit(X[ti], y[ti], eval_set=(X[vi], y[vi]), verbose=0)
        oo3[vi] = cb.predict(X[vi]); tp3 += cb.predict(Xt)/nf
    oof_ens = (oo1+oo2+oo3)/3; test_ens = (tp1+tp2+tp3)/3
except:
    oof_ens = (oo1+oo2)/2; test_ens = (tp1+tp2)/2
final = test_ens
sub = pd.DataFrame({'ID': test_ids, 'TARGET': final}); sub.to_csv('submission.csv', index=False)
print(f'Ensemble OOF R2: {r2_score(y, oof_ens):.6f}')
print(f'Time: {time.time()-T0:.0f}s')
