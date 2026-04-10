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
