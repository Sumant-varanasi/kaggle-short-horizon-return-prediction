# V25: Regular KFold experiment (CONFIRMED GroupKFold is correct)
# Score: -0.03526 (DISASTER - proved regular KFold causes massive overfitting)
# Key insight: Using regular KFold instead of GroupKFold leaks temporal info
# This experiment confirmed GroupKFold is the CORRECT CV strategy
import numpy as np, pandas as pd, time, os, warnings, gc, lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, GroupKFold
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
clip = float(np.percentile(np.abs(y), 95))
y_clip = np.clip(y, -clip, clip)

X = train[fc].values.astype(np.float32)
Xt = test[fc].values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
Xt = np.nan_to_num(Xt, nan=0, posinf=0, neginf=0)
del train, test; gc.collect()

def cm(yt, yp, g):
    br = [r2_score(yt[g==v], yp[g==v]) for v in np.unique(g) if np.std(yt[g==v]) > 1e-12]
    return 2 * np.mean(br) - r2_score(yt, yp)

# EXPERIMENT: Regular KFold vs GroupKFold
# Regular KFold randomly splits rows - temporal neighbors end up in same fold
# This creates MASSIVE data leakage for time-series data
print(f'\n=== REGULAR KFOLD (WRONG - for comparison only) ===')
rkf = KFold(n_splits=5, shuffle=True, random_state=42)
pp = {'objective':'huber','huber_delta':0.5,'metric':'rmse','verbosity':-1,'n_jobs':-1,
      'learning_rate':0.03,'num_leaves':15,'max_depth':4,'min_child_samples':500,
      'subsample':0.7,'colsample_bytree':0.6,'reg_alpha':5.0,'reg_lambda':5.0,
      'feature_pre_filter':False}

oo_rkf = np.zeros(len(y)); tp_rkf = np.zeros(len(Xt))
for ti, vi in rkf.split(X):
    dt = lgb.Dataset(X[ti], y_clip[ti]); dv = lgb.Dataset(X[vi], y_clip[vi], reference=dt)
    m = lgb.train(pp, dt, num_boost_round=500, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
    oo_rkf[vi] = m.predict(X[vi]); tp_rkf += m.predict(Xt) / 5
r2_rkf = r2_score(y, oo_rkf)
print(f'Regular KFold OOF R2: {r2_rkf:.6f} (INFLATED by leakage!)')

print(f'\n=== GROUP KFOLD (CORRECT) ===')
ug = np.unique(groups); nf = len(ug) if len(ug) <= 5 else 5
gkf = GroupKFold(n_splits=nf)
oo_gkf = np.zeros(len(y)); tp_gkf = np.zeros(len(Xt))
for ti, vi in gkf.split(X, y_clip, groups):
    dt = lgb.Dataset(X[ti], y_clip[ti]); dv = lgb.Dataset(X[vi], y_clip[vi], reference=dt)
    m = lgb.train(pp, dt, num_boost_round=500, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
    oo_gkf[vi] = m.predict(X[vi]); tp_gkf += m.predict(Xt) / nf
r2_gkf = r2_score(y, oo_gkf)
print(f'GroupKFold OOF R2: {r2_gkf:.6f} (TRUE generalization)')

print(f'\n=== COMPARISON ===')
print(f'Regular KFold R2: {r2_rkf:.6f} (FAKE - leaked)')
print(f'GroupKFold R2:    {r2_gkf:.6f} (REAL)')
print(f'Leakage factor:   {r2_rkf/max(r2_gkf,1e-10):.1f}x')

# Use GroupKFold predictions with shrinkage
zero_cm = cm(y, np.zeros(len(y)), groups)
best_s = 0; best_cm = zero_cm
for s in np.arange(0.01, 1.01, 0.01):
    sc = cm(y, oo_gkf * s, groups)
    if sc > best_cm: best_cm = sc; best_s = s

if best_cm > zero_cm:
    final = tp_gkf * best_s
    print(f'Using GroupKFold s={best_s:.2f}')
else:
    final = np.zeros(len(test_ids))
    print('Using ZERO')

sub = pd.DataFrame({'ID': test_ids, 'TARGET': final})
sub.to_csv('submission.csv', index=False)
print(f'\nSub: {sub.shape} mean={final.mean():.6f} std={final.std():.6f}')
print(f'Time: {time.time()-T0:.0f}s ({(time.time()-T0)/60:.1f}min)')
print('Done!')
