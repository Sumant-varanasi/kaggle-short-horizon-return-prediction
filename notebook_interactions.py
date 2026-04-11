# V23: Pairwise feature interactions model
# Score: -0.00064 (overfit due to too many interaction features)
# Key: Created pairwise interaction features between top features
# Then trained LGB + Ridge ensemble on raw + interaction features
# Lesson: Feature engineering adds noise in low-SNR regime
import numpy as np, pandas as pd, time, os, warnings, gc, lightgbm as lgb
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

ug = np.unique(groups)
nf = len(ug) if len(ug) <= 5 else 5
gkf = GroupKFold(n_splits=nf)

def cm(yt, yp, g):
    br = [r2_score(yt[g==v], yp[g==v]) for v in np.unique(g) if np.std(yt[g==v]) > 1e-12]
    return 2 * np.mean(br) - r2_score(yt, yp)

# Feature importance for top features
imp = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31,
                         subsample=0.5, colsample_bytree=0.5, verbose=-1, n_jobs=-1)
imp.fit(X, y_clip)
top_idx = np.argsort(imp.feature_importances_)[::-1][:20]
top_names = [fc[i] for i in top_idx]
print(f'Top 20 features for interactions: {top_names[:5]}...')

# Create pairwise interaction features (top 20 x top 20 = 190 pairs)
print(f'\n=== PAIRWISE INTERACTIONS ===')
interaction_train = []
interaction_test = []
for i in range(len(top_idx)):
    for j in range(i+1, len(top_idx)):
        interaction_train.append(X[:, top_idx[i]] * X[:, top_idx[j]])
        interaction_test.append(Xt[:, top_idx[i]] * Xt[:, top_idx[j]])

X_inter = np.column_stack(interaction_train).astype(np.float32)
Xt_inter = np.column_stack(interaction_test).astype(np.float32)
print(f'Interaction features: {X_inter.shape[1]}')

# Combined features
X_all = np.hstack([X, X_inter])
Xt_all = np.hstack([Xt, Xt_inter])
X_all = np.nan_to_num(X_all, nan=0, posinf=0, neginf=0)
Xt_all = np.nan_to_num(Xt_all, nan=0, posinf=0, neginf=0)

# LGB on combined
pp = {'objective':'huber','huber_delta':0.5,'metric':'rmse','verbosity':-1,'n_jobs':-1,
      'learning_rate':0.02,'num_leaves':15,'max_depth':4,'min_child_samples':500,
      'subsample':0.6,'colsample_bytree':0.4,'reg_alpha':5.0,'reg_lambda':5.0,
      'feature_pre_filter':False}

oo = np.zeros(len(y)); tp = np.zeros(len(Xt))
for ti, vi in gkf.split(X_all, y_clip, groups):
    dt = lgb.Dataset(X_all[ti], y_clip[ti]); dv = lgb.Dataset(X_all[vi], y_clip[vi], reference=dt)
    m = lgb.train(pp, dt, num_boost_round=500, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
    oo[vi] = m.predict(X_all[vi]); tp += m.predict(Xt_all) / nf
print(f'LGB+Interactions: R2={r2_score(y,oo):.6f} CM={cm(y,oo,groups):.8f}')

# Shrinkage search
zero_cm = cm(y, np.zeros(len(y)), groups)
best_s = 0; best_cm = zero_cm
for s in np.arange(0.01, 1.01, 0.01):
    sc = cm(y, oo * s, groups)
    if sc > best_cm: best_cm = sc; best_s = s
print(f'Best shrinkage: {best_s:.2f} CM={best_cm:.8f} (zero={zero_cm:.8f})')

if best_cm > zero_cm:
    final = tp * best_s
    print('Using MODEL')
else:
    final = np.zeros(len(test_ids))
    print('Using ZERO')

sub = pd.DataFrame({'ID': test_ids, 'TARGET': final})
sub.to_csv('submission.csv', index=False)
print(f'\nSub: {sub.shape} mean={final.mean():.6f} std={final.std():.6f}')
print(f'Time: {time.time()-T0:.0f}s ({(time.time()-T0)/60:.1f}min)')
print('Done!')
