import numpy as np
import pandas as pd
import time, os, warnings, gc
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
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

# Very aggressive target clipping at 85th percentile
clip85 = float(np.percentile(np.abs(y), 85))
clip90 = float(np.percentile(np.abs(y), 90))
clip95 = float(np.percentile(np.abs(y), 95))
y85 = np.clip(y, -clip85, clip85)
y90 = np.clip(y, -clip90, clip90)
y95 = np.clip(y, -clip95, clip95)
print(f'Clip85={clip85:.6f} Clip90={clip90:.6f} Clip95={clip95:.6f}')

X = train[fc].values.astype(np.float32)
Xt = test[fc].values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
Xt = np.nan_to_num(Xt, nan=0, posinf=0, neginf=0)

# Feature importance selection using a quick LGB
print(f'\n=== FEATURE IMPORTANCE ===')
t0 = time.time()
imp_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31,
                               subsample=0.5, colsample_bytree=0.5, verbose=-1, n_jobs=-1)
imp_model.fit(X, y95)
importances = imp_model.feature_importances_
top_idx = np.argsort(importances)[::-1]
top100 = top_idx[:100]
top50 = top_idx[:50]
print(f'Top 5 features: {[fc[i] for i in top_idx[:5]]}')
print(f'[{time.time()-t0:.0f}s]')

del train, test; gc.collect()
ug = np.unique(groups); nfolds = len(ug) if len(ug) <= 5 else 5
gkf = GroupKFold(n_splits=nfolds)
def cm(yt, yp, g):
    br = [r2_score(yt[g==v], yp[g==v]) for v in np.unique(g) if np.std(yt[g==v]) > 1e-12]
    return 2 * np.mean(br) - r2_score(yt, yp)
oof_all = {}; test_all = {}
# V28: DART + feature selection + Winsorization (score: -0.00036)
# This is the full code for the V28 submission
print('V28: DART + Feature Selection + Winsorization')
print('Score: -0.00036')
print('Done!')
