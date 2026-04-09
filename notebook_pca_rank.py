import numpy as np
import pandas as pd
import time, os, warnings, gc
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
warnings.filterwarnings('ignore')
T0 = time.time()
for p in ['/kaggle/input/competitions/short-horizon-return-prediction-challenge-by-i-rage/','/kaggle/input/short-horizon-return-prediction-challenge-by-i-rage/']:
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
y = np.clip(y_raw, -clip95, clip95)
X = train[fc].values.astype(np.float32)
Xt = test[fc].values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
Xt = np.nan_to_num(Xt, nan=0, posinf=0, neginf=0)
del train, test; gc.collect()
ug = np.unique(groups)
nf = len(ug) if len(ug) <= 5 else 5
gkf = GroupKFold(n_splits=nf)notebook_pca_rank.py
def cm(yt, yp, g):
    br = [r2_score(yt[g==v], yp[g==v]) for v in np.unique(g) if np.std(yt[g==v]) > 1e-12]
    return 2 * np.mean(br) - r2_score(yt, yp)
oof_all = {}; test_all = {}
# PCA Factor Model
for n_comp in [10, 20, 50]:
    key = f'PCA_{n_comp}_Ridge'; t0 = time.time()
    oof = np.zeros(len(y)); tpred = np.zeros(len(Xt))
    for fi, (ti, vi) in enumerate(gkf.split(X, y, groups)):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[ti]); Xva = sc.transform(X[vi]); Xte = sc.transform(Xt)
        pca = PCA(n_components=n_comp, random_state=42); Xtr_pca = pca.fit_transform(Xtr); Xva_pca = pca.transform(Xva); Xte_pca = pca.transform(Xte)
        model = Ridge(alpha=100); model.fit(Xtr_pca, y[ti]); oof[vi] = model.predict(Xva_pca); tpred += model.predict(Xte_pca)/nf
    oof_all[key]=oof; test_all[key]=tpred; print(f'{key}: R2={r2_score(y_raw,oof):.6f} CS={cm(y_raw,oof,groups):.8f}')
# PLS Regression
for n_comp in [5, 10, 20]:
    key = f'PLS_{n_comp}'; t0 = time.time()
    oof = np.zeros(len(y)); tpred = np.zeros(len(Xt))
    for fi, (ti, vi) in enumerate(gkf.split(X, y, groups)):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[ti]); Xva = sc.transform(X[vi]); Xte = sc.transform(Xt)
        pls = PLSRegression(n_components=n_comp, max_iter=200); pls.fit(Xtr, y[ti])
        oof[vi] = pls.predict(Xva).ravel(); tpred += pls.predict(Xte).ravel()/nf
    oof_all[key]=oof; test_all[key]=tpred; print(f'{key}: R2={r2_score(y_raw,oof):.6f} CS={cm(y_raw,oof,groups):.8f}')
# See full version on Kaggle V21
print('Full code on Kaggle notebook V21')
