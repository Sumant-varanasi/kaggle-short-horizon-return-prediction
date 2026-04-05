
import numpy as np, pandas as pd, os, glob, time, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from itertools import product as iprod

# ── Data loading ─────────────────────────────
BASE = '/kaggle/input/competitions/short-horizon-return-prediction-challenge-by-i-rage'
train_path = test_path = None
for ext in ['parquet','csv']:
    cand = os.path.join(BASE, f'train.{ext}')
    if os.path.isfile(cand):
        train_path = cand
        test_path = os.path.join(BASE, f'test.{ext}')
        break
if train_path is None:
    for f in glob.glob(os.path.join(BASE,'**','train.*'), recursive=True):
        train_path = f
        test_path = f.replace('train.','test.')
        break
assert train_path is not None, f'No train file found in {BASE}'
print(f'Loading: {train_path}')
loader = pd.read_parquet if train_path.endswith('.parquet') else pd.read_csv
train = loader(train_path)
test  = loader(test_path)
print(f'Train: {train.shape}, Test: {test.shape}')

#  Columns ──────────────────
target_col = [c for c in train.columns if c.upper()=='TARGET'][0]
id_col     = [c for c in test.columns  if c.upper()=='ID'][0]
cv_col     = [c for c in train.columns if 'CV' in c.upper() or 'GROUP' in c.upper()]
cv_col     = cv_col[0] if cv_col else None
drop_cols  = [target_col, id_col] + ([cv_col] if cv_col else [])
feat_cols  = [c for c in train.columns if c not in drop_cols]
print(f'Features: {len(feat_cols)}, CV column: {cv_col}')

y = train[target_col].values.astype(np.float32)
X = train[feat_cols].values.astype(np.float32)
X_test = test[feat_cols].values.astype(np.float32)
test_ids = test[id_col].values

for arr in [X, X_test]:
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

N_FOLDS = 5
if cv_col:
    groups = train[cv_col].values
    unique_g = np.sort(np.unique(groups))
    fold_map = {g: i % N_FOLDS for i, g in enumerate(unique_g)}
    folds_idx = np.array([fold_map[g] for g in groups])
else:
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    folds_idx = np.zeros(len(y), dtype=int)
    for i, (_, vi) in enumerate(kf.split(X)):
        folds_idx[vi] = i

#  Feature selection (stability) ───
t0 = time.time()
TOP_K = 120
scores_acc = np.zeros(len(feat_cols))
for f in range(N_FOLDS):
    mask = folds_idx == f
    Xv, yv = X[mask], y[mask]
    corrs = np.array([np.corrcoef(Xv[:,j], yv)[0,1] if np.std(Xv[:,j])>0 else 0
                       for j in range(Xv.shape[1])])
    corrs = np.nan_to_num(corrs)
    ranks = np.argsort(np.argsort(-np.abs(corrs)))
    scores_acc += ranks
best_idx = np.argsort(scores_acc)[:TOP_K]
feat_names_sel = [feat_cols[i] for i in best_idx]
X_sel = X[:, best_idx]
X_test_sel = X_test[:, best_idx]
print(f'Feature selection: {len(feat_cols)} -> {TOP_K} in {time.time()-t0:.0f}s')

#  PCA denoised features ───────
t1 = time.time()
N_PCA = 30
mu = np.mean(X_sel, axis=0)
sigma = np.std(X_sel, axis=0) + 1e-8
X_std = (X_sel - mu) / sigma
X_test_std = (X_test_sel - mu) / sigma
pca = PCA(n_components=N_PCA, random_state=42)
X_pca = pca.fit_transform(X_std)
X_test_pca = pca.transform(X_test_std)
X_recon = pca.inverse_transform(X_pca) * sigma + mu
X_test_recon = pca.inverse_transform(X_test_pca) * sigma + mu
print(f'PCA ({N_PCA} components, var explained: {pca.explained_variance_ratio_.sum():.3f}) in {time.time()-t1:.0f}s')

#  Feature interactions (top 20 features) 
t1 = time.time()
TOP_INTER = 15
inter_idx = best_idx[:TOP_INTER]
X_inter_parts = []
X_test_inter_parts = []
inter_names = []
for i in range(TOP_INTER):
    for j in range(i+1, TOP_INTER):
        xi, xj = X_sel[:, i], X_sel[:, j]
        xti, xtj = X_test_sel[:, i], X_test_sel[:, j]
        X_inter_parts.append((xi * xj).reshape(-1,1))
        X_test_inter_parts.append((xti * xtj).reshape(-1,1))
        inter_names.append(f'inter_{i}x{j}')
X_inter = np.hstack(X_inter_parts) if X_inter_parts else np.empty((len(y),0))
X_test_inter = np.hstack(X_test_inter_parts) if X_test_inter_parts else np.empty((len(X_test_sel),0))
print(f'Feature interactions: {X_inter.shape[1]} in {time.time()-t1:.0f}s')

#  Augmented feature sets 
X_aug = np.hstack([X_sel, X_pca, X_inter])
X_test_aug = np.hstack([X_test_sel, X_test_pca, X_test_inter])
print(f'Augmented features: {X_aug.shape[1]}')

oof_dict = {}
test_dict = {}

#  1. Extra Trees (random splits, less overfitting) 
t1 = time.time()
oof_et = np.zeros(len(y))
test_et = np.zeros(len(X_test_sel))
for f in range(N_FOLDS):
    tr = folds_idx != f; va = folds_idx == f
    et = ExtraTreesRegressor(
        n_estimators=500, max_depth=8, min_samples_leaf=200,
        max_features=0.3, n_jobs=-1, random_state=42
    )
    et.fit(X_sel[tr], y[tr])
    oof_et[va] = et.predict(X_sel[va])
    test_et += et.predict(X_test_sel) / N_FOLDS
et_r2 = r2_score(y, oof_et)
print(f'ExtraTrees: R2={et_r2:.6f} ({time.time()-t1:.0f}s)')
oof_dict['ExtraTrees'] = oof_et
test_dict['ExtraTrees'] = test_et

#  2. Extra Trees on PCA-denoised features 
t1 = time.time()
oof_et2 = np.zeros(len(y))
test_et2 = np.zeros(len(X_test_sel))
for f in range(N_FOLDS):
    tr = folds_idx != f; va = folds_idx == f
    et2 = ExtraTreesRegressor(
        n_estimators=500, max_depth=8, min_samples_leaf=200,
        max_features=0.5, n_jobs=-1, random_state=123
    )
    et2.fit(X_recon[tr], y[tr])
    oof_et2[va] = et2.predict(X_recon[va])
    test_et2 += et2.predict(X_test_recon) / N_FOLDS
et2_r2 = r2_score(y, oof_et2)
print(f'ExtraTrees_Denoised: R2={et2_r2:.6f} ({time.time()-t1:.0f}s)')
oof_dict['ET_Denoised'] = oof_et2
test_dict['ET_Denoised'] = test_et2

#  3. Bayesian Ridge on augmented features 
t1 = time.time()
oof_br = np.zeros(len(y))
test_br = np.zeros(len(X_test_sel))
mu_aug = np.mean(X_aug, axis=0)
sig_aug = np.std(X_aug, axis=0) + 1e-8
X_aug_s = (X_aug - mu_aug) / sig_aug
X_test_aug_s = (X_test_aug - mu_aug) / sig_aug
for f in range(N_FOLDS):
    tr = folds_idx != f; va = folds_idx == f
    br = BayesianRidge(max_iter=300, tol=1e-6)
    br.fit(X_aug_s[tr], y[tr])
    oof_br[va] = br.predict(X_aug_s[va])
    test_br += br.predict(X_test_aug_s) / N_FOLDS
br_r2 = r2_score(y, oof_br)
print(f'BayesianRidge_Aug: R2={br_r2:.6f} ({time.time()-t1:.0f}s)')
oof_dict['BayesianRidge'] = oof_br
test_dict['BayesianRidge'] = test_br

#  4. Ridge on PCA features only (ultra-regularized linear) 
t1 = time.time()
oof_rpca = np.zeros(len(y))
test_rpca = np.zeros(len(X_test_sel))
for f in range(N_FOLDS):
    tr = folds_idx != f; va = folds_idx == f
    rr = Ridge(alpha=50.0)
    rr.fit(X_pca[tr], y[tr])
    oof_rpca[va] = rr.predict(X_pca[va])
    test_rpca += rr.predict(X_test_pca) / N_FOLDS
rpca_r2 = r2_score(y, oof_rpca)
print(f'Ridge_PCA: R2={rpca_r2:.6f} ({time.time()-t1:.0f}s)')
oof_dict['Ridge_PCA'] = oof_rpca
test_dict['Ridge_PCA'] = test_rpca

#  5. LightGBM (proven winner) 
import lightgbm as lgb
lgb_base = dict(n_estimators=600, learning_rate=0.03, max_depth=6, num_leaves=40,
                subsample=0.7, colsample_bytree=0.5, reg_alpha=0.1, reg_lambda=1.0,
                min_child_samples=100, n_jobs=-1, verbose=-1, feature_pre_filter=False)
lgb_cfgs = [
    ('LGB_RMSE',  {**lgb_base, 'objective':'regression', 'metric':'rmse'}),
    ('LGB_Huber', {**lgb_base, 'objective':'huber', 'huber_delta':0.01, 'metric':'rmse'}),
]
for name, params in lgb_cfgs:
    t1 = time.time()
    oof_p = np.zeros(len(y))
    test_p = np.zeros(len(X_test_sel))
    for f in range(N_FOLDS):
        tr = folds_idx != f; va = folds_idx == f
        dtrain = lgb.Dataset(X_sel[tr], y[tr], feature_name=feat_names_sel, free_raw_data=False)
        dval   = lgb.Dataset(X_sel[va], y[va], feature_name=feat_names_sel, free_raw_data=False)
        m = lgb.train(params, dtrain, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof_p[va] = m.predict(X_sel[va])
        test_p += m.predict(X_test_sel) / N_FOLDS
    r2 = r2_score(y, oof_p)
    print(f'{name}: R2={r2:.6f} ({time.time()-t1:.0f}s)')
    oof_dict[name] = oof_p
    test_dict[name] = test_p

#  6. CatBoost (proven winner) ──
from catboost import CatBoostRegressor
t1 = time.time()
oof_cb = np.zeros(len(y))
test_cb = np.zeros(len(X_test_sel))
for f in range(N_FOLDS):
    tr = folds_idx != f; va = folds_idx == f
    cb = CatBoostRegressor(iterations=800, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
                           subsample=0.7, rsm=0.5, loss_function='RMSE',
                           boosting_type='Plain', random_seed=42, verbose=0)
    cb.fit(X_sel[tr], y[tr], eval_set=(X_sel[va], y[va]), early_stopping_rounds=50, verbose=0)
    oof_cb[va] = cb.predict(X_sel[va])
    test_cb += cb.predict(X_test_sel) / N_FOLDS
cb_r2 = r2_score(y, oof_cb)
print(f'CB_RMSE: R2={cb_r2:.6f} ({time.time()-t1:.0f}s)')
oof_dict['CB_RMSE'] = oof_cb
test_dict['CB_RMSE'] = test_cb

#  7. Extra Trees on augmented features 
t1 = time.time()
oof_et3 = np.zeros(len(y))
test_et3 = np.zeros(len(X_test_sel))
for f in range(N_FOLDS):
    tr = folds_idx != f; va = folds_idx == f
    et3 = ExtraTreesRegressor(
        n_estimators=500, max_depth=7, min_samples_leaf=300,
        max_features=0.2, n_jobs=-1, random_state=777
    )
    et3.fit(X_aug[tr], y[tr])
    oof_et3[va] = et3.predict(X_aug[va])
    test_et3 += et3.predict(X_test_aug) / N_FOLDS
et3_r2 = r2_score(y, oof_et3)
print(f'ET_Augmented: R2={et3_r2:.6f} ({time.time()-t1:.0f}s)')
oof_dict['ET_Augmented'] = oof_et3
test_dict['ET_Augmented'] = test_et3

#  Ensemble optimization 
names = list(oof_dict.keys())
print(f'\nOptimising ensemble over {names}')
pos_mask = [r2_score(y, oof_dict[n]) > -0.01 for n in names]
pos_names = [n for n, m in zip(names, pos_mask) if m]
print(f'Candidates (R2 > -0.01): {pos_names}')

oof_pos = np.column_stack([oof_dict[n] for n in pos_names])
test_pos = np.column_stack([test_dict[n] for n in pos_names])

wt_grid = np.arange(0, 1.01, 0.05)
best_r2 = -999
best_w = None
n_pos = len(pos_names)

if n_pos <= 5:
    for combo in iprod(wt_grid, repeat=n_pos):
        s = sum(combo)
        if s < 0.05: continue
        w = np.array(combo) / s
        r = r2_score(y, oof_pos @ w)
        if r > best_r2:
            best_r2 = r; best_w = w.copy()
else:
    from scipy.optimize import minimize
    def neg_r2(w):
        w = np.abs(w); w /= w.sum() + 1e-12
        return -r2_score(y, oof_pos @ w)
    w0 = np.ones(n_pos) / n_pos
    res = minimize(neg_r2, w0, method='Nelder-Mead',
                   options={'maxiter':50000, 'xatol':1e-6, 'fatol':1e-10})
    best_w = np.abs(res.x); best_w /= best_w.sum()
    best_r2 = -res.fun

print(f'Best ensemble R2={best_r2:.6f}')
for n, w in zip(pos_names, best_w):
    print(f'  {n}: {w:.4f}')

oof_blend = oof_pos @ best_w
test_blend = test_pos @ best_w

#  Shrinkage 
best_shrink = 0.0
best_sr2 = -999
for s in np.arange(0.05, 1.01, 0.05):
    r = r2_score(y, oof_blend * s)
    if r > best_sr2: best_sr2 = r; best_shrink = s
for s in np.arange(max(0.01, best_shrink-0.05), best_shrink+0.06, 0.01):
    r = r2_score(y, oof_blend * s)
    if r > best_sr2: best_sr2 = r; best_shrink = s

print(f'Shrinkage={best_shrink:.2f}, Final OOF R2={best_sr2:.6f}')
preds = test_blend * best_shrink

#  Submission ─
sub = pd.read_csv(os.path.join(BASE, 'sample_submission.csv'))
id_sub = [c for c in sub.columns if c.upper()=='ID'][0]
tgt_sub = [c for c in sub.columns if c.upper()=='TARGET'][0]
sub = sub[[id_sub]].merge(
    pd.DataFrame({id_sub: test_ids, tgt_sub: preds}), on=id_sub, how='left')
sub[tgt_sub] = sub[tgt_sub].fillna(0.0)
sub.to_csv('submission.csv', index=False)
print(f'Submission saved: {sub.shape}')
print(f'Pred stats: mean={preds.mean():.8f}, std={preds.std():.8f}')
print(f'Total runtime: {time.time()-t0:.0f}s')
