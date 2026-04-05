
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import time, warnings, gc, os
warnings.filterwarnings('ignore')

t0 = time.time()
print("=== ULTIMATE SUPER-ENSEMBLE: LightGBM + CatBoost + Ridge ===")

DATA = '/kaggle/input/competitions/short-horizon-return-prediction-challenge-by-i-rage'

# Try multiple path patterns
for base in [DATA,
             '/kaggle/input/short-horizon-return-prediction-challenge-by-i-rage',
             '/kaggle/input/short-horizon-return-prediction-challenge-by-irage']:
    for ext in ['.parquet', '.csv']:
        p = f'{base}/train{ext}'
        if os.path.exists(p):
            DATA = base
            USE_PARQUET = (ext == '.parquet')
            print(f"Found data at: {base} (format: {ext})")
            break
    else:
        continue
    break
else:
    # list what's available
    for d in ['/kaggle/input', '/kaggle/input/competitions']:
        if os.path.exists(d):
            print(f"Contents of {d}: {os.listdir(d)}")
    raise FileNotFoundError("Cannot find competition data")

TARGET_COL = 'TARGET'
CV_COL = 'CV_GROUP'
ID_COL = 'ID'
N_FOLDS = 5
TOP_K = 120
SHRINKAGE = 0.40

def load(name):
    if USE_PARQUET:
        return pd.read_parquet(f'{DATA}/{name}.parquet')
    return pd.read_csv(f'{DATA}/{name}.csv')

train = load('train')
test = load('test')
print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Columns: {list(train.columns[:10])} ...")

feat_cols = [c for c in train.columns if c not in [TARGET_COL, CV_COL, ID_COL]]
y = train[TARGET_COL].values
clip_lo, clip_hi = np.percentile(y, 1), np.percentile(y, 99)
y_clip = np.clip(y, clip_lo, clip_hi)
print(f"Target clipped to [{clip_lo:.6f}, {clip_hi:.6f}]")

print(f"\nFeature selection from {len(feat_cols)} features ...")
np.random.seed(42)
# Use CV_GROUP if available, else KFold
has_cv = CV_COL in train.columns
if has_cv:
    groups = train[CV_COL].unique()
    sel_groups = np.random.choice(groups, size=min(10, len(groups)), replace=False)
    imp_sum = np.zeros(len(feat_cols))
    for g in sel_groups:
        mask = train[CV_COL] == g
        ds = lgb.Dataset(train.loc[~mask, feat_cols], y_clip[~mask], free_raw_data=False)
        m = lgb.train({'objective':'regression','learning_rate':0.05,'num_leaves':31,
                        'min_data_in_leaf':500,'feature_pre_filter':False,
                        'verbosity':-1}, ds, num_boost_round=50)
        imp = m.feature_importance('gain')
        imp_sum += imp / (imp.sum() + 1e-15)
else:
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    imp_sum = np.zeros(len(feat_cols))
    for tr_idx, va_idx in kf.split(train):
        ds = lgb.Dataset(train.iloc[tr_idx][feat_cols], y_clip[tr_idx], free_raw_data=False)
        m = lgb.train({'objective':'regression','learning_rate':0.05,'num_leaves':31,
                        'min_data_in_leaf':500,'feature_pre_filter':False,
                        'verbosity':-1}, ds, num_boost_round=50)
        imp = m.feature_importance('gain')
        imp_sum += imp / (imp.sum() + 1e-15)

top_idx = np.argsort(imp_sum)[-TOP_K:]
sel_feats = [feat_cols[i] for i in top_idx]
print(f"Selected {len(sel_feats)} features in {time.time()-t0:.0f}s")

X_tr = train[sel_feats].values
X_te = test[sel_feats].values

# Build folds
fold_ids = np.zeros(len(y), dtype=int)
if has_cv:
    uniq_cv = np.sort(train[CV_COL].unique())
    for i, g in enumerate(uniq_cv):
        fold_ids[train[CV_COL].values == g] = i % N_FOLDS
else:
    kf2 = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for i, (_, vi) in enumerate(kf2.split(X_tr)):
        fold_ids[vi] = i
print(f"Folds built: {np.bincount(fold_ids)}")
del train; gc.collect()

print(f"\n--- Model 1: LightGBM RMSE --- [{time.time()-t0:.0f}s]")
lgb_p1 = {'objective':'regression','metric':'rmse','learning_rate':0.03,
    'num_leaves':63,'min_data_in_leaf':500,'feature_fraction':0.5,
    'bagging_fraction':0.7,'bagging_freq':1,'lambda_l1':1.0,'lambda_l2':5.0,
    'feature_pre_filter':False,'verbosity':-1,'seed':42}
oof1 = np.zeros(len(y)); pred1 = np.zeros(len(X_te))
for f in range(N_FOLDS):
    tr_i, va_i = fold_ids!=f, fold_ids==f
    dtr = lgb.Dataset(X_tr[tr_i], y_clip[tr_i], free_raw_data=False)
    dva = lgb.Dataset(X_tr[va_i], y_clip[va_i], reference=dtr, free_raw_data=False)
    m = lgb.train(lgb_p1, dtr, 500, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(30), lgb.log_evaluation(200)])
    oof1[va_i] = m.predict(X_tr[va_i]); pred1 += m.predict(X_te)/N_FOLDS
    del m, dtr, dva; gc.collect()
print(f"LGB1 R2={r2_score(y, oof1):.6f} [{time.time()-t0:.0f}s]")

print(f"\n--- Model 2: LightGBM Huber --- [{time.time()-t0:.0f}s]")
lgb_p2 = {'objective':'huber','huber_delta':0.01,'metric':'mae',
    'learning_rate':0.03,'num_leaves':31,'min_data_in_leaf':800,
    'feature_fraction':0.4,'bagging_fraction':0.7,'bagging_freq':1,
    'lambda_l1':2.0,'lambda_l2':10.0,'feature_pre_filter':False,
    'verbosity':-1,'seed':123}
oof2 = np.zeros(len(y)); pred2 = np.zeros(len(X_te))
for f in range(N_FOLDS):
    tr_i, va_i = fold_ids!=f, fold_ids==f
    dtr = lgb.Dataset(X_tr[tr_i], y_clip[tr_i], free_raw_data=False)
    dva = lgb.Dataset(X_tr[va_i], y_clip[va_i], reference=dtr, free_raw_data=False)
    m = lgb.train(lgb_p2, dtr, 500, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(30), lgb.log_evaluation(200)])
    oof2[va_i] = m.predict(X_tr[va_i]); pred2 += m.predict(X_te)/N_FOLDS
    del m, dtr, dva; gc.collect()
print(f"LGB2 R2={r2_score(y, oof2):.6f} [{time.time()-t0:.0f}s]")

print(f"\n--- Model 3: CatBoost Plain RMSE --- [{time.time()-t0:.0f}s]")
oof3 = np.zeros(len(y)); pred3 = np.zeros(len(X_te))
for f in range(N_FOLDS):
    tr_i, va_i = fold_ids!=f, fold_ids==f
    m = CatBoostRegressor(loss_function='RMSE', iterations=400, learning_rate=0.05,
        depth=6, l2_leaf_reg=10, random_strength=2, bagging_temperature=0.5,
        boosting_type='Plain', random_seed=42, verbose=200, early_stopping_rounds=30)
    m.fit(X_tr[tr_i], y_clip[tr_i], eval_set=(X_tr[va_i], y_clip[va_i]), verbose=200)
    oof3[va_i] = m.predict(X_tr[va_i]); pred3 += m.predict(X_te)/N_FOLDS
    del m; gc.collect()
print(f"CB1 R2={r2_score(y, oof3):.6f} [{time.time()-t0:.0f}s]")

print(f"\n--- Model 4: CatBoost Plain Huber --- [{time.time()-t0:.0f}s]")
oof4 = np.zeros(len(y)); pred4 = np.zeros(len(X_te))
for f in range(N_FOLDS):
    tr_i, va_i = fold_ids!=f, fold_ids==f
    m = CatBoostRegressor(loss_function='Huber:delta=0.01', iterations=300,
        learning_rate=0.05, depth=4, l2_leaf_reg=15, random_strength=3,
        bagging_temperature=1.0, boosting_type='Plain', random_seed=99,
        verbose=200, early_stopping_rounds=30)
    m.fit(X_tr[tr_i], y_clip[tr_i], eval_set=(X_tr[va_i], y_clip[va_i]), verbose=200)
    oof4[va_i] = m.predict(X_tr[va_i]); pred4 += m.predict(X_te)/N_FOLDS
    del m; gc.collect()
print(f"CB2 R2={r2_score(y, oof4):.6f} [{time.time()-t0:.0f}s]")

print(f"\n--- Model 5: Ridge --- [{time.time()-t0:.0f}s]")
scl = StandardScaler(); Xts = scl.fit_transform(X_tr); Xes = scl.transform(X_te)
oof5 = np.zeros(len(y)); pred5 = np.zeros(len(X_te))
for f in range(N_FOLDS):
    tr_i, va_i = fold_ids!=f, fold_ids==f
    m = Ridge(alpha=100.0); m.fit(Xts[tr_i], y_clip[tr_i])
    oof5[va_i] = m.predict(Xts[va_i]); pred5 += m.predict(Xes)/N_FOLDS
del Xts, Xes; gc.collect()
print(f"Ridge R2={r2_score(y, oof5):.6f} [{time.time()-t0:.0f}s]")

print("\n--- Optimal ensemble weights ---")
oofs = [oof1, oof2, oof3, oof4, oof5]
preds = [pred1, pred2, pred3, pred4, pred5]
names = ['LGB_RMSE','LGB_Huber','CB_RMSE','CB_Huber','Ridge']
best_r2 = -999; best_w = None; cnt = 0
wgrid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]
for w0 in wgrid:
    for w1 in wgrid:
        if w0+w1 > 1.01: break
        for w2 in wgrid:
            if w0+w1+w2 > 1.01: break
            for w3 in wgrid:
                if w0+w1+w2+w3 > 1.01: break
                w4 = round(1.0-w0-w1-w2-w3, 2)
                if w4 < -0.001: continue
                w4 = max(0, w4)
                blend = w0*oofs[0]+w1*oofs[1]+w2*oofs[2]+w3*oofs[3]+w4*oofs[4]
                sc2 = r2_score(y, blend); cnt += 1
                if sc2 > best_r2: best_r2 = sc2; best_w = (w0,w1,w2,w3,w4)
print(f"Searched {cnt} combos")
print(f"Best weights: {dict(zip(names, best_w))}")
print(f"Best OOF R2: {best_r2:.6f}")

final_pred = sum(w*p for w,p in zip(best_w, preds))
final_pred = final_pred * SHRINKAGE
print(f"\nShrinkage={SHRINKAGE}, pred mean={final_pred.mean():.8f}, std={final_pred.std():.8f}")

sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: final_pred})
sub.to_csv('submission.csv', index=False)
print(f"Submission: {sub.shape}")
print(f"TOTAL RUNTIME: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")
print("=== DONE ===")
