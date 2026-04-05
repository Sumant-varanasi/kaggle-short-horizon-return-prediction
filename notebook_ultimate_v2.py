
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
import time, warnings, gc, os
warnings.filterwarnings('ignore')

t0 = time.time()
print("=== ULTIMATE V2: LGB + CatBoost + XGBoost + MLP + Ridge ===")

DATA = '/kaggle/input/competitions/short-horizon-return-prediction-challenge-by-i-rage'
for base in [DATA,
             '/kaggle/input/short-horizon-return-prediction-challenge-by-i-rage']:
    for ext in ['.parquet', '.csv']:
        p = f'{base}/train{ext}'
        if os.path.exists(p):
            DATA = base
            USE_PARQUET = (ext == '.parquet')
            print(f"Data: {base} ({ext})")
            break
    else: continue
    break
else:
    for d in ['/kaggle/input', '/kaggle/input/competitions']:
        if os.path.exists(d): print(f"{d}: {os.listdir(d)}")
    raise FileNotFoundError("No data found")

TARGET_COL = 'TARGET'; CV_COL = 'CV_GROUP'; ID_COL = 'ID'
N_FOLDS = 5; TOP_K = 120; SHRINKAGE = 0.35

def load(name):
    if USE_PARQUET: return pd.read_parquet(f'{DATA}/{name}.parquet')
    return pd.read_csv(f'{DATA}/{name}.csv')

train = load('train'); test = load('test')
print(f"Train: {train.shape}, Test: {test.shape}")

feat_cols = [c for c in train.columns if c not in [TARGET_COL, CV_COL, ID_COL]]
y = train[TARGET_COL].values
clip_lo, clip_hi = np.percentile(y, 1), np.percentile(y, 99)
y_clip = np.clip(y, clip_lo, clip_hi)
print(f"Clipped [{clip_lo:.6f}, {clip_hi:.6f}]")

print(f"\nFeature selection from {len(feat_cols)} ...")
np.random.seed(42)
has_cv = CV_COL in train.columns
if has_cv:
    groups = train[CV_COL].unique()
    sel_g = np.random.choice(groups, size=min(10, len(groups)), replace=False)
    imp_sum = np.zeros(len(feat_cols))
    for g in sel_g:
        mask = train[CV_COL] == g
        ds = lgb.Dataset(train.loc[~mask, feat_cols], y_clip[~mask], free_raw_data=False)
        m = lgb.train({'objective':'regression','learning_rate':0.05,'num_leaves':31,
            'min_data_in_leaf':500,'feature_pre_filter':False,'verbosity':-1}, ds, 50)
        imp = m.feature_importance('gain')
        imp_sum += imp / (imp.sum()+1e-15)
else:
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    imp_sum = np.zeros(len(feat_cols))
    for tr_idx, _ in kf.split(train):
        ds = lgb.Dataset(train.iloc[tr_idx][feat_cols], y_clip[tr_idx], free_raw_data=False)
        m = lgb.train({'objective':'regression','learning_rate':0.05,'num_leaves':31,
            'min_data_in_leaf':500,'feature_pre_filter':False,'verbosity':-1}, ds, 50)
        imp = m.feature_importance('gain')
        imp_sum += imp / (imp.sum()+1e-15)
top_idx = np.argsort(imp_sum)[-TOP_K:]
sel_feats = [feat_cols[i] for i in top_idx]
print(f"Selected {len(sel_feats)} in {time.time()-t0:.0f}s")

X_tr = train[sel_feats].values; X_te = test[sel_feats].values

fold_ids = np.zeros(len(y), dtype=int)
if has_cv:
    uniq_cv = np.sort(train[CV_COL].unique())
    for i, g in enumerate(uniq_cv):
        fold_ids[train[CV_COL].values == g] = i % N_FOLDS
else:
    kf2 = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for i, (_, vi) in enumerate(kf2.split(X_tr)): fold_ids[vi] = i
print(f"Folds: {np.bincount(fold_ids)}")
del train; gc.collect()

# Scale for MLP + Ridge
scl = StandardScaler(); Xts = scl.fit_transform(X_tr); Xes = scl.transform(X_te)

# ========== MODEL 1: LightGBM RMSE ==========
print(f"\n--- M1: LGB RMSE --- [{time.time()-t0:.0f}s]")
p1 = {'objective':'regression','metric':'rmse','learning_rate':0.03,
    'num_leaves':63,'min_data_in_leaf':500,'feature_fraction':0.5,
    'bagging_fraction':0.7,'bagging_freq':1,'lambda_l1':1.0,'lambda_l2':5.0,
    'feature_pre_filter':False,'verbosity':-1,'seed':42}
oof1=np.zeros(len(y)); pr1=np.zeros(len(X_te))
for f in range(N_FOLDS):
    ti,vi = fold_ids!=f, fold_ids==f
    d1=lgb.Dataset(X_tr[ti],y_clip[ti],free_raw_data=False)
    d2=lgb.Dataset(X_tr[vi],y_clip[vi],reference=d1,free_raw_data=False)
    m=lgb.train(p1,d1,500,valid_sets=[d2],callbacks=[lgb.early_stopping(30),lgb.log_evaluation(500)])
    oof1[vi]=m.predict(X_tr[vi]); pr1+=m.predict(X_te)/N_FOLDS
    del m,d1,d2; gc.collect()
print(f"M1 R2={r2_score(y,oof1):.6f} [{time.time()-t0:.0f}s]")

# ========== MODEL 2: LightGBM Huber ==========
print(f"\n--- M2: LGB Huber --- [{time.time()-t0:.0f}s]")
p2 = {'objective':'huber','huber_delta':0.01,'metric':'mae',
    'learning_rate':0.03,'num_leaves':31,'min_data_in_leaf':800,
    'feature_fraction':0.4,'bagging_fraction':0.7,'bagging_freq':1,
    'lambda_l1':2.0,'lambda_l2':10.0,'feature_pre_filter':False,
    'verbosity':-1,'seed':123}
oof2=np.zeros(len(y)); pr2=np.zeros(len(X_te))
for f in range(N_FOLDS):
    ti,vi = fold_ids!=f, fold_ids==f
    d1=lgb.Dataset(X_tr[ti],y_clip[ti],free_raw_data=False)
    d2=lgb.Dataset(X_tr[vi],y_clip[vi],reference=d1,free_raw_data=False)
    m=lgb.train(p2,d1,500,valid_sets=[d2],callbacks=[lgb.early_stopping(30),lgb.log_evaluation(500)])
    oof2[vi]=m.predict(X_tr[vi]); pr2+=m.predict(X_te)/N_FOLDS
    del m,d1,d2; gc.collect()
print(f"M2 R2={r2_score(y,oof2):.6f} [{time.time()-t0:.0f}s]")

# ========== MODEL 3: CatBoost Plain RMSE ==========
print(f"\n--- M3: CB RMSE --- [{time.time()-t0:.0f}s]")
oof3=np.zeros(len(y)); pr3=np.zeros(len(X_te))
for f in range(N_FOLDS):
    ti,vi = fold_ids!=f, fold_ids==f
    m=CatBoostRegressor(loss_function='RMSE',iterations=400,learning_rate=0.05,
        depth=6,l2_leaf_reg=10,random_strength=2,bagging_temperature=0.5,
        boosting_type='Plain',random_seed=42,verbose=0,early_stopping_rounds=30)
    m.fit(X_tr[ti],y_clip[ti],eval_set=(X_tr[vi],y_clip[vi]),verbose=0)
    oof3[vi]=m.predict(X_tr[vi]); pr3+=m.predict(X_te)/N_FOLDS
    del m; gc.collect()
print(f"M3 R2={r2_score(y,oof3):.6f} [{time.time()-t0:.0f}s]")

# ========== MODEL 4: XGBoost RMSE ==========
print(f"\n--- M4: XGB RMSE --- [{time.time()-t0:.0f}s]")
xp = {'objective':'reg:squarederror','eval_metric':'rmse','learning_rate':0.03,
    'max_depth':6,'min_child_weight':500,'subsample':0.7,'colsample_bytree':0.5,
    'reg_alpha':1.0,'reg_lambda':5.0,'tree_method':'hist','seed':77,'verbosity':0}
oof4=np.zeros(len(y)); pr4=np.zeros(len(X_te))
for f in range(N_FOLDS):
    ti,vi = fold_ids!=f, fold_ids==f
    dtrain=xgb.DMatrix(X_tr[ti],label=y_clip[ti])
    dval=xgb.DMatrix(X_tr[vi],label=y_clip[vi])
    dtest=xgb.DMatrix(X_te)
    m=xgb.train(xp,dtrain,500,evals=[(dval,'val')],early_stopping_rounds=30,verbose_eval=0)
    oof4[vi]=m.predict(dval); pr4+=m.predict(dtest)/N_FOLDS
    del m,dtrain,dval,dtest; gc.collect()
print(f"M4 R2={r2_score(y,oof4):.6f} [{time.time()-t0:.0f}s]")

# ========== MODEL 5: XGBoost Huber ==========
print(f"\n--- M5: XGB Huber --- [{time.time()-t0:.0f}s]")
xp2 = {'objective':'reg:pseudohubererror','huber_slope':0.01,'eval_metric':'mae',
    'learning_rate':0.03,'max_depth':4,'min_child_weight':800,'subsample':0.7,
    'colsample_bytree':0.4,'reg_alpha':2.0,'reg_lambda':10.0,'tree_method':'hist',
    'seed':88,'verbosity':0}
oof5=np.zeros(len(y)); pr5=np.zeros(len(X_te))
for f in range(N_FOLDS):
    ti,vi = fold_ids!=f, fold_ids==f
    dtrain=xgb.DMatrix(X_tr[ti],label=y_clip[ti])
    dval=xgb.DMatrix(X_tr[vi],label=y_clip[vi])
    dtest=xgb.DMatrix(X_te)
    m=xgb.train(xp2,dtrain,500,evals=[(dval,'val')],early_stopping_rounds=30,verbose_eval=0)
    oof5[vi]=m.predict(dval); pr5+=m.predict(dtest)/N_FOLDS
    del m,dtrain,dval,dtest; gc.collect()
print(f"M5 R2={r2_score(y,oof5):.6f} [{time.time()-t0:.0f}s]")

# ========== MODEL 6: MLP Neural Network ==========
print(f"\n--- M6: MLP --- [{time.time()-t0:.0f}s]")
oof6=np.zeros(len(y)); pr6=np.zeros(len(X_te))
for f in range(N_FOLDS):
    ti,vi = fold_ids!=f, fold_ids==f
    m=MLPRegressor(hidden_layer_sizes=(128,64),activation='relu',solver='adam',
        alpha=0.01,batch_size=4096,learning_rate='adaptive',learning_rate_init=0.001,
        max_iter=100,early_stopping=True,n_iter_no_change=10,validation_fraction=0.1,
        random_state=42,verbose=False)
    m.fit(Xts[ti],y_clip[ti])
    oof6[vi]=m.predict(Xts[vi]); pr6+=m.predict(Xes)/N_FOLDS
    del m; gc.collect()
print(f"M6 R2={r2_score(y,oof6):.6f} [{time.time()-t0:.0f}s]")

# ========== MODEL 7: Ridge ==========
print(f"\n--- M7: Ridge --- [{time.time()-t0:.0f}s]")
oof7=np.zeros(len(y)); pr7=np.zeros(len(X_te))
for f in range(N_FOLDS):
    ti,vi = fold_ids!=f, fold_ids==f
    m=Ridge(alpha=100.0); m.fit(Xts[ti],y_clip[ti])
    oof7[vi]=m.predict(Xts[vi]); pr7+=m.predict(Xes)/N_FOLDS
print(f"M7 R2={r2_score(y,oof7):.6f} [{time.time()-t0:.0f}s]")
del Xts, Xes; gc.collect()

# ========== OPTIMAL ENSEMBLE ==========
print(f"\n--- Ensemble optimization --- [{time.time()-t0:.0f}s]")
oofs=[oof1,oof2,oof3,oof4,oof5,oof6,oof7]
preds=[pr1,pr2,pr3,pr4,pr5,pr6,pr7]
names=['LGB_R','LGB_H','CB_R','XGB_R','XGB_H','MLP','Ridge']

# Two-stage: first find top-3 models, then fine-tune weights
ind_r2 = [(r2_score(y,o),i) for i,o in enumerate(oofs)]
ind_r2.sort(reverse=True)
print("Individual R2:", [(names[i],f"{s:.6f}") for s,i in ind_r2])

# Use top-5 models for ensemble search
top5 = [i for _,i in ind_r2[:5]]
top_oofs = [oofs[i] for i in top5]
top_preds = [preds[i] for i in top5]
top_names = [names[i] for i in top5]
print(f"Top 5: {top_names}")

best_r2=-999; best_w=None; cnt=0
wg=[0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.8,1.0]
for w0 in wg:
    for w1 in wg:
        if w0+w1>1.01: break
        for w2 in wg:
            if w0+w1+w2>1.01: break
            for w3 in wg:
                if w0+w1+w2+w3>1.01: break
                w4=round(1.0-w0-w1-w2-w3,2)
                if w4<-0.001: continue
                w4=max(0,w4)
                bl=w0*top_oofs[0]+w1*top_oofs[1]+w2*top_oofs[2]+w3*top_oofs[3]+w4*top_oofs[4]
                s=r2_score(y,bl); cnt+=1
                if s>best_r2: best_r2=s; best_w=(w0,w1,w2,w3,w4)
print(f"Searched {cnt}, Best weights: {dict(zip(top_names,best_w))}")
print(f"Best OOF R2: {best_r2:.6f}")

fp=sum(w*p for w,p in zip(best_w,top_preds))

# Try multiple shrinkage values
for sh in [0.30, 0.35, 0.40]:
    pred_sh = fp * sh
    print(f"  Shrinkage={sh}: mean={pred_sh.mean():.8f}, std={pred_sh.std():.8f}")

fp = fp * SHRINKAGE
print(f"\nFinal shrinkage={SHRINKAGE}")

sub=pd.DataFrame({ID_COL:test[ID_COL],TARGET_COL:fp})
sub.to_csv('submission.csv',index=False)
print(f"Submission: {sub.shape}")
print(f"RUNTIME: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")
print("=== DONE ===")
