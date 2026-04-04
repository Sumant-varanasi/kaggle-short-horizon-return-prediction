import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import gc, warnings, os
warnings.filterwarnings('ignore')
print('Loading data...')
DATA = '/kaggle/input/competitions/short-horizon-return-prediction-challenge-by-i-rage/'
train = pd.read_parquet(DATA + 'train.parquet')
test = pd.read_parquet(DATA + 'test.parquet')
TARGET_COL = 'TARGET'
GROUP_COL = 'CV_GROUP'
ID_COL = 'ID'
y = train[TARGET_COL].values
groups = train[GROUP_COL].values
exclude_cols = [TARGET_COL, GROUP_COL, ID_COL]
feature_cols = [c for c in train.columns if c not in exclude_cols]
y_clip_val = float(np.percentile(np.abs(y), 99))
y_clipped = np.clip(y, -y_clip_val, y_clip_val)
X_train = train[feature_cols]
X_test = test[feature_cols]
test_ids = test[ID_COL].values
del train, test
gc.collect()
p1 = dict(objective='regression',metric='rmse',boosting_type='gbdt',learning_rate=0.03,num_leaves=63,max_depth=7,min_child_samples=100,subsample=0.7,colsample_bytree=0.5,reg_alpha=0.1,reg_lambda=1.0,n_estimators=800,verbose=-1,n_jobs=-1,random_state=42)
p2 = dict(objective='mae',metric='mae',boosting_type='gbdt',learning_rate=0.05,num_leaves=31,max_depth=6,min_child_samples=150,subsample=0.6,colsample_bytree=0.4,reg_alpha=0.5,reg_lambda=2.0,n_estimators=600,verbose=-1,n_jobs=-1,random_state=123)
p3 = dict(objective='huber',metric='rmse',boosting_type='gbdt',learning_rate=0.04,num_leaves=47,max_depth=7,min_child_samples=120,subsample=0.65,colsample_bytree=0.45,reg_alpha=0.3,reg_lambda=1.5,n_estimators=700,verbose=-1,n_jobs=-1,random_state=77)
def tm(par, nm, Xtr, ytr, grp, Xte, ns=5):
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=ns)
    oof = np.zeros(len(ytr))
    tp = np.zeros(len(Xte))
    sc = []
    for i,(tri,vi) in enumerate(gkf.split(Xtr, ytr, grp)):
        Xt,yt = Xtr.iloc[tri], ytr[tri]
        Xv,yv = Xtr.iloc[vi], ytr[vi]
        m = lgb.LGBMRegressor(**par)
        m.fit(Xt, yt, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(0)])
        oof[vi] = m.predict(Xv)
        tp += m.predict(Xte) / ns
        r = r2_score(yv, oof[vi])
        sc.append(r)
        print(nm, 'fold', i+1, 'R2:', round(r, 6), 'iter:', m.best_iteration_)
        del m, Xt, yt, Xv, yv
        gc.collect()
    print(nm, 'OOF R2:', round(r2_score(ytr, oof), 6))
    return oof, tp
print('Training M1 RMSE')
o1,t1 = tm(p1,'M1',X_train,y_clipped,groups,X_test)
print('Training M2 MAE')
o2,t2 = tm(p2,'M2',X_train,y_clipped,groups,X_test)
print('Training M3 Huber')
o3,t3 = tm(p3,'M3',X_train,y_clipped,groups,X_test)
print('Optimizing weights')
br,bw = -999,(1/3,1/3,1/3)
for w1 in np.arange(0.1,0.8,0.1):
    for w2 in np.arange(0.1,0.9-w1,0.1):
        w3 = round(1.0-w1-w2,2)
        if w3 < 0.05: continue
        r = r2_score(y_clipped, w1*o1+w2*o2+w3*o3)
        if r > br: br,bw = r,(w1,w2,w3)
w1,w2,w3 = bw
print('Best weights:', w1, w2, w3, 'OOF R2:', round(br,6))
ft = w1*t1 + w2*t2 + w3*t3
pc = float(np.percentile(np.abs(ft), 99.5))
ft = np.clip(ft, -pc, pc)
import pandas as pd
sub = pd.DataFrame({'ID': test_ids, 'TARGET': ft})
sub.to_csv('submission.csv', index=False)
print('Done. submission.csv shape:', sub.shape)
