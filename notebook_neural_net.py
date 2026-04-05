
import numpy as np, pandas as pd, os, glob, time, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from itertools import product as iprod

# ── Data loading (auto-detect parquet/csv) ────────────────
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

#  Columns ───────────────────────────
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

# Replace inf/nan
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

#  Feature selection (stability) ─────
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

#  Standardize for NN ──────
mu = np.mean(X_sel, axis=0)
sigma = np.std(X_sel, axis=0) + 1e-8
X_nn = (X_sel - mu) / sigma
X_test_nn = (X_test_sel - mu) / sigma

#  PyTorch Neural Network ──
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cpu')
print(f'PyTorch device: {device}')

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.act(x + self.net(x)))

class DeepNet(nn.Module):
    def __init__(self, in_dim, hidden=256, n_blocks=3, dropout=0.3):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        x = self.input_layer(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x).squeeze(-1)

def train_nn_fold(X_tr, y_tr, X_va, y_va, seed=42, epochs=80, batch_size=4096,
                  lr=1e-3, wd=1e-3, hidden=256, n_blocks=3, dropout=0.3, patience=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds_tr = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)
    model = DeepNet(X_tr.shape[1], hidden, n_blocks, dropout).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    loss_fn = nn.SmoothL1Loss(beta=0.01)
    Xv_t = torch.tensor(X_va).to(device)
    best_val = 1e18
    best_state = None
    wait = 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xv_t).cpu().numpy()
        val_mse = np.mean((vp - y_va)**2)
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(Xv_t).cpu().numpy()
    return model, val_pred

#  Train NN with 5-fold CV ─
t_nn = time.time()
oof_nn = np.zeros(len(y), dtype=np.float32)
nn_models = []

# Train 3 seeds per fold for stability, then average
NN_SEEDS = [42, 123, 777]

for f in range(N_FOLDS):
    tr_mask = folds_idx != f
    va_mask = folds_idx == f
    fold_preds = []
    fold_models = []
    for seed in NN_SEEDS:
        model, vp = train_nn_fold(
            X_nn[tr_mask], y[tr_mask], X_nn[va_mask], y[va_mask],
            seed=seed, epochs=60, batch_size=4096, lr=5e-4, wd=5e-3,
            hidden=192, n_blocks=2, dropout=0.35, patience=8
        )
        fold_preds.append(vp)
        fold_models.append(model)
    oof_nn[va_mask] = np.mean(fold_preds, axis=0)
    nn_models.append(fold_models)
    r2_f = r2_score(y[va_mask], oof_nn[va_mask])
    print(f'  NN fold {f}: R2={r2_f:.6f}')

nn_r2 = r2_score(y, oof_nn)
print(f'NN OOF R2={nn_r2:.6f} ({time.time()-t_nn:.0f}s)')

#  LightGBM models ──
import lightgbm as lgb

lgb_base = dict(n_estimators=600, learning_rate=0.03, max_depth=6, num_leaves=40,
                subsample=0.7, colsample_bytree=0.5, reg_alpha=0.1, reg_lambda=1.0,
                min_child_samples=100, n_jobs=-1, verbose=-1, feature_pre_filter=False)
lgb_cfgs = [
    ('LGB_RMSE',  {**lgb_base, 'objective':'regression',       'metric':'rmse'}),
    ('LGB_Huber', {**lgb_base, 'objective':'huber', 'huber_delta':0.01, 'metric':'rmse'}),
]

oof_dict = {}
test_dict = {}
model_store = {}

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

#  CatBoost model ────
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

#  Ridge baseline 
oof_ridge = np.zeros(len(y))
test_ridge = np.zeros(len(X_test_sel))
for f in range(N_FOLDS):
    tr = folds_idx != f; va = folds_idx == f
    rr = Ridge(alpha=100.0)
    rr.fit(X_sel[tr], y[tr])
    oof_ridge[va] = rr.predict(X_sel[va])
    test_ridge += rr.predict(X_test_sel) / N_FOLDS
oof_dict['Ridge'] = oof_ridge
test_dict['Ridge'] = test_ridge
print(f'Ridge: R2={r2_score(y, oof_ridge):.6f}')

#  Add NN to dict ───
oof_dict['DeepNet'] = oof_nn.astype(np.float64)
# NN test predictions
Xt_t = torch.tensor(X_test_nn).to(device)
test_nn = np.zeros(len(X_test_sel), dtype=np.float64)
for fold_models in nn_models:
    for model in fold_models:
        model.eval()
        with torch.no_grad():
            test_nn += model(Xt_t).cpu().numpy()
test_nn /= (N_FOLDS * len(NN_SEEDS))
test_dict['DeepNet'] = test_nn

#  Ensemble optimization 
names = list(oof_dict.keys())
oof_arr = np.column_stack([oof_dict[n] for n in names])
test_arr = np.column_stack([test_dict[n] for n in names])
print(f'\nOptimising ensemble over {names}')

# Filter to models with positive R2
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
    from itertools import product as iprod
    for combo in iprod(wt_grid, repeat=n_pos):
        s = sum(combo)
        if s < 0.05:
            continue
        w = np.array(combo) / s
        blend = oof_pos @ w
        r = r2_score(y, blend)
        if r > best_r2:
            best_r2 = r
            best_w = w.copy()

print(f'Best ensemble R2={best_r2:.6f}')
for n, w in zip(pos_names, best_w):
    print(f'  {n}: {w:.2f}')

oof_blend = oof_pos @ best_w
test_blend = test_pos @ best_w

#  Shrinkage ────────────
best_shrink = 0.0
best_sr2 = -999
for s in np.arange(0.05, 1.01, 0.05):
    r = r2_score(y, oof_blend * s)
    if r > best_sr2:
        best_sr2 = r
        best_shrink = s

# Fine grid
for s in np.arange(max(0.01, best_shrink-0.05), best_shrink+0.06, 0.01):
    r = r2_score(y, oof_blend * s)
    if r > best_sr2:
        best_sr2 = r
        best_shrink = s

print(f'Shrinkage={best_shrink:.2f}, Final OOF R2={best_sr2:.6f}')
preds = test_blend * best_shrink

#  Submission 
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
