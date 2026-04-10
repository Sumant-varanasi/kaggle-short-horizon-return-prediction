# V24: Quant Signals Model - Score: -0.00014
# Unique: Constructs proper quant features from lag-difference data
# pct_change = LagT1 / (current - LagT1) -> matches % return target space
# acceleration = LagT1 - LagT2 -> momentum signal
# mean_reversion = sign(LagT1) * sign(LagT2) -> +1 momentum, -1 reversal
# Creates 444 new quant features from 111 base features
# LGB_all had R2=+0.000129 but model chose Ridge (wrong pick)
import numpy as np, pandas as pd, time, os, warnings, gc, lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
# Full code on Kaggle V24 - notebook_quant_signals.py
print('V24: Quant signals - pct_change, momentum, mean_reversion features')
