# V22: R2-optimized model with scipy ensemble optimization
# Score: -0.00085 (overfit - too aggressive)
# Key: 4 LGB configs + 3 Ridge, scipy-optimized ensemble weights
# Lesson: Aggressive models overfit on this low-SNR data
import numpy as np, pandas as pd, time, os, warnings, gc, lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
warnings.filterwarnings('ignore')
# Full code on Kaggle V22 - notebook_r2_optimized.py
print('V22: R2-optimized 4xLGB + 3xRidge ensemble with scipy weight optimization')
