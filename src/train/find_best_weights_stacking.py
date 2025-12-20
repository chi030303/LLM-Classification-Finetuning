import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from scipy.optimize import minimize

# 1. 加载所有模型的 OOF 预测
# 这些是你 Stage 2 训练完生成的，或者是 Stage 1 的 OOF
print("Loading OOF predictions...")

# DeBERTa (Stage 1)
oof_deb = pd.read_csv("data/processed/oof_deberta_v3_large.csv")
p_deb = oof_deb[['pred_a', 'pred_b', 'pred_tie']].values

# LightGBM (Stage 2) - 假设你保存了
# 如果还没保存，请去 train_stacking.py 把 lgb_oof 存下来
oof_lgb = pd.read_csv("outputs/stacking_models/oof_lgbm.csv") 
p_lgb = oof_lgb[['pred_a', 'pred_b', 'pred_tie']].values

# XGBoost (Stage 2)
oof_xgb = pd.read_csv("outputs/stacking_models/oof_xgboost.csv")
p_xgb = oof_xgb[['pred_a', 'pred_b', 'pred_tie']].values

# 真实标签
y_true = oof_deb[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1).map({
    'winner_model_a': 0, 'winner_model_b': 1, 'winner_tie': 2
}).values

# 2. 定义目标函数
# weights = [w_deb, w_lgb, w_xgb]
def log_loss_func(weights):
    # 归一化权重，确保和为1
    final_weights = weights / np.sum(weights)
    
    # 加权平均
    p_final = (final_weights[0] * p_deb + 
               final_weights[1] * p_lgb + 
               final_weights[2] * p_xgb)
    
    # 稍微截断防止 log(0)
    p_final = np.clip(p_final, 1e-15, 1-1e-15)
    
    return log_loss(y_true, p_final)

# 3. 求解最佳权重
print("🔍 Optimizing Ensemble Weights...")
# 初始权重 [0.33, 0.33, 0.33]
init_guess = [1/3, 1/3, 1/3] 
# 约束：权重在 0-1 之间
bounds = [(0, 1), (0, 1), (0, 1)]
# 约束：权重之和为 1
constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})

res = minimize(
    log_loss_func, 
    init_guess, 
    method='SLSQP', 
    bounds=bounds, 
    constraints=constraints
)

best_weights = res.x / np.sum(res.x)
print("-" * 30)
print(f"🏆 Optimization Success: {res.success}")
print(f"📉 Best Ensemble LogLoss: {res.fun:.5f}")
print("-" * 30)
print(f"Weights Distribution:")
print(f"  DeBERTa: {best_weights[0]:.4f}")
print(f"  LightGBM: {best_weights[1]:.4f}")
print(f"  XGBoost:  {best_weights[2]:.4f}")
print("-" * 30)

# 4. 单独对比
print("Individual Scores:")
print(f"  DeBERTa Only: {log_loss(y_true, p_deb):.5f}")
print(f"  LightGBM Only: {log_loss(y_true, p_lgb):.5f}")
print(f"  XGBoost Only:  {log_loss(y_true, p_xgb):.5f}")