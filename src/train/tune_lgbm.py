import optuna
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import os
import numpy as np
import pandas as pd

# --- 配置 ---
FEATURE_PATH = "data/processed/train_features_structured.parquet"
OOF_PATHS = {
    "deberta": "data/processed/oof_deberta_v3_large.csv",
}
OUTPUT_DIR = "outputs/stacking_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. 加载并合并数据 ---
print("🔄 Loading Data...")
df = pd.read_parquet(FEATURE_PATH)

# 清理冗余列
if 'cluster_id' in df.columns:
    df = df.drop(columns=['cluster_id'])

# 加载 LLM OOF
for model_name, path in OOF_PATHS.items():
    if os.path.exists(path):
        print(f"   -> Merging {model_name} OOF...")
        oof = pd.read_csv(path)
        df[f'{model_name}_a'] = oof['pred_a']
        df[f'{model_name}_b'] = oof['pred_b']
        df[f'{model_name}_tie'] = oof['pred_tie']

# 准备 Target
df['target'] = df[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1).map({
    'winner_model_a': 0, 'winner_model_b': 1, 'winner_tie': 2
})

# 准备特征列表
drop_cols = ['id', 'winner_model_a', 'winner_model_b', 'winner_tie', 'fold', 'target', 
             'prompt_text', 'res_a_text', 'res_b_text']
features = [c for c in df.columns if c not in drop_cols]
print(f"✅ Training on {len(features)} features.")

# 处理 categorical 特征
cat_feats = [c for c in features if 'cluster' in c or 'is_code' in c]
for c in cat_feats:
    df[c] = df[c].astype('category')

X = df[features]
y = df['target']

def objective(trial):
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'seed': 42
    }
    
    # 使用 Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    # 这里的 X 和 y 现在可以被正确访问了
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(
            params, dtrain, 
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)] # 静默模式
        )
        
        preds = model.predict(X_val)
        scores.append(log_loss(y_val, preds))
        
    return np.mean(scores)

print("🚀 Starting Optuna Tuning...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50) # 跑50次搜索

print('🏆 Best hyperparameters:', study.best_params)
print('📉 Best LogLoss:', study.best_value)