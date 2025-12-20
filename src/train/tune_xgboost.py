import optuna
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

# --- 配置 ---
FEATURE_PATH = "data/processed/train_features_structured.parquet"
OOF_PATHS = {"deberta": "data/processed/oof_deberta_v3_large.csv"}

# --- 加载数据 (同之前) ---
print("🔄 Loading Data...")
df = pd.read_parquet(FEATURE_PATH)
for model_name, path in OOF_PATHS.items():
    oof = pd.read_csv(path)
    df[f'{model_name}_a'] = oof['pred_a']
    df[f'{model_name}_b'] = oof['pred_b']
    df[f'{model_name}_tie'] = oof['pred_tie']

df = df.replace([np.inf, -np.inf], np.nan)

df['target'] = df[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1).map({
    'winner_model_a': 0, 'winner_model_b': 1, 'winner_tie': 2
})

drop_cols = ['id', 'winner_model_a', 'winner_model_b', 'winner_tie', 'fold', 'target', 'prompt_text', 'res_a_text', 'res_b_text']
features = [c for c in df.columns if c not in drop_cols]
X = df[features]
y = df['target']

def objective(trial):
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'verbosity': 0,
        'tree_method': 'hist', # 加速
        # 调参空间
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True), # L2
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),   # L1
        'seed': 42
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # XGBoost DMatrix (enable_categorical=True 如果你有category列)
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
        
        model = xgb.train(
            params, dtrain, 
            num_boost_round=1000,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        preds = model.predict(dval)
        scores.append(log_loss(y_val, preds))
        
    return np.mean(scores)

print("🚀 Starting XGBoost Tuning...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print('🏆 Best hyperparameters:', study.best_params)
print('📉 Best LogLoss:', study.best_value)