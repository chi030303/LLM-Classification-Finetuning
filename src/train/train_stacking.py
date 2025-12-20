import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# --- 配置 ---
FEATURE_PATH = "data/processed/train_features_structured.parquet"
OOF_PATHS = {
    "deberta": "data/processed/oof_deberta_v3_large.csv",
    # "qwen": "data/processed/oof_qwen_14b.csv" # 如果有Qwen就解开注释
}
OUTPUT_DIR = "outputs/stacking_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. 加载并合并数据 ---
print("🔄 Loading Data...")
# 加载手工特征
df = pd.read_parquet(FEATURE_PATH)

# 检查并删除 cluster_id
if 'cluster_id' in df.columns:
    print(f"⚠️ Found redundant column 'cluster_id'. Removing...")
    df = df.drop(columns=['cluster_id'])
    
    # 覆盖保存
    df.to_parquet(FEATURE_PATH, index=False)
    print(f"✅ Saved cleaned dataframe to {FEATURE_PATH}")
    print(f"Current columns: {[c for c in df.columns if 'cluster' in c]}")
else:
    print("✅ 'cluster_id' not found. Data is already clean.")

# 加载 LLM OOF 并合并
for model_name, path in OOF_PATHS.items():
    if os.path.exists(path):
        print(f"   -> Merging {model_name} OOF...")
        oof = pd.read_csv(path)
        # 假设行顺序一致（通常是一致的），直接赋值
        df[f'{model_name}_a'] = oof['pred_a']
        df[f'{model_name}_b'] = oof['pred_b']
        df[f'{model_name}_tie'] = oof['pred_tie']
    else:
        print(f"⚠️ Warning: {path} not found, skipping...")

# 准备 Target
df['target'] = df[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1).map({
    'winner_model_a': 0, 'winner_model_b': 1, 'winner_tie': 2
})

# 准备特征列表 (排除 ID, Label 等非特征列)
drop_cols = ['id', 'winner_model_a', 'winner_model_b', 'winner_tie', 'fold', 'target', 
             'prompt_text', 'res_a_text', 'res_b_text'] # 确保排除文本列
features = [c for c in df.columns if c not in drop_cols]
print(f"✅ Training on {len(features)} features.")

# --- 2. 训练 LightGBM ---
print("\n🚀 Starting LightGBM Training...")
lgb_params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'verbosity': -1,
    'seed': 42,
    'learning_rate': 0.0169, 
    'num_leaves': 32, 
    'max_depth': 9, 
    'feature_fraction': 0.67, 
    'bagging_fraction': 0.88, 
    'bagging_freq': 6, 
    'min_child_samples': 16, 
    'lambda_l1': 0.0001, 
    'lambda_l2': 0.006
}

# 转换 Categorical 特征 (Cluster ID)
cat_feats = [c for c in features if 'cluster' in c or 'is_code' in c]
for c in cat_feats:
    df[c] = df[c].astype('category')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lgb_oof = np.zeros((len(df), 3))

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['target'])):
    X_train, y_train = df.iloc[train_idx][features], df.iloc[train_idx]['target']
    X_val, y_val = df.iloc[val_idx][features], df.iloc[val_idx]['target']
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)
    
    model = lgb.train(
        lgb_params, dtrain, 
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)] # 0 表示不打印刷屏
    )
    
    lgb_oof[val_idx] = model.predict(X_val)
    # 保存模型
    model.save_model(f"{OUTPUT_DIR}/lgbm_fold{fold}.txt")
    print(f"   LGBM Fold {fold} LogLoss: {log_loss(y_val, lgb_oof[val_idx]):.5f}")

print(f"🏆 Final LightGBM CV Score: {log_loss(df['target'], lgb_oof):.5f}")

# --- 3. 训练 XGBoost ---
print("\n🚀 Starting XGBoost Training...")
# XGBoost 需要将 category 转回 int 或 enable_categorical
for c in cat_feats:
    df[c] = df[c].astype('int')
df = df.replace([np.inf, -np.inf], np.nan)

xgb_params = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'n_jobs': -1,
    'seed': 42,
    'enable_categorical': True,
    
    # --- Optuna Best Params ---
    'learning_rate': 0.015,
    'max_depth': 6,
    'subsample': 0.625,
    'colsample_bytree': 0.867,
    'min_child_weight': 4,
    'lambda': 9e-8,
    'alpha': 2e-6,
    # 增加一点 boost round 因为学习率变低了
    'n_estimators': 2000 
}

xgb_oof = np.zeros((len(df), 3))

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['target'])):
    X_train, y_train = df.iloc[train_idx][features], df.iloc[train_idx]['target']
    X_val, y_val = df.iloc[val_idx][features], df.iloc[val_idx]['target']
    
    # 再次确保切分后的数据也没有 inf (双重保险)
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)

    # dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True) 
    # 注意：如果 enable_categorical=True 报错，可以尝试去掉它，
    # 因为前面如果你没把 category 转 int，包含 NaN 的 category 可能会有问题
    # 建议保持 enable_categorical=True，XGBoost 新版支持 NaN category
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    
    model = xgb.train(
        xgb_params, dtrain, 
        num_boost_round=1000,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    xgb_oof[val_idx] = model.predict(dval)
    # 保存模型
    model.save_model(f"{OUTPUT_DIR}/xgb_fold{fold}.json")
    print(f"   XGB Fold {fold} LogLoss: {log_loss(y_val, xgb_oof[val_idx]):.5f}")

print(f"🏆 Final XGBoost CV Score: {log_loss(df['target'], xgb_oof):.5f}")

from scipy.optimize import minimize

print("\n⚖️ Finding Best Ensemble Weights...")

# 准备用于融合的预测结果列表
# 注意：这里要确保顺序！
# 1. LightGBM OOF
# 2. XGBoost OOF
# 3. DeBERTa OOF (作为正则项加入)
deb_keys = [k for k in OOF_PATHS.keys() if 'deberta' in k]

if deb_keys:
    target_key = deb_keys[0] # 取第一个匹配的 key
    
    # 2. 构造列名列表
    target_cols = [f'{target_key}_a', f'{target_key}_b', f'{target_key}_tie']
    
    # 3. 从 DataFrame 中提取数据
    deberta_oof_probs = df[target_cols].values
    
    print(f"   -> Extracted DeBERTa probs from columns: {target_cols}")
else:
    raise ValueError("❌ Could not find 'deberta' key in OOF_PATHS configuration!")

predictions_list = [lgb_oof, xgb_oof, deberta_oof_probs]
model_names = ["LightGBM", "XGBoost", "DeBERTa"]

# 如果你有 Qwen，记得加进去：
# qwen_cols = [c for c in df.columns if 'qwen' in c and c.endswith(('_a', '_b', '_tie'))]
# if qwen_cols:
#     qwen_oof_probs = df[qwen_cols].values
#     predictions_list.append(qwen_oof_probs)
#     model_names.append("Qwen")

def log_loss_func(weights):
    final_preds = np.zeros_like(predictions_list[0])
    # 归一化权重
    weights = np.array(weights) / np.sum(weights)
    
    for i, pred in enumerate(predictions_list):
        final_preds += weights[i] * pred
        
    return log_loss(df['target'], final_preds)

# 初始权重均分
init_w = [1/len(predictions_list)] * len(predictions_list)
# 约束：范围 0-1，和为 1
bounds = [(0, 1)] * len(predictions_list)
constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})

res = minimize(log_loss_func, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
best_w = res.x / np.sum(res.x)

print("-" * 30)
print(f"🔥 Best Ensemble CV Score: {res.fun:.5f}")
print("Weights:")
for name, w in zip(model_names, best_w):
    print(f"  {name}: {w:.4f}")
print("-" * 30)