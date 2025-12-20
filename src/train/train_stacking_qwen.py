import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import os
import argparse
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# --- 1. 参数配置 ---
parser = argparse.ArgumentParser()
parser.add_argument("--use_deberta", action="store_true", help="Include DeBERTa OOF as features")
parser.add_argument("--use_qwen", action="store_true", help="Include Qwen OOF as features")
parser.add_argument("--use_manual_feats", action="store_true", help="Include manual engineered features")
parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
args = parser.parse_args()

# --- 2. 文件路径 ---
FEATURE_PATH = "data/processed/train_features_structured.parquet"
DEBERTA_OOF_PATH = "data/processed/oof_deberta_v3_large.csv"
QWEN_OOF_PATH = "data/processed/oof_qwen_14b.csv"
OUTPUT_DIR = "outputs/stacking_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 3. 数据加载与特征拼接 ---
def load_data():
    print("🔄 Loading and preparing data...")
    
    # [A] 基础数据 (手工特征 + Target)
    df = pd.read_parquet(FEATURE_PATH)
    
    df['target'] = df[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1).map({
        'winner_model_a': 0, 'winner_model_b': 1, 'winner_tie': 2
    })
    
    # [B] 手工特征
    # 删掉所有标签列，剩下的就是特征
    drop_cols = ['winner_model_a', 'winner_model_b', 'winner_tie', 'target']
    manual_feats = [c for c in df.columns if c not in drop_cols]
    
    # --- [关键修改] 使用 concat 代替 merge ---
    
    # [C] LLM OOF 特征
    llm_feats = []
    data_to_concat = [df] # 准备一个列表存放要拼接的 df

    if args.use_deberta:
        print("   -> Merging DeBERTa OOF...")
        oof_deb = pd.read_csv(DEBERTA_OOF_PATH)
        # 重命名
        oof_deb = oof_deb[['pred_a', 'pred_b', 'pred_tie']].rename(columns={
            'pred_a': 'deberta_a', 'pred_b': 'deberta_b', 'pred_tie': 'deberta_tie'
        })
        data_to_concat.append(oof_deb)
        llm_feats.extend(['deberta_a', 'deberta_b', 'deberta_tie'])

    if args.use_qwen:
        print("   -> Merging Qwen OOF...")
        oof_qwen = pd.read_csv(QWEN_OOF_PATH)
        oof_qwen = oof_qwen[['pred_a', 'pred_b', 'pred_tie']].rename(columns={
            'pred_a': 'qwen_a', 'pred_b': 'qwen_b', 'pred_tie': 'qwen_tie'
        })
        data_to_concat.append(oof_qwen)
        llm_feats.extend(['qwen_a', 'qwen_b', 'qwen_tie'])
    
    # 横向拼接所有 DataFrame
    df = pd.concat(data_to_concat, axis=1)

    # [D] 确定最终特征列表
    final_features = []
    if args.use_manual_feats:
        final_features.extend(manual_feats)
    if llm_feats:
        final_features.extend(llm_feats)
        
    if not final_features:
        raise ValueError("No features selected!")
        
    df = df.replace([np.inf, -np.inf], np.nan)
    
    print(f"✅ Training with {len(final_features)} features.")
    return df, final_features

# --- 4. 训练/调优逻辑 ---

def train_lgbm(params, X, y, features):
    """训练 LightGBM 并返回 CV 分数和模型"""
    print("\n🚀 Starting LightGBM Training...")
    
    # 处理类别特征
    cat_feats = [c for c in features if 'cluster' in c or 'is_code' in c]
    for c in cat_feats:
        X[c] = X[c].astype('category')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(X), 3))
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        dtrain = lgb.Dataset(X_train[features], label=y_train)
        dval = lgb.Dataset(X_val[features], label=y_val)
        
        model = lgb.train(
            params, dtrain, 
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        oof_preds[val_idx] = model.predict(X_val[features])
        models.append(model)
    
    score = log_loss(y, oof_preds)
    print(f"🏆 LightGBM CV Score: {score:.5f}")
    return models, score

def main():
    df, features = load_data()
    X = df[features]
    y = df['target']
    
    # 固定的最佳参数 (从之前的 Optuna 获得)
    best_lgbm_params = {
        'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
        'verbosity': -1, 'seed': 42, 'learning_rate': 0.0169, 
        'num_leaves': 32, 'max_depth': 9, 'feature_fraction': 0.67, 
        'bagging_fraction': 0.88, 'bagging_freq': 6, 
        'min_child_samples': 16, 'lambda_l1': 0.0001, 'lambda_l2': 0.006
    }
    
    models, _ = train_lgbm(best_lgbm_params, X, y, features)
    
    # 保存模型
    print("\n💾 Saving LGBM models...")
    exp_name = ""
    if args.use_deberta: exp_name += "deb_"
    if args.use_qwen: exp_name += "qwen_"
    if args.use_manual_feats: exp_name += "feats"
    
    for i, model in enumerate(models):
        model.save_model(f"{OUTPUT_DIR}/lgbm_{exp_name}_fold{i}.txt")
    print(f"   -> Models saved with prefix: lgbm_{exp_name}")

if __name__ == "__main__":
    main()