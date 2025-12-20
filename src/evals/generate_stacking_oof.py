import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import os
from sklearn.model_selection import StratifiedKFold

# --- 配置 ---
FEATURE_PATH = "data/processed/train_features_structured.parquet"
DEBERTA_OOF = "data/processed/oof_deberta_v3_large.csv"
MODEL_DIR = "outputs/stacking_models"
OUTPUT_DIR = "outputs/stacking_models" # 结果保存到这里

# --- 1. 准备数据 (必须与训练时完全一致) ---
print("🔄 Loading Data...")
df = pd.read_parquet(FEATURE_PATH)

# 如果之前的清洗没删掉 cluster_id，这里确保删掉
if 'cluster_id' in df.columns:
    df = df.drop(columns=['cluster_id'])

# 合并 DeBERTa OOF
if os.path.exists(DEBERTA_OOF):
    oof = pd.read_csv(DEBERTA_OOF)
    df['deberta_a'] = oof['pred_a']
    df['deberta_b'] = oof['pred_b']
    df['deberta_tie'] = oof['pred_tie']
else:
    raise FileNotFoundError("DeBERTa OOF file missing!")

# 准备 Target (用于 KFold 切分)
df['target'] = df[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1).map({
    'winner_model_a': 0, 'winner_model_b': 1, 'winner_tie': 2
})

# 清洗 inf
df = df.replace([np.inf, -np.inf], np.nan)

# 特征列表
drop_cols = ['id', 'winner_model_a', 'winner_model_b', 'winner_tie', 'fold', 'target', 'prompt_text', 'res_a_text', 'res_b_text']
features = [c for c in df.columns if c not in drop_cols]

# 转换 Categorical (给 XGBoost/LGBM 用)
cat_feats = [c for c in features if 'cluster' in c or 'is_code' in c]
for c in cat_feats:
    df[c] = df[c].astype('category')

print(f"✅ Data Prepared. Features: {len(features)}")

# --- 2. 准备容器 ---
lgb_oof_preds = np.zeros((len(df), 3))
xgb_oof_preds = np.zeros((len(df), 3))

# --- 3. 循环 5 Folds 进行推理 ---
# 必须使用随机种子 42，确保和训练时的切分一模一样
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n🚀 Starting OOF Generation...")

for fold, (_, val_idx) in enumerate(skf.split(df, df['target'])):
    print(f"   -> Processing Fold {fold}...")
    
    # 获取验证集数据
    X_val = df.iloc[val_idx][features]
    
    # ---------------- LightGBM 推理 ----------------
    model_path = os.path.join(MODEL_DIR, f"lgbm_fold{fold}.txt")
    if os.path.exists(model_path):
        bst = lgb.Booster(model_file=model_path)
        
        # [关键] 对齐特征顺序
        lgb_feats = bst.feature_name()
        X_val_lgb = X_val[lgb_feats]
        
        # 传入 .values 避开 category 检查
        lgb_oof_preds[val_idx] = bst.predict(X_val_lgb.values)
    else:
        print(f"      ⚠️ LightGBM model not found for fold {fold}")

    # ---------------- XGBoost 推理 ----------------
    model_path = os.path.join(MODEL_DIR, f"xgb_fold{fold}.json")
    if os.path.exists(model_path):
        bst = xgb.Booster()
        bst.load_model(model_path)
        
        # [关键] 对齐特征顺序
        xgb_feats = bst.feature_names
        X_val_xgb = X_val[xgb_feats]
        
        # 构造 DMatrix (保留 category 类型)
        dval = xgb.DMatrix(X_val_xgb, enable_categorical=True)
        xgb_oof_preds[val_idx] = bst.predict(dval)
    else:
        print(f"      ⚠️ XGBoost model not found for fold {fold}")

# --- 4. 保存结果 ---
print("\n💾 Saving OOF files...")

# 保存 LightGBM OOF
lgb_df = pd.DataFrame(lgb_oof_preds, columns=['pred_a', 'pred_b', 'pred_tie'])
lgb_df['id'] = df['id'] if 'id' in df.columns else df.index # 最好有ID
lgb_df.to_csv(f"{OUTPUT_DIR}/oof_lgbm.csv", index=False)
print(f"   -> Saved {OUTPUT_DIR}/oof_lgbm.csv")

# 保存 XGBoost OOF
xgb_df = pd.DataFrame(xgb_oof_preds, columns=['pred_a', 'pred_b', 'pred_tie'])
xgb_df['id'] = df['id'] if 'id' in df.columns else df.index
xgb_df.to_csv(f"{OUTPUT_DIR}/oof_xgboost.csv", index=False)
print(f"   -> Saved {OUTPUT_DIR}/oof_xgboost.csv")

print("✅ All Done!")