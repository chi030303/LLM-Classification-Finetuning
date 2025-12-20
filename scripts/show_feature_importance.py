import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# --- 配置 ---
MODEL_DIR = "outputs/stacking_models"

def show_importance():
    print("="*60)
    print("📈 Feature Importance Report")
    print("="*60)
    
    # --- 1. LightGBM Importance ---
    print("\n--- LightGBM Top 10 Features ---")
    lgb_files = glob.glob(os.path.join(MODEL_DIR, "lgbm_fold*.txt"))
    if not lgb_files:
        print("   -> No LightGBM models found.")
    else:
        all_importances = []
        for f in lgb_files:
            bst = lgb.Booster(model_file=f)
            # 记录特征名和重要性
            importance = pd.DataFrame({
                'feature': bst.feature_name(),
                'importance': bst.feature_importance(importance_type='gain'), # 'gain' 更能反映贡献度
            })
            all_importances.append(importance)
            
        # 计算 5 折的平均重要性
        avg_importance = pd.concat(all_importances).groupby('feature').mean().reset_index()
        avg_importance = avg_importance.sort_values('importance', ascending=False)
        
        # 打印 Top 10
        print(avg_importance.head(10).to_string(index=False))
        
        # 可视化
        plt.figure(figsize=(10, 6))
        sns.barplot(data=avg_importance.head(10), x='importance', y='feature')
        plt.title("LightGBM Feature Importance (Top 10 - Mean Gain)")
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "lgbm_importance.png"))
        print(f"   -> Plot saved to {MODEL_DIR}/lgbm_importance.png")
        plt.show()

    # --- 2. XGBoost Importance ---
    print("\n--- XGBoost Top 10 Features ---")
    xgb_files = glob.glob(os.path.join(MODEL_DIR, "xgb_fold*.json"))
    if not xgb_files:
        print("   -> No XGBoost models found.")
    else:
        all_importances_xgb = []
        for f in xgb_files:
            bst = xgb.Booster()
            bst.load_model(f)
            # 获取重要性
            importance = bst.get_score(importance_type='gain')
            importance_df = pd.DataFrame(list(importance.items()), columns=['feature', 'importance'])
            all_importances_xgb.append(importance_df)
            
        # 计算平均重要性
        avg_importance_xgb = pd.concat(all_importances_xgb).groupby('feature').mean().reset_index()
        avg_importance_xgb = avg_importance_xgb.sort_values('importance', ascending=False)
        
        # 打印 Top 10
        print(avg_importance_xgb.head(10).to_string(index=False))
        
        # 可视化
        plt.figure(figsize=(10, 6))
        sns.barplot(data=avg_importance_xgb.head(10), x='importance', y='feature')
        plt.title("XGBoost Feature Importance (Top 10 - Mean Gain)")
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "xgb_importance.png"))
        print(f"   -> Plot saved to {MODEL_DIR}/xgb_importance.png")
        plt.show()

if __name__ == "__main__":
    show_importance()