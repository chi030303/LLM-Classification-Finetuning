import pandas as pd
import numpy as np
import os

# --- 配置 ---
FEATURE_PATH = "data/processed/train_features_structured.parquet" # 或者 .csv

def inspect_structured_features(file_path):
    print("="*80)
    print(f"📊 Feature Overview for: {os.path.basename(file_path)}")
    print("="*80)
    
    try:
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        return

    # --- 定义特征分组 ---
    feature_groups = {
        "Length & Ratio": ['len_p', 'len_a', 'len_b', 'len_diff', 'len_ratio', 'expansion_a', 'expansion_b', 'expansion_diff'],
        "Formatting": [c for c in df.columns if 'has_' in c and c not in ['has_emoji_a', 'has_emoji_b', 'has_disclaimer_a', 'has_disclaimer_b']],
        "Style & Semantic": [
            'cosine_sim', 'sentiment_gap', 'read_diff',
            'has_emoji_a', 'has_emoji_b', 'is_polite_a', 'is_polite_b',
            'has_disclaimer_a', 'has_disclaimer_b', 'is_refusal_a', 'is_refusal_b'
        ],
        "Topic & Task": [c for c in df.columns if 'cluster_id' in c or 'is_code_task' in c],
        "Interaction": [c for c in df.columns if '_x_' in c],
    }

    # 打印每个组的统计信息
    for group_name, cols in feature_groups.items():
        # 筛选出 DataFrame 中实际存在的列
        existing_cols = [c for c in cols if c in df.columns]
        if not existing_cols:
            continue
            
        print(f"\n--- {group_name} Features ---")
        
        # 为了简洁，只展示前5列的统计
        subset_df = df[existing_cols]
        print(subset_df.describe().round(2).T) # .T 转置后更好看
        
        # 打印前几行示例
        print("\n   Sample values:")
        print(subset_df.head(3))
        
    print("\n" + "="*80)
    print("✅ Overview generation complete.")

if __name__ == "__main__":
    inspect_structured_features(FEATURE_PATH)