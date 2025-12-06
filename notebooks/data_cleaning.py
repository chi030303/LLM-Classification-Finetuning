import pandas as pd
import numpy as np

def clean_features_final(df):
    # 1. 删除重复列
    if 'cluster_id' in df.columns:
        df = df.drop(columns=['cluster_id'])
    
    # 2. 转换 Cluster 列为 Category 类型 (给 LightGBM 用)
    cluster_cols = [c for c in df.columns if c.startswith('cluster_id_k')]
    for c in cluster_cols:
        df[c] = df[c].astype('category')
        
    # 3. 处理无限值 (Infinity)
    # 将 inf 替换为该列的最大值，-inf 替换为最小值
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan) # 先转 NaN
        df[c] = df[c].fillna(0) # 再填 0 (或者均值)
        
    print(f"✅ Features Cleaned. Retained clusters: {cluster_cols}")
    return df

# 加载 -> 清洗 -> 覆盖保存
df = pd.read_parquet("data/processed/train_features_structured.parquet")
df = clean_features_final(df)
df.to_parquet("data/processed/train_features_structured.parquet", index=False)

df = pd.read_csv("data/processed/train_features_structured.csv")
df = clean_features_final(df)
df.to_csv("data/processed/train_features_structured.csv", index=False)

df = pd.read_csv("data/processed/train_text_finetuning.csv")
df = clean_features_final(df)
df.to_csv("data/processed/train_text_finetuning.csv", index=False)