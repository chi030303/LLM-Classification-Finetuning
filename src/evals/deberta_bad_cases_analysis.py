import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

# 1. 读取 OOF
oof = pd.read_csv("data/processed/oof_deberta_v3_large.csv")
# 假设有 'prompt_text', 'res_a_text', 'res_b_text' 列，如果没有，需要 merge 原始数据
# train = pd.read_csv("data/processed/train_with_folds.csv")
# oof = oof.merge(train[['id', 'prompt_text', 'res_a_text', 'res_b_text', 'cluster_id_k20']], on='id')

# 2. 计算每个样本的 Log Loss
def calculate_sample_loss(row):
    # 真实标签 One-Hot
    target = np.array([row['winner_model_a'], row['winner_model_b'], row['winner_tie']])
    # 预测概率
    pred = np.array([row['pred_a'], row['pred_b'], row['pred_tie']])
    # 避免 log(0)
    pred = np.clip(pred, 1e-15, 1 - 1e-15)
    # Cross Entropy
    return -np.sum(target * np.log(pred))

oof['sample_loss'] = oof.apply(calculate_sample_loss, axis=1)

# 3. 排序：找出 Loss 最高的 50 个样本
bad_cases = oof.sort_values('sample_loss', ascending=False).head(50)

# 4. 分析
print("🔥 Top Bad Cases Clusters:")
print(bad_cases['cluster_id_k20'].value_counts())

# 5. 打印具体的文本看一眼
for idx, row in bad_cases.head(3).iterrows():
    print(f"\n=== Loss: {row['sample_loss']:.4f} | Cluster: {row['cluster_id_k20']} ===")
    print(f"Prompt: {row['prompt_text'][:200]}...")
    print(f"Winner: {'A' if row['winner_model_a'] else 'B' if row['winner_model_b'] else 'Tie'}")
    print(f"Pred: A={row['pred_a']:.2f}, B={row['pred_b']:.2f}, Tie={row['pred_tie']:.2f}")