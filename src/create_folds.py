# src/create_folds.py
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# 1. 读取含有 cluster_id 的文本数据
# 注意：这里读取的是你之前保存的 csv
df = pd.read_csv("data/processed/train_text_finetuning.csv")


# 2. 创建 Stratified K-Fold
# 我们希望每个 Fold 里，Cluster（题目类型）和 Winner（胜负分布）都保持平衡
# 简单的做法是把 cluster_id 和 winner 组合成一个 stratify_label
df['stratify_group'] = df['cluster_id_k20'].astype(str) + "_" + df[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df['fold'] = -1

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['stratify_group'])):
    df.loc[val_idx, 'fold'] = fold

# 3. 保存
df.to_csv("data/processed/train_with_folds.csv", index=False)
print("✅ Folds created. Saved to output/train_with_folds.csv")
print(df['fold'].value_counts())