import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

# 1. 读取 OOF 文件
oof_df = pd.read_csv("outputs/models/deberta_v3_large_fold0/oof_preds.csv")

# 2. 准备真实标签
# 将 One-Hot 标签转回 0, 1, 2
y_true = oof_df[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1).map({
    'winner_model_a': 0, 'winner_model_b': 1, 'winner_tie': 2
})

# 3. 准备预测概率
y_pred = oof_df[['pred_a', 'pred_b', 'pred_tie']].values

# 4. 计算分数
loss = log_loss(y_true, y_pred)
acc = accuracy_score(y_true, y_pred.argmax(axis=1))

print(f"✅ Fold 0 Evaluation Results:")
print(f"   Log Loss: {loss:.5f} (越低越好)")
print(f"   Accuracy: {acc:.2%} (越高越好)")