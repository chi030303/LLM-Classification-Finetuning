import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

# 1. 读取生成的 OOF 文件
oof_file = "data/processed/oof_deberta_v3_large.csv"
df = pd.read_csv(oof_file)

# 2. 准备真实标签 (Target)
# 假设你的 OOF 文件里保留了原始的 winner 列
# 如果没有，你需要读取 train.csv 并通过 id merge 进去
if 'winner_model_a' in df.columns:
    y_true = df[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1).map({
        'winner_model_a': 0, 'winner_model_b': 1, 'winner_tie': 2
    })
else:
    print("⚠️ OOF 文件缺少标签列，请合并原始数据！")

# 3. 准备预测概率
y_pred = df[['pred_a', 'pred_b', 'pred_tie']].values

# 4. 计算指标
cv_log_loss = log_loss(y_true, y_pred)
cv_accuracy = accuracy_score(y_true, y_pred.argmax(axis=1))

print(f"📊 DeBERTa-v3-Large 5-Fold OOF Results:")
print(f"   📉 Log Loss: {cv_log_loss:.5f} (越低越好)")
print(f"   📈 Accuracy: {cv_accuracy:.2%} (越高越好)")