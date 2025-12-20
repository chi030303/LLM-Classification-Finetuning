import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, classification_report
matplotlib.use('Agg')

# --- 1. 配置路径 ---
DEBERTA_OOF_PATH = "data/processed/oof_deberta_v3_large.csv"
QWEN_OOF_PATH = "data/processed/oof_qwen_14b.csv" # 确保这是完整版或对齐版
# [关键] 引入原始数据文件，用于找回 Cluster 和 Label
SOURCE_DATA_PATH = "data/processed/train_with_folds.csv" 

# --- 2. 加载数据 ---
print("📊 加载 OOF 和源数据...")
df_deb_oof = pd.read_csv(DEBERTA_OOF_PATH)
df_qwen_oof = pd.read_csv(QWEN_OOF_PATH)
source_df = pd.read_csv(SOURCE_DATA_PATH)

# [关键修复] 检查行数
if not (len(df_deb_oof) == len(source_df) and len(df_qwen_oof) == len(source_df)):
    raise ValueError("Row count mismatch! Cannot align data.")

# --- 构造 All-in-One DataFrame (基于 Index) ---
# 1. 从源文件获取 Label 和 Cluster
df_all = source_df[['winner_model_a', 'winner_model_b', 'winner_tie', 'cluster_id_k20', 'fold']].copy()

# 拼接 DeBERTa 预测
df_all['deb_pred_a'] = df_deb_oof['pred_a']
df_all['deb_pred_b'] = df_deb_oof['pred_b']
df_all['deb_pred_tie'] = df_deb_oof['pred_tie']

# 拼接 Qwen 预测
df_all['qwen_pred_a'] = df_qwen_oof['pred_a']
df_all['qwen_pred_b'] = df_qwen_oof['pred_b']
df_all['qwen_pred_tie'] = df_qwen_oof['pred_tie']

# 4. 转换真实标签 (只需一次)
y_true = df_all[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1).map({
    'winner_model_a': 0, 'winner_model_b': 1, 'winner_tie': 2
}).values

print("✅ Data Merged Successfully.")

# ================= 3. 核心指标对比表 =================
metrics_data = []

# 计算 DeBERTa 指标
y_pred_deb = df_all[['deb_pred_a', 'deb_pred_b', 'deb_pred_tie']].values
metrics_data.append({
    "Model": "DeBERTa-v3-Large",
    "Log Loss (↓)": log_loss(y_true, y_pred_deb),
    "Accuracy (↑)": accuracy_score(y_true, y_pred_deb.argmax(axis=1))
})

# 计算 Qwen 指标
y_pred_qwen = df_all[['qwen_pred_a', 'qwen_pred_b', 'qwen_pred_tie']].values
metrics_data.append({
    "Model": "Qwen3-14B",
    "Log Loss (↓)": log_loss(y_true, y_pred_qwen),
    "Accuracy (↑)": accuracy_score(y_true, y_pred_qwen.argmax(axis=1))
})

metrics_df = pd.DataFrame(metrics_data)
print("\n" + "="*45)
print("📊 Overall Model Performance (5-Fold CV)")
print("="*45)
print(metrics_df.to_string(index=False))
print("="*45)


# ================= 4. 按 Fold 分析 Loss (诊断坏 Fold) =================
print("\n🔍 Analyzing Loss per Fold...")

for model_prefix, y_pred in [("DeBERTa", y_pred_deb), ("Qwen", y_pred_qwen)]:
    print(f"\n--- {model_prefix} ---")
    
    for fold in range(5):
        # 找到属于当前 Fold 的行索引
        fold_mask = (df_all['fold'] == fold)
        
        if np.sum(fold_mask) == 0:
            print(f"Fold {fold}: No data found.")
            continue
            
        # 提取当前 Fold 的真实标签和预测概率
        fold_y_true = y_true[fold_mask]
        fold_y_pred = y_pred[fold_mask]
        
        fold_loss = log_loss(fold_y_true, fold_y_pred)
        print(f"Fold {fold} LogLoss: {fold_loss:.4f}")
        
# ==============================================================================
# 📈 图表 2: 归一化混淆矩阵 (以 Qwen 为例)
# ==============================================================================
def plot_cm(y_true, y_pred_probs, title):
    labels = ['A Win', 'B Win', 'Tie']
    cm = confusion_matrix(y_true, y_pred_probs.argmax(axis=1))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{title.replace(' ', '_')}.png") 
    print(f"✅ Saved plot: {title.replace(' ', '_')}.png")
    # plt.show()

plot_cm(y_true, df_all[['qwen_pred_a', 'qwen_pred_b', 'qwen_pred_tie']].values, "Normalized Confusion Matrix (Qwen3-14B)")

# ==============================================================================
# 📈 图表 3: 按 Cluster 性能对比 (证明互补性)
# ==============================================================================
cluster_perf = df_all.groupby('cluster_id_k20')[['deb_loss', 'qwen_loss']].mean()

plt.figure(figsize=(14, 6))
cluster_perf.plot(kind='bar', ax=plt.gca(), color=['#3498db', '#e74c3c'])
plt.axhline(df_all['deb_loss'].mean(), color='#3498db', linestyle='--', alpha=0.5, label='Deb Avg')
plt.axhline(df_all['qwen_loss'].mean(), color='#e74c3c', linestyle='--', alpha=0.5, label='Qwen Avg')
plt.title("Log Loss per Cluster: DeBERTa vs Qwen")
plt.ylabel("Average Log Loss (Lower is Better)")
plt.xlabel("Cluster ID (Topic Category)")
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.savefig("Log_Loss_per_Cluster.png")
print("✅ Saved plot: Log_Loss_per_Cluster.png")
# plt.show()

# ==============================================================================
# 📈 图表 4: 预测分布密度图 (Confidence Analysis)
# ==============================================================================
plt.figure(figsize=(10, 5))
sns.kdeplot(df_all['deb_pred_a'], label='DeBERTa - Pred A Prob', fill=True, alpha=0.3)
sns.kdeplot(df_all['qwen_pred_a'], label='Qwen - Pred A Prob', fill=True, alpha=0.3)
plt.title("Prediction Confidence Distribution (Class A)")
plt.xlabel("Probability assigned to Model A")
plt.legend()
plt.savefig("Prediction_Confidence_Distribution.png")
print("✅ Saved plot: Prediction_Confidence_Distribution.png")
# plt.show()