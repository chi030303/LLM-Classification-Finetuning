import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

# 1. 读取您的 OOF 文件
OOF_PATH = "/root/autodl-tmp/llm_classification_finetuning/data/processed/oof_deberta_v3_large.csv"

# 2. 标签映射 (根据您的训练代码)
# 确保这里和您训练时的 map 一致
label_map = {'winner_model_a': 0, 'winner_model_b': 1, 'winner_tie': 2}

def calculate_weights():
    print(f"📖 Reading {OOF_PATH}...")
    df = pd.read_csv(OOF_PATH)
    
    # 构造真实标签 (将 One-Hot 列转为 0,1,2)
    # 假设您的 CSV 里有 winner_model_a/b/tie 这些原始列
    # 如果没有，您需要从原始 train.csv merge 过来，或者如果您的 OOF 只有 pred 列，那无法计算 loss
    
    # 为了保险，我们重新读取原始数据 merge 标签 (CPU操作，内存足够)
    # 如果您的 OOF CSV 里已经保留了 label 列，可以跳过这一步
    if 'winner_model_a' not in df.columns:
        print("⚠️ OOF file missing labels, merging from train_with_folds.csv...")
        train_df = pd.read_csv("data/processed/train_with_folds.csv") # 修改为您实际的训练数据路径
        # 假设通过 id 或者 index 对齐
        # 简单起见，这里假设 OOF 是按顺序或者有 id
        # 建议您确认 OOF CSV 里是否有 id 列
        if 'id' in df.columns and 'id' in train_df.columns:
            df = df.merge(train_df[['id', 'winner_model_a', 'winner_model_b', 'winner_tie']], on='id', how='left')
        else:
            print("❌ 无法对齐标签，无法计算 Loss。请检查 CSV。")
            return

    # 提取真实标签索引 (0, 1, 2)
    df['target'] = df[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1).map(label_map)
    
    fold_scores = []
    
    print("\n📊 Analyzing Per-Fold Performance:")
    print("-" * 40)
    for fold in range(5):
        # 筛选该 Fold 的数据
        fold_data = df[df['fold'] == fold]
        
        if len(fold_data) == 0:
            print(f"Fold {fold}: No data found!")
            fold_scores.append(10.0) # 惩罚
            continue
            
        y_true = fold_data['target'].values
        y_pred = fold_data[['pred_a', 'pred_b', 'pred_tie']].values
        
        # 计算 LogLoss
        loss = log_loss(y_true, y_pred)
        fold_scores.append(loss)
        
        print(f"   Fold {fold} LogLoss: {loss:.5f}")
        
    print("-" * 40)
    
    # === 核心：计算权重 ===
    scores = np.array(fold_scores)
    
    # 策略：Softmax(负 Loss)
    # 温度系数 T：越小，对好模型的偏向越重（惩罚坏模型越狠）
    T = 0.05 
    
    # 归一化权重计算
    exp_scores = np.exp((scores.min() - scores) / T)
    weights = exp_scores / exp_scores.sum()
    
    print("\n⚖️  Recommended Weights (T=0.05):")
    print(weights)
    print(f"   Sum: {weights.sum():.2f}")
    
    # 格式化输出方便复制
    weights_str = ", ".join([f"{w:.4f}" for w in weights])
    print(f"\n📋 Copy this to your inference script:\nMODEL_WEIGHTS = [{weights_str}]")

if __name__ == "__main__":
    calculate_weights()