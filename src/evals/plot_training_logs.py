import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

import matplotlib
matplotlib.use('Agg') 

# --- 配置 ---
LOG_FILES = {
    # 你的日志文件列表...
    "DeBERTa_Fold0": "outputs/logs/deberta_fold0.log",
    "DeBERTa_Fold1": "outputs/logs/deberta_fold1.log",
    "DeBERTa_Fold2": "outputs/logs/deberta_fold2.log",
    "DeBERTa_Fold3": "outputs/logs/deberta_fold3.log",
    "DeBERTa_Fold4": "outputs/logs/deberta_fold4.log",
    "Qwen_Fold0": "outputs/logs/qwen_fold0.log",
    "Qwen_Fold1": "outputs/logs/qwen_fold1.log",
    "Qwen_Fold2": "outputs/logs/qwen_fold2.log",
    "Qwen_Fold3": "outputs/logs/qwen_fold3.log",
    "Qwen_Fold4": "outputs/logs/qwen_fold4.log",
}
OUTPUT_DIR = "outputs/analysis_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 日志解析器 (保持不变) ---
def parse_log(file_path):
    train_data = []
    eval_data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                json_part = line[line.find('{'):]
                log_dict = json.loads(json_part.replace("'", '"'))
                
                if 'eval_loss' in log_dict and 'epoch' in log_dict:
                    eval_data.append({"epoch": log_dict['epoch'], "eval_loss": log_dict['eval_loss']})
                elif 'loss' in log_dict and 'epoch' in log_dict:
                    train_data.append({"epoch": log_dict['epoch'], "train_loss": log_dict['loss']})
            except:
                continue
    return pd.DataFrame(train_data), pd.DataFrame(eval_data)

# --- [新增] 模块化绘图函数 ---
def plot_curves(df, title, y_col, filename):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='epoch', y=y_col, hue='model', marker='o')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(y_col.replace('_', ' ').title())
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    print(f"✅ Saved plot: {os.path.join(OUTPUT_DIR, filename)}")
    plt.close()

def plot_combined_curves(df, title, y_col, filename):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='epoch', y=y_col, hue='model_type', ci='sd')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(y_col.replace('_', ' ').title())
    plt.grid(True)
    plt.legend(title='Model Type')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    print(f"✅ Saved plot: {os.path.join(OUTPUT_DIR, filename)}")
    plt.close()

# --- 主程序 ---
def main():
    all_train_logs = {}
    all_eval_logs = {}

    for model_name, path in LOG_FILES.items():
        if os.path.exists(path):
            train_df, eval_df = parse_log(path)
            if not train_df.empty: all_train_logs[model_name] = train_df
            if not eval_df.empty: all_eval_logs[model_name] = eval_df
        else:
            print(f"⚠️ Log file not found: {path}")

    # --- 准备数据 ---
    df_train = pd.concat(all_train_logs.values(), keys=all_train_logs.keys(), names=['model'])
    df_eval = pd.concat(all_eval_logs.values(), keys=all_eval_logs.keys(), names=['model'])
    
    df_train['model_type'] = df_train.index.get_level_values('model').str.split('_').str[0]
    df_eval['model_type'] = df_eval.index.get_level_values('model').str.split('_').str[0]

    # 筛选数据
    deb_train_df = df_train[df_train['model_type'] == 'DeBERTa']
    qwen_train_df = df_train[df_train['model_type'] == 'Qwen']
    deb_eval_df = df_eval[df_eval['model_type'] == 'DeBERTa']
    qwen_eval_df = df_eval[df_eval['model_type'] == 'Qwen']

    # --- 绘制 6 张图 ---
    print("\n🎨 Generating 6 plots...")
    
    # 1. Training Loss - DeBERTa
    plot_curves(deb_train_df, "DeBERTa - Training Loss Curves (5 Folds)", "train_loss", "train_curve_deberta.png")
    
    # 2. Training Loss - Qwen
    plot_curves(qwen_train_df, "Qwen - Training Loss Curves (5 Folds)", "train_loss", "train_curve_qwen.png")
    
    # 3. Training Loss - Combined
    plot_combined_curves(df_train, "Combined Training Loss (Mean ± Std Dev)", "train_loss", "train_curve_combined.png")
    
    # 4. Validation Loss - DeBERTa
    plot_curves(deb_eval_df, "DeBERTa - Validation Loss Curves (5 Folds)", "eval_loss", "validation_curve_deberta.png")
    
    # 5. Validation Loss - Qwen
    plot_curves(qwen_eval_df, "Qwen - Validation Loss Curves (5 Folds)", "eval_loss", "validation_curve_qwen.png")
    
    # 6. Validation Loss - Combined
    plot_combined_curves(df_eval, "Combined Validation Loss (Mean ± Std Dev)", "eval_loss", "validation_curve_combined.png")

if __name__ == "__main__":
    main()