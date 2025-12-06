import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# 配置
CHECKPOINT_PATH = "outputs/models/deberta_v3_large_fold0/checkpoint-4600"
DATA_PATH = "data/processed/train_with_folds.csv"
OUTPUT_FILE = "outputs/models/deberta_v3_large_fold0/oof_preds.csv"
FOLD = 0

# 1. 加载数据 (只取验证集)
df = pd.read_csv(DATA_PATH)
df['prompt_text'] = df['prompt_text'].fillna("")
df['res_a_text'] = df['res_a_text'].fillna("")
df['res_b_text'] = df['res_b_text'].fillna("")

valid_df = df[df['fold'] == FOLD].reset_index(drop=True)
valid_ds = Dataset.from_pandas(valid_df)

# 2. 加载模型和 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_PATH, num_labels=3)

# 3. 预处理函数 (必须和训练时完全一致!)
def preprocess_function(examples):
    sep = tokenizer.sep_token
    combined_texts = [
        f"{p} {sep} {a} {sep} {b}" 
        for p, a, b in zip(examples["prompt_text"], examples["res_a_text"], examples["res_b_text"])
    ]
    return tokenizer(combined_texts, truncation=True, max_length=1024, padding=False)

valid_ds = valid_ds.map(preprocess_function, batched=True)

# 4. 推理
inference_args = TrainingArguments(
    output_dir="./tmp_inference_output",  # 临时目录，不重要
    report_to="none",                     # <--- 关键：关闭 wandb/tensorboard
    per_device_eval_batch_size=8,         # 确保这里设置了合理的 batch size
    fp16=True                             # 如果是 V100，开启 fp16 加速推理
)

trainer = Trainer(
    model=model, 
    tokenizer=tokenizer, 
    args=inference_args
)

preds = trainer.predict(valid_ds)
preds = trainer.predict(valid_ds)
logits = preds.predictions
probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

# 5. 保存结果
valid_df['pred_a'] = probs[:, 0]
valid_df['pred_b'] = probs[:, 1]
valid_df['pred_tie'] = probs[:, 2]

valid_df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ OOF Predictions saved to {OUTPUT_FILE}")