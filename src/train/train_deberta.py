import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset

# --- 1. 参数解析 (已对齐路径) ---
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=0, help="Run training on which fold")
# 这里 model_name 依然传模型ID，路径通过 HF_HOME 环境变量控制
parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-large") 
parser.add_argument("--data_path", type=str, default="data/processed/train_with_folds.csv") # 对齐数据路径
parser.add_argument("--output_dir", type=str, default="outputs/models/deberta_v3_large")   # 对齐输出路径
parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
parser.add_argument("--max_len", type=int, default=1536, help="Max sequence length") 

parser.add_argument("--per_device_train_batch_size", type=int, default=2)
parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
parser.add_argument("--num_train_epochs", type=int, default=3)
parser.add_argument("--learning_rate", type=float, default=5e-6)
parser.add_argument("--dataloader_num_workers", type=int, default=4)
# Boolean 开关
parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
parser.add_argument("--bf16", action="store_true", help="Enable bf16")
parser.add_argument("--fp16", action="store_true", help="Enable fp16")

args = parser.parse_args()

REAL_OUTPUT_DIR = f"{args.output_dir}_fold{args.fold}"
print(f"🎯 Real Output Directory: {REAL_OUTPUT_DIR}")
# --- 2. 数据处理 ---
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

def preprocess_function(examples):
    # DeBERTa 处理 1024 长度通常占用约 12G-15G 显存 (BS=4)
    prompts = [str(x) if x is not None else "" for x in examples["prompt_text"]]
    res_a = [str(x) if x is not None else "" for x in examples["res_a_text"]]
    res_b = [str(x) if x is not None else "" for x in examples["res_b_text"]]

    sep = tokenizer.sep_token 
    combined_texts = [
        f"{p} {sep} {a} {sep} {b}" 
        for p, a, b in zip(prompts, res_a, res_b)
    ]
    
    # 3. 传给 Tokenizer (现在只传一个列表)
    return tokenizer(
        combined_texts,
        truncation=True,
        max_length=args.max_len, 
        padding=False # DataCollator 会负责动态 Padding，这里 False 是对的
    )

# 读取 CSV (如果是 parquet 改为 pd.read_parquet)
df = pd.read_csv(args.data_path) 

# Label 转换
df['label'] = df[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1).map({
    'winner_model_a': 0, 'winner_model_b': 1, 'winner_tie': 2
})

train_df = df[df['fold'] != args.fold].reset_index(drop=True)
valid_df = df[df['fold'] == args.fold].reset_index(drop=True)

train_ds = Dataset.from_pandas(train_df)
valid_ds = Dataset.from_pandas(valid_df)

train_ds = train_ds.map(preprocess_function, batched=True)
valid_ds = valid_ds.map(preprocess_function, batched=True)

# --- 3. 模型 ---
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name, 
    num_labels=3,
    attention_probs_dropout_prob=0.1,
    hidden_dropout_prob=0.1
)

# --- 4. 训练参数 (单卡 V100 调优) ---
training_args = TrainingArguments(
    output_dir=REAL_OUTPUT_DIR,
    learning_rate=args.learning_rate,             # 单卡通常调低一点 LR 比较稳
    per_device_train_batch_size=args.per_device_train_batch_size,  # 48G 显存跑 1536 长度，BS=4 应该很轻松
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=args.gradient_accumulation_steps,  # 4 * 4 = 16 (等效 Batch Size)
    num_train_epochs=args.num_train_epochs,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="log_loss",
    greater_is_better=False,
    bf16=args.bf16,
    fp16=args.fp16,
    report_to="none",
    disable_tqdm=True,
    gradient_checkpointing=args.gradient_checkpointing,
    dataloader_num_workers=args.dataloader_num_workers,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    loss = log_loss(labels, probs)
    acc = accuracy_score(labels, np.argmax(probs, axis=1))
    return {"log_loss": loss, "accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

last_checkpoint = None
if args.resume_from_checkpoint:
    print(f"🛑 Manual Resume: Forcing resume from {args.resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

# 否则尝试自动寻找
elif os.path.isdir(REAL_OUTPUT_DIR):
    from transformers.trainer_utils import get_last_checkpoint
    last_checkpoint = get_last_checkpoint(REAL_OUTPUT_DIR)
    if last_checkpoint is not None:
        print(f"🔄 Auto Resume: Found checkpoint {last_checkpoint}")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("🚀 No checkpoint found. Starting new training run...")
        trainer.train()
# 全新训练
else:
    print("🚀 New output directory. Starting new training run...")
    trainer.train()