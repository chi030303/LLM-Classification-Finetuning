import os
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
from sklearn.metrics import log_loss, accuracy_score

# --- 1. 参数 ---
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=0)
# 默认模型名改为 Qwen2.5，如果你确实下载的是 Qwen3，请保持你的设置
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct") 
parser.add_argument("--data_path", type=str, default="data/processed/train_with_folds.csv")
parser.add_argument("--output_dir", type=str, default="outputs/models/qwen_finetuned")
args = parser.parse_args()

# --- 2. 构造 ChatML 风格 Prompt (效果优于普通文本) ---
# Qwen 对 <|im_start|> 这种 tag 很敏感
TEMPLATE = """<|im_start|>system
You are a helpful assistant acting as a judge. Please evaluate which response is better.<|im_end|>
<|im_start|>user
Question: {prompt}

Response A: {res_a}

Response B: {res_b}

Which response is better? Output 'A', 'B' or 'Tie'.<|im_end|>
<|im_start|>assistant
"""

def preprocess_function(examples):
    inputs = []
    # 稍微缩短长度以适配单卡显存，1536 是个安全值
    # 注意：这里直接取 string，因为前面已经清洗过了
    for p, a, b in zip(examples["prompt_text"], examples["res_a_text"], examples["res_b_text"]):
        text = TEMPLATE.format(
            prompt=str(p)[:600],  # 稍微缩短截断长度，留空间给 System Prompt
            res_a=str(a)[:600],
            res_b=str(b)[:600]
        )
        inputs.append(text)
    
    # 显式指定 padding 策略
    return tokenizer(inputs, truncation=True, max_length=1536, padding=False)

# 读取数据
df = pd.read_csv(args.data_path)

# [关键修复] 填充 NaN，防止 str() 转换出 "nan" 字符串干扰模型
df['prompt_text'] = df['prompt_text'].fillna("")
df['res_a_text'] = df['res_a_text'].fillna("")
df['res_b_text'] = df['res_b_text'].fillna("")

df['label'] = df[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1).map({
    'winner_model_a': 0, 'winner_model_b': 1, 'winner_tie': 2
})

train_ds = Dataset.from_pandas(df[df['fold'] != args.fold])
valid_ds = Dataset.from_pandas(df[df['fold'] == args.fold])

# --- 3. 模型配置 ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, 
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
# Qwen 的 pad token 设置
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# 对于 SequenceClassification，右填充通常没问题，Transformers 会自动处理
tokenizer.padding_side = 'right'

model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=3,
    quantization_config=bnb_config,
    attn_implementation="sdpa", # V100 使用 PyTorch 原生 Attention
    device_map="auto" 
)

# 准备 k-bit 训练
model = prepare_model_for_kbit_training(model)

# LoRA 配置
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    # Qwen 的全量模块，这能带来更好的效果
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    bias="none",
    task_type=TaskType.SEQ_CLS,
    modules_to_save=["score"] # 必须训练分类头
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# --- 4. 训练参数 ---
training_args = TrainingArguments(
    output_dir=f"{args.output_dir}_fold{args.fold}",
    learning_rate=2e-4,             # QLoRA 标准 LR
    per_device_train_batch_size=2,  # 8B 模型在 V100 上应该能开到 2 或 4
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,  # 2 * 8 = 16 (等效 Batch Size)
    num_train_epochs=1,
    fp16=True,
    logging_steps=10,
    eval_strategy="steps",          # [关键修复] evaluation_strategy -> eval_strategy
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="log_loss",
    greater_is_better=False,
    report_to="none"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # 防止 logits 中出现 NaN
    if isinstance(logits, tuple):
        logits = logits[0]
    
    # 转换为概率
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    
    # 计算指标
    loss = log_loss(labels, probs)
    acc = accuracy_score(labels, np.argmax(probs, axis=1))
    return {"log_loss": loss, "accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds.map(preprocess_function, batched=True),
    eval_dataset=valid_ds.map(preprocess_function, batched=True),
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

print("Starting Training...")
trainer.train()