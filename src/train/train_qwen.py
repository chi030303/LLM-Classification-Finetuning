import os
# [关键] 解决 Tokenizer 死锁警告，必须放在最前面
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import pandas as pd
import numpy as np
import torch
import textwrap
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from sklearn.metrics import log_loss, accuracy_score

# --- 1. 参数设置 ---
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=0)
# 你的本地模型路径 (Qwen2.5/Qwen3)
parser.add_argument("--model_name", type=str, default="/root/autodl-tmp/base_models/Qwen3-14") 
parser.add_argument("--data_path", type=str, default="data/processed/train_with_folds.csv")
parser.add_argument("--output_dir", type=str, default="outputs/models/qwen_14b")
parser.add_argument("--max_len", type=int, default=2048) # 保持你要求的长度
parser.add_argument("--learning_rate", type=float, default=2e-4)
args = parser.parse_args()

REAL_OUTPUT_DIR = f"{args.output_dir}_fold{args.fold}"
print(f"🚀 Training Fold {args.fold} | Model: {args.model_name}")

# --- 2. 增强版 Prompt (ChatML 格式) ---
def preprocess_function(examples):
    inputs = []
    # 使用清洗过的列
    for p, a, b in zip(examples["prompt_text"], examples["res_a_text"], examples["res_b_text"]):
        # 使用 textwrap.dedent 去除代码缩进，确保 Prompt 干净
        text = textwrap.dedent(f"""\
            <|im_start|>system
            You are a helpful assistant acting as a judge. Please evaluate which response is better.<|im_end|>
            <|im_start|>user
            Evaluate the two responses to the user question.

            Question:
            {p}

            Candidate Response A:
            {a}

            Candidate Response B:
            {b}

            You must choose the preferred answer.
            Respond strictly with one of: A, B, Tie.
            Do not provide any explanation.<|im_end|>
            <|im_start|>assistant
            """)
        inputs.append(text)
    
    return tokenizer(inputs, truncation=True, max_length=args.max_len, padding=False)

# --- 3. 加载数据 ---
# 填充空值防止报错
df = pd.read_csv(args.data_path).fillna("")
# Label Mapping
df['label'] = df[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1).map({
    'winner_model_a': 0, 'winner_model_b': 1, 'winner_tie': 2
})

train_ds = Dataset.from_pandas(df[df['fold'] != args.fold])
valid_ds = Dataset.from_pandas(df[df['fold'] == args.fold])

# --- 4. 模型与 Tokenizer ---
# 4-bit 量化配置 (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # RTX 6000 必须开 BF16
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right' # 分类任务必须右填充

model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=3,
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2", # 🔥 关键加速：开启 Flash Attention 2
    device_map="auto",
    torch_dtype=torch.bfloat16
)

model = prepare_model_for_kbit_training(model)

# LoRA 配置
peft_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    lora_dropout=0.05,
    # Qwen 全模块微调效果最好
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    task_type=TaskType.SEQ_CLS,
    modules_to_save=["score"] # 必须训练分类头
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.config.pad_token_id = tokenizer.pad_token_id 

# --- 5. 训练参数 (已优化频率) ---
train_ds = train_ds.map(preprocess_function, batched=True)
valid_ds = valid_ds.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir=REAL_OUTPUT_DIR,
    learning_rate=args.learning_rate,
    
    # [显存优化] RTX 6000 96G 显存巨大，BS 开大提速
    # per_device_train_batch_size=16,   
    per_device_train_batch_size=8,  
    gradient_accumulation_steps=4,  # 总 Batch 还是 32
    per_device_eval_batch_size=16,
    
    num_train_epochs=1,               # 14B 跑 1 轮
    bf16=True,                        # 必须开启 BF16
    
    logging_steps=10,
    
    # [评估频率优化] 减少评估次数，专注训练
    # 假设数据量 4.6w, Batch 32 -> epoch steps ≈ 1437
    # 设为 0.2 表示每跑 20% (约280步) 评估一次
    eval_strategy="steps",
    eval_steps=300,                   
    save_strategy="steps",
    save_steps=300,
    save_total_limit=2,               # 只留最近2个
    
    load_best_model_at_end=True,
    metric_for_best_model="log_loss",
    greater_is_better=False,
    
    gradient_checkpointing=True,      # 显存优化
    max_grad_norm=1.0,                # 防止梯度爆炸
    warmup_ratio=0.05,                
    
    report_to="none",
    disable_tqdm=True,
    dataloader_num_workers=4          
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple): logits = logits[0]
    # [关键修复] 必须用 numpy 且防止溢出
    # 使用 float32 确保精度
    probs = torch.nn.functional.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
    
    return {
        "log_loss": log_loss(labels, probs), 
        "accuracy": accuracy_score(labels, np.argmax(probs, axis=1))
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    # tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

print("🚀 Starting Qwen Training...")
trainer.train()

# 保存最终模型
trainer.save_model(REAL_OUTPUT_DIR)
print(f"✅ Training Complete. Model saved to {REAL_OUTPUT_DIR}")