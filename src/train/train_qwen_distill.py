import os
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
    DataCollatorWithPadding,
    PreTrainedTokenizerBase
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from sklearn.metrics import log_loss, accuracy_score
from dataclasses import dataclass

# ================= 蒸馏 Trainer =================
class DistillationTrainer(Trainer):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # inputs 字典里现在包含了 Dataset 返回的所有东西
        labels = inputs.pop("labels")
        teacher_probs = inputs.pop("teacher_probs").to(model.device) # 挪到 GPU
        
        outputs = model(**inputs)
        student_logits = outputs.logits
        
        loss_hard = self.ce_loss(student_logits, labels)
        
        student_log_softmax = torch.nn.functional.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_softmax = torch.nn.functional.softmax(teacher_probs / self.temperature, dim=-1)
        
        loss_soft = torch.nn.functional.kl_div(student_log_softmax, teacher_softmax, reduction='batchmean') * (self.temperature ** 2)
        
        loss = self.alpha * loss_hard + (1 - self.alpha) * loss_soft
        
        return (loss, outputs) if return_outputs else loss

# ================= 参数配置 =================
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--student_model", type=str, default="/root/autodl-tmp/llm_classification_finetuning/base_models/Qwen3-8B")
parser.add_argument("--data_path", type=str, default="data/processed/train_with_teacher_labels.csv")
parser.add_argument("--output_dir", type=str, default="outputs/models/qwen_8b_distilled")
parser.add_argument("--max_len", type=int, default=2048)
args = parser.parse_args()

REAL_OUTPUT_DIR = f"{args.output_dir}_fold{args.fold}"
print(f"🚀 Distilling Fold {args.fold} | Student: {os.path.basename(args.student_model)}")

# ================= 数据处理 =================
df = pd.read_csv(args.data_path).fillna("")
teacher_col_mapping = {
    'pred_a': 'teacher_a',
    'pred_b': 'teacher_b',
    'pred_tie': 'teacher_tie'
}
# 检查列是否存在
if all(col in df.columns for col in teacher_col_mapping.keys()):
    df.rename(columns=teacher_col_mapping, inplace=True)
    print("   -> Renamed teacher label columns.")
else:
    # 如果已经是 teacher_a 了，就不用改
    if 'teacher_a' not in df.columns:
        raise KeyError("Could not find 'pred_a' or 'teacher_a' columns in the dataset!")

if 'labels' not in df.columns:
    df['labels'] = df[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1).map({
        'winner_model_a': 0, 'winner_model_b': 1, 'winner_tie': 2
    })
    print("   -> Created 'label' column for training.")
# df['label'] = df[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1).map({
#     'winner_model_a': 0, 'winner_model_b': 1, 'winner_tie': 2
# })

train_df = df[df['fold'] != args.fold].reset_index(drop=True)
valid_df = df[df['fold'] == args.fold].reset_index(drop=True)

train_ds_raw = Dataset.from_pandas(train_df)
valid_ds_raw = Dataset.from_pandas(valid_df)

tokenizer = AutoTokenizer.from_pretrained(args.student_model)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# Prompt 模板
TEMPLATE = textwrap.dedent("""\
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

def map_function(examples):
    # Tokenize
    texts = []
    for p, a, b in zip(examples["prompt_text"], examples["res_a_text"], examples["res_b_text"]):
        text = TEMPLATE.format(
            p=str(p), 
            a=str(a),
            b=str(b)
        )
        texts.append(text)
        
    tokenized = tokenizer(texts, truncation=True, max_length=args.max_len, padding=False)
    
    tokenized["teacher_probs"] = [
        [a, b, t] for a, b, t in zip(examples['teacher_a'], examples['teacher_b'], examples['teacher_tie'])
    ]
    return tokenized

class DistillDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.prompts = df['prompt_text'].tolist()
        self.res_a = df['res_a_text'].tolist()
        self.res_b = df['res_b_text'].tolist()
        self.labels = df['labels'].tolist()
        # self.teacher_probs = df[['teacher_a', 'teacher_b', 'teacher_tie']].values
        self.teacher_probs_df = df[['teacher_a', 'teacher_b', 'teacher_tie']]


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = TEMPLATE.format(p=self.prompts[idx], a=self.res_a[idx], b=self.res_b[idx])
        # Tokenizer 直接返回字典
        inputs = self.tokenizer(text, truncation=True, max_length=self.max_len)
        inputs['labels'] = self.labels[idx]
        # inputs['teacher_probs'] = self.teacher_probs[idx]
        inputs['teacher_probs'] = self.teacher_probs_df.iloc[idx].tolist()
        return inputs

train_ds = DistillDataset(train_df, tokenizer, args.max_len)
valid_ds = DistillDataset(valid_df, tokenizer, args.max_len)

# ================= 模型加载 (QLoRA) =================
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

model = AutoModelForSequenceClassification.from_pretrained(
    args.student_model,
    num_labels=3,
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    task_type=TaskType.SEQ_CLS,
    modules_to_save=["score"]
)
model = get_peft_model(model, peft_config)
model.config.pad_token_id = tokenizer.pad_token_id
model.print_trainable_parameters()

# ================= 训练 =================
training_args = TrainingArguments(
    output_dir=REAL_OUTPUT_DIR,
    learning_rate=2e-4,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=64,
    num_train_epochs=2,
    bf16=True,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="log_loss",
    greater_is_better=False,
    gradient_checkpointing=True,
    report_to="none",
    disable_tqdm=True,
    remove_unused_columns=False,
)

@dataclass
class DistillDataCollator:
    tokenizer: PreTrainedTokenizerBase
    
    def __call__(self, features):
        # 分离 teacher_probs
        teacher_probs = [f.pop("teacher_probs") for f in features]
        
        # 用默认的 collator 处理剩下的
        batch = self.tokenizer.pad(features, return_tensors="pt")
        
        # 把 teacher_probs 加回来
        batch["teacher_probs"] = torch.tensor(teacher_probs, dtype=torch.float)
        
        return batch
    
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple): logits = logits[0]
    probs = torch.nn.functional.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
    return {"log_loss": log_loss(labels, probs), "accuracy": accuracy_score(labels, np.argmax(probs, axis=1))}

trainer = DistillationTrainer( # 使用自定义 Trainer
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    # tokenizer=tokenizer,
    # data_collator=DataCollatorWithPadding(tokenizer),
    data_collator=DistillDataCollator(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
    alpha=0.5,      # 硬软标签权重各一半
    temperature=2.0 # 蒸馏温度
)

print(f"🚀 Starting Distillation Training...")
trainer.train()
trainer.save_model(REAL_OUTPUT_DIR)
print(f"✅ Distillation Complete for Fold {args.fold}.")