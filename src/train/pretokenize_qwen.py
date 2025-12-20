import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset
import os

# 配置
MODEL_NAME = "/root/autodl-tmp/base_models/Qwen3-14B"
DATA_PATH = "data/processed/train_with_folds.csv"
SAVE_PATH = "data/processed/qwen_pretokenized"
MAX_LEN = 2048

print("🚀 Starting Qwen Pre-tokenization on CPU...")

# 1. 加载数据
df = pd.read_csv(DATA_PATH).fillna("")
dataset = Dataset.from_pandas(df)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

# 2. 定义处理函数
TEMPLATE = """<|im_start|>system
You are a helpful assistant acting as a judge.<|im_end|>
<|im_start|>user
Question: {prompt}
Response A: {res_a}
Response B: {res_b}
Which response is better?<|im_end|>
<|im_start|>assistant
"""

def process_fn(examples):
    inputs = [
        TEMPLATE.format(prompt=str(p)[:1200], res_a=str(a)[:1200], res_b=str(b)[:1200])
        for p, a, b in zip(examples["prompt_text"], examples["res_a_text"], examples["res_b_text"])
    ]
    return tokenizer(inputs, truncation=True, max_length=MAX_LEN, padding=False)

# 3. 利用多核 CPU 并行处理 (这就把闲置 CPU 用起来了！)
print(f"Tokenizing with {os.cpu_count()} CPU cores...")
tokenized_ds = dataset.map(
    process_fn, 
    batched=True, 
    num_proc=20,  # 🔥 直接开 20 个核！反正闲着也是闲着
    remove_columns=dataset.column_names # 移除原始文本，只留 input_ids
)

# 4. 保存到磁盘
tokenized_ds.save_to_disk(SAVE_PATH)
print(f"✅ Saved pre-tokenized data to {SAVE_PATH}")