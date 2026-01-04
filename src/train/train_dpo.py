import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

# --- 1. 配置 ---
# [关键] 你的 SFT 好的模型路径 (基座 + Adapter)
BASE_MODEL_PATH = "/root/autodl-tmp/llm_classification_finetuning/base_models/Qwen3-14B"
SFT_ADAPTER_REPO_ID = "chi10969/qwen3-14b-fold2"

# DPO 数据路径
DPO_DATA_PATH = "data/processed/dpo_train_data.jsonl"
OUTPUT_DIR = "outputs/models/qwen_14b_dpo_fold2"

# --- 2. 加载模型 ---
# 注意：DPO 必须用 CausalLM (生成模型)，而不是 SequenceClassification
print("🚀 Loading SFT model for DPO training...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    device_map={"": 0},
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)

base_model = prepare_model_for_kbit_training(base_model)#？

# 加载 SFT Adapter
from huggingface_hub import login
login("") # 如果你的 token 已经通过 CLI 登录过，可能不需要这行
#？
model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_REPO_ID)#？
model.train()

# Reference model
# ref_model = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL_PATH,
#     quantization_config=bnb_config,
#     device_map={"": 0},
#     attn_implementation="flash_attention_2",
#     torch_dtype=torch.bfloat16,
# )

# ref_model.eval()
# for p in ref_model.parameters():
#     p.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"#？

train_dataset = load_dataset("json", data_files=DPO_DATA_PATH, split="train")

dpo_peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

dpo_config = DPOConfig(
    beta=0.1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-5,
    bf16=True,
    logging_steps=10,
    output_dir=OUTPUT_DIR,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    gradient_checkpointing=True,
    report_to="none",
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_config, # DPOConfig 里已经包含了 beta
    train_dataset=train_dataset,
    # tokenizer=tokenizer,
    peft_config=dpo_peft_config,
    # max_length=2048,
    # max_prompt_length=1024
)

dpo_trainer.train()
dpo_trainer.save_model(OUTPUT_DIR)