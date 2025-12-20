import os
import glob
import re
import json
import argparse
import pandas as pd
import numpy as np
import torch
import textwrap
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import PeftModel
from datasets import Dataset
from torch.nn.functional import softmax

# --- 配置 ---
parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default="/root/autodl-tmp/base_models/Qwen3-14B")
parser.add_argument("--data_path", type=str, default="data/processed/train_with_folds.csv")
parser.add_argument("--adapters_dir", type=str, default="outputs/models/qwen_14b") # LoRA Adapter 的父目录
parser.add_argument("--output_path", type=str, default="data/processed/oof_qwen_14b.csv")
parser.add_argument("--max_len", type=int, default=2048)
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()

def get_best_checkpoint(fold_dir):
    """智能选择最佳 LoRA Adapter Checkpoint"""
    state_file = os.path.join(fold_dir, "trainer_state.json")
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            best_path_raw = state.get("best_model_checkpoint")
            if best_path_raw:
                ckpt_name = os.path.basename(best_path_raw)
                best_ckpt_path = os.path.join(fold_dir, ckpt_name)
                if os.path.exists(best_ckpt_path):
                    print(f"   🏆 Found Best Checkpoint from JSON: {ckpt_name}")
                    return best_ckpt_path
        except Exception:
            pass # fallback

    # Fallback: 按步数取最大的
    checkpoints = glob.glob(os.path.join(fold_dir, "checkpoint-*"))
    if not checkpoints: return None
    def get_step(path):
        match = re.search(r"checkpoint-(\d+)", path)
        return int(match.group(1)) if match else 0
    checkpoints.sort(key=get_step)
    return checkpoints[-1]

def main():
    print(f"🚀 Starting Qwen OOF Generation...")
    
    # 1. 加载数据
    df = pd.read_csv(args.data_path).fillna("")
    
    # 2. 加载 4-bit 基座模型 (只加载一次，节约时间)
    print("   -> Loading 4-bit Base Model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    base_tokenizer.padding_side = 'right'

    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_path,
        num_labels=3,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    base_model.config.pad_token_id = base_tokenizer.pad_token_id

    oof_results = []
    
    # 3. 循环处理 5 个 Fold
    for fold in range(5):
        print(f"\n================ Processing FOLD {fold} ================")
        
        # A. 确定 Adapter 路径
        fold_adapter_dir = f"{args.adapters_dir}_fold{fold}"
        adapter_path = get_best_checkpoint(fold_adapter_dir)
        
        if not adapter_path:
            print(f"❌ Error: No Adapter found for Fold {fold}")
            continue
            
        print(f"   -> Loading Adapter: {adapter_path}")
        
        # B. [关键] 将 Adapter 加载到基座模型上
        model = PeftModel.from_pretrained(base_model, adapter_path)
        # 合并权重以加速推理
        model = model.merge_and_unload()
        
        # C. 准备验证集
        val_df = df[df['fold'] == fold].copy()
        val_ds = Dataset.from_pandas(val_df)
        
        # D. 预处理 (必须与训练一致)
        def preprocess(examples):
            inputs = []
            for p, a, b in zip(examples["prompt_text"], examples["res_a_text"], examples["res_b_text"]):
                text = textwrap.dedent(f"""\
                    <|im_start|>system
                    You are a helpful assistant acting as a judge. Please evaluate which response is better.<|im_end|>
                    <|im_start|>user
                    Evaluate the two responses to the user question.
                    Question: {str(p)[:1200]}
                    Candidate Response A: {str(a)[:1200]}
                    Candidate Response B: {str(b)[:1200]}
                    You must choose the preferred answer.
                    Respond strictly with one of: A, B, Tie.
                    Do not provide any explanation.<|im_end|>
                    <|im_start|>assistant
                    """)
                inputs.append(text)
            return base_tokenizer(inputs, truncation=True, max_length=args.max_len, padding=False)
        
        val_ds = val_ds.map(preprocess, batched=True)
        
        # E. 推理
        training_args = TrainingArguments(
            output_dir="./tmp_qwen_infer",
            per_device_eval_batch_size=args.batch_size,
            bf16=True,
            report_to="none",
            dataloader_num_workers=4
        )
        
        trainer = Trainer(model=model, args=training_args, tokenizer=base_tokenizer)
        preds_output = trainer.predict(val_ds)
        
        logits = torch.tensor(preds_output.predictions)
        probs = softmax(logits.float(), dim=-1).numpy() # 转 float32
        
        val_df['pred_a'] = probs[:, 0]
        val_df['pred_b'] = probs[:, 1]
        val_df['pred_tie'] = probs[:, 2]
        
        oof_results.append(val_df)

    # 4. 合并与保存
    if oof_results:
        oof_full = pd.concat(oof_results).sort_values('fold').reset_index(drop=True)
        oof_full.to_csv(args.output_path, index=False)
        print(f"\n✅ Qwen OOF Predictions saved to: {args.output_path}")
    else:
        print("❌ No OOF generated.")

if __name__ == "__main__":
    main()