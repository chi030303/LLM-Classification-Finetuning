import os
import glob
import re
import json
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from torch.nn.functional import softmax
from sklearn.metrics import log_loss, accuracy_score
from huggingface_hub import login

# --- 配置 ---
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data/processed/train_with_folds.csv")
parser.add_argument("--hf_user", type=str, default="chi10969", help="Your Hugging Face username")
parser.add_argument("--model_prefix", type=str, default="deberta-v3-large-fold", help="Prefix for your HF model repos")
parser.add_argument("--output_path", type=str, default="data/processed/oof_deberta_v3_large.csv")
parser.add_argument("--max_len", type=int, default=1536)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--hf_token", type=str, required=True, help="Your Hugging Face token")
args = parser.parse_args()

def get_best_checkpoint(fold_dir):
    """智能选择最佳 Checkpoint"""
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
        except Exception as e:
            print(f"   ⚠️ Could not read trainer_state.json: {e}")

    # Fallback
    print("   ⚠️ JSON not found or invalid. Falling back to largest step number.")
    checkpoints = glob.glob(os.path.join(fold_dir, "checkpoint-*"))
    if not checkpoints: return None
    def get_step(path):
        match = re.search(r"checkpoint-(\d+)", path)
        return int(match.group(1)) if match else 0
    checkpoints.sort(key=get_step)
    return checkpoints[-1]

def main():
    print(f"🚀 Starting DeBERTa OOF Generation...")
    login(token=args.hf_token)
    # 1. 加载数据
    df = pd.read_csv(args.data_path).fillna("")
    
    oof_results = []
    
    # 2. 循环处理 5 个 Fold
    for fold in range(5):
        print(f"\n================ Processing FOLD {fold} ================")
        
        # A. 确定模型路径
        repo_id = f"{args.hf_user}/{args.model_prefix}{fold}"
        print(f"   -> Loading Model from Hub: {repo_id}")
        
        try:
            # B. 直接从 Hub 加载模型与 Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(repo_id)
            model = AutoModelForSequenceClassification.from_pretrained(
                repo_id, 
                num_labels=3,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            model.eval()
        except Exception as e:
            print(f"❌ Error: Failed to load model from {repo_id}. Skipping... Error: {e}")
            continue
        
        # C. 准备验证集
        val_df = df[df['fold'] == fold].copy()
        print(f"   -> Inference on {len(val_df)} samples")
        
        val_ds = Dataset.from_pandas(val_df)
        
        # D. 预处理
        def preprocess(examples):
            prompts = [str(x) for x in examples["prompt_text"]]
            res_a = [str(x) for x in examples["res_a_text"]]
            res_b = [str(x) for x in examples["res_b_text"]]
            sep = tokenizer.sep_token
            combined = [f"{p} {sep} {a} {sep} {b}" for p, a, b in zip(prompts, res_a, res_b)]
            return tokenizer(combined, truncation=True, max_length=args.max_len, padding=False)
        
        val_ds = val_ds.map(preprocess, batched=True, num_proc=4)
        
        # E. 推理
        training_args = TrainingArguments(
            output_dir="./tmp_deberta_infer", 
            per_device_eval_batch_size=args.batch_size,
            bf16=True, # 开启 BF16
            report_to="none",
            dataloader_num_workers=4
        )
        
        trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer)
        preds_output = trainer.predict(val_ds)
        
        logits = torch.tensor(preds_output.predictions)
        probs = softmax(logits.float(), dim=-1).numpy()
        
        val_df['pred_a'] = probs[:, 0]
        val_df['pred_b'] = probs[:, 1]
        val_df['pred_tie'] = probs[:, 2]
        
        oof_results.append(val_df)
        
        # 释放显存
        del model, tokenizer, trainer
        torch.cuda.empty_cache()

    # 3. 合并与保存
    if len(oof_results) == 5:
        # [关键] 恢复原始顺序
        oof_full = pd.concat(oof_results).sort_index()
        
        # 验证行数
        if len(oof_full) != len(df):
            print(f"⚠️ Row count mismatch!")
        
        # 保存
        oof_full.to_csv(args.output_path, index=False)
        print(f"\n✅ DeBERTa OOF Predictions saved to: {args.output_path}")
        
        # --- [新增] 最终 CV 评估 ---
        y_true = oof_full[['winner_model_a', 'winner_model_b', 'winner_tie']].idxmax(axis=1).map({
            'winner_model_a': 0, 'winner_model_b': 1, 'winner_tie': 2
        })
        y_pred = oof_full[['pred_a', 'pred_b', 'pred_tie']].values
        
        loss = log_loss(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred.argmax(axis=1))
        
        print("-" * 30)
        print(f"🏆 Final 5-Fold CV Score:")
        print(f"   Log Loss: {loss:.5f}")
        print(f"   Accuracy: {acc:.2%}")
        print("-" * 30)
    else:
        print(f"❌ OOF Generation Incomplete. Found {len(oof_results)}/5 folds.")

if __name__ == "__main__":
    main()