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
    BitsAndBytesConfig,
    DataCollatorWithPadding
)
from peft import PeftModel
from datasets import Dataset
from torch.nn.functional import softmax

# --- 配置 ---
parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default="/root/autodl-tmp/base_models/Qwen3-14B")
parser.add_argument("--data_path", type=str, default="data/processed/train_with_folds.csv")
parser.add_argument("--adapters_dir", type=str, default="outputs/models/qwen_14b") 
parser.add_argument("--output_path", type=str, default="data/processed/oof_qwen_14b.csv")
parser.add_argument("--max_len", type=int, default=2048)
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()

def get_best_checkpoint(fold_dir):
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
            pass
    checkpoints = glob.glob(os.path.join(fold_dir, "checkpoint-*"))
    if not checkpoints: return None
    def get_step(path):
        match = re.search(r"checkpoint-(\d+)", path)
        return int(match.group(1)) if match else 0
    checkpoints.sort(key=get_step)
    return checkpoints[-1]

def main():
    print(f"🚀 Starting Qwen OOF Generation...")
    df = pd.read_csv(args.data_path).fillna("")
    oof_results = []
    
    for fold in range(5):
        print(f"\n================ Processing FOLD {fold} ================")
        
        fold_adapter_dir = f"{args.adapters_dir}_fold{fold}"
        adapter_path = get_best_checkpoint(fold_adapter_dir)
        
        if not adapter_path:
            print(f"❌ Error: No Adapter found for Fold {fold}. Skipping...")
            continue
            
        print(f"   -> Loading Adapter: {adapter_path}")
        print("   -> Reloading CLEAN 4-bit Base Model...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        base_tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
        if base_tokenizer.pad_token is None:
            base_tokenizer.pad_token = base_tokenizer.eos_token
            
        # 🔥🔥🔥 核心修复：必须是 left padding !!! 🔥🔥🔥
        # 否则模型读取的是 pad token 的 embedding，导致结果为随机猜测
        base_tokenizer.padding_side = 'left'

        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model_path,
            num_labels=3,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        base_model.config.pad_token_id = base_tokenizer.pad_token_id
        
        model = PeftModel.from_pretrained(base_model, adapter_path)
        # 强制设为 eval 模式，避免 dropout 影响结果
        model.eval()
        
        val_df = df[df['fold'] == fold].copy()
        val_ds = Dataset.from_pandas(val_df)
        
        def preprocess(examples):
            inputs = []
            for p, a, b in zip(examples["prompt_text"], examples["res_a_text"], examples["res_b_text"]):
                # ⚠️ 请确保这里的 Prompt 格式和训练时是一个字都不差的！
                # 任何换行符或空格的差异都会导致性能下降，但不会导致随机猜测(Loss 1.35)
                # 随机猜测纯粹是因为 Padding Side 错了。
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
            return base_tokenizer(inputs, truncation=True, max_length=args.max_len, padding=False)
        
        val_ds = val_ds.map(preprocess, batched=True)
        
        training_args = TrainingArguments(
            output_dir="./tmp_qwen_infer",
            per_device_eval_batch_size=args.batch_size,
            bf16=True,
            report_to="none",
            dataloader_num_workers=4
        )
        
        # 显式使用 DataCollatorWithPadding 确保使用 tokenizer 的 padding_side 设置
        data_collator = DataCollatorWithPadding(tokenizer=base_tokenizer)
        
        trainer = Trainer(
            model=model, 
            args=training_args, 
            tokenizer=base_tokenizer,
            data_collator=data_collator
        )
        
        preds_output = trainer.predict(val_ds)
        
        logits = torch.tensor(preds_output.predictions)
        # 使用 float32 进行 softmax 保证精度
        probs = softmax(logits.float(), dim=-1).numpy()
        
        val_df['pred_a'] = probs[:, 0]
        val_df['pred_b'] = probs[:, 1]
        val_df['pred_tie'] = probs[:, 2]
        
        oof_results.append(val_df)
        
        del model, base_model, trainer
        torch.cuda.empty_cache()

    if len(oof_results) == 5:
        oof_full = pd.concat(oof_results).sort_values('fold').reset_index(drop=True)
        oof_full.to_csv(args.output_path, index=False)
        print(f"\n✅ Qwen OOF Predictions saved to: {args.output_path}")
        
        # 计算一下 CV 看看
        from sklearn.metrics import log_loss
        labels = oof_full[['winner_model_a', 'winner_model_b', 'winner_tie']].values
        preds = oof_full[['pred_a', 'pred_b', 'pred_tie']].values
        # 只有当 label 是 one-hot 时 log_loss 才准确，或者自己手动实现
        # 这里简单打印一下
        try:
             # 将 label 转换为 class index (0, 1, 2)
            y_true = np.argmax(labels, axis=1)
            loss = log_loss(y_true, preds)
            print(f"📊 Estimated CV LogLoss: {loss:.4f}")
        except:
            pass
    else:
        print("❌ No OOF generated.")

if __name__ == "__main__":
    main()