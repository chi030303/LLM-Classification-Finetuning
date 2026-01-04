import pandas as pd
import json
from tqdm import tqdm

# 配置
DATA_PATH = "data/processed/train_with_folds.csv"
OUTPUT_PATH = "data/processed/dpo_train_data.jsonl"

def main():
    print("🚀 Preparing DPO dataset...")
    df = pd.read_csv(DATA_PATH).fillna("")
    
    dpo_data = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # DPO 无法处理 Tie，必须跳过
        if row['winner_tie'] == 1:
            continue
            
        prompt = str(row['prompt_text'])
        
        # 确定哪个是 chosen，哪个是 rejected
        if row['winner_model_a'] == 1:
            chosen = str(row['res_a_text'])
            rejected = str(row['res_b_text'])
        else: # winner_model_b == 1
            chosen = str(row['res_b_text'])
            rejected = str(row['res_a_text'])
            
        dpo_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })
        
    # 保存为 jsonl 文件
    with open(OUTPUT_PATH, "w") as f:
        for item in dpo_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"✅ DPO data saved to {OUTPUT_PATH}. Total samples: {len(dpo_data)}")

if __name__ == "__main__":
    main()