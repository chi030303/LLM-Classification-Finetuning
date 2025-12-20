import pandas as pd

# 检查所有文件的行数是否都是 57477
FILES_TO_CHECK = [
    "data/processed/train_features_structured.parquet",
    "data/processed/oof_deberta_v3_large.csv",
    "data/processed/oof_qwen_14b.csv"
]

for f in FILES_TO_CHECK:
    if f.endswith('.parquet'):
        df = pd.read_parquet(f)
    else:
        df = pd.read_csv(f)
    print(f"File: {f} | Rows: {len(df)}")