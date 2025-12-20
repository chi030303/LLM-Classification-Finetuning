import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# --- 配置 ---
FEATURE_PATH = "data/processed/train_features_structured.parquet"
OOF_PATH = "data/processed/oof_deberta_v3_large.csv"
# 为了展示，我们需要原始文本，从 train_with_folds.csv 里拿
SOURCE_PATH = "data/processed/train_with_folds.csv"

def find_conflict_samples(num_samples=3):
    print("🔍 正在寻找“直觉冲突”的样本...")

    # 1. 加载数据
    feats_df = pd.read_parquet(FEATURE_PATH)
    oof_df = pd.read_csv(OOF_PATH)
    source_df = pd.read_csv(SOURCE_PATH)
    
    # 合并 (假设行顺序一致，否则用 id merge)
    df = pd.concat([
        source_df[['prompt_text', 'res_a_text', 'res_b_text', 'winner_model_a', 'winner_model_b', 'winner_tie']],
        feats_df[['len_diff']],
        oof_df[['pred_a', 'pred_b']]
    ], axis=1)

    # 2. 定义“冲突”条件
    # 人类直觉：A 比 B 长很多 (len_diff > 500)
    # AI 直觉：DeBERTa 却认为 B 赢面更大 (pred_b > pred_a)
    conflict_df = df[
        (df['len_diff'] > 500) & 
        (df['pred_b'] > df['pred_a'])
    ].copy()

    if len(conflict_df) < num_samples:
        print("⚠️ 没找到足够多的冲突样本，请调整阈值。")
        return

    # 随机抽取
    samples = conflict_df.sample(n=num_samples, random_state=42)
    
    console = Console()
    console.print(f"[bold yellow]找到 {len(conflict_df)} 个冲突样本。展示其中 {num_samples} 个:[/bold yellow]\n")

    for i, (_, row) in enumerate(samples.iterrows()):
        # 准备数据
        winner = "A" if row['winner_model_a'] == 1 else "B" if row['winner_model_b'] == 1 else "Tie"
        
        # --- 创建一个表格来展示“直觉” ---
        intuition_table = Table(title="Intuition Analysis", show_header=True, header_style="bold magenta")
        intuition_table.add_column("Source", style="dim")
        intuition_table.add_column("Verdict", justify="center")
        intuition_table.add_column("Reason / Evidence")

        # 人类直觉
        intuition_table.add_row(
            "[bold cyan]Human Intuition[/bold cyan]", 
            "[bold red]A should win[/bold red]", 
            f"Response A is significantly longer (len_diff = {row['len_diff']:.0f})"
        )
        # AI 直觉
        intuition_table.add_row(
            "[bold yellow]AI Intuition (DeBERTa)[/bold yellow]", 
            "[bold green]B should win[/bold green]", 
            f"Model predicts B is better (Prob B = {row['pred_b']:.2f} > Prob A = {row['pred_a']:.2f})"
        )
        
        # 真实结果
        intuition_table.add_row(
            "[bold white]Ground Truth[/bold white]", 
            f"[bold blue]Winner is {winner}[/bold blue]",
            "This is who the Stacking model must learn to predict."
        )

        console.print(Panel(
            intuition_table,
            title=f"Conflict Case #{i+1}",
            border_style="white"
        ))
        
        # 打印部分文本供参考
        console.print(Panel(
            Text(f"Prompt: {str(row['prompt_text'])[:200]}..."),
            title="Context",
            border_style="dim"
        ))
        console.print("\n")

if __name__ == "__main__":
    find_conflict_samples()