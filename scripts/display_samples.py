import pandas as pd
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

def display_samples(csv_path, num_samples=3):
    """
    Reads a CSV and prints formatted samples to the terminal.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {csv_path}")
        return

    # 随机抽取样本
    samples = df.sample(n=num_samples, random_state=42)
    
    console = Console()

    for i, (_, row) in enumerate(samples.iterrows()):
        
        # --- 1. 准备数据 ---
        prompt = str(row.get('prompt', 'N/A'))
        response_a = str(row.get('response_a', 'N/A'))
        response_b = str(row.get('response_b', 'N/A'))
        
        # 确定胜者
        winner = "Tie"
        if row.get('winner_model_a') == 1:
            winner = "Model A"
        elif row.get('winner_model_b') == 1:
            winner = "Model B"
        
        # --- 2. 创建一个大面板 (Panel) ---
        title = f"Sample #{i+1} | Winner: [bold green]{winner}[/bold green]"
        
        # --- 3. 创建一个表格来放 Prompt, ResA, ResB ---
        grid = Table.grid(expand=True)
        grid.add_column(width=45) # 左栏
        grid.add_column(width=45) # 右栏

        # --- 4. 格式化 Prompt (高亮代码) ---
        prompt_panel = Panel(
            Syntax(prompt, "python", theme="monokai", line_numbers=False, word_wrap=True),
            title="[bold yellow]Prompt[/bold yellow]",
            border_style="yellow"
        )
        
        # --- 5. 格式化 Response A & B ---
        response_a_panel = Panel(
            Text(response_a, justify="left"),
            title="[bold cyan]Response A[/bold cyan]",
            border_style="cyan"
        )
        
        response_b_panel = Panel(
            Text(response_b, justify="left"),
            title="[bold magenta]Response B[/bold magenta]",
            border_style="magenta"
        )
        
        # --- 6. 组合 ---
        # Prompt 独占一行
        grid.add_row(prompt_panel)
        # ResA 和 ResB 并排
        grid.add_row(response_a_panel, response_b_panel)
        
        # --- 7. 在大面板里打印表格 ---
        final_panel = Panel(grid, title=title, border_style="white")
        console.print(final_panel)
        console.print("\n" + "="*90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display formatted samples from a CSV file.")
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file.")
    parser.add_argument("-n", "--num_samples", type=int, default=3, help="Number of samples to display.")
    
    args = parser.parse_args()
    
    display_samples(args.csv_path, args.num_samples)