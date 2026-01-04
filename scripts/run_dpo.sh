#!/bin/bash

# 1. 环境与路径
PYTHON_EXEC="/root/autodl-tmp/envs/llm_finetune/bin/python"
PROJECT_ROOT="/root/autodl-tmp/llm_classification_finetuning"
cd "$PROJECT_ROOT" || exit 1

export HF_HOME="/root/autodl-tmp/.cache/huggingface"
export PYTHONUNBUFFERED=1

# 2. 定义日志文件路径
mkdir -p outputs/logs
LOG_FILE="outputs/logs/dpo_qwen_fold2.log"

echo "🚀 Starting DPO Training... Logs will be saved to $LOG_FILE"

# 3. 运行 Python 脚本，并重定向输出
# > "$LOG_FILE": 把标准输出 (print) 写入到日志文件
# 2>&1: 把标准错误 (报错信息) 也一并写入到同一个文件
$PYTHON_EXEC src/train/train_dpo.py > "$LOG_FILE" 2>&1