#!/bin/bash

# 1. 路径与环境
PYTHON_EXEC="/root/autodl-tmp/envs/llm_finetune/bin/python"
PROJECT_ROOT="/root/autodl-tmp/llm_classification_finetuning"
cd "$PROJECT_ROOT" || exit 1

export HF_HOME="/root/autodl-tmp/.cache/huggingface"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "🚀 Starting Qwen 14B Serial Training (Fold 0-4)..."

# 2. 循环跑 5 个 Fold
for fold in {0..4}
# for fold in 2
do
    echo "=================================================="
    echo "▶️  Running Qwen FOLD $fold"
    echo "=================================================="
    
    mkdir -p outputs/logs
    LOG_FILE="outputs/logs/qwen_fold${fold}.log"
    
    # 使用优化后的 python 脚本
    $PYTHON_EXEC src/train/train_qwen.py \
        --fold $fold \
        --model_name "/root/autodl-tmp/llm_classification_finetuning/base_models/Qwen3-14B" \
        --max_len 3072 \
        --learning_rate 1e-4 \
        > "$LOG_FILE" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "❌ Fold $fold Failed! Check $LOG_FILE"
    else
        echo "✅ Fold $fold Completed."
    fi
    
    # 清理显存缓存
    $PYTHON_EXEC -c "import torch; torch.cuda.empty_cache()"
done

echo "🏆 All Qwen Folds Finished!"