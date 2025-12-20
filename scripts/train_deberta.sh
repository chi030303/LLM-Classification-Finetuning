#!/bin/bash

# 1. 路径与环境
PYTHON_EXEC="/root/autodl-tmp/envs/llm_finetune/bin/python"
PROJECT_ROOT="/root/autodl-tmp/llm_classification_finetuning"
cd "$PROJECT_ROOT" || exit 1

# 环境变量 (OFFLINE 保持开启，确保绝对不联网)
export HF_HOME="/root/autodl-tmp/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 定义本地模型路径 [关键修改]
MODEL_PATH="/root/autodl-tmp/base_models/deberta-v3-large"

echo "🚀 Starting Serial Training on RTX 6000 (Max Performance)..."

# 2. 循环跑 Fold 0 到 Fold 4 (串行)
for fold in {0..4}
do
    echo "----------------------------------------------------------------"
    echo "▶️  Running FOLD $fold"
    echo "----------------------------------------------------------------"
    
    mkdir -p outputs/logs
    LOG_FILE="outputs/logs/deberta_fold${fold}.log"
    
    # 3. 运行参数优化
    # max_len: 1800 (保持)
    # batch_size: 提升到 4 (显存够用)
    # grad_acc: 降低到 4 (保持总batch=16)
    # num_workers: 提升到 8 (单任务独占更多CPU)
    
    $PYTHON_EXEC src/train/train_deberta.py \
        --fold $fold \
        --model_name "$MODEL_PATH" \
        --data_path "data/processed/train_with_folds.csv" \
        --output_dir "outputs/models/deberta_v3_large" \
        --max_len 1800 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --num_train_epochs 3 \
        --learning_rate 5e-6 \
        --gradient_checkpointing \
        --bf16 \
        --dataloader_num_workers 8 \
        > "$LOG_FILE" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "❌ Fold $fold Failed! Check $LOG_FILE"
        exit 1
    fi
    
    echo "✅ Fold $fold Completed."
done

echo "🏆 All Folds Finished!"