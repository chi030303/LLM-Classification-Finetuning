#!/bin/bash

# 1. 路径与环境
PYTHON_EXEC="/root/autodl-tmp/envs/llm_finetune/bin/python"
PROJECT_ROOT="/root/autodl-tmp/llm_classification_finetuning"
cd "$PROJECT_ROOT" || exit 1

export HF_HOME="/root/autodl-tmp/.cache/huggingface"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

# --- [关键] 补跑 Fold 2 ---
FOLD_ID=2
echo "🚀 Resuming/Starting Qwen 14B Training for FOLD $FOLD_ID..."

# 2. 自动寻找最新的 Checkpoint
OUTPUT_DIR="outputs/models/qwen_14b_fold${FOLD_ID}"
LATEST_CHECKPOINT=$(ls -d ${OUTPUT_DIR}/checkpoint-*/ 2>/dev/null | sort -V | tail -n 1)

RESUME_ARG=""
if [ -d "$LATEST_CHECKPOINT" ]; then
    echo "🔄 Found checkpoint, resuming from: $LATEST_CHECKPOINT"
    # [关键] 构造续训参数
    RESUME_ARG="--resume_from_checkpoint $LATEST_CHECKPOINT"
else
    echo "🚀 No checkpoint found, starting new run for Fold $FOLD_ID."
fi

# 3. 日志文件
mkdir -p outputs/logs
LOG_FILE="outputs/logs/qwen_fold${FOLD_ID}_retry.log"

# 4. 运行命令 (加入显存优化参数)
# 96GB 显存跑 3072 长度，BS=8, GradAcc=2 是比较稳的配置
$PYTHON_EXEC src/train/train_qwen.py \
    --fold $FOLD_ID \
    --model_name "/root/autodl-tmp/base_models/Qwen3-14B" \
    --max_len 3072 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing \
    --bf16 \
    $RESUME_ARG \
    > "$LOG_FILE" 2>&1

if [ $? -ne 0 ]; then
    echo "❌ Fold $FOLD_ID Failed! Check $LOG_FILE"
else
    echo "✅ Fold $FOLD_ID Completed."
fi