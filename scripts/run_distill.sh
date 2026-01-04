#!/bin/bash
PYTHON_EXEC="/root/autodl-tmp/envs/llm_finetune/bin/python"
PROJECT_ROOT="/root/autodl-tmp/llm_classification_finetuning"
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
mkdir -p outputs/logs

echo "🚀 Starting Qwen-4B Distillation for Fold 0..."
LOG_FILE="outputs/logs/distill_qwen4b_fold0.log"

$PYTHON_EXEC src/train/train_qwen_distill.py \
    --fold 0 \
    --student_model "/root/autodl-tmp/llm_classification_finetuning/base_models/Qwen3-8B" \
    --data_path "data/processed/train_with_teacher_labels.csv" \
    --output_dir "outputs/models/qwen_8b_distilled" \
    --max_len 2048 \
    > "$LOG_FILE" 2>&1 &