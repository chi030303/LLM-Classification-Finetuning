#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --job-name=qwen_train

source /scratch-shared/tc1proj043/miniconda3/etc/profile.d/conda.sh
conda activate llm_finetune

# 4卡并行训练 Fold 0
# 注意：QLoRA 多卡训练有时需要 DeepSpeed，或者直接用 Accelerate
accelerate launch --multi_gpu --num_processes 4 src/train_qwen.py \
    --fold 0 \
    --model_name "Qwen/Qwen3-8B" \
    --output_dir "/scratch-shared/tc1proj043/llm_classification_finetuning/outputs/models/qwen"