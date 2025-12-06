#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --job-name=deberta_train
#SBATCH --output=/scratch-shared/tc1proj043/llm_classification_finetuning/outputs/logs/deberta_%j.log
#SBATCH --error=/scratch-shared/tc1proj043/llm_classification_finetuning/outputs/logs/deberta_%j.err

# 1. 获取命令行参数
# 第一个参数是 fold (默认为 0)
FOLD_ID=${1:-0}
# 第二个参数是 checkpoint 路径 (可选)
RESUME_PATH=$2

echo "Running Fold: $FOLD_ID"

# 2. 开启实时日志
export PYTHONUNBUFFERED=1 

# 3. 环境设置
source /scratch-shared/tc1proj043/miniconda3/etc/profile.d/conda.sh
conda activate llm_finetune

export HF_HUB_OFFLINE=1
export HF_HOME="/scratch-shared/tc1proj043/llm_classification_finetuning/code/.cache/huggingface"

PROJECT_ROOT="/scratch-shared/tc1proj043/llm_classification_finetuning"
cd "$PROJECT_ROOT" || exit 1

# 4. 构建 Python 命令
CMD="accelerate launch --num_processes 1 $PROJECT_ROOT/src/train/train_deberta.py \
    --fold $FOLD_ID \
    --model_name microsoft/deberta-v3-large \
    --data_path data/processed/train_with_folds.csv \
    --output_dir outputs/models/deberta_v3_large"

# [关键逻辑] 如果传入了第二个参数(checkpoint)，则拼接到命令后面
if [ -n "$RESUME_PATH" ]; then
    echo "Resuming from: $RESUME_PATH"
    CMD="$CMD --resume_from_checkpoint $RESUME_PATH"
fi

# 5. 执行命令
echo "Executing: $CMD"
eval $CMD