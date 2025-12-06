#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1          # 推理只需要单卡
#SBATCH --time=01:00:00
#SBATCH --job-name=infer_oof
#SBATCH --output=outputs/logs/infer_%j.log
#SBATCH --error=outputs/logs/infer_%j.err

export PYTHONUNBUFFERED=1
source /scratch-shared/tc1proj043/miniconda3/etc/profile.d/conda.sh
conda activate llm_finetune

mkdir -p outputs/models/deberta_v3_large_fold0/

echo "Starting Inference..."
python src/inference_oof.py
echo "Inference script finished."