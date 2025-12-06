#!/bin/bash
# scripts/submit_chain.sh
# 用法: bash scripts/submit_chain.sh

# 定义任务名和脚本路径
JOB_NAME="deberta_chain"
SCRIPT_PATH="scripts/train_deberta.sh"

echo "🚀 Launching Job Chain for $JOB_NAME..."

# --- 第 1 棒 ---
# 提交第一个任务，并获取 Job ID
# --parsable 让 sbatch 只输出纯数字 ID
JOB_ID_1=$(sbatch --parsable --job-name=${JOB_NAME}_1 $SCRIPT_PATH)
echo "✅ Job 1 Submitted: $JOB_ID_1"

# --- 第 2 棒 ---
# 依赖 Job 1：无论 Job 1 是跑完了还是超时被杀 (afterany)，Job 2 都会启动
# Job 2 会自动读取 checkpoints 目录，继续训练
JOB_ID_2=$(sbatch --parsable --dependency=afterany:$JOB_ID_1 --job-name=${JOB_NAME}_2 $SCRIPT_PATH)
echo "✅ Job 2 Submitted: $JOB_ID_2 (Depends on $JOB_ID_1)"

# --- 第 3 棒 (保险起见) ---
JOB_ID_3=$(sbatch --parsable --dependency=afterany:$JOB_ID_2 --job-name=${JOB_NAME}_3 $SCRIPT_PATH)
echo "✅ Job 3 Submitted: $JOB_ID_3 (Depends on $JOB_ID_2)"

echo "🎉 Chain complete! Monitor logs at outputs/logs/"