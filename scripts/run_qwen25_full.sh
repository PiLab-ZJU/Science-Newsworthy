#!/bin/bash
# Full pipeline: Qwen2.5-7B-Instruct training + evaluation
set -e

cd /home/ubuntu/pilab_jiang/lcx/LLaMA-Factory
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sft_cxlin

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OPENBLAS_NUM_THREADS=8

CONFIG=/home/ubuntu/pilab_jiang/lcx/academic_new_policy/configs/qwen25_combined_r32.yaml
ADAPTER=/mnt/nvme1/lcx/academic_social_impact/qwen25_combined_r32
BASE_MODEL=/mnt/nvme1/hf-model/Qwen2.5-7B-Instruct
LOG_DIR=/home/ubuntu/pilab_jiang/lcx/academic_new_policy/logs
SCRIPTS=/home/ubuntu/pilab_jiang/lcx/academic_new_policy/scripts

echo "========================================" >> $LOG_DIR/qwen25_full.log
echo "$(date): Starting Qwen2.5 training" >> $LOG_DIR/qwen25_full.log
echo "========================================" >> $LOG_DIR/qwen25_full.log

# Step 1: Training
echo "$(date): Training started" >> $LOG_DIR/qwen25_full.log
llamafactory-cli train $CONFIG >> $LOG_DIR/qwen25_train.log 2>&1
echo "$(date): Training completed" >> $LOG_DIR/qwen25_full.log

# Step 2: Evaluation on test set (4 GPU parallel)
echo "$(date): Evaluation started" >> $LOG_DIR/qwen25_full.log
cd /home/ubuntu/pilab_jiang/lcx/academic_new_policy
python -u $SCRIPTS/step6_evaluate_fast.py \
    --adapter_path $ADAPTER \
    --base_model $BASE_MODEL \
    --field combined \
    --gpus 0,1,2,3 >> $LOG_DIR/qwen25_eval.log 2>&1
echo "$(date): Evaluation completed" >> $LOG_DIR/qwen25_full.log

# Step 3: Generate explanations for analysis
echo "$(date): Generating explanations" >> $LOG_DIR/qwen25_full.log
python -u $SCRIPTS/analysis_1_generate_explanations.py \
    --adapter_path $ADAPTER \
    --base_model $BASE_MODEL \
    --gpus 0,1,2,3 >> $LOG_DIR/qwen25_explanations.log 2>&1
echo "$(date): Explanations completed" >> $LOG_DIR/qwen25_full.log

echo "========================================" >> $LOG_DIR/qwen25_full.log
echo "$(date): All done!" >> $LOG_DIR/qwen25_full.log
echo "========================================" >> $LOG_DIR/qwen25_full.log
