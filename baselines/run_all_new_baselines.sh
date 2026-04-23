#!/usr/bin/env bash
# Run every new baseline on the combined/ split.
# Assumes:
#   DATA:      /root/pilab_jiang/cxlin/academic_new_policy/data/processed/combined
#   SCIBERT:   /root/pilab_jiang/hf-model/scibert_scivocab_uncased
#   SPECTER2:  /root/pilab_jiang/hf-model/specter2_base (+ /specter2 adapter)

set -e

PROJECT=/root/pilab_jiang/cxlin/academic_new_policy
DATA=$PROJECT/data/processed/combined
SCIBERT=/root/pilab_jiang/hf-model/scibert_scivocab_uncased
SPECTER2_BASE=/root/pilab_jiang/hf-model/specter2_base
SPECTER2_ADAPTER=/root/pilab_jiang/hf-model/specter2

cd "$PROJECT"

mkdir -p logs/baselines

echo "========== 1. Trivial single-feature baselines =========="
python baselines/trivial_baselines.py --data_dir "$DATA" 2>&1 \
    | tee logs/baselines/trivial.log

echo "========== 2. Enhanced metadata XGBoost =========="
python baselines/enhanced_metadata_xgb.py --data_dir "$DATA" 2>&1 \
    | tee logs/baselines/enhanced_xgb.log

echo "========== 3. Wallace-style LR (TF-IDF + metadata) =========="
python baselines/wallace_lr.py --data_dir "$DATA" 2>&1 \
    | tee logs/baselines/wallace.log

echo "========== 4. SciBERT embedding + metadata + XGBoost =========="
python baselines/scibert_embed_xgb.py \
    --data_dir "$DATA" --model_path "$SCIBERT" --batch_size 64 2>&1 \
    | tee logs/baselines/scibert_embed_xgb.log

echo "========== 4b. SciBERT embedding only (ablation) =========="
python baselines/scibert_embed_xgb.py \
    --data_dir "$DATA" --model_path "$SCIBERT" --batch_size 64 --no_metadata 2>&1 \
    | tee logs/baselines/scibert_embed_only.log

echo "========== 5. SPECTER2 base + [PRX] adapter + LR =========="
python baselines/specter2_baseline.py \
    --data_dir "$DATA" \
    --base_path "$SPECTER2_BASE" \
    --adapter_path "$SPECTER2_ADAPTER" \
    --batch_size 64 2>&1 \
    | tee logs/baselines/specter2_adapter.log

echo "========== 5b. SPECTER2 base only (fallback) =========="
python baselines/specter2_baseline.py \
    --data_dir "$DATA" \
    --base_path "$SPECTER2_BASE" --no_adapter --batch_size 64 2>&1 \
    | tee logs/baselines/specter2_base.log

echo "========== 6. SciBERT fine-tuned (slowest, ~hours on 1 GPU) =========="
python baselines/scibert_baseline.py \
    --data_dir "$DATA" --model_name "$SCIBERT" \
    --epochs 3 --batch_size 32 --lr 2e-5 2>&1 \
    | tee logs/baselines/scibert_ft.log

echo ""
echo "All baselines done. Results:"
find outputs/baselines -name '*.json' -newer logs/baselines/trivial.log
