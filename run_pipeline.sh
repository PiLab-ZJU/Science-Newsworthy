#!/bin/bash
# =============================================================================
# Full pipeline for Academic Social Impact Prediction
# =============================================================================
# Usage:
#   bash run_pipeline.sh          # Run all stages
#   bash run_pipeline.sh step0    # Run specific stage
# =============================================================================

set -e
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Export OpenAlex API key (set your key here or via environment)
# export OPENALEX_API_KEY="your_key_here"

stage=${1:-all}

# ============================================================
# Stage 0: Data Exploration
# ============================================================
if [[ "$stage" == "all" || "$stage" == "step0" ]]; then
    echo "========== Stage 0: Data Exploration =========="
    python scripts/step0_explore.py
fi

# ============================================================
# Stage 1: Data Collection
# ============================================================
if [[ "$stage" == "all" || "$stage" == "step1" ]]; then
    echo "========== Stage 1: Data Collection (Medicine) =========="
    python scripts/step1_fetch_data.py

    echo "========== Stage 1: Data Collection (Generalization Fields) =========="
    python scripts/step1_fetch_data.py --field_id fields/23 --field_name "Environmental Science" \
        --media_pos 2000 --media_neg 2000 --policy_pos 1000 --policy_neg 1000
    python scripts/step1_fetch_data.py --field_id fields/33 --field_name "Social Sciences" \
        --media_pos 2000 --media_neg 2000 --policy_pos 1000 --policy_neg 1000
fi

# ============================================================
# Stage 2: Data Cleaning
# ============================================================
if [[ "$stage" == "all" || "$stage" == "step2" ]]; then
    echo "========== Stage 2: Data Cleaning =========="
    python scripts/step2_clean_data.py
    python scripts/step2_clean_data.py --field_name "Environmental Science"
    python scripts/step2_clean_data.py --field_name "Social Sciences"
fi

# ============================================================
# Stage 3: CoT Generation
# ============================================================
if [[ "$stage" == "all" || "$stage" == "step3" ]]; then
    echo "========== Stage 3: CoT Generation =========="
    python scripts/step3_generate_cot.py --task media --split train
    python scripts/step3_generate_cot.py --task policy --split train
fi

# ============================================================
# Stage 4: SFT Data Formatting
# ============================================================
if [[ "$stage" == "all" || "$stage" == "step4" ]]; then
    echo "========== Stage 4: SFT Data Formatting =========="
    python scripts/step4_format_sft.py
    python scripts/step4_format_sft.py --no_cot  # For ablation A7
fi

# ============================================================
# Stage 5: Baseline Experiments
# ============================================================
if [[ "$stage" == "all" || "$stage" == "baselines" ]]; then
    echo "========== Stage 5: Baseline Experiments =========="
    python baselines/traditional_ml.py --task media
    python baselines/traditional_ml.py --task policy
    python baselines/scibert_baseline.py --task media
    python baselines/scibert_baseline.py --task policy
    # LLM baselines (requires API keys)
    # python baselines/llm_zeroshot.py --task media --model gpt4o --mode zero_shot --max_test 500
    # python baselines/llm_zeroshot.py --task media --model gpt4o --mode few_shot --max_test 500
    # python baselines/llm_zeroshot.py --task policy --model gpt4o --mode zero_shot --max_test 500
fi

# ============================================================
# Stage 6: SFT Training (via LLaMA-Factory)
# ============================================================
if [[ "$stage" == "all" || "$stage" == "train" ]]; then
    echo "========== Stage 6: SFT Training =========="
    echo "Copy SFT data to LLaMA-Factory/data/ directory:"
    echo "  cp data/sft/medicine/media_train.json LLaMA-Factory/data/"
    echo "  cp data/sft/medicine/policy_train.json LLaMA-Factory/data/"
    echo "  cp data/sft/medicine/joint_train.json LLaMA-Factory/data/"
    echo ""
    echo "Merge configs/dataset_info.json into LLaMA-Factory/data/dataset_info.json"
    echo ""
    echo "Then run:"
    echo "  llamafactory-cli train configs/media_lora_sft.yaml"
    echo "  llamafactory-cli train configs/policy_lora_sft.yaml"
    echo "  llamafactory-cli train configs/joint_lora_sft.yaml"
fi

# ============================================================
# Stage 7: Evaluation
# ============================================================
if [[ "$stage" == "all" || "$stage" == "eval" ]]; then
    echo "========== Stage 7: Evaluation =========="
    python evaluation/inference.py --task media --adapter_path outputs/media_lora_sft
    python evaluation/inference.py --task policy --adapter_path outputs/policy_lora_sft
    python evaluation/metrics.py --predictions outputs/predictions/medicine/media_test_predictions.json
    python evaluation/metrics.py --predictions outputs/predictions/medicine/policy_test_predictions.json
fi

# ============================================================
# Stage 8: Analysis
# ============================================================
if [[ "$stage" == "all" || "$stage" == "analysis" ]]; then
    echo "========== Stage 8: Analysis =========="
    python analysis/keyword_analysis.py --task media
    python analysis/keyword_analysis.py --task policy
    python analysis/keyword_analysis.py --task compare
    python analysis/signal_taxonomy.py \
        --media_predictions outputs/predictions/medicine/media_test_predictions.json \
        --policy_predictions outputs/predictions/medicine/policy_test_predictions.json
    python analysis/ablation.py --experiment all --task media
    python analysis/visualize.py --type all
fi

# ============================================================
# Stage 9: Cross-domain Evaluation
# ============================================================
if [[ "$stage" == "all" || "$stage" == "cross_domain" ]]; then
    echo "========== Stage 9: Cross-domain Evaluation =========="
    python evaluation/cross_domain.py --task media --adapter_path outputs/media_lora_sft
    python evaluation/cross_domain.py --task policy --adapter_path outputs/policy_lora_sft
fi

echo ""
echo "Pipeline complete!"
