"""
Cross-domain generalization evaluation (RQ3 / Ablation A6).

Train on Medicine, test on Environmental Science and Social Sciences.

Usage:
    python evaluation/cross_domain.py --task media --adapter_path outputs/media_lora_sft
"""
import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROCESSED_DATA_DIR, OUTPUTS_DIR, GENERALIZATION_FIELDS,
    BASE_MODEL, MEDIA_INSTRUCTION, POLICY_INSTRUCTION,
)
from evaluation.inference import load_model, predict_single
from evaluation.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["media", "policy"], required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--base_model", default=BASE_MODEL)
    parser.add_argument("--max_samples", type=int, default=500)
    args = parser.parse_args()

    model, tokenizer = load_model(args.base_model, args.adapter_path)

    all_results = {}

    for field_id, field_name in GENERALIZATION_FIELDS.items():
        field_dir = field_name.lower().replace(" ", "_")
        test_path = PROCESSED_DATA_DIR / field_dir / args.task / "test.json"

        if not test_path.exists():
            print(f"\n[SKIP] {test_path} not found. Run step1_fetch_data.py for {field_name} first.")
            continue

        with open(test_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        if args.max_samples and len(test_data) > args.max_samples:
            test_data = test_data[:args.max_samples]

        print(f"\n{'='*60}")
        print(f"Cross-domain evaluation: {field_name} ({len(test_data)} samples)")
        print("=" * 60)

        labels, predictions = [], []
        for sample in tqdm(test_data, desc=field_name):
            result = predict_single(
                model, tokenizer, sample["title"], sample["abstract"], args.task
            )
            labels.append(sample["label"])
            predictions.append(result["prediction"])

        metrics = compute_metrics(labels, predictions)
        all_results[field_name] = metrics

        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1:       {metrics['f1']:.4f}")
        print(f"  MCC:      {metrics['mcc']:.4f}")

    # Save
    output_dir = OUTPUTS_DIR / "cross_domain"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.task}_cross_domain_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCross-domain results saved to {out_path}")


if __name__ == "__main__":
    main()
