"""
Compute comprehensive evaluation metrics from prediction results.

Generates overall, per-subfield, and per-year metrics.

Usage:
    python evaluation/metrics.py --predictions outputs/predictions/medicine/media_test_predictions.json
"""
import os
import sys
import json
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    classification_report, confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUTS_DIR


def compute_metrics(labels, predictions):
    """Compute all evaluation metrics."""
    labels = np.array(labels)
    predictions = np.array(predictions)
    results = {
        "n_samples": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    results["confusion_matrix"] = cm.tolist()
    results["true_positive"] = int(cm[1, 1]) if cm.shape == (2, 2) else 0
    results["true_negative"] = int(cm[0, 0]) if cm.shape == (2, 2) else 0
    results["false_positive"] = int(cm[0, 1]) if cm.shape == (2, 2) else 0
    results["false_negative"] = int(cm[1, 0]) if cm.shape == (2, 2) else 0

    return results


def compute_grouped_metrics(data: list, group_key: str) -> dict:
    """Compute metrics grouped by a specific key (e.g., subfield, year)."""
    groups = defaultdict(lambda: {"labels": [], "predictions": []})

    for item in data:
        key = item.get(group_key, "unknown")
        if not key:
            key = "unknown"
        groups[key]["labels"].append(item["label"])
        groups[key]["predictions"].append(item["prediction"])

    results = {}
    for key, group in sorted(groups.items()):
        if len(group["labels"]) >= 10:  # Only report groups with sufficient samples
            results[key] = compute_metrics(group["labels"], group["predictions"])

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON")
    args = parser.parse_args()

    with open(args.predictions, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = [d["label"] for d in data]
    predictions = [d["prediction"] for d in data]

    # Overall metrics
    print("=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)
    overall = compute_metrics(labels, predictions)
    for k, v in overall.items():
        if k != "confusion_matrix":
            print(f"  {k:20s}: {v}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={overall['true_negative']}  FP={overall['false_positive']}")
    print(f"    FN={overall['false_negative']}  TP={overall['true_positive']}")

    # Per-subfield metrics
    print(f"\n{'='*60}")
    print("PER-SUBFIELD METRICS")
    print("=" * 60)
    subfield_metrics = compute_grouped_metrics(data, "subfield")
    for sf, metrics in subfield_metrics.items():
        print(f"\n  {sf} (n={metrics['n_samples']}):")
        print(f"    F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}  "
              f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}")

    # Per-year metrics
    print(f"\n{'='*60}")
    print("PER-YEAR METRICS")
    print("=" * 60)
    year_metrics = compute_grouped_metrics(data, "year")
    for year, metrics in year_metrics.items():
        print(f"  {year} (n={metrics['n_samples']}): "
              f"F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}")

    # Full classification report
    print(f"\n{'='*60}")
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(labels, predictions, target_names=["No", "Yes"]))

    # Save all results
    all_results = {
        "overall": overall,
        "per_subfield": subfield_metrics,
        "per_year": year_metrics,
    }

    out_path = Path(args.predictions).with_suffix(".metrics.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull metrics saved to {out_path}")


if __name__ == "__main__":
    main()
