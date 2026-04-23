"""
RQ3 Core: Build text signal taxonomy from CoT explanations.

Steps:
1. Collect CoT explanations from correct predictions on test set
2. Use LLM to cluster/categorize explanation themes
3. Build media vs policy signal comparison

Usage:
    python analysis/signal_taxonomy.py --predictions outputs/predictions/medicine/media_test_predictions.json
"""
import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUTS_DIR

# Pre-defined signal categories (to be refined with data)
MEDIA_SIGNAL_CATEGORIES = [
    "public_health_relevance",
    "everyday_consumer_impact",
    "novelty_surprise",
    "emotional_human_interest",
    "actionable_takeaway",
    "controversy_debate",
    "large_population_scale",
    "clear_causal_claim",
    "relatable_topic",
    "visual_dramatic_finding",
]

POLICY_SIGNAL_CATEGORIES = [
    "regulatory_relevance",
    "population_level_evidence",
    "cost_effectiveness",
    "policy_recommendation",
    "current_policy_debate",
    "methodological_rigor",
    "intervention_evaluation",
    "health_system_impact",
    "equity_disparity",
    "environmental_impact",
]


def extract_correct_predictions(predictions: list) -> list:
    """Get samples where model predicted correctly."""
    return [p for p in predictions if p["prediction"] == p["label"] and p.get("explanation")]


def categorize_explanations_rule_based(explanations: list, task: str) -> dict:
    """
    Rule-based keyword matching for initial categorization.
    This serves as a baseline; LLM-based categorization follows.
    """
    categories = MEDIA_SIGNAL_CATEGORIES if task == "media" else POLICY_SIGNAL_CATEGORIES
    keyword_map = {
        "public_health_relevance": ["public health", "health outcome", "mortality", "morbidity", "disease"],
        "everyday_consumer_impact": ["diet", "exercise", "consumer", "lifestyle", "daily", "food", "beverage"],
        "novelty_surprise": ["novel", "first", "surprising", "unexpected", "unprecedented", "breakthrough"],
        "emotional_human_interest": ["children", "patient", "suffering", "death", "cancer", "mental health"],
        "actionable_takeaway": ["recommendation", "should", "advise", "guideline", "practical"],
        "controversy_debate": ["controversial", "debate", "disagree", "opposing", "conflict"],
        "large_population_scale": ["population", "nationwide", "global", "millions", "cohort", "large-scale"],
        "clear_causal_claim": ["cause", "causal", "leads to", "results in", "associated with"],
        "relatable_topic": ["common", "everyday", "widespread", "prevalent", "familiar"],
        "visual_dramatic_finding": ["dramatic", "significant increase", "doubled", "tripled", "stark"],
        "regulatory_relevance": ["regulation", "regulatory", "legislation", "law", "mandate", "ban"],
        "population_level_evidence": ["epidemiolog", "prevalence", "incidence", "surveillance", "population"],
        "cost_effectiveness": ["cost", "economic", "budget", "spending", "financial", "affordable"],
        "policy_recommendation": ["policy", "recommend", "implement", "strategy", "framework"],
        "current_policy_debate": ["reform", "policy debate", "legislative", "government", "federal"],
        "methodological_rigor": ["meta-analysis", "systematic review", "randomized", "RCT", "rigorous"],
        "intervention_evaluation": ["intervention", "program", "effectiveness", "efficacy", "trial"],
        "health_system_impact": ["healthcare system", "hospital", "clinical practice", "healthcare delivery"],
        "equity_disparity": ["disparity", "inequality", "equity", "underserved", "vulnerable", "minority"],
        "environmental_impact": ["environment", "climate", "pollution", "emission", "sustainability"],
    }

    signal_counts = Counter()
    sample_signals = []

    for item in explanations:
        text = item["explanation"].lower()
        item_signals = []
        for cat in categories:
            keywords = keyword_map.get(cat, [])
            if any(kw in text for kw in keywords):
                signal_counts[cat] += 1
                item_signals.append(cat)
        sample_signals.append({
            "id": item["id"],
            "signals": item_signals,
            "label": item["label"],
        })

    return {
        "signal_counts": dict(signal_counts),
        "sample_signals": sample_signals,
    }


def compare_media_policy_signals(media_results: dict, policy_results: dict) -> dict:
    """Compare signal distributions between media and policy tasks."""
    media_counts = media_results["signal_counts"]
    policy_counts = policy_results["signal_counts"]

    all_signals = set(list(media_counts.keys()) + list(policy_counts.keys()))
    comparison = {}
    for signal in sorted(all_signals):
        m = media_counts.get(signal, 0)
        p = policy_counts.get(signal, 0)
        comparison[signal] = {
            "media_count": m,
            "policy_count": p,
            "dominant": "media" if m > p else ("policy" if p > m else "equal"),
        }

    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--media_predictions", help="Path to media predictions JSON")
    parser.add_argument("--policy_predictions", help="Path to policy predictions JSON")
    parser.add_argument("--task", choices=["media", "policy"], help="Single task mode")
    parser.add_argument("--predictions", help="Single predictions file")
    args = parser.parse_args()

    output_dir = OUTPUTS_DIR / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.media_predictions and args.policy_predictions:
        # Comparison mode
        with open(args.media_predictions, "r", encoding="utf-8") as f:
            media_preds = json.load(f)
        with open(args.policy_predictions, "r", encoding="utf-8") as f:
            policy_preds = json.load(f)

        media_correct = extract_correct_predictions(media_preds)
        policy_correct = extract_correct_predictions(policy_preds)

        print(f"Media correct predictions with explanations: {len(media_correct)}")
        print(f"Policy correct predictions with explanations: {len(policy_correct)}")

        media_results = categorize_explanations_rule_based(media_correct, "media")
        policy_results = categorize_explanations_rule_based(policy_correct, "policy")

        comparison = compare_media_policy_signals(media_results, policy_results)

        print(f"\n{'='*60}")
        print("Signal Comparison (Media vs Policy)")
        print(f"{'='*60}")
        for signal, info in comparison.items():
            arrow = "<-MEDIA" if info["dominant"] == "media" else ("POLICY->" if info["dominant"] == "policy" else "==")
            print(f"  {signal:35s}  M={info['media_count']:4d}  P={info['policy_count']:4d}  {arrow}")

        with open(output_dir / "signal_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

    elif args.task and args.predictions:
        # Single task mode
        with open(args.predictions, "r", encoding="utf-8") as f:
            preds = json.load(f)

        correct = extract_correct_predictions(preds)
        print(f"Correct predictions with explanations: {len(correct)}")

        results = categorize_explanations_rule_based(correct, args.task)

        print(f"\nSignal distribution for {args.task}:")
        for signal, count in sorted(results["signal_counts"].items(), key=lambda x: -x[1]):
            print(f"  {signal:35s}  {count}")

        with open(output_dir / f"{args.task}_signals.json", "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
