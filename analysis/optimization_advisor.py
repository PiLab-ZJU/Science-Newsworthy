"""
Application experiment: Abstract optimization advisor.

Select papers predicted as "no social impact" but with high academic quality,
analyze missing signals, and generate optimization suggestions.

Usage:
    python analysis/optimization_advisor.py --predictions outputs/predictions/medicine/media_test_predictions.json
"""
import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUTS_DIR, MEDIA_INSTRUCTION, POLICY_INSTRUCTION

try:
    import anthropic
    HAS_CLAUDE = True
except ImportError:
    HAS_CLAUDE = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


OPTIMIZATION_PROMPT = """You are an expert in science communication and research impact.

A machine learning model predicted that the following high-quality academic paper will NOT receive {impact_type}.

Paper:
Title: {title}
Abstract: {abstract}

The model's analysis identified these missing signals:
{missing_signals}

Please provide 3-5 specific suggestions for how the authors could revise their abstract
to increase the paper's {impact_type} potential, WITHOUT changing the underlying research content.
Focus on framing, emphasis, and language choices.

Format your response as:
1. [Suggestion]
2. [Suggestion]
...

Then provide a revised version of the abstract that implements these suggestions."""


def find_optimization_candidates(predictions: list, min_citations: int = 50, max_candidates: int = 20) -> list:
    """
    Find papers that are predicted as "no impact" but have high academic citations
    (indicating they are high-quality but poorly communicated for public impact).
    """
    candidates = []
    for p in predictions:
        if p["prediction"] == 0 and p["label"] == 0:
            # True negatives with high citations
            candidates.append(p)

    # Sort by academic citation count (if available in data)
    candidates.sort(key=lambda x: x.get("cited_by_count", 0), reverse=True)
    return candidates[:max_candidates]


def generate_optimization(title: str, abstract: str, explanation: str, task: str) -> str:
    """Generate optimization suggestions using LLM."""
    impact_type = "mainstream media news coverage" if task == "media" else "policy document citations"

    prompt = OPTIMIZATION_PROMPT.format(
        impact_type=impact_type,
        title=title,
        abstract=abstract,
        missing_signals=explanation if explanation else "No specific signals identified.",
    )

    if HAS_CLAUDE:
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    elif HAS_OPENAI:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    else:
        return "[ERROR] No LLM API available. Install anthropic or openai package."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--task", choices=["media", "policy"], required=True)
    parser.add_argument("--max_candidates", type=int, default=10)
    args = parser.parse_args()

    with open(args.predictions, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    candidates = find_optimization_candidates(predictions, max_candidates=args.max_candidates)
    print(f"Found {len(candidates)} optimization candidates")

    results = []
    for i, candidate in enumerate(candidates):
        print(f"\n[{i+1}/{len(candidates)}] Processing: {candidate['title'][:80]}...")

        optimization = generate_optimization(
            candidate["title"],
            candidate.get("abstract", ""),
            candidate.get("explanation", ""),
            args.task,
        )

        results.append({
            "id": candidate["id"],
            "title": candidate["title"],
            "original_abstract": candidate.get("abstract", ""),
            "model_explanation": candidate.get("explanation", ""),
            "optimization_suggestions": optimization,
        })

    # Save
    output_dir = OUTPUTS_DIR / "analysis" / "optimization"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.task}_optimization_suggestions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nOptimization suggestions saved to {out_path}")


if __name__ == "__main__":
    main()
