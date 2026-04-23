"""
Step 3: Generate Chain-of-Thought explanations using Claude/GPT-4 API.

Given a paper's title, abstract, and known label, generate a CoT analysis
explaining WHY this paper would (or would not) attract media/policy attention.

Usage:
    python scripts/step3_generate_cot.py --task media --split train
    python scripts/step3_generate_cot.py --task policy --split train
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, PRIMARY_FIELD_NAME

# Try anthropic first, fallback to openai
try:
    import anthropic
    USE_CLAUDE = True
except ImportError:
    USE_CLAUDE = False

try:
    import openai
    USE_OPENAI = True
except ImportError:
    USE_OPENAI = False


MEDIA_COT_SYSTEM = """You are an expert in science communication and media studies.
Given an academic paper's title, abstract, and whether it actually received mainstream media news coverage,
generate a brief analysis (3-5 sentences) explaining the key factors that make this paper
likely or unlikely to attract media attention. Consider factors like:
- Public health relevance and everyday impact
- Novelty and surprise factor
- Emotional resonance and human interest
- Clear actionable takeaways for general audience
- Controversy or debate potential
- Scale of affected population

End with: "Prediction: Yes" or "Prediction: No" matching the known label."""

POLICY_COT_SYSTEM = """You are an expert in science policy and evidence-based policymaking.
Given an academic paper's title, abstract, and whether it was actually cited in policy documents,
generate a brief analysis (3-5 sentences) explaining the key factors that make this paper
likely or unlikely to be cited in policy documents. Consider factors like:
- Direct relevance to regulatory or legislative decisions
- Population-level evidence (epidemiology, public health)
- Cost-effectiveness or economic impact data
- Clear policy recommendations or implications
- Alignment with current policy debates
- Methodological rigor (RCT, meta-analysis, systematic review)

End with: "Prediction: Yes" or "Prediction: No" matching the known label."""


def generate_cot_claude(title: str, abstract: str, label: int, task: str) -> str:
    """Generate CoT using Claude API."""
    client = anthropic.Anthropic()
    system = MEDIA_COT_SYSTEM if task == "media" else POLICY_COT_SYSTEM
    label_str = "Yes (it DID receive media coverage)" if task == "media" and label == 1 else \
                "No (it did NOT receive media coverage)" if task == "media" and label == 0 else \
                "Yes (it WAS cited in policy documents)" if label == 1 else \
                "No (it was NOT cited in policy documents)"

    prompt = f"Title: {title}\nAbstract: {abstract}\n\nKnown outcome: {label_str}\n\nGenerate the analysis:"

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def generate_cot_openai(title: str, abstract: str, label: int, task: str) -> str:
    """Generate CoT using OpenAI API."""
    client = openai.OpenAI()
    system = MEDIA_COT_SYSTEM if task == "media" else POLICY_COT_SYSTEM
    label_str = "Yes (it DID receive media coverage)" if task == "media" and label == 1 else \
                "No (it did NOT receive media coverage)" if task == "media" and label == 0 else \
                "Yes (it WAS cited in policy documents)" if label == 1 else \
                "No (it was NOT cited in policy documents)"

    prompt = f"Title: {title}\nAbstract: {abstract}\n\nKnown outcome: {label_str}\n\nGenerate the analysis:"

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=512,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["media", "policy"], required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--field_name", default=PRIMARY_FIELD_NAME)
    parser.add_argument("--batch_size", type=int, default=50, help="Save checkpoint every N samples")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples to process")
    args = parser.parse_args()

    field_dir = args.field_name.lower().replace(" ", "_")
    input_path = PROCESSED_DATA_DIR / field_dir / args.task / f"{args.split}.json"
    output_path = PROCESSED_DATA_DIR / field_dir / args.task / f"{args.split}_cot.json"

    with open(input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if args.max_samples:
        samples = samples[:args.max_samples]

    # Resume from checkpoint
    processed = []
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            processed = json.load(f)
        processed_ids = {s["id"] for s in processed}
        samples = [s for s in samples if s["id"] not in processed_ids]
        print(f"Resuming: {len(processed)} already done, {len(samples)} remaining")

    # Select API
    if USE_CLAUDE:
        generate_fn = generate_cot_claude
        print("Using Claude API")
    elif USE_OPENAI:
        generate_fn = generate_cot_openai
        print("Using OpenAI API")
    else:
        print("ERROR: Neither anthropic nor openai package installed.")
        sys.exit(1)

    for i, sample in enumerate(tqdm(samples, desc=f"Generating CoT ({args.task})")):
        try:
            cot = generate_fn(sample["title"], sample["abstract"], sample["label"], args.task)
            sample["cot_explanation"] = cot
            processed.append(sample)
        except Exception as e:
            print(f"\n  Error on {sample['id']}: {e}")
            time.sleep(5)
            continue

        # Checkpoint
        if (i + 1) % args.batch_size == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(processed, f, ensure_ascii=False, indent=2)
            print(f"\n  Checkpoint saved: {len(processed)} samples")

        time.sleep(0.5)  # Rate limiting

    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    print(f"\nDone! {len(processed)} samples saved to {output_path}")


if __name__ == "__main__":
    main()
