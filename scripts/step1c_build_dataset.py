"""
Step 1C: Build balanced positive/negative datasets from Altmetric-labeled papers.

Takes papers_with_altmetric.json and creates media_data.json / policy_data.json
with balanced positive and negative samples.

Usage:
    python scripts/step1c_build_dataset.py
"""
import os
import sys
import json
import argparse
import random
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_DATA_DIR, PRIMARY_FIELD_NAME, RANDOM_SEED,
    MEDIA_POSITIVE_COUNT, MEDIA_NEGATIVE_COUNT,
    POLICY_POSITIVE_COUNT, POLICY_NEGATIVE_COUNT,
)


def build_task_dataset(papers: list, count_field: str, pos_count: int, neg_count: int, seed: int):
    """Build balanced dataset for one task (media or policy)."""
    rng = random.Random(seed)

    positive = [p for p in papers if p.get(count_field, 0) > 0]
    negative = [p for p in papers if p.get(count_field, 0) == 0]

    print(f"  Available: {len(positive)} positive, {len(negative)} negative")

    # Sample
    if len(positive) > pos_count:
        positive = rng.sample(positive, pos_count)
    if len(negative) > neg_count:
        # Stratified sampling: match subfield distribution of positive set
        pos_subfields = Counter(p.get("subfield", "") for p in positive)
        neg_by_subfield = {}
        for p in negative:
            sf = p.get("subfield", "")
            neg_by_subfield.setdefault(sf, []).append(p)

        sampled_neg = []
        total_pos = sum(pos_subfields.values())
        for sf, count in pos_subfields.items():
            target = int(neg_count * count / total_pos)
            available = neg_by_subfield.get(sf, [])
            if len(available) >= target:
                sampled_neg.extend(rng.sample(available, target))
            else:
                sampled_neg.extend(available)

        # Fill remaining from any subfield
        remaining = neg_count - len(sampled_neg)
        if remaining > 0:
            used_ids = {p["id"] for p in sampled_neg}
            leftover = [p for p in negative if p["id"] not in used_ids]
            sampled_neg.extend(rng.sample(leftover, min(remaining, len(leftover))))

        negative = sampled_neg

    print(f"  Sampled:   {len(positive)} positive, {len(negative)} negative")
    return positive, negative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field_name", default=PRIMARY_FIELD_NAME)
    args = parser.parse_args()

    field_dir = args.field_name.lower().replace(" ", "_")
    input_path = RAW_DATA_DIR / field_dir / "papers_with_altmetric.json"
    output_dir = RAW_DATA_DIR / field_dir

    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run step1b_altmetric_labels.py first.")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    print(f"Loaded {len(papers)} papers with Altmetric data")

    # Media task
    print(f"\n{'='*40} MEDIA {'='*40}")
    media_pos, media_neg = build_task_dataset(
        papers, "news_count",
        MEDIA_POSITIVE_COUNT, MEDIA_NEGATIVE_COUNT,
        RANDOM_SEED,
    )
    media_path = output_dir / "media_data.json"
    with open(media_path, "w", encoding="utf-8") as f:
        json.dump({"positive": media_pos, "negative": media_neg}, f, ensure_ascii=False, indent=2)
    print(f"  Saved to {media_path}")

    # Policy task
    print(f"\n{'='*40} POLICY {'='*40}")
    policy_pos, policy_neg = build_task_dataset(
        papers, "policy_count",
        POLICY_POSITIVE_COUNT, POLICY_NEGATIVE_COUNT,
        RANDOM_SEED,
    )
    policy_path = output_dir / "policy_data.json"
    with open(policy_path, "w", encoding="utf-8") as f:
        json.dump({"positive": policy_pos, "negative": policy_neg}, f, ensure_ascii=False, indent=2)
    print(f"  Saved to {policy_path}")

    print(f"\nDone! Next: run step2_clean_data.py")


if __name__ == "__main__":
    main()
