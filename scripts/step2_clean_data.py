"""
Step 2: Data cleaning — deduplication, length filtering, author-based split.

Usage:
    python scripts/step2_clean_data.py
"""
import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
    PRIMARY_FIELD_NAME,
)


def deduplicate(samples: list) -> list:
    """Remove duplicates by OpenAlex work ID."""
    seen = set()
    unique = []
    for s in samples:
        wid = s["id"]
        if wid not in seen:
            seen.add(wid)
            unique.append(s)
    return unique


def filter_length(samples: list, min_abstract_len: int = 100, min_title_len: int = 10) -> list:
    """Filter out samples with too short titles or abstracts."""
    return [
        s for s in samples
        if len(s.get("abstract", "")) >= min_abstract_len
        and len(s.get("title", "")) >= min_title_len
    ]


def author_based_split(positive: list, negative: list, seed: int = RANDOM_SEED):
    """
    Split data by first author to prevent data leakage.
    Papers by the same first author always go into the same split.
    """
    rng = np.random.RandomState(seed)

    # Collect all unique first authors
    author_to_samples = defaultdict(list)
    for s in positive:
        s["label"] = 1
        author_to_samples[s.get("first_author_id", s["id"])].append(s)
    for s in negative:
        s["label"] = 0
        author_to_samples[s.get("first_author_id", s["id"])].append(s)

    # Shuffle author groups
    authors = list(author_to_samples.keys())
    rng.shuffle(authors)

    # Calculate split points by sample count
    total = sum(len(v) for v in author_to_samples.values())
    train_target = int(total * TRAIN_RATIO)
    val_target = int(total * VAL_RATIO)

    train, val, test = [], [], []
    cumulative = 0

    for author in authors:
        samples = author_to_samples[author]
        if cumulative < train_target:
            train.extend(samples)
        elif cumulative < train_target + val_target:
            val.extend(samples)
        else:
            test.extend(samples)
        cumulative += len(samples)

    # Shuffle within each split
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def process_task(task_name: str, raw_dir: Path, output_dir: Path):
    """Process one task (media or policy)."""
    raw_path = raw_dir / f"{task_name}_data.json"
    if not raw_path.exists():
        print(f"  [SKIP] {raw_path} not found")
        return

    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    positive = data["positive"]
    negative = data["negative"]

    print(f"\n  Raw: {len(positive)} positive + {len(negative)} negative")

    # Deduplication
    positive = deduplicate(positive)
    negative = deduplicate(negative)
    print(f"  After dedup: {len(positive)} positive + {len(negative)} negative")

    # Length filtering
    positive = filter_length(positive)
    negative = filter_length(negative)
    print(f"  After length filter: {len(positive)} positive + {len(negative)} negative")

    # Cross-dedup: remove papers that appear in both positive and negative
    pos_ids = {s["id"] for s in positive}
    neg_ids = {s["id"] for s in negative}
    overlap = pos_ids & neg_ids
    if overlap:
        print(f"  WARNING: {len(overlap)} papers in both pos and neg, removing from neg")
        negative = [s for s in negative if s["id"] not in overlap]

    # Author-based split
    train, val, test = author_based_split(positive, negative)
    print(f"  Split: train={len(train)}, val={len(val)}, test={len(test)}")

    # Check label distribution in each split
    for name, split in [("train", train), ("val", val), ("test", test)]:
        pos = sum(1 for s in split if s["label"] == 1)
        neg = len(split) - pos
        print(f"    {name}: {pos} pos + {neg} neg (pos ratio: {pos/len(split):.2%})")

    # Save
    task_dir = output_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = task_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)
        print(f"  Saved {path} ({len(split)} samples)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field_name", default=PRIMARY_FIELD_NAME)
    args = parser.parse_args()

    field_dir = args.field_name.lower().replace(" ", "_")
    raw_dir = RAW_DATA_DIR / field_dir
    output_dir = PROCESSED_DATA_DIR / field_dir

    print(f"Processing data for {args.field_name}")
    print(f"  Raw dir: {raw_dir}")
    print(f"  Output dir: {output_dir}")

    for task in ["media", "policy"]:
        print(f"\n{'='*40} {task.upper()} {'='*40}")
        process_task(task, raw_dir, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
