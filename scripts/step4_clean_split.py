"""
Step 4: Clean data and split into train/val/test.

- Merge positive and negative papers
- Deduplicate by DOI
- Filter by abstract length
- Author-based split to prevent data leakage
- Save per-field and combined datasets

Usage:
    python scripts/step4_clean_split.py
"""
import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
)


def load_and_label(pos_path, neg_path, year_start=2017, year_end=2023):
    """Load positive and negative papers, add labels, filter by year."""
    with open(pos_path) as f:
        positives = json.load(f)
    with open(neg_path) as f:
        negatives = json.load(f)

    years = set(str(y) for y in range(year_start, year_end + 1))

    for p in positives:
        p["label"] = 1
    for p in negatives:
        p["label"] = 0

    # Filter by year
    positives = [p for p in positives if p.get("publication_date", "")[:4] in years]
    negatives = [p for p in negatives if p.get("publication_date", "")[:4] in years]

    return positives, negatives


def deduplicate(samples):
    """Remove duplicates by DOI."""
    seen = set()
    unique = []
    for s in samples:
        doi = s.get("doi", "")
        if doi and doi not in seen:
            seen.add(doi)
            unique.append(s)
    return unique


def filter_quality(samples, min_abstract=100, min_title=10):
    """Filter out low-quality samples."""
    filtered = []
    for s in samples:
        abstract = s.get("abstract") or ""
        title = s.get("title") or ""
        if len(abstract) >= min_abstract and len(title) >= min_title:
            filtered.append(s)
    return filtered


def author_split(samples, seed=RANDOM_SEED):
    """Split by first author to prevent data leakage."""
    rng = np.random.RandomState(seed)

    author_groups = defaultdict(list)
    for s in samples:
        author_id = s.get("first_author_id", "") or s.get("doi", "")
        author_groups[author_id].append(s)

    authors = list(author_groups.keys())
    rng.shuffle(authors)

    total = len(samples)
    train_target = int(total * TRAIN_RATIO)
    val_target = int(total * VAL_RATIO)

    train, val, test = [], [], []
    count = 0

    for author in authors:
        group = author_groups[author]
        if count < train_target:
            train.extend(group)
        elif count < train_target + val_target:
            val.extend(group)
        else:
            test.extend(group)
        count += len(group)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def print_split_stats(name, data):
    """Print statistics for a split."""
    pos = sum(1 for s in data if s["label"] == 1)
    neg = len(data) - pos
    ratio = pos / len(data) if data else 0
    print(f"    {name:6s}: {len(data):>7,d} samples (pos={pos:,d}, neg={neg:,d}, pos_ratio={ratio:.2%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year_start", type=int, default=2017)
    parser.add_argument("--year_end", type=int, default=2023)
    args = parser.parse_args()

    pos_path = RAW_DATA_DIR / "openalex" / "news_papers.json"
    neg_path = RAW_DATA_DIR / "openalex" / "negative_papers.json"

    print("Loading data...")
    positives, negatives = load_and_label(pos_path, neg_path, args.year_start, args.year_end)
    print(f"  Positives (after year filter): {len(positives):,d}")
    print(f"  Negatives (after year filter): {len(negatives):,d}")

    # Merge
    all_samples = positives + negatives

    # Dedup
    before = len(all_samples)
    all_samples = deduplicate(all_samples)
    print(f"  After dedup: {len(all_samples):,d} (removed {before - len(all_samples):,d})")

    # Quality filter
    before = len(all_samples)
    all_samples = filter_quality(all_samples)
    print(f"  After quality filter: {len(all_samples):,d} (removed {before - len(all_samples):,d})")

    # Overall stats
    pos_count = sum(1 for s in all_samples if s["label"] == 1)
    neg_count = len(all_samples) - pos_count
    print(f"\n  Final: {len(all_samples):,d} samples (pos={pos_count:,d}, neg={neg_count:,d})")

    # === Save combined (all fields) ===
    print(f"\n{'='*60}")
    print("Combined dataset (all fields)")
    print("=" * 60)

    output_dir = PROCESSED_DATA_DIR / "combined"
    output_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = author_split(all_samples)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        print_split_stats(name, split)
        path = output_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)

    # === Save per-field datasets ===
    print(f"\n{'='*60}")
    print("Per-field datasets")
    print("=" * 60)

    field_samples = defaultdict(list)
    for s in all_samples:
        field = s.get("field", "Unknown")
        if field:
            field_samples[field].append(s)

    field_stats = []
    for field, samples in sorted(field_samples.items(), key=lambda x: -len(x[1])):
        pos = sum(1 for s in samples if s["label"] == 1)
        neg = len(samples) - pos

        if len(samples) < 100:
            continue

        field_dir = PROCESSED_DATA_DIR / field.lower().replace(" ", "_").replace(",", "")
        field_dir.mkdir(parents=True, exist_ok=True)

        train, val, test = author_split(samples)

        print(f"\n  {field} ({len(samples):,d} samples, pos={pos:,d}, neg={neg:,d}):")
        for name, split in [("train", train), ("val", val), ("test", test)]:
            print_split_stats(name, split)
            path = field_dir / f"{name}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(split, f, ensure_ascii=False, indent=2)

        field_stats.append({
            "field": field,
            "total": len(samples),
            "positive": pos,
            "negative": neg,
            "train": len(train),
            "val": len(val),
            "test": len(test),
        })

    # Save field stats summary
    stats_path = PROCESSED_DATA_DIR / "field_stats.json"
    with open(stats_path, "w") as f:
        json.dump(field_stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All datasets saved to {PROCESSED_DATA_DIR}")
    print(f"Field stats: {stats_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
