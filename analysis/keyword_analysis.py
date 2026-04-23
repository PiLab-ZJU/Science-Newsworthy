"""
Keyword-level contrastive analysis using chi-squared test.

Extracts high-discrimination keywords between:
- Media positive vs negative
- Policy positive vs negative
- Media positive vs Policy positive

Usage:
    python analysis/keyword_analysis.py --task media
"""
import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import chi2_contingency

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, OUTPUTS_DIR, PRIMARY_FIELD_NAME


def chi_squared_keywords(texts_a: list, texts_b: list, top_n: int = 50) -> list:
    """
    Find keywords that best discriminate between two text groups
    using chi-squared test.
    """
    vectorizer = CountVectorizer(
        max_features=5000, stop_words="english",
        ngram_range=(1, 2), min_df=5,
    )
    all_texts = texts_a + texts_b
    X = vectorizer.fit_transform(all_texts)
    feature_names = vectorizer.get_feature_names_out()

    n_a = len(texts_a)
    results = []

    for i, word in enumerate(feature_names):
        col = X[:, i].toarray().flatten()
        a_present = col[:n_a].sum()
        a_absent = n_a - a_present
        b_present = col[n_a:].sum()
        b_absent = len(texts_b) - b_present

        contingency = np.array([[a_present, a_absent], [b_present, b_absent]])

        if contingency.min() == 0 and contingency.sum() < 10:
            continue

        try:
            chi2, p_value, _, _ = chi2_contingency(contingency)
            # Direction: positive means more frequent in group A
            ratio_a = a_present / n_a if n_a > 0 else 0
            ratio_b = b_present / len(texts_b) if len(texts_b) > 0 else 0

            results.append({
                "keyword": word,
                "chi2": float(chi2),
                "p_value": float(p_value),
                "freq_group_a": float(ratio_a),
                "freq_group_b": float(ratio_b),
                "direction": "A" if ratio_a > ratio_b else "B",
            })
        except ValueError:
            continue

    # Sort by chi2 score
    results.sort(key=lambda x: x["chi2"], reverse=True)
    return results[:top_n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["media", "policy", "compare"], default="compare")
    parser.add_argument("--field_name", default=PRIMARY_FIELD_NAME)
    parser.add_argument("--top_n", type=int, default=50)
    args = parser.parse_args()

    field_dir = args.field_name.lower().replace(" ", "_")
    output_dir = OUTPUTS_DIR / "analysis" / "keywords"
    output_dir.mkdir(parents=True, exist_ok=True)

    def load_texts(task, split="train"):
        path = PROCESSED_DATA_DIR / field_dir / task / f"{split}.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pos = [f"{s['title']} {s['abstract']}" for s in data if s["label"] == 1]
        neg = [f"{s['title']} {s['abstract']}" for s in data if s["label"] == 0]
        return pos, neg

    if args.task in ["media", "policy"]:
        pos, neg = load_texts(args.task)
        print(f"\n{args.task.upper()} positive vs negative")
        print(f"  Positive: {len(pos)} texts, Negative: {len(neg)} texts")

        results = chi_squared_keywords(pos, neg, args.top_n)

        print(f"\n  Top discriminating keywords (pos vs neg):")
        for r in results[:20]:
            dir_label = "POS" if r["direction"] == "A" else "NEG"
            print(f"    {r['keyword']:30s}  chi2={r['chi2']:8.1f}  "
                  f"p={r['p_value']:.2e}  dir={dir_label}  "
                  f"freq_pos={r['freq_group_a']:.3f}  freq_neg={r['freq_group_b']:.3f}")

        with open(output_dir / f"{args.task}_pos_vs_neg.json", "w") as f:
            json.dump(results, f, indent=2)

    elif args.task == "compare":
        # Compare media positive vs policy positive
        media_pos, _ = load_texts("media")
        policy_pos, _ = load_texts("policy")

        print(f"\nMedia positive vs Policy positive")
        print(f"  Media positive: {len(media_pos)}, Policy positive: {len(policy_pos)}")

        results = chi_squared_keywords(media_pos, policy_pos, args.top_n)

        print(f"\n  Top discriminating keywords (media vs policy):")
        for r in results[:20]:
            dir_label = "MEDIA" if r["direction"] == "A" else "POLICY"
            print(f"    {r['keyword']:30s}  chi2={r['chi2']:8.1f}  "
                  f"p={r['p_value']:.2e}  dir={dir_label}")

        with open(output_dir / "media_vs_policy_positive.json", "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
