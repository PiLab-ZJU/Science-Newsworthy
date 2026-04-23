"""
Analysis 4: Error analysis with news cross-reference.

For False Positives and False Negatives, analyze patterns and
cross-reference with actual news coverage.

Usage:
    python scripts/analysis_4_error_analysis.py
"""
import json, sys
from pathlib import Path
from collections import Counter

def main():
    analysis_dir = Path("/mnt/nvme1/lcx/academic_social_impact/analysis")

    with open(analysis_dir / "test_predictions_with_explanations.json") as f:
        predictions = json.load(f)

    # Load news
    news_path = Path("/home/ubuntu/pilab_jiang/lcx/academic_new_policy/data/raw/news_text/news_articles.json")
    with open(news_path) as f:
        news = json.load(f)
    doi_news = {a["doi"]: a for a in news if a.get("success")}

    # Categorize
    tp = [p for p in predictions if p["true_label"] == 1 and p["predicted"] == 1]
    tn = [p for p in predictions if p["true_label"] == 0 and p["predicted"] == 0]
    fp = [p for p in predictions if p["true_label"] == 0 and p["predicted"] == 1]
    fn = [p for p in predictions if p["true_label"] == 1 and p["predicted"] == 0]

    print(f"TP={len(tp)} TN={len(tn)} FP={len(fp)} FN={len(fn)}")

    # === False Positive Analysis ===
    print(f"\n{'='*60}")
    print(f"FALSE POSITIVES (model says Yes, actually No): {len(fp)}")
    print(f"{'='*60}")

    fp_fields = Counter(p["field"] for p in fp)
    print("\nBy field:")
    for field, count in fp_fields.most_common(10):
        total_in_field = sum(1 for p in predictions if p["field"] == field and p["true_label"] == 0)
        rate = count / total_in_field if total_in_field > 0 else 0
        print(f"  {field:<45s} {count:>4d} (FP rate: {rate:.2%})")

    print("\nSample FP cases:")
    for p in fp[:5]:
        print(f"  Title: {p['title'][:80]}...")
        print(f"  Field: {p['field']}")
        print(f"  Explanation: {p['explanation'][:150]}...")
        print()

    # === False Negative Analysis ===
    print(f"\n{'='*60}")
    print(f"FALSE NEGATIVES (model says No, actually Yes): {len(fn)}")
    print(f"{'='*60}")

    fn_fields = Counter(p["field"] for p in fn)
    print("\nBy field:")
    for field, count in fn_fields.most_common(10):
        total_in_field = sum(1 for p in predictions if p["field"] == field and p["true_label"] == 1)
        rate = count / total_in_field if total_in_field > 0 else 0
        print(f"  {field:<45s} {count:>4d} (FN rate: {rate:.2%})")

    # FN with news available - what did the news say?
    print("\nFN cases WITH news coverage (what did model miss?):")
    fn_with_news = 0
    case_studies = []
    for p in fn:
        if p["doi"] in doi_news:
            fn_with_news += 1
            if len(case_studies) < 5:
                article = doi_news[p["doi"]]
                news_text = article.get("text", "")[:300]
                case_studies.append({
                    "title": p["title"],
                    "field": p["field"],
                    "model_explanation": p["explanation"],
                    "news_excerpt": news_text,
                })

    print(f"  FN with available news: {fn_with_news}/{len(fn)}")
    for cs in case_studies:
        print(f"\n  Title: {cs['title'][:80]}...")
        print(f"  Field: {cs['field']}")
        print(f"  Model said: {cs['model_explanation'][:150]}...")
        print(f"  News said:  {cs['news_excerpt'][:150]}...")

    # Save
    results = {
        "summary": {"tp": len(tp), "tn": len(tn), "fp": len(fp), "fn": len(fn)},
        "fp_by_field": dict(fp_fields),
        "fn_by_field": dict(fn_fields),
        "fp_samples": [{"title": p["title"], "field": p["field"], "explanation": p["explanation"]} for p in fp[:20]],
        "fn_samples": [{"title": p["title"], "field": p["field"], "explanation": p["explanation"]} for p in fn[:20]],
        "fn_case_studies": case_studies,
    }

    out_path = analysis_dir / "error_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
