"""
Analysis 3: Signal taxonomy — extract and compare model signals vs news signals.

1. From model explanations: categorize WHY model predicts Yes/No
2. From news articles: extract what journalists actually highlighted
3. Cross-validate: how well do model signals match news signals

Usage:
    python scripts/analysis_3_signal_taxonomy.py
"""
import json, re, sys
from pathlib import Path
from collections import Counter, defaultdict

def extract_signal_keywords(text):
    """Extract signal categories from explanation text."""
    text_lower = text.lower()
    signals = []

    signal_map = {
        "public_health": ["public health", "health outcome", "mortality", "disease", "cancer", "obesity", "diabetes", "mental health"],
        "consumer_relevance": ["diet", "food", "exercise", "lifestyle", "consumer", "daily life", "everyday"],
        "novelty": ["novel", "first", "new", "breakthrough", "unprecedented", "surprising", "unexpected"],
        "large_scale": ["population", "nationwide", "global", "large-scale", "cohort", "million", "large study"],
        "controversy": ["controversial", "debate", "disagree", "contentious", "conflict"],
        "policy_relevance": ["policy", "regulation", "government", "legislation", "guideline"],
        "environmental": ["climate", "environment", "pollution", "sustainability", "carbon", "ecosystem"],
        "technology": ["technology", "ai", "artificial intelligence", "robot", "digital", "innovation"],
        "emotional": ["children", "death", "suffering", "crisis", "tragedy", "fear", "risk"],
        "actionable": ["recommendation", "practical", "advice", "prevention", "treatment", "intervention"],
        "specialized_technical": ["specialized", "technical", "niche", "narrow", "specific methodology"],
        "limited_audience": ["limited audience", "limited interest", "narrow audience", "academic community only"],
    }

    for signal, keywords in signal_map.items():
        if any(kw in text_lower for kw in keywords):
            signals.append(signal)

    return signals


def main():
    analysis_dir = Path("/mnt/nvme1/lcx/academic_social_impact/analysis")

    # Load predictions with explanations
    with open(analysis_dir / "test_predictions_with_explanations.json") as f:
        predictions = json.load(f)

    # Load news articles
    news_path = Path("/home/ubuntu/pilab_jiang/lcx/academic_new_policy/data/raw/news_text/news_articles.json")
    with open(news_path) as f:
        news = json.load(f)
    doi_news = {a["doi"]: a for a in news if a.get("success")}

    # Load paper titles for matching
    papers_path = Path("/home/ubuntu/pilab_jiang/lcx/academic_new_policy/data/raw/openalex/news_papers.json")
    with open(papers_path) as f:
        papers = json.load(f)
    doi_title = {p["doi"]: p.get("title", "") for p in papers}

    print(f"Predictions: {len(predictions)}")
    print(f"News articles available: {len(doi_news)}")

    # === Model signal analysis ===
    print(f"\n{'='*60}")
    print("MODEL SIGNAL ANALYSIS")
    print(f"{'='*60}")

    tp_signals = Counter()  # True Positive explanations
    tn_signals = Counter()  # True Negative explanations
    fp_signals = Counter()
    fn_signals = Counter()

    for p in predictions:
        signals = extract_signal_keywords(p["explanation"])
        if p["true_label"] == 1 and p["predicted"] == 1:
            for s in signals: tp_signals[s] += 1
        elif p["true_label"] == 0 and p["predicted"] == 0:
            for s in signals: tn_signals[s] += 1
        elif p["true_label"] == 0 and p["predicted"] == 1:
            for s in signals: fp_signals[s] += 1
        elif p["true_label"] == 1 and p["predicted"] == 0:
            for s in signals: fn_signals[s] += 1

    print("\nSignals in TRUE POSITIVE predictions (model correctly says Yes):")
    for sig, count in tp_signals.most_common(10):
        print(f"  {sig:<25s} {count:>5d}")

    print("\nSignals in TRUE NEGATIVE predictions (model correctly says No):")
    for sig, count in tn_signals.most_common(10):
        print(f"  {sig:<25s} {count:>5d}")

    # === News signal analysis ===
    print(f"\n{'='*60}")
    print("NEWS SIGNAL ANALYSIS")
    print(f"{'='*60}")

    news_signals = Counter()
    matched_count = 0

    for p in predictions:
        if p["true_label"] == 1 and p["doi"] in doi_news:
            article = doi_news[p["doi"]]
            text = article.get("text", "")
            # Verify match
            title = doi_title.get(p["doi"], "")
            if p["doi"].lower() in text.lower() or (title and title.lower() in text.lower()):
                signals = extract_signal_keywords(text)
                for s in signals:
                    news_signals[s] += 1
                matched_count += 1

    print(f"\nMatched test predictions with news: {matched_count}")
    print("\nSignals in actual NEWS ARTICLES:")
    for sig, count in news_signals.most_common(10):
        print(f"  {sig:<25s} {count:>5d}")

    # === Cross-validation ===
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION: Model vs News signals")
    print(f"{'='*60}")

    all_signals = set(list(tp_signals.keys()) + list(news_signals.keys()))
    print(f"\n{'Signal':<25s} {'Model(TP)':>10s} {'News':>10s} {'Ratio':>10s}")
    print("-" * 57)

    comparison = []
    for sig in sorted(all_signals):
        m = tp_signals.get(sig, 0)
        n = news_signals.get(sig, 0)
        ratio = m / max(n, 1)
        print(f"  {sig:<23s} {m:>10d} {n:>10d} {ratio:>10.2f}")
        comparison.append({"signal": sig, "model_count": m, "news_count": n, "ratio": round(ratio, 2)})

    # Save
    results = {
        "tp_signals": dict(tp_signals),
        "tn_signals": dict(tn_signals),
        "fp_signals": dict(fp_signals),
        "fn_signals": dict(fn_signals),
        "news_signals": dict(news_signals),
        "comparison": comparison,
        "matched_count": matched_count,
    }

    out_path = analysis_dir / "signal_taxonomy.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
