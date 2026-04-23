"""
Analysis 3j: Signal category ratio comparison on the SAME papers.

For each of the ~2900 triplet papers, check if each signal category
appears in the model explanation vs the news article.
Ratio > 1 = model overestimates, Ratio < 1 = model underestimates.

Usage:
    python scripts/analysis_3j_signal_ratio.py
"""
import os, sys, json, re
from pathlib import Path
from collections import Counter

ANALYSIS_DIR = Path("/mnt/nvme1/lcx/academic_social_impact/analysis")
NEWS_PATH = Path("/home/ubuntu/pilab_jiang/lcx/academic_new_policy/data/raw/news_text/news_articles.json")
PAPERS_PATH = Path("/home/ubuntu/pilab_jiang/lcx/academic_new_policy/data/raw/openalex/news_papers.json")

# Signal categories with expanded keyword lists (derived from data observation)
SIGNALS = {
    "Novelty & Surprise": [
        "novel", "new", "first", "discovery", "discover", "breakthrough",
        "surprising", "unexpected", "unprecedented", "remarkable", "groundbreaking",
    ],
    "Controversy & Debate": [
        "controversial", "controversy", "debate", "debated", "disputed",
        "contentious", "polariz", "disagree", "conflict", "provocative",
        "challenge", "question", "rethink", "overturn",
    ],
    "Emotional & Human Interest": [
        "death", "dying", "suffer", "fear", "anxiety", "tragic",
        "children", "child", "baby", "infant", "elderly", "vulnerable",
        "hope", "inspiring", "heartbreak", "crisis", "victim",
    ],
    "Public Health Impact": [
        "public health", "epidemic", "pandemic", "outbreak", "mortality",
        "disease burden", "prevention", "vaccination", "risk factor",
        "health risk", "exposure", "toxicity", "carcinogen",
    ],
    "Large-Scale Evidence": [
        "million", "billion", "nationwide", "national", "global",
        "population", "large-scale", "cohort", "meta-analysis",
        "systematic review", "longitudinal", "census",
    ],
    "Consumer & Daily Life": [
        "diet", "food", "exercise", "sleep", "weight", "smoking",
        "alcohol", "coffee", "sugar", "lifestyle", "consumer",
        "everyday", "daily", "household", "screen time",
    ],
    "Actionable & Practical": [
        "recommend", "advice", "guideline", "practical", "tip",
        "should", "prevent", "reduce risk", "improve", "intervention",
        "solution", "strategy", "treatment option",
    ],
    "Environmental & Climate": [
        "climate", "warming", "carbon", "emission", "pollution",
        "biodiversity", "extinction", "conservation", "endangered",
        "deforestation", "ocean", "sea level", "arctic", "coral",
    ],
    "Technology & Innovation": [
        "artificial intelligence", "robot", "machine learning", "algorithm",
        "autonomous", "drone", "virtual reality", "3d print",
        "renewable energy", "solar", "battery", "quantum comput",
    ],
    "Economic & Policy": [
        "cost", "economic", "gdp", "spending", "inequality",
        "poverty", "policy", "regulation", "government", "tax",
        "legislation", "reform", "budget",
    ],
    "Space & Astronomy": [
        "planet", "star", "galaxy", "universe", "asteroid",
        "mars", "moon", "spacecraft", "telescope", "cosmic",
        "black hole", "exoplanet", "nasa",
    ],
    "Animal & Species": [
        "species", "animal", "bird", "fish", "insect", "mammal",
        "whale", "dolphin", "shark", "dinosaur", "fossil", "extinct",
        "wildlife", "predator", "prey",
    ],
}


def check_signals(text, signals_dict):
    """Check which signal categories are present in text."""
    text_lower = text.lower()
    present = {}
    for signal, keywords in signals_dict.items():
        present[signal] = any(kw in text_lower for kw in keywords)
    return present


def main():
    # Load predictions with explanations
    with open(ANALYSIS_DIR / "test_predictions_with_explanations.json") as f:
        predictions = json.load(f)

    # Load abstracts
    proc = Path("/home/ubuntu/pilab_jiang/lcx/academic_new_policy/data/processed/combined/test.json")
    with open(proc) as f:
        test_data = json.load(f)
    doi_abstract = {d["doi"]: d.get("abstract", "") for d in test_data}

    # Load news
    with open(NEWS_PATH) as f:
        news = json.load(f)
    doi_news = {a["doi"]: a for a in news if a.get("success")}
    with open(PAPERS_PATH) as f:
        papers = json.load(f)
    doi_title = {p["doi"]: p.get("title", "") for p in papers}

    # Build triplets
    triplets = []
    for p in predictions:
        if p["true_label"] == 1 and p["predicted"] == 1 and p["doi"] in doi_news:
            abstract = doi_abstract.get(p["doi"], "")
            explanation = p.get("explanation", "")
            article = doi_news[p["doi"]]
            news_text = article.get("text", "")
            title = doi_title.get(p["doi"], "")

            if not (p["doi"].lower() in news_text.lower() or
                    (title and title.lower() in news_text.lower())):
                continue

            if abstract and explanation and news_text:
                triplets.append({
                    "abstract": abstract,
                    "explanation": explanation,
                    "news": news_text[:3000],
                    "field": p.get("field", ""),
                })

    print(f"Triplets: {len(triplets)}")

    # Count signals in each source
    model_counts = Counter()
    news_counts = Counter()
    abstract_counts = Counter()

    for t in triplets:
        m = check_signals(t["explanation"], SIGNALS)
        n = check_signals(t["news"], SIGNALS)
        a = check_signals(t["abstract"], SIGNALS)
        for signal in SIGNALS:
            if m[signal]: model_counts[signal] += 1
            if n[signal]: news_counts[signal] += 1
            if a[signal]: abstract_counts[signal] += 1

    # Compare
    print(f"\n{'='*90}")
    print(f"{'Signal':<30s} {'Abstract':>8s} {'Model':>8s} {'News':>8s} {'M/N Ratio':>10s} {'Interpretation':<20s}")
    print("-" * 90)

    results = []
    for signal in SIGNALS:
        a = abstract_counts[signal]
        m = model_counts[signal]
        n = news_counts[signal]
        ratio = m / max(n, 1)

        if ratio > 1.5:
            interp = "Model OVERESTIMATES"
        elif ratio < 0.67:
            interp = "Model UNDERESTIMATES"
        else:
            interp = "Aligned"

        print(f"  {signal:<28s} {a:>8d} {m:>8d} {n:>8d} {ratio:>10.2f} {interp:<20s}")

        results.append({
            "signal": signal,
            "abstract_count": a,
            "model_count": m,
            "news_count": n,
            "ratio": round(ratio, 2),
            "interpretation": interp,
        })

    # Per-field breakdown for top signals
    print(f"\n{'='*60}")
    print("Per-field signal ratio (Model/News) for key signals")
    print("=" * 60)

    from collections import defaultdict
    field_triplets = defaultdict(list)
    for t in triplets:
        field_triplets[t["field"]].append(t)

    key_signals = ["Novelty & Surprise", "Controversy & Debate",
                   "Emotional & Human Interest", "Large-Scale Evidence",
                   "Consumer & Daily Life", "Public Health Impact"]

    print(f"\n  {'Field':<35s}", end="")
    for sig in key_signals:
        print(f" {sig[:10]:>10s}", end="")
    print()
    print(f"  {'-'*95}")

    for field in sorted(field_triplets.keys(), key=lambda x: -len(field_triplets[x])):
        ft = field_triplets[field]
        if len(ft) < 30:
            continue
        print(f"  {field:<35s}", end="")
        for sig in key_signals:
            m = sum(1 for t in ft if any(kw in t["explanation"].lower() for kw in SIGNALS[sig]))
            n = sum(1 for t in ft if any(kw in t["news"].lower() for kw in SIGNALS[sig]))
            ratio = m / max(n, 1)
            print(f" {ratio:>10.2f}", end="")
        print()

    # Save
    out_path = ANALYSIS_DIR / "signal_ratios.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
