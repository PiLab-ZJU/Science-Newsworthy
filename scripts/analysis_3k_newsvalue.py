"""
Analysis 3k: Signal analysis based on news value theory (Harcup & O'Neill, 2017).

8 news value categories operationalized with keywords.
Compared on the SAME 2900+ triplet papers (abstract, model explanation, news).

References:
- Harcup, T., & O'Neill, D. (2017). What is news? News values revisited (again).
  Journalism Studies, 18(12), 1470-1488.
- Galtung, J., & Ruge, M. H. (1965). The structure of foreign news.
  Journal of Peace Research, 2(1), 64-90.

Usage:
    python scripts/analysis_3k_newsvalue.py
"""
import json, re
from pathlib import Path
from collections import Counter, defaultdict

ANALYSIS_DIR = Path("/mnt/nvme1/lcx/academic_social_impact/analysis")
NEWS_PATH = Path("/home/ubuntu/pilab_jiang/lcx/academic_new_policy/data/raw/news_text/news_articles.json")
PAPERS_PATH = Path("/home/ubuntu/pilab_jiang/lcx/academic_new_policy/data/raw/openalex/news_papers.json")

# News value categories based on Harcup & O'Neill (2017), operationalized for science news
NEWS_VALUES = {
    "Surprise": {
        "description": "Unexpected, surprising, or counter-intuitive findings",
        "keywords": [
            "surprising", "surprised", "unexpected", "unexpectedly",
            "counter-intuitive", "counterintuitive", "contrary to",
            "challenge", "challenged", "overturns", "rethink",
            "for the first time", "first time", "never before",
            "unprecedented", "remarkable", "unusual", "rare",
            "paradox", "mystery", "puzzle",
        ],
    },
    "Bad News": {
        "description": "Threats, risks, dangers, harm to health/environment",
        "keywords": [
            "risk", "danger", "dangerous", "threat", "threaten",
            "harm", "harmful", "toxic", "death", "mortality",
            "cancer", "disease", "epidemic", "pandemic", "crisis",
            "pollution", "contamination", "decline", "loss",
            "extinction", "collapse", "damage", "warning",
            "alarming", "concern", "worried",
        ],
    },
    "Good News": {
        "description": "Breakthroughs, cures, solutions, positive outcomes",
        "keywords": [
            "breakthrough", "cure", "solution", "solve", "solved",
            "success", "successful", "promising", "hope", "hopeful",
            "improve", "improved", "improvement", "benefit",
            "protect", "protection", "prevent", "prevention",
            "advance", "advancement", "progress", "discovery",
            "discover", "discovered", "innovation", "innovative",
        ],
    },
    "Magnitude": {
        "description": "Large scale, affecting many people, significant numbers",
        "keywords": [
            "million", "billion", "thousand", "percent",
            "global", "worldwide", "nationwide", "national",
            "population", "large-scale", "massive", "vast",
            "widespread", "common", "prevalent", "majority",
            "significant", "substantially", "dramatically",
            "double", "triple", "half",
        ],
    },
    "Relevance": {
        "description": "Directly relevant to audience's daily life",
        "keywords": [
            "diet", "food", "eat", "drink", "coffee", "alcohol",
            "exercise", "sleep", "weight", "obesity",
            "smoking", "screen", "phone", "social media",
            "children", "pregnancy", "aging", "elderly",
            "cost", "price", "afford", "income", "salary",
            "school", "education", "work", "workplace",
            "commut", "driving", "travel",
        ],
    },
    "Power Elite": {
        "description": "Involves prestigious institutions, authorities, WHO/CDC",
        "keywords": [
            "harvard", "oxford", "cambridge", "stanford", "mit",
            "nasa", "who", "cdc", "fda", "nih",
            "lancet", "nature", "science", "nejm",
            "world health", "united nations",
            "government", "federal", "congress", "parliament",
            "professor", "leading", "expert", "authority",
        ],
    },
    "Entertainment": {
        "description": "Amusing, quirky, fun, animal stories",
        "keywords": [
            "funny", "humor", "amusing", "quirky", "weird",
            "bizarre", "strange", "curious", "fascinating",
            "cute", "adorable", "pet", "dog", "cat",
            "dinosaur", "shark", "whale", "dolphin",
            "sex", "sexual", "love", "dating", "attraction",
            "chocolate", "beer", "wine", "pizza",
            "robot", "alien", "zombie",
        ],
    },
    "Conflict": {
        "description": "Scientific debate, conflicting evidence, policy disagreements",
        "keywords": [
            "debate", "debated", "controversial", "controversy",
            "conflict", "conflicting", "disagree", "dispute",
            "oppose", "opposition", "critic", "criticism",
            "skeptic", "question", "questioned", "doubt",
            "refute", "contradict", "versus",
        ],
    },
}


def check_signals(text):
    text_lower = text.lower()
    present = {}
    for signal, info in NEWS_VALUES.items():
        present[signal] = any(kw in text_lower for kw in info["keywords"])
    return present


def main():
    # Load predictions
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
    abs_counts = Counter()
    model_counts = Counter()
    news_counts = Counter()

    for t in triplets:
        a = check_signals(t["abstract"])
        m = check_signals(t["explanation"])
        n = check_signals(t["news"])
        for signal in NEWS_VALUES:
            if a[signal]: abs_counts[signal] += 1
            if m[signal]: model_counts[signal] += 1
            if n[signal]: news_counts[signal] += 1

    # Main results table
    n_papers = len(triplets)
    print(f"\n{'='*100}")
    print(f"News Value Signal Comparison (n={n_papers} papers)")
    print(f"Based on Harcup & O'Neill (2017) news value framework")
    print(f"{'='*100}")
    print(f"\n{'Signal':<25s} {'Abstract':>10s} {'Model':>10s} {'News':>10s} {'M/N':>8s} {'M vs N':>20s}")
    print(f"{'-'*85}")

    results = []
    for signal in NEWS_VALUES:
        a = abs_counts[signal]
        m = model_counts[signal]
        n = news_counts[signal]
        a_pct = a / n_papers * 100
        m_pct = m / n_papers * 100
        n_pct = n / n_papers * 100
        ratio = m / max(n, 1)

        if ratio > 1.5:
            interp = "OVERESTIMATES"
        elif ratio < 0.67:
            interp = "UNDERESTIMATES"
        else:
            interp = "Aligned"

        print(f"  {signal:<23s} {a:>5d}({a_pct:>4.1f}%) {m:>5d}({m_pct:>4.1f}%) "
              f"{n:>5d}({n_pct:>4.1f}%) {ratio:>8.2f} {interp:<20s}")

        results.append({
            "signal": signal,
            "description": NEWS_VALUES[signal]["description"],
            "abstract_count": a, "abstract_pct": round(a_pct, 1),
            "model_count": m, "model_pct": round(m_pct, 1),
            "news_count": n, "news_pct": round(n_pct, 1),
            "ratio": round(ratio, 2),
            "interpretation": interp,
        })

    # Per-field breakdown
    print(f"\n{'='*100}")
    print("Per-field Model/News ratio")
    print(f"{'='*100}")

    field_triplets = defaultdict(list)
    for t in triplets:
        field_triplets[t["field"]].append(t)

    signals_short = list(NEWS_VALUES.keys())
    header = f"{'Field':<30s}" + "".join(f"{s[:8]:>10s}" for s in signals_short)
    print(f"\n{header}")
    print(f"{'-'*(30+10*len(signals_short))}")

    for field in sorted(field_triplets.keys(), key=lambda x: -len(field_triplets[x])):
        ft = field_triplets[field]
        if len(ft) < 30:
            continue
        row = f"  {field[:28]:<28s}"
        for signal in signals_short:
            m = sum(1 for t in ft if check_signals(t["explanation"])[signal])
            n = sum(1 for t in ft if check_signals(t["news"])[signal])
            ratio = m / max(n, 1)
            row += f" {ratio:>9.2f}"
        print(row)

    # Save
    out_path = ANALYSIS_DIR / "newsvalue_signals.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
