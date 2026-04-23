"""
Analysis 3g: Three-way contrastive signal analysis.

1. Model signals: YES explanations vs NO explanations (log-odds ratio)
2. News signals: News articles vs paper abstracts (log-odds ratio)
3. Cross-validation: overlap between model signals and news signals

Usage:
    OPENBLAS_NUM_THREADS=8 python scripts/analysis_3g_contrastive.py
"""
import os, sys, json, re, math
import numpy as np
from pathlib import Path
from collections import Counter

os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

ANALYSIS_DIR = Path("/mnt/nvme1/lcx/academic_social_impact/analysis")
NEWS_PATH = Path("/home/ubuntu/pilab_jiang/lcx/academic_new_policy/data/raw/news_text/news_articles.json")
PAPERS_PATH = Path("/home/ubuntu/pilab_jiang/lcx/academic_new_policy/data/raw/openalex/news_papers.json")
PROC_PATH = Path("/home/ubuntu/pilab_jiang/lcx/academic_new_policy/data/processed/combined")

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "must", "need",
    "i", "we", "they", "he", "she", "it", "you", "me", "him", "her",
    "this", "that", "these", "those", "my", "your", "his", "its", "our",
    "of", "in", "to", "for", "with", "on", "at", "by", "from", "as",
    "into", "about", "between", "through", "during", "before", "after",
    "and", "or", "but", "not", "if", "than", "so", "because", "while",
    "also", "just", "very", "more", "most", "only", "even", "still",
    "such", "each", "both", "all", "any", "some", "no", "other",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "there", "here", "then", "now", "well", "however", "although",
    "whether", "since", "until", "unless", "yet", "already",
    # Common filler
    "paper", "study", "research", "article", "work", "results",
    "using", "used", "based", "method", "approach", "analysis",
    "data", "model", "figure", "table", "shown", "found", "reported",
}


def tokenize(text):
    """Simple tokenization: lowercase, keep alpha words > 2 chars."""
    return [w for w in re.findall(r'\b[a-z]{3,}\b', text.lower())
            if w not in STOPWORDS]


def log_odds_ratio(counts_a, counts_b, n_a, n_b, prior=1.0):
    """
    Compute log-odds ratio with informative Dirichlet prior.
    Positive = more in A, negative = more in B.
    """
    results = {}
    all_words = set(counts_a.keys()) | set(counts_b.keys())

    for word in all_words:
        fa = counts_a.get(word, 0) + prior
        fb = counts_b.get(word, 0) + prior
        odds_a = fa / (n_a - fa + prior)
        odds_b = fb / (n_b - fb + prior)
        lor = math.log2(odds_a / odds_b)

        # Z-score approximation
        var = 1.0 / fa + 1.0 / fb
        z = lor / math.sqrt(var)

        total = counts_a.get(word, 0) + counts_b.get(word, 0)
        if total >= 20:  # Min frequency threshold
            results[word] = {"lor": lor, "z": z, "freq_a": counts_a.get(word, 0),
                             "freq_b": counts_b.get(word, 0), "total": total}

    return results


def print_top_words(results, name_a, name_b, top_n=25):
    """Print top signal words for each side."""
    sorted_words = sorted(results.items(), key=lambda x: x[1]["z"], reverse=True)

    print(f"\n  Top {name_a} signals (highest log-odds):")
    print(f"  {'Word':<25s} {'Z-score':>8s} {name_a:>8s} {name_b:>8s} {'Total':>8s}")
    print(f"  {'-'*57}")
    for word, info in sorted_words[:top_n]:
        print(f"  {word:<25s} {info['z']:>8.2f} {info['freq_a']:>8d} {info['freq_b']:>8d} {info['total']:>8d}")

    print(f"\n  Top {name_b} signals (lowest log-odds):")
    print(f"  {'Word':<25s} {'Z-score':>8s} {name_a:>8s} {name_b:>8s} {'Total':>8s}")
    print(f"  {'-'*57}")
    for word, info in sorted_words[-top_n:]:
        print(f"  {word:<25s} {info['z']:>8.2f} {info['freq_a']:>8d} {info['freq_b']:>8d} {info['total']:>8d}")


def main():
    # ============================================================
    # 1. Model signals: YES explanations vs NO explanations
    # ============================================================
    print("=" * 60)
    print("1. MODEL SIGNALS: YES vs NO explanations")
    print("=" * 60)

    with open(ANALYSIS_DIR / "test_predictions_with_explanations.json") as f:
        predictions = json.load(f)

    yes_texts = [p["explanation"] for p in predictions if p["predicted"] == 1 and p["explanation"]]
    no_texts = [p["explanation"] for p in predictions if p["predicted"] == 0 and p["explanation"]]
    print(f"YES explanations: {len(yes_texts)}, NO explanations: {len(no_texts)}")

    yes_words = Counter()
    no_words = Counter()
    for t in yes_texts:
        yes_words.update(set(tokenize(t)))  # set() for document frequency
    for t in no_texts:
        no_words.update(set(tokenize(t)))

    model_lor = log_odds_ratio(yes_words, no_words, len(yes_texts), len(no_texts))
    print_top_words(model_lor, "YES", "NO", top_n=30)

    # ============================================================
    # 2. News signals: News articles vs paper abstracts
    # ============================================================
    print(f"\n{'='*60}")
    print("2. NEWS SIGNALS: News text vs Paper abstracts")
    print("=" * 60)

    # Load news
    with open(NEWS_PATH) as f:
        news = json.load(f)
    doi_news = {a["doi"]: a for a in news if a.get("success")}

    # Load papers
    with open(PAPERS_PATH) as f:
        papers = json.load(f)
    doi_title = {p["doi"]: p.get("title", "") for p in papers}
    doi_abstract = {p["doi"]: p.get("abstract", "") for p in papers}

    # Get matched pairs (news text, abstract) from test set TP
    news_texts_clean = []
    abstract_texts_clean = []
    for p in predictions:
        if p["true_label"] == 1 and p["predicted"] == 1 and p["doi"] in doi_news:
            article = doi_news[p["doi"]]
            text = article.get("text", "")
            title = doi_title.get(p["doi"], "")
            abstract = doi_abstract.get(p["doi"], "")
            if (p["doi"].lower() in text.lower() or
                    (title and title.lower() in text.lower())):
                news_texts_clean.append(text[:2000])
                abstract_texts_clean.append(abstract)

    print(f"Matched pairs: {len(news_texts_clean)}")

    news_words = Counter()
    abstract_words = Counter()
    for t in news_texts_clean:
        news_words.update(set(tokenize(t)))
    for t in abstract_texts_clean:
        abstract_words.update(set(tokenize(t)))

    news_lor = log_odds_ratio(news_words, abstract_words, len(news_texts_clean), len(abstract_texts_clean))
    print_top_words(news_lor, "News", "Abstract", top_n=30)

    # ============================================================
    # 3. Cross-validation: overlap analysis
    # ============================================================
    print(f"\n{'='*60}")
    print("3. CROSS-VALIDATION: Model signals vs News signals")
    print("=" * 60)

    # Get top model YES signals and top news signals
    model_yes_words = {w for w, info in model_lor.items() if info["z"] > 3.0}
    model_no_words = {w for w, info in model_lor.items() if info["z"] < -3.0}
    news_added_words = {w for w, info in news_lor.items() if info["z"] > 3.0}
    news_absent_words = {w for w, info in news_lor.items() if info["z"] < -3.0}

    print(f"\nModel YES signals (z>3): {len(model_yes_words)} words")
    print(f"Model NO signals (z<-3): {len(model_no_words)} words")
    print(f"News added signals (z>3): {len(news_added_words)} words")
    print(f"News absent signals (z<-3): {len(news_absent_words)} words")

    # Overlap
    overlap_yes_news = model_yes_words & news_added_words
    model_only = model_yes_words - news_added_words
    news_only = news_added_words - model_yes_words

    print(f"\n--- Shared signals (model YES ∩ news added): {len(overlap_yes_news)} ---")
    print(f"  {', '.join(sorted(overlap_yes_news)[:30])}")

    print(f"\n--- Model-only signals (model sees, news doesn't emphasize): {len(model_only)} ---")
    print(f"  {', '.join(sorted(model_only)[:30])}")

    print(f"\n--- News-only signals (news emphasizes, model misses): {len(news_only)} ---")
    print(f"  {', '.join(sorted(news_only)[:30])}")

    # Jaccard similarity
    union = model_yes_words | news_added_words
    jaccard = len(overlap_yes_news) / len(union) if union else 0
    print(f"\nJaccard similarity (model YES vs news added): {jaccard:.4f}")

    # Save
    results = {
        "model_yes_top30": [(w, model_lor[w]) for w in sorted(model_lor, key=lambda x: -model_lor[x]["z"])[:30]],
        "model_no_top30": [(w, model_lor[w]) for w in sorted(model_lor, key=lambda x: model_lor[x]["z"])[:30]],
        "news_added_top30": [(w, news_lor[w]) for w in sorted(news_lor, key=lambda x: -news_lor[x]["z"])[:30]],
        "news_absent_top30": [(w, news_lor[w]) for w in sorted(news_lor, key=lambda x: news_lor[x]["z"])[:30]],
        "shared_signals": sorted(overlap_yes_news),
        "model_only_signals": sorted(model_only),
        "news_only_signals": sorted(news_only),
        "jaccard": jaccard,
    }

    out_path = ANALYSIS_DIR / "contrastive_signals.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
