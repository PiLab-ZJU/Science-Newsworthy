"""
Analysis 3i: Triplet contrastive analysis on the SAME set of papers.

For TP papers with matched news (~2900), compare:
1. Model explanation vs Abstract → model's added emphasis
2. News article vs Abstract → journalist's added emphasis
3. Model emphasis vs News emphasis → alignment

All three comparisons are on the SAME papers, making them directly comparable.

Usage:
    OPENBLAS_NUM_THREADS=8 python scripts/analysis_3i_triplet.py
"""
import os, sys, json, re, math
from pathlib import Path
from collections import Counter

os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

ANALYSIS_DIR = Path("/mnt/nvme1/lcx/academic_social_impact/analysis")
NEWS_PATH = Path("/home/ubuntu/pilab_jiang/lcx/academic_new_policy/data/raw/news_text/news_articles.json")
PAPERS_PATH = Path("/home/ubuntu/pilab_jiang/lcx/academic_new_policy/data/raw/openalex/news_papers.json")

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
    "using", "used", "based", "show", "shown", "found",
}


def tokenize(text):
    return [w for w in re.findall(r'\b[a-z]{3,}\b', text.lower()) if w not in STOPWORDS]


def get_doc_freq(texts):
    counter = Counter()
    for t in texts:
        counter.update(set(tokenize(t)))
    return counter


def log_odds_ratio(counts_a, counts_b, n_a, n_b, prior=1.0, min_freq=10):
    results = {}
    for word in set(counts_a.keys()) | set(counts_b.keys()):
        fa = counts_a.get(word, 0) + prior
        fb = counts_b.get(word, 0) + prior
        total = counts_a.get(word, 0) + counts_b.get(word, 0)
        if total < min_freq:
            continue
        lor = math.log2((fa / (n_a - fa + prior)) / (fb / (n_b - fb + prior)))
        z = lor / math.sqrt(1.0 / fa + 1.0 / fb)
        results[word] = {"z": round(z, 2), "freq_a": counts_a.get(word, 0),
                         "freq_b": counts_b.get(word, 0)}
    return results


def print_top(results, name_a, name_b, n=25):
    s = sorted(results.items(), key=lambda x: x[1]["z"], reverse=True)
    print(f"\n  '{name_a}' emphasis (top {n}):")
    print(f"  {'Word':<22s} {'Z':>7s} {name_a:>7s} {name_b:>7s}")
    print(f"  {'-'*45}")
    for w, info in s[:n]:
        print(f"  {w:<22s} {info['z']:>7.1f} {info['freq_a']:>7d} {info['freq_b']:>7d}")


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

    # Build triplets: (abstract, model_explanation, news_text) for same papers
    triplets = []
    for p in predictions:
        if p["true_label"] == 1 and p["predicted"] == 1 and p["doi"] in doi_news:
            abstract = doi_abstract.get(p["doi"], "")
            explanation = p.get("explanation", "")
            article = doi_news[p["doi"]]
            news_text = article.get("text", "")
            title = doi_title.get(p["doi"], "")

            # Verify news matches this paper
            if not (p["doi"].lower() in news_text.lower() or
                    (title and title.lower() in news_text.lower())):
                continue

            if abstract and explanation and news_text:
                triplets.append({
                    "doi": p["doi"],
                    "field": p.get("field", ""),
                    "abstract": abstract,
                    "explanation": explanation,
                    "news": news_text[:2000],
                })

    print(f"Triplets (same papers with abstract + explanation + news): {len(triplets)}")

    abstracts = [t["abstract"] for t in triplets]
    explanations = [t["explanation"] for t in triplets]
    news_texts = [t["news"] for t in triplets]
    n = len(triplets)

    abs_freq = get_doc_freq(abstracts)
    exp_freq = get_doc_freq(explanations)
    news_freq = get_doc_freq(news_texts)

    # ============================================================
    # 1. Model explanation vs Abstract (same papers)
    # ============================================================
    print(f"\n{'='*60}")
    print("1. MODEL EMPHASIS: What model highlights beyond abstract")
    print("=" * 60)

    model_vs_abs = log_odds_ratio(exp_freq, abs_freq, n, n)
    print_top(model_vs_abs, "Model", "Abstract")

    # ============================================================
    # 2. News vs Abstract (same papers)
    # ============================================================
    print(f"\n{'='*60}")
    print("2. NEWS EMPHASIS: What journalists highlight beyond abstract")
    print("=" * 60)

    news_vs_abs = log_odds_ratio(news_freq, abs_freq, n, n)
    print_top(news_vs_abs, "News", "Abstract")

    # ============================================================
    # 3. Model emphasis vs News emphasis (direct comparison)
    # ============================================================
    print(f"\n{'='*60}")
    print("3. ALIGNMENT: Model emphasis vs News emphasis")
    print("=" * 60)

    # Get signal words (emphasized beyond abstract) for each
    model_signals = {w for w, info in model_vs_abs.items() if info["z"] > 3.0}
    news_signals = {w for w, info in news_vs_abs.items() if info["z"] > 3.0}

    shared = model_signals & news_signals
    model_only = model_signals - news_signals
    news_only = news_signals - model_signals
    union = model_signals | news_signals
    jaccard = len(shared) / len(union) if union else 0

    print(f"\n  Model emphasis words (z>3): {len(model_signals)}")
    print(f"  News emphasis words (z>3):  {len(news_signals)}")
    print(f"  Shared:                     {len(shared)}")
    print(f"  Model-only:                 {len(model_only)}")
    print(f"  News-only:                  {len(news_only)}")
    print(f"  Jaccard similarity:         {jaccard:.4f}")

    # Rank shared by combined z-score
    shared_ranked = sorted(shared,
        key=lambda w: model_vs_abs.get(w, {}).get("z", 0) + news_vs_abs.get(w, {}).get("z", 0),
        reverse=True)

    print(f"\n  SHARED emphasis (both model and news highlight these beyond abstract):")
    print(f"  {'Word':<20s} {'Model Z':>9s} {'News Z':>9s}")
    print(f"  {'-'*40}")
    for w in shared_ranked[:25]:
        mz = model_vs_abs.get(w, {}).get("z", 0)
        nz = news_vs_abs.get(w, {}).get("z", 0)
        print(f"  {w:<20s} {mz:>9.1f} {nz:>9.1f}")

    # Model-only: model emphasizes but news doesn't
    model_only_ranked = sorted(model_only, key=lambda w: -model_vs_abs.get(w, {}).get("z", 0))
    print(f"\n  MODEL-ONLY emphasis (model highlights, news doesn't):")
    print(f"  {', '.join(model_only_ranked[:25])}")

    # News-only: news emphasizes but model doesn't
    news_only_ranked = sorted(news_only, key=lambda w: -news_vs_abs.get(w, {}).get("z", 0))
    print(f"\n  NEWS-ONLY emphasis (news highlights, model doesn't):")
    print(f"  {', '.join(news_only_ranked[:25])}")

    # ============================================================
    # 4. Per-field comparison
    # ============================================================
    print(f"\n{'='*60}")
    print("4. PER-FIELD: Jaccard similarity between model and news emphasis")
    print("=" * 60)

    from collections import defaultdict
    field_triplets = defaultdict(list)
    for t in triplets:
        field_triplets[t["field"]].append(t)

    print(f"\n  {'Field':<45s} {'N':>5s} {'Jaccard':>8s}")
    print(f"  {'-'*60}")

    for field in sorted(field_triplets.keys(), key=lambda x: -len(field_triplets[x])):
        ft = field_triplets[field]
        if len(ft) < 30:
            continue
        fn = len(ft)
        f_abs = get_doc_freq([t["abstract"] for t in ft])
        f_exp = get_doc_freq([t["explanation"] for t in ft])
        f_news = get_doc_freq([t["news"] for t in ft])

        f_model_lor = log_odds_ratio(f_exp, f_abs, fn, fn, min_freq=5)
        f_news_lor = log_odds_ratio(f_news, f_abs, fn, fn, min_freq=5)

        f_model_sig = {w for w, info in f_model_lor.items() if info["z"] > 2.0}
        f_news_sig = {w for w, info in f_news_lor.items() if info["z"] > 2.0}
        f_union = f_model_sig | f_news_sig
        f_jaccard = len(f_model_sig & f_news_sig) / len(f_union) if f_union else 0

        print(f"  {field:<45s} {fn:>5d} {f_jaccard:>8.4f}")

    # Save
    results = {
        "n_triplets": len(triplets),
        "model_emphasis_top50": [(w, model_vs_abs[w]) for w in sorted(model_vs_abs, key=lambda x: -model_vs_abs[x]["z"])[:50]],
        "news_emphasis_top50": [(w, news_vs_abs[w]) for w in sorted(news_vs_abs, key=lambda x: -news_vs_abs[x]["z"])[:50]],
        "shared_emphasis": shared_ranked[:50],
        "model_only_emphasis": model_only_ranked[:50],
        "news_only_emphasis": news_only_ranked[:50],
        "jaccard": round(jaccard, 4),
    }

    out_path = ANALYSIS_DIR / "triplet_signals.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
