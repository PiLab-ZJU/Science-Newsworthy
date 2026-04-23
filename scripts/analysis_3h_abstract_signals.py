"""
Analysis 3h: Three-way contrastive signal analysis based on abstracts.

1. Model signals: Abstracts predicted YES vs predicted NO
2. News signals: News articles vs corresponding abstracts
3. Cross-validation: overlap between model signals and news signals
4. Error analysis: TP vs FN abstracts, TN vs FP abstracts

Usage:
    OPENBLAS_NUM_THREADS=8 python scripts/analysis_3h_abstract_signals.py
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
    "using", "used", "based", "show", "shown", "found", "results",
    "method", "approach", "propose", "proposed",
}


def tokenize(text):
    return [w for w in re.findall(r'\b[a-z]{3,}\b', text.lower()) if w not in STOPWORDS]


def log_odds_ratio(counts_a, counts_b, n_a, n_b, prior=1.0, min_freq=10):
    results = {}
    all_words = set(counts_a.keys()) | set(counts_b.keys())
    for word in all_words:
        fa = counts_a.get(word, 0) + prior
        fb = counts_b.get(word, 0) + prior
        total = counts_a.get(word, 0) + counts_b.get(word, 0)
        if total < min_freq:
            continue
        odds_a = fa / (n_a - fa + prior)
        odds_b = fb / (n_b - fb + prior)
        lor = math.log2(odds_a / odds_b)
        var = 1.0 / fa + 1.0 / fb
        z = lor / math.sqrt(var)
        results[word] = {"lor": round(lor, 4), "z": round(z, 2),
                         "freq_a": counts_a.get(word, 0),
                         "freq_b": counts_b.get(word, 0)}
    return results


def get_doc_freq(texts):
    counter = Counter()
    for t in texts:
        counter.update(set(tokenize(t)))
    return counter


def print_top(results, name_a, name_b, n=25):
    sorted_w = sorted(results.items(), key=lambda x: x[1]["z"], reverse=True)
    print(f"\n  Top '{name_a}' words (what distinguishes {name_a}):")
    print(f"  {'Word':<22s} {'Z':>7s} {name_a:>7s} {name_b:>7s}")
    print(f"  {'-'*45}")
    for w, info in sorted_w[:n]:
        print(f"  {w:<22s} {info['z']:>7.1f} {info['freq_a']:>7d} {info['freq_b']:>7d}")

    print(f"\n  Top '{name_b}' words (what distinguishes {name_b}):")
    print(f"  {'Word':<22s} {'Z':>7s} {name_a:>7s} {name_b:>7s}")
    print(f"  {'-'*45}")
    for w, info in sorted_w[-n:]:
        print(f"  {w:<22s} {info['z']:>7.1f} {info['freq_a']:>7d} {info['freq_b']:>7d}")


def overlap_analysis(results_a, results_b, name_a, name_b, z_thresh=3.0):
    set_a = {w for w, info in results_a.items() if info["z"] > z_thresh}
    set_b = {w for w, info in results_b.items() if info["z"] > z_thresh}
    shared = set_a & set_b
    only_a = set_a - set_b
    only_b = set_b - set_a
    union = set_a | set_b
    jaccard = len(shared) / len(union) if union else 0

    print(f"\n  {name_a} signal words (z>{z_thresh}): {len(set_a)}")
    print(f"  {name_b} signal words (z>{z_thresh}): {len(set_b)}")
    print(f"  Shared: {len(shared)}  |  {name_a}-only: {len(only_a)}  |  {name_b}-only: {len(only_b)}")
    print(f"  Jaccard similarity: {jaccard:.4f}")

    # Sort shared by combined z-score
    shared_ranked = sorted(shared, key=lambda w: results_a.get(w, {}).get("z", 0) + results_b.get(w, {}).get("z", 0), reverse=True)
    print(f"\n  Shared signal words (top 30): {', '.join(shared_ranked[:30])}")
    print(f"  {name_a}-only (top 20): {', '.join(sorted(only_a, key=lambda w: -results_a.get(w,{}).get('z',0))[:20])}")
    print(f"  {name_b}-only (top 20): {', '.join(sorted(only_b, key=lambda w: -results_b.get(w,{}).get('z',0))[:20])}")

    return {"shared": sorted(shared_ranked[:50]), "only_a": sorted(list(only_a)[:50]),
            "only_b": sorted(list(only_b)[:50]), "jaccard": round(jaccard, 4)}


def main():
    # Load predictions (with abstracts from processed data)
    with open(ANALYSIS_DIR / "test_predictions_with_explanations.json") as f:
        predictions = json.load(f)

    # Load abstracts
    from pathlib import Path
    proc = Path("/home/ubuntu/pilab_jiang/lcx/academic_new_policy/data/processed/combined/test.json")
    with open(proc) as f:
        test_data = json.load(f)
    doi_abstract = {d["doi"]: d.get("abstract", "") for d in test_data}

    # Attach abstracts to predictions
    for p in predictions:
        p["abstract"] = doi_abstract.get(p.get("doi", ""), "")

    tp = [p for p in predictions if p["true_label"] == 1 and p["predicted"] == 1]
    tn = [p for p in predictions if p["true_label"] == 0 and p["predicted"] == 0]
    fp = [p for p in predictions if p["true_label"] == 0 and p["predicted"] == 1]
    fn = [p for p in predictions if p["true_label"] == 1 and p["predicted"] == 0]
    print(f"TP={len(tp)} TN={len(tn)} FP={len(fp)} FN={len(fn)}")

    # ============================================================
    # 1. Model signals: abstracts predicted YES vs predicted NO
    # ============================================================
    print(f"\n{'='*60}")
    print("1. MODEL SIGNALS: Abstracts of YES vs NO predictions")
    print("=" * 60)

    yes_abstracts = [p["abstract"] for p in tp + fp if p["abstract"]]
    no_abstracts = [p["abstract"] for p in tn + fn if p["abstract"]]
    print(f"  YES abstracts: {len(yes_abstracts)}, NO abstracts: {len(no_abstracts)}")

    yes_freq = get_doc_freq(yes_abstracts)
    no_freq = get_doc_freq(no_abstracts)
    model_lor = log_odds_ratio(yes_freq, no_freq, len(yes_abstracts), len(no_abstracts))
    print_top(model_lor, "YES", "NO")

    # ============================================================
    # 2. News signals: news text vs corresponding abstract
    # ============================================================
    print(f"\n{'='*60}")
    print("2. NEWS SIGNALS: News articles vs corresponding abstracts")
    print("=" * 60)

    with open(NEWS_PATH) as f:
        news = json.load(f)
    doi_news = {a["doi"]: a for a in news if a.get("success")}
    with open(PAPERS_PATH) as f:
        papers = json.load(f)
    doi_title = {p["doi"]: p.get("title", "") for p in papers}

    news_texts = []
    paired_abstracts = []
    for p in tp:
        if p["doi"] in doi_news and p["abstract"]:
            article = doi_news[p["doi"]]
            text = article.get("text", "")
            title = doi_title.get(p["doi"], "")
            if p["doi"].lower() in text.lower() or (title and title.lower() in text.lower()):
                news_texts.append(text[:2000])
                paired_abstracts.append(p["abstract"])

    print(f"  Matched pairs: {len(news_texts)}")

    news_freq = get_doc_freq(news_texts)
    abs_freq = get_doc_freq(paired_abstracts)
    news_lor = log_odds_ratio(news_freq, abs_freq, len(news_texts), len(paired_abstracts))
    print_top(news_lor, "News", "Abstract")

    # ============================================================
    # 3. Cross-validation
    # ============================================================
    print(f"\n{'='*60}")
    print("3. CROSS-VALIDATION: Model abstract signals vs News added signals")
    print("=" * 60)

    cross = overlap_analysis(model_lor, news_lor, "Model", "News")

    # ============================================================
    # 4. Error analysis: TP vs FN, TN vs FP
    # ============================================================
    print(f"\n{'='*60}")
    print("4a. ERROR: TP abstracts vs FN abstracts (both actually reported)")
    print("=" * 60)

    tp_abs = [p["abstract"] for p in tp if p["abstract"]]
    fn_abs = [p["abstract"] for p in fn if p["abstract"]]
    print(f"  TP: {len(tp_abs)}, FN: {len(fn_abs)}")

    tp_freq = get_doc_freq(tp_abs)
    fn_freq = get_doc_freq(fn_abs)
    tp_fn_lor = log_odds_ratio(tp_freq, fn_freq, len(tp_abs), len(fn_abs))
    print_top(tp_fn_lor, "TP(caught)", "FN(missed)", n=20)

    print(f"\n{'='*60}")
    print("4b. ERROR: TN abstracts vs FP abstracts (both actually NOT reported)")
    print("=" * 60)

    tn_abs = [p["abstract"] for p in tn if p["abstract"]]
    fp_abs = [p["abstract"] for p in fp if p["abstract"]]
    print(f"  TN: {len(tn_abs)}, FP: {len(fp_abs)}")

    tn_freq = get_doc_freq(tn_abs)
    fp_freq = get_doc_freq(fp_abs)
    tn_fp_lor = log_odds_ratio(tn_freq, fp_freq, len(tn_abs), len(fp_abs))
    print_top(tn_fp_lor, "TN(correct)", "FP(fooled)", n=20)

    # Save all results
    def top_n(lor_dict, n=50):
        s = sorted(lor_dict.items(), key=lambda x: x[1]["z"], reverse=True)
        return {"positive": [(w, info) for w, info in s[:n]],
                "negative": [(w, info) for w, info in s[-n:]]}

    results = {
        "model_signals": top_n(model_lor),
        "news_signals": top_n(news_lor),
        "cross_validation": cross,
        "tp_vs_fn": top_n(tp_fn_lor, 30),
        "tn_vs_fp": top_n(tn_fp_lor, 30),
    }

    out_path = ANALYSIS_DIR / "abstract_signals.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
