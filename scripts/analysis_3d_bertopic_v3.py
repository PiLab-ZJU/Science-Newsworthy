"""
Analysis 3d (v3): JOINT BERTopic across model rationales and news articles.

v3 changes vs v2:
  - We now fit a SINGLE BERTopic model on the union of three provenances:
        (1) YES: TP rationales whose papers have verified news coverage
        (2) NO : FN rationales whose papers have verified news coverage
        (3) NEWS: matched news articles for (1) ∪ (2)
  - Every topic therefore lives in the same semantic space. The per-panel
    bar chart becomes a share decomposition of the SAME topics across the
    three provenances — apples-to-apples.

Output:
    analysis/bertopic_results_v3.json
        {
          "topics": [{id, words, total, share_by_src: {yes, no, news}}, ...],
          "totals": {yes_n, no_n, news_n},
          ...
        }

Usage:
    OPENBLAS_NUM_THREADS=8 python scripts/analysis_3d_bertopic_v3.py
"""
import os
import sys
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

PROJECT_DIR = Path("/root/pilab_jiang/cxlin/academic_new_policy")
ANALYSIS_DIR = PROJECT_DIR / "analysis"
NEWS_PATH = PROJECT_DIR / "data" / "raw" / "news_text" / "news_articles.json"
ST_MODEL_PATH = Path("/root/pilab_jiang/hf-model/all-MiniLM-L6-v2")

EXTRA_STOPWORDS = {
    "paper", "papers", "news", "coverage", "mainstream",
    "media", "receive", "received", "attention", "newsworthy",
    "mention", "mentioned",
    "study", "studies", "research", "researcher", "researchers",
    "scientific", "science", "scientist", "scientists",
    "article", "articles", "published", "publishing",
    "academic", "academia",
    "technical", "specialized", "field", "fields",
    "appeal", "appealing", "attract", "attracts",
    "topic", "topics", "subject",
    "discuss", "discusses", "discussed", "describes", "described",
    "involves", "involving", "presents", "discussing",
    "significant", "significantly", "important", "interesting",
    "fascinating", "relevant", "relevance",
    "paper receive", "news coverage", "receive news",
    "mainstream media", "media attention",
    "due", "via", "may", "might", "likely",
}


def build_vectorizer(min_df: int = 3):
    stops = list(ENGLISH_STOP_WORDS) + sorted(EXTRA_STOPWORDS)
    return CountVectorizer(stop_words=stops, ngram_range=(1, 2), min_df=min_df)


def verified_news_doi(predictions, doi_news):
    verified = {}
    for p in predictions:
        if p["true_label"] != 1:
            continue
        art = doi_news.get(p["doi"])
        if not art:
            continue
        text = art.get("text", "") or ""
        title = (p.get("title") or "").strip()
        if not (p["doi"].lower() in text.lower() or
                (title and title.lower() in text.lower())):
            continue
        if not text:
            continue
        verified[p["doi"]] = text
    return verified


def main():
    # ---- load ----
    with open(ANALYSIS_DIR / "test_predictions_with_explanations.json") as f:
        predictions = json.load(f)

    tp = [p for p in predictions
          if p["true_label"] == 1 and p["predicted"] == 1]
    fn = [p for p in predictions
          if p["true_label"] == 1 and p["predicted"] == 0]

    with open(NEWS_PATH) as f:
        news = json.load(f)
    doi_news = {a["doi"]: a for a in news if a.get("success")}
    verified = verified_news_doi(predictions, doi_news)
    print(f"verified news DOIs: {len(verified)}")

    tp_matched = [p for p in tp if p["doi"] in verified]
    fn_matched = [p for p in fn if p["doi"] in verified]
    print(f"TP w/ news = {len(tp_matched)}   FN w/ news = {len(fn_matched)}")

    # ---- assemble joint corpus with provenance labels ----
    texts = []
    labels = []  # "yes" | "no" | "news"

    for p in tp_matched:
        if p.get("explanation"):
            texts.append(p["explanation"])
            labels.append("yes")
    for p in fn_matched:
        if p.get("explanation"):
            texts.append(p["explanation"])
            labels.append("no")
    for p in tp_matched + fn_matched:
        text = verified.get(p["doi"], "")
        if not text:
            continue
        # Cap news articles at 500 tokens so one long piece does not dominate
        texts.append(" ".join(text.split()[:500]))
        labels.append("news")

    labels = np.array(labels)
    counts_by_src = Counter(labels)
    print(f"joint corpus: {len(texts)}  "
          f"(yes={counts_by_src['yes']}, "
          f"no={counts_by_src['no']}, "
          f"news={counts_by_src['news']})")

    # ---- run a single BERTopic on the joint corpus ----
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN

    print("\nencoding with MiniLM...")
    st_model = SentenceTransformer(str(ST_MODEL_PATH))
    emb = st_model.encode(texts, batch_size=256, show_progress_bar=True)

    umap_m = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                  metric="cosine", random_state=42)
    hdbscan_m = HDBSCAN(min_cluster_size=50, min_samples=10,
                        prediction_data=True)
    vectorizer = build_vectorizer(min_df=3)
    ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)

    model = BERTopic(
        umap_model=umap_m,
        hdbscan_model=hdbscan_m,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
        nr_topics="auto",
        verbose=True,
    )
    topic_ids, _ = model.fit_transform(texts, embeddings=emb)
    topic_ids = np.array(topic_ids)

    info = model.get_topic_info()
    print("\n--- topic summary (joint) ---")
    print(f"{'Topic':>5s} {'Count':>6s}  Words")
    for _, row in info.iterrows():
        tag = "[Outliers]" if row["Topic"] == -1 else row["Name"][:70]
        print(f"  {row['Topic']:>3d} {row['Count']:>6d}  {tag}")

    # ---- per-topic breakdown by provenance ----
    topics_out = []
    for tid in sorted(set(topic_ids)):
        if tid == -1:
            continue
        mask = topic_ids == tid
        yes_n = int(((labels == "yes") & mask).sum())
        no_n  = int(((labels == "no")  & mask).sum())
        news_n = int(((labels == "news") & mask).sum())
        total = yes_n + no_n + news_n
        words_scored = model.get_topic(int(tid))
        if not words_scored:
            continue
        topics_out.append({
            "id": int(tid),
            "words": [w for w, _ in words_scored[:8]],
            "total": total,
            "yes_n": yes_n,
            "no_n":  no_n,
            "news_n": news_n,
            # normalized share WITHIN each provenance (for the per-panel bars)
            "yes_share": yes_n / max(counts_by_src["yes"], 1),
            "no_share":  no_n  / max(counts_by_src["no"], 1),
            "news_share": news_n / max(counts_by_src["news"], 1),
        })
    topics_out.sort(key=lambda t: -t["total"])

    # ---- save ----
    results = {
        "topics": topics_out,
        "totals": {
            "yes_n":  counts_by_src["yes"],
            "no_n":   counts_by_src["no"],
            "news_n": counts_by_src["news"],
        },
        "composition": {
            "yes":  "TP rationales with verified news (model correctly YES)",
            "no":   "FN rationales with verified news (model wrongly NO)",
            "news": "news articles of TP∪FN matched papers",
        },
        "stopwords_extra": sorted(EXTRA_STOPWORDS),
    }
    out_path = ANALYSIS_DIR / "bertopic_results_v3.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
