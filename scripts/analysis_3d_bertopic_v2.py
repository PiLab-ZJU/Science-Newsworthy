"""
Analysis 3d (v2): BERTopic on model explanations and news articles.

v2 improvements over v1:
  - Paths patched for a800 ( /root/pilab_jiang/cxlin/... )
  - Extended stopword list filters out boilerplate task-meta tokens
    ('paper', 'news', 'coverage', 'mainstream', 'media', 'study', …)
    so T0 no longer swallows 45–75% of each corpus as "paper, news, coverage"
  - Drops the need for news_papers.json (use title from the predictions file)

Usage:
    OPENBLAS_NUM_THREADS=8 python scripts/analysis_3d_bertopic_v2.py
"""
import os
import sys
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

# ---- paths (a800) ----
PROJECT_DIR = Path("/root/pilab_jiang/cxlin/academic_new_policy")
ANALYSIS_DIR = PROJECT_DIR / "analysis"
NEWS_PATH = PROJECT_DIR / "data" / "raw" / "news_text" / "news_articles.json"
ST_MODEL_PATH = Path("/root/pilab_jiang/hf-model/all-MiniLM-L6-v2")

# ---- extended stoplist (boilerplate + task-meta tokens) ----
EXTRA_STOPWORDS = {
    # prompt echoes & classification boilerplate
    "paper", "papers", "news", "coverage", "mainstream",
    "media", "receive", "received", "attention", "newsworthy",
    "mention", "mentioned",
    # generic research meta
    "study", "studies", "research", "researcher", "researchers",
    "scientific", "science", "scientist", "scientists",
    "article", "articles", "published", "publishing",
    "academic", "academia",
    # generic descriptors overused by both sides
    "technical", "specialized", "field", "fields",
    "appeal", "appealing", "attract", "attracts",
    "topic", "topics", "subject",
    # hedge verbs every rationale uses
    "discuss", "discusses", "discussed", "describes", "described",
    "involves", "involving", "presents", "discussing",
    # meta-adjectives
    "significant", "significantly", "important", "interesting",
    "fascinating", "relevant", "relevance",
    # misc connectors
    "paper receive", "news coverage", "receive news",
    "mainstream media", "media attention",
    "due", "via", "may", "might", "likely",
}


def build_vectorizer(min_df: int = 2):
    # CountVectorizer wants list-like; combine sklearn + extras
    stops = list(ENGLISH_STOP_WORDS) + sorted(EXTRA_STOPWORDS)
    return CountVectorizer(stop_words=stops, ngram_range=(1, 2),
                           min_df=min_df)


def run_bertopic(texts, name="", min_topic_size=100, min_df=2):
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN

    print(f"\n[{name}] loading MiniLM and encoding {len(texts)} texts...")
    st_model = SentenceTransformer(str(ST_MODEL_PATH))
    embeddings = st_model.encode(texts, batch_size=256, show_progress_bar=True)

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                       metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=min_topic_size,
                             min_samples=10, prediction_data=True)
    vectorizer = build_vectorizer(min_df=min_df)
    ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)

    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
        nr_topics="auto",
        verbose=True,
    )
    topics, _ = model.fit_transform(texts, embeddings=embeddings)

    info = model.get_topic_info()
    print(f"\n[{name}] {len(set(topics)) - 1} topics after HDBSCAN+reduce")
    print(f"{'Topic':>6s} {'Count':>7s} {'Name':<65s}")
    print("-" * 80)
    for _, row in info.iterrows():
        tag = "[Outliers]" if row["Topic"] == -1 else row["Name"][:65]
        print(f"  {row['Topic']:>4d} {row['Count']:>7d}  {tag}")

    return model, topics, embeddings


def topic_summary(model, tid):
    """Extract a topic's top words and document count."""
    words = model.get_topic(tid)
    if not words:
        return None
    info = model.get_topic_info()
    row = info[info["Topic"] == tid]
    if row.empty:
        return None
    return {
        "id": int(tid),
        "words": [w for w, _ in words[:8]],
        "count": int(row["Count"].values[0]),
    }


def collect_topics(model):
    out = []
    info = model.get_topic_info()
    for tid in info["Topic"]:
        if tid == -1:
            continue
        s = topic_summary(model, int(tid))
        if s:
            out.append(s)
    return out


def verified_news_doi(predictions):
    """Return the set of DOIs whose paper has a verified matched news article."""
    with open(NEWS_PATH) as f:
        news = json.load(f)
    doi_news = {a["doi"]: a for a in news if a.get("success")}
    verified = {}
    for p in predictions:
        if p["true_label"] != 1:  # news-covered means actual positive
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
    return verified, doi_news


def main():
    # ---- load predictions ----
    with open(ANALYSIS_DIR / "test_predictions_with_explanations.json") as f:
        predictions = json.load(f)

    tp = [p for p in predictions
          if p["true_label"] == 1 and p["predicted"] == 1]
    tn = [p for p in predictions
          if p["true_label"] == 0 and p["predicted"] == 0]
    fp = [p for p in predictions
          if p["true_label"] == 0 and p["predicted"] == 1]
    fn = [p for p in predictions
          if p["true_label"] == 1 and p["predicted"] == 0]
    print(f"TP={len(tp)} TN={len(tn)} FP={len(fp)} FN={len(fn)}")

    # ---- build the verified-news DOI table (same filter as §5.4.2/§5.4.3) --
    verified_news_text, _ = verified_news_doi(predictions)
    print(f"DOIs with verified news text: {len(verified_news_text)}")

    # ---- YES = TP that also have a verified news article --------------------
    tp_matched = [p for p in tp if p["doi"] in verified_news_text]
    yes_texts = [p["explanation"] for p in tp_matched if p.get("explanation")]
    print(f"\nYES panel — TP with news (model correctly flagged): {len(yes_texts)}")

    # ---- NO = FN that also have a verified news article --------------------
    fn_matched = [p for p in fn if p["doi"] in verified_news_text]
    no_texts = [p["explanation"] for p in fn_matched if p.get("explanation")]
    print(f"NO panel  — FN with news (model wrongly missed): {len(no_texts)}")

    # ---- News articles for TP+FN with verified news (symmetric) -----------
    combined_matched = tp_matched + fn_matched
    news_texts = []
    for p in combined_matched:
        text = verified_news_text.get(p["doi"], "")
        if not text:
            continue
        news_texts.append(" ".join(text.split()[:500]))
    print(f"News panel (TP+FN matched articles): {len(news_texts)}")

    # ---- fit BERTopic on all three corpora --------------------------------
    yes_model, _, _ = run_bertopic(
        yes_texts, "YES explanations (TP w/ news)", min_topic_size=60)
    no_model, _, _ = run_bertopic(
        no_texts, "NO explanations (FN w/ news)", min_topic_size=20)
    news_model = None
    if len(news_texts) > 500:
        news_model, _, _ = run_bertopic(
            news_texts, "NEWS articles (TP+FN)", min_topic_size=50)

    # ---- save ----
    results = {
        "yes_topics": collect_topics(yes_model),
        "no_topics": collect_topics(no_model),
        "news_topics": collect_topics(news_model) if news_model else [],
        "totals": {
            "yes_n": len(yes_texts),
            "no_n": len(no_texts),
            "news_n": len(news_texts),
        },
        "composition": {
            "yes_source": "TP with verified news article (model correctly said YES)",
            "no_source":  "FN with verified news article (model wrongly said NO)",
            "news_source": "news articles for the union of TP-w/-news and FN-w/-news",
        },
        "stopwords_extra": sorted(EXTRA_STOPWORDS),
    }

    out_path = ANALYSIS_DIR / "bertopic_results_v2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
